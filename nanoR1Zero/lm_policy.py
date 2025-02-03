# LM models of actor critor model
import torch.nn
from transformers import Qwen2PreTrainedModel, GenerationConfig, Qwen2ForCausalLM
from torch import nn
import requests
import json
import subprocess
import time

class PolicyModel(nn.Module):
    def __init__(self, policy: Qwen2ForCausalLM, ref: Qwen2ForCausalLM, gen_model: Qwen2ForCausalLM=None, model_path: str=None):
        super().__init__()
        self.vllm_process = None
        self.devices = []
        self.worker_urls = []

        self.policy_model = policy
        self.ref_model = ref
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.gen_model = gen_model
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

        empty_generation_config = GenerationConfig(**{
            "bos_token_id": 151643,
            "do_sample": True,
            "eos_token_id": [
                151645,
                151643
            ],
            "pad_token_id": 151643,
        })
        self.policy_model.generation_config = empty_generation_config
        self.model_path = model_path

    def sync_policy_model(self):
        policy_state = self.policy_model.state_dict()
        self.gen_model.load_state_dict(policy_state)
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.policy_model = self.policy_model.to('cuda')
        self.policy_model.train(mode)
        self.ref_model.eval()
        torch.cuda.empty_cache()

    def eval(self):
        self.policy_model = self.policy_model.cpu()
        self.policy_model.eval()
        self.ref_model.eval()
        torch.cuda.empty_cache()

    def generate(self, input_ids, max_len=128, number_responses=4, eos_token=0, logit_fn=torch.nn.Softmax(dim=-1)):
        start_index = input_ids.shape[1]
        outputs = self.policy_model.generate(
            input_ids,
            max_length=max_len,
            do_sample=True,
            num_return_sequences=number_responses,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True
        )
        logits = torch.stack(outputs.logits, dim=1)
        return outputs.sequences, logits, start_index

    def generate_vllm(self, prompt_token_ids=None, output_token_ids_list=None, eos_token=0):
        max_len = max([len(output_token_ids) for output_token_ids in output_token_ids_list])
        outputs = [prompt_token_ids + output_token_ids + [eos_token] * (max_len - len(output_token_ids)) for output_token_ids in output_token_ids_list]                
        input_ids = torch.LongTensor(outputs)
        start_index = len(prompt_token_ids)
        token_gen_probs = self.calc_probs(input_ids, start_index, batch_size=8, model='gen')
        token_ref_probs = self.calc_probs(input_ids, start_index, batch_size=8, model='ref')
        return input_ids, token_gen_probs, token_ref_probs, start_index
    
    def forward_policy(self, input_ids):
        logits = self.policy_model.forward(input_ids).logits
        return logits

    def get_input_logits(self, input_ids, logits):
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1: , None]
        logits = torch.gather(logits, 2, targets).view(input_ids.shape[0], input_ids.shape[1]-1)
        return logits

    def calc_probs(self, input_ids, start_index, batch_size=16, model='ref'):
        if model == 'ref':
            model = self.ref_model
        else:
            model = self.gen_model
        input_batch = input_ids.shape[0]
        token_probs_all = None
        if input_batch > batch_size:
            for i in range(0, input_batch, batch_size):
                input_ids_batch = input_ids[i:i+batch_size].to(model.device)
                logits = model.forward(input_ids_batch).logits[:, start_index-1:-1, :]
                probs = torch.nn.Softmax(dim=-1)(logits)
                token_probs = torch.log(torch.gather(probs, -1, input_ids_batch[:, start_index:, None])).squeeze(-1).detach().cpu()
                if i == 0:
                    token_probs_all = token_probs
                else:
                    token_probs_all = torch.cat([token_probs_all, token_probs], dim=0)
        else:
            input_ids_batch = input_ids.to(model.device)
            logits = model.forward(input_ids_batch).logits[:, start_index-1:-1, :]
            probs = torch.nn.Softmax(dim=-1)(logits)
            token_probs = torch.log(torch.gather(probs, -1, input_ids_batch[:, start_index:, None])).squeeze(-1).detach().cpu()
            token_probs_all = token_probs
        return token_probs_all

    def detect_vllm_server(self):
        is_ready = True
        for url in self.worker_urls:
            try:
                response = requests.get(f'{url}/ping')
                print (f'{url} status: {response.json()}')
                if response.status_code == 200 and response.json()['status'] == 'ok':
                    pass
                else:
                    is_ready = False
            except Exception as e:
                print (f'{url} error: {e}')
                is_ready = False
        return is_ready
    
    def start_vllm_server(self, path, devices=[]):
        # 利用subprocess启动vllm_server，并返回进程，以供后续停止，查看启动状态
        self.vllm_process = []
        for device, port in devices:
            log_file = open(f'vllm_server_{device}.log', 'w')
            self.vllm_process.append(subprocess.Popen(['python', './nanoR1Zero/vllm_server.py', path, str(device), str(port)], stdout=log_file, stderr=log_file))
            self.worker_urls.append(f'http://localhost:{port}')
        self.devices = devices

        # 等待vllm_server启动，等待5分钟timeout
        for i in range(30):
            if self.detect_vllm_server():
                print ("vllm_server started")
                return
            time.sleep(10)

    def stop_vllm_server(self):
        # 强制停止进程
        try:
            for process in self.vllm_process:
                process.kill()
        except:
            pass

        try:
            for url in self.worker_urls:
                requests.get(f'{url}/stop')
        except:
            pass

        self.vllm_process = None
        self.devices = []
        self.worker_urls = []

    def is_vllm_server_running(self):
        return self.vllm_process is not None

    def save_policy_model(self, path):
        policy_state = self.policy_model.state_dict()
        self.policy_model.save_pretrained(path)
        self.gen_model.load_state_dict(policy_state)
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    def load_policy_model(self, path):
        self.policy_model.load_state_dict(torch.load(path + '/policy_model.pth'))
        self.sync_policy_model()
