# LM models of actor critor model
import os
import torch.nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import requests
import subprocess
import time

class PolicyModel(nn.Module):
    def __init__(self, model_path: str=None, torch_dtype=torch.bfloat16, device=None):
        super().__init__()
        self.model_path = model_path
        self.update_path = f'{model_path}-update'
        self.device = device

        self.ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if os.path.exists(self.update_path):
            self.policy_model = AutoModelForCausalLM.from_pretrained(self.update_path, torch_dtype=torch_dtype, use_cache=False)
            self.gen_model = AutoModelForCausalLM.from_pretrained(self.update_path, torch_dtype=torch_dtype)
        else:
            self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch_dtype, use_cache=False)
            self.gen_model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch_dtype)
            self.policy_model.save_pretrained(self.update_path)
            self.tokenizer.save_pretrained(self.update_path)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    def sync_policy_model(self):
        policy_state = self.policy_model.state_dict()
        self.gen_model.load_state_dict(policy_state)
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.policy_model = self.policy_model.to(self.device)
        self.policy_model.gradient_checkpointing_enable()
        self.policy_model.train(mode)


    def eval(self):
        self.policy_model = self.policy_model.cpu()
        self.policy_model.gradient_checkpointing_disable()
        self.policy_model.eval()
        

    def generate_probs(self, prompt_token_ids=None, output_token_ids_list=None, eos_token=0):
        max_len = max([len(output_token_ids) for output_token_ids in output_token_ids_list])
        outputs = [prompt_token_ids + output_token_ids + [eos_token] * (max_len - len(output_token_ids)) for output_token_ids in output_token_ids_list]
        input_ids = torch.LongTensor(outputs)
        start_index = len(prompt_token_ids)
        token_gen_probs = self.calc_probs(input_ids, start_index, batch_size=2, model='gen')
        token_ref_probs = self.calc_probs(input_ids, start_index, batch_size=2, model='ref')
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


    def save_policy_model(self):
        policy_state = self.policy_model.state_dict()
        self.policy_model.save_pretrained(self.update_path)
        self.gen_model.load_state_dict(policy_state)
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    def load_policy_model(self, path):
        self.policy_model.load_state_dict(torch.load(path + '/policy_model.pth'))
        self.sync_policy_model()