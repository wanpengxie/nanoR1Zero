import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim
from transformers import GPT2LMHeadModel, GPT2Model, BertTokenizer, Qwen2PreTrainedModel, Qwen2Tokenizer
from transformers import AutoModel, AutoTokenizer
import time
import numpy as np
from nanoRL4GPT.lm_grpo import GRPO, softmax_fn
from nanoRL4GPT.lm_policy import PolicyModel
from nanoRL4GPT.reward import BaseReward, CounterReward, MathReward
from nanoRL4GPT.collector import GRPOCollector
import tqdm
from torch.utils.tensorboard import SummaryWriter
from nanoRL4GPT.data import DataLoader



if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 2
    epoch = 10
    inner_epoch = 1
    kl_coe = 0.1
    clip = 0.2
    max_sentence_len = 1024*8
    max_prompt_len = 1024
    train_batch = 128
    lr = 5e-6
    buffer = 4
    device = "cuda"
    value_coe = 0.1
    entropy_coe = 0.005
    max_grad_norm = 0.5
    number_responses = 10


    reward = MathReward()

    model_path = '/hy-tmp/Qwen2.5-1.5B-Instruct'
    update_path = '/hy-tmp/Qwen2.5-1.5B-Instruct-update'
    base_model = AutoModel.from_pretrained(model_path)
    ref_model = AutoModel.from_pretrained(model_path)
    gen_model = AutoModel.from_pretrained(model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    policy_model = PolicyModel(base_model, ref_model, gen_model, update_path).to(device)
    reward_model = MathReward()

    ppo = GRPO(policy_model, reward, clip, logit_post_fn=softmax_fn(mask_ids=[0, 100]))
    params = list(policy_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    collector = GRPOCollector(buffer, kl_coe, eos_token=tokenizer.eos_token_id)

    dataset = DataLoader('data/test_data.json', batch_size=2)

    sample_step = 0
    train_step = 0

    policy_model.save_policy_model()

    for i in range(epoch):
        for prompts in dataset:
            sample_step += 1

            print (f'start sample {sample_step}----------------------------')
            policy_model.eval()
            policy_model.start_vllm_server()
            for prompt in prompts:
                prompt_text = prompt['text']
                answer_text = prompt['answer']
                input_ids, gen_log_probs, ref_log_probs, start_index = ppo.generate_episode(
                    None, 
                    max_sentence_len, 
                    number_responses=number_responses, 
                    eos_token=tokenizer.eos_token_id,
                    prompt=prompt_text
                )  

                response_texts = tokenizer.batch_decode(input_ids[:, start_index:], skip_special_tokens=True)
                rewards = []
                for response in response_texts:
                    reward = reward_model.rule_reward(response, answer_text)
                    rewards.append(reward)
                rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-6)

                for i in range(number_responses):
                    eos_index = (input_ids[i, start_index:] == tokenizer.eos_token_id).nonzero(as_tuple=True)
                    if len(eos_index) == 0:
                        continue
                    else:
                        eos_index = eos_index[0].item() + start_index
                    episode = (prompt_text, response_texts[i], input_ids[i][:eos_index].tolist(), gen_log_probs[i][:eos_index-start_index+1].tolist(), ref_log_probs[i][:eos_index-start_index+1].tolist(), start_index, rewards[i])
                    collector.add_buffer([episode])
            
            print (f'end sample {sample_step}----------------------------')
            policy_model.stop_vllm_server()
            print ("episodes summary info: ", collector.summary())
            policy_model.train()
            for samples in collector.sample(inner_epoch, batch=train_batch, device=device):
                train_step += 1
                policy_loss, entropy_loss = ppo.forward(samples)
                loss = policy_loss + entropy_loss * entropy_coe
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                opt.zero_grad()
            collector.reset()
            policy_model.save_policy_model()

            # cur_rewards = torch.mean(torch.stack([x[-1] for x in episodes])).detach().item()
