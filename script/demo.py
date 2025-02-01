import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim
from transformers import GPT2LMHeadModel, GPT2Model, BertTokenizer, Qwen2PreTrainedModel, Qwen2Tokenizer, Qwen2ForCausalLM
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import time
import numpy as np
from nanoRL4GPT.lm_grpo import GRPO, softmax_fn
from nanoRL4GPT.lm_policy import PolicyModel
from nanoRL4GPT.reward import BaseReward, CounterReward, MathReward
from nanoRL4GPT.collector import GRPOCollector
import tqdm
from torch.utils.tensorboard import SummaryWriter
from nanoRL4GPT.data import DataLoader
import wandb  # 添加wandb导入



if __name__ == "__main__":
    import sys 
    device = sys.argv[1]
    # 初始化wandb
    # api: 
    wandb.init(
        project="RL-Zero",  # 项目名称
        name="Qwen2.5-1.5B-Instruct",
        group="baseline",
        tags=["math", "Qwen2.5-1.5B-Instruct", "baseline", "rl-zero"],
        config={
            "batch_size": 2,
            "epoch": 2,
            "inner_epoch": 1,
            "kl_coe": 0.1,
            "clip": 0.2,
            "max_sentence_len": 1024*8,
            "max_prompt_len": 1024,
            "train_batch": 128,
            "lr": 5e-6,
            "buffer": 4,
            "value_coe": 0.1,
            "entropy_coe": 0.005,
            "max_grad_norm": 0.5,
            "number_responses": 2,
            "model": "Qwen2.5-1.5B-Instruct",
            "random_seed": 42,
        }
    )
    
    # 从wandb配置中获取超参数
    config = wandb.config


    batch_size = config.batch_size
    epoch = config.epoch
    inner_epoch = config.inner_epoch
    random_seed = config.random_seed
    kl_coe = config.kl_coe
    clip = config.clip
    max_sentence_len = config.max_sentence_len
    max_prompt_len = config.max_prompt_len
    train_batch = config.train_batch
    lr = config.lr
    buffer = config.buffer
    value_coe = config.value_coe
    entropy_coe = config.entropy_coe
    max_grad_norm = config.max_grad_norm
    number_responses = config.number_responses

    torch.manual_seed(random_seed)

    device = f'cuda:{device}'
    torch.cuda.set_device(device)
    print (f'using device: {device}')     
    reward = MathReward()

    model_path = '/hy-tmp/Qwen2.5-1.5B-Instruct'
    update_path = '/hy-tmp/Qwen2.5-1.5B-Instruct-update'
    # base_model = AutoModel.from_pretrained(model_path)
    base_model = Qwen2ForCausalLM.from_pretrained(model_path)
    ref_model = Qwen2ForCausalLM.from_pretrained(model_path)
    gen_model = Qwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    policy_model = PolicyModel(base_model, ref_model, gen_model, model_path)
    reward_model = MathReward()

    ppo = GRPO(policy_model, reward, clip, logit_post_fn=softmax_fn(mask_ids=[0, 100]))
    params = list(policy_model.policy_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    collector = GRPOCollector(buffer, kl_coe, eos_token=tokenizer.eos_token_id)

    dataset = DataLoader('data/test_data.json', batch_size=2)

    sample_step = 0
    train_step = 0

    # policy_model.save_policy_model()
    # tokenizer.save_pretrained(update_path)
    # policy_model.start_vllm_server()


    for i in range(epoch):
        for prompts in dataset:
            sample_step += 1

            print (f'start sample {sample_step}----------------------------')
            policy_model.eval()
            for prompt in prompts:
                prompt_text = prompt['input']
                answer_text = prompt['answer']
                # if reward parse answer is None, skip this prompt
                if reward_model.parse_ground_truth(answer_text) is None:
                    print (f'skip {prompt_text} because answer is None, {answer_text}')
                    continue

                level = prompt['level']
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
                    eos_index = (input_ids[i, start_index:] == tokenizer.eos_token_id).nonzero()
                    if len(eos_index) == 0:
                        continue
                    else:
                        eos_index = eos_index[0][0].item() + start_index
                    episode = (prompt_text, response_texts[i], [answer_text, str(reward_model.parse_ground_truth(answer_text)), str(reward_model.parse_answer(response_texts[i]))], input_ids[i][:eos_index].tolist(), gen_log_probs[i][:eos_index-start_index+1].tolist(), ref_log_probs[i][:eos_index-start_index+1].tolist(), start_index, rewards[i])
                    collector.add_buffer([episode])
            
            print (f'end sample {sample_step}----------------------------')

            collector.dump_buffer(f'buffer_{i}.pkl', mode='pickle')
            collector.dump_buffer(f'buffer_{i}.json', mode='json')
            wandb.finish()
            os._exit(0)
            # average reward
            average_reward = np.mean([x[6] for x in collector.episodes])
            average_length = np.mean([len(x[2]) - x[5] for x in collector.episodes])
            print (f'average reward: {average_reward}')
            print (f'average input_ids - start_index length: {average_length}')

            # 记录采样阶段的指标
            wandb.log({
                "sample_step": sample_step,
                "average_reward": average_reward,
                "average_length": average_length,
            })

            policy_model.train()
            for samples in collector.sample(inner_epoch, batch=train_batch, device=device):
                train_step += 1
                policy_loss, entropy_loss = ppo.forward(samples)
                loss = policy_loss + entropy_loss * entropy_coe
                loss = loss.mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                opt.zero_grad()
                
                # 记录训练阶段的指标
                wandb.log({
                    "train_step": train_step,
                    "policy_loss": policy_loss.mean().item(),
                    "entropy_loss": entropy_loss.mean().item(),
                    "total_loss": loss.item(),
                })
                wandb.log({
                    "gradient_norm": grad_norm,
                })

            collector.reset()
            # policy_model.save_policy_model()

            # cur_rewards = torch.mean(torch.stack([x[-1] for x in episodes])).detach().item()

    # 实验结束时关闭wandb
    wandb.finish()
