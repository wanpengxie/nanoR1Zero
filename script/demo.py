import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim
from transformers import GPT2LMHeadModel, GPT2Model, BertTokenizer, Qwen2PreTrainedModel, Qwen2Tokenizer, Qwen2ForCausalLM
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import time
import numpy as np
from nanoR1Zero.lm_grpo import GRPO, softmax_fn
from nanoR1Zero.lm_policy import PolicyModel
from nanoR1Zero.reward import BaseReward, CounterReward, MathReward
from nanoR1Zero.collector import GRPOCollector
import tqdm
from torch.utils.tensorboard import SummaryWriter
from nanoR1Zero.data import DataLoader
import wandb  # 添加wandb导入
import requests
import json
from tqdm import tqdm

def eval_aime24(dataset, reward_model, batch_size=128, tokenizer=None, number_responses=2):
    n = len(dataset)
    pass_rewards = []    
    mean_rewards = []
    for i in range(0, n, batch_size):
        batch = dataset[i:i+batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": x['problem']}], 
            tokenize=False, 
            add_generation_prompt=True) 
            for x in batch]
        answers = [x['answer'] for x in batch]
        results = requests.post(
            f'http://localhost:8000/generate_batch', 
            json={'prompts': prompts, 'max_len': 8192, 'temperature': 0.01, 'top_p': 1.0, 'number_responses': number_responses}
        )
        results = results.json()
        for i, result in enumerate(results['results']):
            prompt_text = result['prompt_text']
            answer_texts = result['output_text']
            gt_answer = answers[i]
            reward = [reward_model.rule_reward(answer_text, gt_answer) for answer_text in answer_texts]
            pass_rewards.append(max(reward))
            mean_rewards.append(np.mean(reward))
        print (f'batch {i} pass reward: {max(reward)}, mean reward: {np.mean(reward)}')
    print (f'average pass reward: {np.mean(pass_rewards)}')
    print (f'average mean reward: {np.mean(mean_rewards)}')
    return np.mean(pass_rewards), np.mean(mean_rewards)

def eval_dataset(dataset, reward_model, batch_size=128, max_len=128):
    n = len(dataset)
    rewards = []
    pass_rewards = []    
    mean_rewards = []
    answer_lens = []
    for i in range(0, n, batch_size):
        if max_len is not None and i >= max_len:
            break
        batch = dataset[i:i+batch_size]
        prompts = [x['input'] for x in batch]
        answers = [x['answer'] for x in batch]
        results = requests.post(
            f'http://localhost:8000/generate_batch', 
            json={'prompts': prompts, 'max_len': 8192, 'temperature': 1.0, 'top_p': 1.0, 'number_responses': 4}
        )
        results = results.json()
        batch_pass_rewards = []
        batch_mean_rewards = []
        batch_answer_lens = []
        for k, result in enumerate(results['results']):
            prompt_text = result['prompt_text']
            answer_texts = result['output_text']
            answer_token_ids = result['output_token_ids']
            gt_answer = answers[k]
            reward = [reward_model.rule_reward(answer, gt_answer) for answer in answer_texts]
            answer_len = [len(x) for x in answer_token_ids]
            pass_rewards.append(max(reward))
            mean_rewards.append(np.mean(reward))
            answer_lens.append(answer_len)
            batch_pass_rewards.append(max(reward))
            batch_mean_rewards.append(np.mean(reward))
            batch_answer_lens.append(np.mean(answer_len))
        print (f'batch {i} pass reward: {np.mean(batch_pass_rewards)}, mean reward: {np.mean(batch_mean_rewards)}, answer len: {np.mean(batch_answer_lens)}')
    print (f'average pass reward: {np.mean(pass_rewards)}')
    print (f'average mean reward: {np.mean(mean_rewards)}')
    print (f'average answer len: {np.mean(answer_lens)}')
    return np.mean(pass_rewards), np.mean(mean_rewards)

if __name__ == "__main__":
    import sys 
    device = sys.argv[1]
    # 初始化wandb
    # api: 
    wandb.init(
        project="RL-Zero",  # 项目名称
        group="baseline",
        tags=["math", "Qwen2.5-1.5B-Instruct", "baseline", "rl-zero"],
        config={
            "batch_size": 4,
            "epoch": 2,
            "inner_epoch": 1,
            "kl_coe": 0.1,
            "clip": 0.2,
            "max_sentence_len": 1024*8,
            "max_prompt_len": 1024,
            "train_batch": 64,
            "micro_batch": 2,
            "lr": 5e-6,
            "buffer": 4,
            "value_coe": 0.1,
            "entropy_coe": 0.005,
            "max_grad_norm": 0.5,
            "number_responses": 8,
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
    micro_batch = config.micro_batch
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
    

    model_path = '/hy-tmp/Qwen2.5-1.5B-Instruct'
    update_path = '/hy-tmp/Qwen2.5-1.5B-Instruct-update'
    # base_model = AutoModel.from_pretrained(model_path)
    base_model = Qwen2ForCausalLM.from_pretrained(model_path)
    ref_model = Qwen2ForCausalLM.from_pretrained(model_path)
    gen_model = Qwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    policy_model = PolicyModel(base_model, ref_model, gen_model, model_path)
    reward_model = MathReward()

    ppo = GRPO(policy_model, reward_model, clip, logit_post_fn=softmax_fn(mask_ids=[0, 100]))
    params = list(policy_model.policy_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    collector = GRPOCollector(buffer, kl_coe, eos_token=tokenizer.eos_token_id)

    dataset = DataLoader('data/math_verify_train.json', batch_size=batch_size)

    sample_step = 0
    train_step = 0

    # policy_model.save_policy_model()
    # tokenizer.save_pretrained(update_path)
    # policy_model.start_vllm_server()
    # test_dataset = DataLoader('data/aime24_eval.json', batch_size=8)
    # eval_aime24(test_dataset, reward_model, batch_size=8, tokenizer=tokenizer, number_responses=16)

    test_dataset = DataLoader('data/math_verify_test.json', batch_size=64)
    policy_model.save_policy_model(update_path)
    tokenizer.save_pretrained(update_path)
    policy_model.start_vllm_server(update_path, '0')

    policy_model.policy_model.gradient_checkpointing_enable()
    for e in range(epoch):
        # eval_dataset(test_dataset, reward_model, batch_size=64, max_len=128)
        for prompts in dataset:
            # eval before sample and training
            max_reward, mean_reward = eval_dataset(test_dataset, reward_model, batch_size=64, max_len=128)
            print (f'max reward: {max_reward}, mean reward: {mean_reward}')
            wandb.log({
                "eval_max_reward": max_reward,
                "eval_mean_reward": mean_reward,
            })

            sample_step += 1
            t = time.time()
            print (f'start sample {sample_step}----------------------------, At {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            policy_model.eval()
            sample_rewards = []
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
                print (f'question level: {level}, rewards size: {len(rewards)}, mean: {np.mean(rewards)}, std: {np.std(rewards)}')
                sample_rewards.append(np.mean(rewards))
                rewards = [(r - np.mean(rewards)) / (np.std(rewards) + 1e-6) for r in rewards]
                for i in range(number_responses):
                    eos_index = (input_ids[i, start_index:] == tokenizer.eos_token_id).nonzero()
                    if len(eos_index) == 0:
                        continue
                    else:
                        eos_index = eos_index[0][0].item() + start_index
                    episode = (prompt_text, 
                               response_texts[i], 
                               [answer_text, str(reward_model.parse_ground_truth(answer_text)), str(reward_model.parse_answer(response_texts[i]))], 
                               input_ids[i][:eos_index].tolist(), 
                               gen_log_probs[i][:eos_index-start_index+1].tolist(), 
                               ref_log_probs[i][:eos_index-start_index+1].tolist(), 
                               start_index, 
                               rewards[i])
                    collector.add_buffer([episode])
            
            print (f'end sample {sample_step}----------------------------, time: {int(time.time() - t)}s')

            collector.dump_buffer(f'sample_buffer_{sample_step}_{e}.pkl', mode='pickle')
            collector.dump_buffer(f'sample_buffer_{sample_step}_{e}.json', mode='json')
            # average reward
            average_reward = np.mean(sample_rewards)
            average_length = np.mean([len(x[3]) - x[6] for x in collector.episodes])
            print (f'average reward: {average_reward}')
            print (f'average input_ids - start_index length: {average_length}')

            # 记录采样阶段的指标
            wandb.log({
                "sample_average_reward": average_reward,
                "sample_average_length": average_length,
            })

            policy_model.train()
            accumulated_loss = 0
            train_samples = 0
            micro_train_samples = 0
            t = time.time()
            accumulated_policy_loss = 0
            accumulated_entropy_loss = 0
            print (f'start train step {sample_step}----------------------------, At {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            for batch_idx, samples in enumerate(collector.sample(inner_epoch, batch=micro_batch, device=device)):
                # train_step += micro_batch
                samples_num = samples[0].shape[0]
                micro_train_samples += samples_num
                print (f'start train batch {batch_idx}, micro samples_num: {micro_train_samples}, train samples_num: {train_samples}')
                policy_loss, entropy_loss = ppo.forward(samples)
                loss = - policy_loss + entropy_loss * entropy_coe
                loss = loss.mean()
                loss.backward()
                accumulated_loss += loss.detach().cpu().item() * samples_num
                policy_loss_acc = policy_loss.detach().cpu().item() * samples_num
                entropy_loss_acc = entropy_loss.detach().cpu().item() * samples_num
                accumulated_policy_loss += np.sqrt(policy_loss_acc * policy_loss_acc)
                accumulated_entropy_loss += np.sqrt(entropy_loss_acc * entropy_loss_acc)

                if micro_train_samples >= train_batch:
                    train_step += 1
                    train_samples += micro_train_samples
                    print (f'-----accumulated batch {batch_idx}, micro train samples: {micro_train_samples}, train samples: {train_samples}-----')
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    opt.step()
                    opt.zero_grad()                
                    wandb.log({
                        "train_step": train_step,
                        "train_samples": train_samples,
                        "time": time.time() - t,
                        "policy_loss": accumulated_policy_loss / micro_train_samples,
                        "entropy_loss": accumulated_entropy_loss * entropy_coe / micro_train_samples,
                        "accumulated_loss": accumulated_loss / micro_train_samples,
                    })
                    wandb.log({
                        "gradient_norm": grad_norm,
                    })
                    micro_train_samples = 0
                    accumulated_loss = 0
                    accumulated_policy_loss = 0
                    accumulated_entropy_loss = 0
            # final train step
            if micro_train_samples > 0:
                train_step += 1
                train_samples += micro_train_samples
                print (f'-----accumulated batch {batch_idx}, micro train samples: {micro_train_samples}, train samples: {train_samples}-----')
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                opt.zero_grad()                
                wandb.log({
                    "train_step": train_step,
                    "train_samples": train_samples,
                    "time": time.time() - t,
                    "policy_loss": accumulated_policy_loss / micro_train_samples,
                    "entropy_loss": accumulated_entropy_loss * entropy_coe / micro_train_samples,
                    "accumulated_loss": accumulated_loss / micro_train_samples,
                })
                wandb.log({
                    "gradient_norm": grad_norm,
                })
                
            print (f'end train step {sample_step}----------------------------, time: {int(time.time() - t)}s')
            collector.reset()
            print (f'sync vllm server: after train numbers: {train_step}-step/{train_samples}-samples')
            policy_model.stop_vllm_server()
            policy_model.save_policy_model(update_path)
            policy_model.start_vllm_server(update_path, '0')

            # cur_rewards = torch.mean(torch.stack([x[-1] for x in episodes])).detach().item()

    # 实验结束时关闭wandb
    wandb.finish()
