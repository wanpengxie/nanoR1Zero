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
from torch.cuda.amp import autocast
from nanoR1Zero.vllm_client import batch_generate

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
            json={'prompts': prompts, 'max_len': 8192, 'temperature': 0.3, 'top_p': 1.0, 'number_responses': number_responses}
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

def eval_dataset(urls, dataset, reward_model, batch_size=32, number_responses=4):
    pass_rewards = []
    mean_rewards = []
    answer_lens = []
    args = {
        'temperature': 1.0,
        'top_p': 1.0,
        'max_tokens': 8192,
        'number_responses': number_responses,
    }
    results = batch_generate(urls, dataset, batch_size, **args)
    for result in results:
        prompt_text = result['prompt']
        answer_texts = result['output_text_list']
        answer_token_ids = result['output_token_ids_list']
        gt_answer = result['answer']
        reward = [reward_model.rule_reward(answer_text, gt_answer) for answer_text in answer_texts]
        answer_len = [len(x) for x in answer_token_ids]
        pass_rewards.append(max(reward))
        mean_rewards.append(np.mean(reward))
        answer_lens.append(answer_len)
    print (f'average pass reward: {np.mean(pass_rewards)}')
    print (f'average mean reward: {np.mean(mean_rewards)}')
    print (f'average answer len: {np.mean(answer_lens)}')
    return np.mean(pass_rewards), np.mean(mean_rewards), np.mean(answer_lens)


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
            "batch_size": 8,
            "epoch": 2,
            "inner_epoch": 1,
            "kl_coe": 0.1,
            "clip": 0.2,
            "max_sentence_len": 1024*8,
            "max_prompt_len": 1024,
            "train_batch": 16,
            "micro_batch": 1,
            "lr": 8e-5,
            "buffer": 4,
            "value_coe": 0.1,
            "entropy_coe": 0.1,
            "max_grad_norm": 0.5,
            "number_responses": 4,
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
    
    torch_dtype = torch.bfloat16
    model_path = '/hy-tmp/Qwen2.5-1.5B-Instruct'
    update_path = '/hy-tmp/Qwen2.5-1.5B-Instruct-update'
    ref_model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)

    if os.path.exists(update_path):
        base_model = Qwen2ForCausalLM.from_pretrained(update_path, use_cache=False, torch_dtype=torch_dtype)
        gen_model = Qwen2ForCausalLM.from_pretrained(update_path, torch_dtype=torch_dtype)
    else:
        base_model = Qwen2ForCausalLM.from_pretrained(model_path, use_cache=False, torch_dtype=torch_dtype)
        gen_model = Qwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)

    policy_model = PolicyModel(base_model, ref_model, gen_model, model_path)
    reward_model = MathReward()

    ppo = GRPO(policy_model, reward_model, clip)
    params = list(policy_model.policy_model.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    collector = GRPOCollector(buffer, kl_coe, eos_token=tokenizer.eos_token_id)

    dataset = DataLoader('data/math_verify_train.json', batch_size=batch_size)

    sample_step = 0
    train_step = 0

    # scaler = torch.cuda.amp.GradScaler()
    test_dataset = DataLoader('data/math_verify_test.json', batch_size=64)
    if not os.path.exists(update_path):
        policy_model.save_policy_model(update_path)
        tokenizer.save_pretrained(update_path)
    policy_model.start_vllm_server(update_path, [('0', '8000'), ('1', '8001')])

    policy_model.policy_model.gradient_checkpointing_enable()
    for e in range(epoch):
        # eval_dataset(test_dataset, reward_model, batch_size=64, max_len=128)
        policy_model.eval()
        last_train_prompts = None
        for prompts in dataset:
            # eval before sample and training
            max_reward, mean_reward, resp_len = eval_dataset([f'{url}/generate_batch' for url in policy_model.worker_urls], 
                                                             test_dataset, reward_model, batch_size=32, number_responses=4)
            if last_train_prompts is not None:
                max_train_reward, mean_train_reward, train_resp_len = eval_dataset([f'{url}/generate_batch' for url in policy_model.worker_urls], 
                                                                                last_train_prompts, reward_model, batch_size=32, number_responses=4)
                wandb.log({
                    "train_max_reward": max_train_reward,
                    "train_mean_reward": mean_train_reward,
                    "train_resp_len": train_resp_len,
                })
            wandb.log({
                "eval_max_reward": max_reward,
                "eval_mean_reward": mean_reward,
                "eval_resp_len": resp_len,
            })
            last_train_prompts = prompts

            sample_step += 1
            t = time.time()
            print (f'start sample {sample_step}----------------------------, At {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
            args = {
                'temperature': 1.0,
                'top_p': 1.0,
                'max_tokens': 8192,
                'number_responses': number_responses,
            }
            results = batch_generate([f'{url}/generate_batch' for url in policy_model.worker_urls], prompts, 8, **args)
            print (f'batch generate time: {time.time() - t}s, size: {len(results) * number_responses}')

            t = time.time()
            sample_rewards = []

            policy_model.stop_vllm_server()
            policy_model.gen_model = policy_model.gen_model.to(device)
            policy_model.ref_model = policy_model.ref_model.to(device)
            policy_model.gen_model.eval()
            policy_model.ref_model.eval()
            with torch.no_grad():
                for result in tqdm(results, desc='calc sample probs'):
                    prompt_text = result['prompt']
                    answer_text = result['answer']
                    # if reward parse answer is None, skip this prompt
                    if reward_model.parse_ground_truth(answer_text) is None:
                        print (f'skip {prompt_text} because answer is None, {answer_text}')
                        continue
                    response_texts = result['output_text_list']
                    rewards = [reward_model.rule_reward(response, answer_text) for response in response_texts]
                    # print (f'rewards size: {len(rewards)}, mean: {np.mean(rewards)}, std: {np.std(rewards)}')
                    sample_rewards.append(np.mean(rewards))
                    rewards = [(r - np.mean(rewards)) / (np.std(rewards) + 1e-6) for r in rewards]

                    input_ids, gen_log_probs, ref_log_probs, start_index = policy_model.generate_vllm(
                        prompt_token_ids=result['prompt_token_ids'],
                        output_token_ids_list=result['output_token_ids_list'],
                        eos_token=tokenizer.eos_token_id,
                    )  
                    n = input_ids.shape[0]
                    for i in range(n):
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
            policy_model.gen_model.cpu()
            policy_model.ref_model.cpu()
            torch.cuda.empty_cache()

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
            for batch_idx, samples in enumerate(collector.sample(inner_epoch, batch=micro_batch)):
                # train_step += micro_batch
                samples_num = samples[0].shape[0]
                micro_train_samples += samples_num
                with torch.cuda.amp.autocast(dtype=torch_dtype):
                    policy_loss, entropy_loss = ppo.forward(samples)
                    loss = - policy_loss + entropy_loss * entropy_coe
                    loss.backward()

                accumulated_loss += loss.detach().cpu().item() * samples_num
                policy_loss_acc = policy_loss.detach().cpu().item() * samples_num
                entropy_loss_acc = entropy_loss.detach().cpu().item() * samples_num
                accumulated_policy_loss += np.sqrt(policy_loss_acc * policy_loss_acc)
                accumulated_entropy_loss += np.sqrt(entropy_loss_acc * entropy_loss_acc)
                print (f'start train batch {batch_idx}, entropy loss: {entropy_loss.detach().cpu()}, policy loss: {policy_loss.detach().cpu()}, loss: {loss.detach().cpu()}')

                if micro_train_samples >= train_batch:
                    train_step += 1
                    train_samples += micro_train_samples
                    print (f'-----accumulated batch {batch_idx}, micro train samples: {micro_train_samples}, train samples: {train_samples}-----')
                    # scaler.unscale_(opt)
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
            # 清理最后一次训练的变量和显存
            policy_model.policy_model = policy_model.policy_model.cpu()
            del loss, policy_loss, entropy_loss
            torch.cuda.empty_cache()

            print (f'sync vllm server: after train numbers: {train_step}-step/{train_samples}-samples')
            # policy_model.stop_vllm_server()
            policy_model.save_policy_model(update_path)
            policy_model.start_vllm_server(update_path, [('0', 8000), ('1', 8001)])

            # cur_rewards = torch.mean(torch.stack([x[-1] for x in episodes])).detach().item()

    # 实验结束时关闭wandb
    wandb.finish()
