import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb  # 添加wandb导入
import time
import numpy as np

from torch.cuda.amp import autocast
import torch
import torch.optim
from nanoR1Zero.grpo import GRPO
from nanoR1Zero.policy import PolicyModel
from nanoR1Zero.reward import MathReward
from nanoR1Zero.collector import Collector
from nanoR1Zero.vllm_client import VLLMClient

def eval_dataset(vllm_client, dataset, reward_model, batch_size=32, number_responses=4):
    pass_rewards = []
    mean_rewards = []
    answer_lens = []
    args = {
        'temperature': 1.0,
        'top_p': 1.0,
        'max_tokens': 8192,
        'number_responses': number_responses,
    }
    results = vllm_client.generate_batch(dataset, batch_size, **args)
    for result in results:
        answer_texts = result['output_text_list']
        answer_token_ids = result['output_token_ids_list']
        gt_answer = result['answer']
        reward = [reward_model.rule_reward(answer_text, gt_answer) for answer_text in answer_texts]
        answer_len = [len(x) for x in answer_token_ids]
        pass_rewards.append(max(reward))
        mean_rewards.append(np.mean(reward))
        answer_lens.append(answer_len)
    return np.mean(pass_rewards), np.mean(mean_rewards), np.mean(answer_lens)


if __name__ == "__main__":
    import sys
    import json
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    wandb.init(
        project="RL Zero",  # 项目名称
        group="baseline",
        tags=["math", "Qwen2.5-1.5B-Instruct", "baseline", "rl-zero"],
        config=config,
    )
    config = wandb.config
    batch_size = config.batch_size
    epoch = config.epoch
    inner_epoch = config.inner_epoch
    clip = config.clip
    max_sentence_len = config.max_sentence_len
    max_prompt_len = config.max_prompt_len
    train_batch = config.train_batch
    micro_batch = config.micro_batch
    lr = config.lr
    entropy_coe = config.entropy_coe
    max_grad_norm = config.max_grad_norm
    number_responses = config.number_responses
    sample_mix = config.sample_mix

    torch.manual_seed(config.random_seed)

    device = f'cuda:{config.gpus[0]}'
    torch.cuda.set_device(device)
    print (f'using device: {device}')

    policy_model = PolicyModel(config.model_path, torch_dtype=torch.bfloat16, device=device)
    reward_model = MathReward()
    grpo = GRPO(policy_model, reward_model, clip)

    # params = list(policy_model.policy_model.parameters())
    # opt = torch.optim.AdamW(params, lr=lr)

    collector = Collector(eos_token=policy_model.tokenizer.eos_token_id)
    train_dataset = json.load(open('data/math_verify_train.json', 'r'))
    test_dataset = json.load(open('data/math_verify_test.json', 'r'))

    vllm_client = VLLMClient(policy_model.update_path, config.gpus)

    sample_step = 0
    train_step = 0


    vllm_client.start_vllm_server()
    for e in range(epoch):
        for i in range(0, len(train_dataset), batch_size):
            prompts = train_dataset[i:i+batch_size]
            max_reward, mean_reward, resp_len = eval_dataset(vllm_client, test_dataset[:128], reward_model, batch_size=64, number_responses=4)
            print (f'eval max_reward: {max_reward}, mean_reward: {mean_reward}, resp_len: {resp_len}')
            wandb.log({
                "eval_max_reward": max_reward,
                "eval_mean_reward": mean_reward,
                "eval_resp_len": resp_len,
            })

            sample_step += 1
            t = time.time()
            args = {
                'temperature': 1.0,
                'top_p': 1.0,
                'max_tokens': 8192,
                'number_responses': number_responses,
            }
            results = vllm_client.generate_batch(prompts, 8, **args)
            print (f'batch generate time: {time.time() - t}s, size: {len(results) * number_responses}')
            vllm_client.stop_vllm_server()

            t = time.time()
            sample_rewards = []
            episodes = grpo.generate_episodes(results)
            torch.cuda.empty_cache()

            collector.add_episodes(episodes)
            print (f'end sample {sample_step}----------------------------, time: {int(time.time() - t)}s')
            collector.dump_episodes(f'episodes.pkl')

            # 记录采样阶段的指标
            average_reward = np.mean([x[-1] for x in episodes])
            average_length = np.mean([len(x[3]) - x[6] for x in episodes])
            wandb.log({
                "sample_average_reward": average_reward,
                "sample_average_length": average_length,
            })
            train_step = grpo.train(collector.sample(inner_epoch, batch=micro_batch, mix=sample_mix), train_batch, train_step)
            torch.cuda.empty_cache()

            policy_model.save_policy_model()
            vllm_client.start_vllm_server()

    wandb.finish()