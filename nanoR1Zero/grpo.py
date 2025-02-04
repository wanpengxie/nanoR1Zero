import torch.nn
import numpy as np
from nanoR1Zero.reward import BaseReward
from nanoR1Zero.policy import PolicyModel
from tqdm import tqdm
import time
import wandb

class GRPO(torch.nn.Module):
    def __init__(self, policy_model: PolicyModel, reward: BaseReward, clip=0.1, lr=1e-5, entropy_coe=0.01, max_grad_norm=1.0):
        super().__init__()
        self.policy_model = policy_model
        self.clip_range = clip
        self.lr = lr
        self.entropy_coe = entropy_coe
        self.max_grad_norm = max_grad_norm
        self.params = list(policy_model.policy_model.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=lr)
        self.device = policy_model.device
        self.reward_model = reward

    def forward(self, samples):
        device = self.policy_model.policy_model.device
        if device is None:
            device = 'cuda'
        token_ids, label_ids, log_gen_probs, log_ref_probs, start_index, rewards = samples
        token_ids = token_ids.to(device)
        label_ids = label_ids.to(device)
        log_gen_probs = log_gen_probs.to(device)
        log_ref_probs = log_ref_probs.to(device)
        rewards = rewards.to(device)
        
        logits = self.policy_model.forward_policy(token_ids)
        safe_label_ids = label_ids.clamp(min=0)
        mask_ids = label_ids.ne(-100)

        token_probs = torch.nn.Softmax(dim=-1)(logits)
        token_probs = torch.gather(token_probs, -1, safe_label_ids[:, :, None]).squeeze(-1)
        log_cur_probs = torch.log(token_probs)

        # grpo loss
        prob_ratio = torch.exp(log_cur_probs - log_gen_probs)
        policy_loss = torch.min(prob_ratio * rewards[:, None], torch.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range)  * rewards[:, None])
        policy_loss = torch.sum(policy_loss * mask_ids, dim=-1) / torch.sum(mask_ids, dim=-1)
        policy_loss = torch.mean(policy_loss)
        
        # kl divergence
        kl_divergence = torch.exp(log_ref_probs - log_cur_probs) - 1 - log_ref_probs + log_cur_probs
        kl_divergence = torch.sum(kl_divergence * mask_ids, dim=-1) / torch.sum(mask_ids, dim=-1)
        kl_divergence = torch.mean(kl_divergence)

        return policy_loss, kl_divergence
    
    def generate_episodes(self, results):
        eos_token_id = self.policy_model.tokenizer.eos_token_id

        self.policy_model.gen_model = self.policy_model.gen_model.to(self.device)
        self.policy_model.ref_model = self.policy_model.ref_model.to(self.device)
        self.policy_model.gen_model.eval()
        self.policy_model.ref_model.eval()
        sample_rewards = []
        sample_episodes = []
        with torch.no_grad():
            for result in tqdm(results, desc='calc sample probs'):
                prompt_text = result['prompt']
                answer_text = result['answer']

                # if reward parse answer is None, skip this prompt
                if self.reward_model.parse_ground_truth(answer_text) is None:
                    print (f'skip {prompt_text} because answer is None, {answer_text}')
                    continue
                response_texts = result['output_text_list']
                rewards = [self.reward_model.rule_reward(response, answer_text) for response in response_texts]

                sample_rewards.append(np.mean(rewards))
                rewards = [(r - np.mean(rewards)) / (np.std(rewards) + 1e-6) for r in rewards]

                input_ids, gen_log_probs, ref_log_probs, start_index = self.policy_model.generate_probs(
                    prompt_token_ids=result['prompt_token_ids'],
                    output_token_ids_list=result['output_token_ids_list'],
                    eos_token=eos_token_id,
                )
                n = input_ids.shape[0]
                for i in range(n):
                    eos_index = (input_ids[i, start_index:] == eos_token_id).nonzero()
                    if len(eos_index) == 0:
                        continue
                    else:
                        eos_index = eos_index[0][0].item() + start_index
                    episode = (prompt_text,
                            response_texts[i],
                            [answer_text, str(self.reward_model.parse_ground_truth(answer_text)), str(self.reward_model.parse_answer(response_texts[i]))],
                            input_ids[i][:eos_index].tolist(),
                            gen_log_probs[i][:eos_index-start_index+1].tolist(),
                            ref_log_probs[i][:eos_index-start_index+1].tolist(),
                            start_index,
                            rewards[i])
                    sample_episodes.append(episode)
        self.policy_model.gen_model.cpu()
        self.policy_model.ref_model.cpu()
        return sample_episodes
    
    def train(self, samples_iter, train_batch):
        self.policy_model.train()
        accumulated_loss = 0
        train_samples = 0
        micro_train_samples = 0
        t = time.time()
        accumulated_policy_loss = 0
        accumulated_entropy_loss = 0
        pbar = tqdm(samples_iter)        
        for samples in pbar:
            samples_num = samples[0].shape[0]
            micro_train_samples += samples_num
            with torch.cuda.amp.autocast(dtype=self.policy_model.torch_dtype):
                policy_loss, entropy_loss = self.forward(samples)
                loss = - policy_loss + entropy_loss * self.entropy_coe
                loss.backward()

            accumulated_loss += loss.detach().cpu().item() * samples_num
            policy_loss_acc = policy_loss.detach().cpu().item() * samples_num
            entropy_loss_acc = entropy_loss.detach().cpu().item() * samples_num
            accumulated_policy_loss += np.sqrt(policy_loss_acc * policy_loss_acc)
            accumulated_entropy_loss += np.sqrt(entropy_loss_acc * entropy_loss_acc)
            pbar.set_postfix({
                'entropy_loss': f'{entropy_loss.detach().cpu().item():.4f}',
                'policy_loss': f'{policy_loss.detach().cpu().item():.4f}',
                'loss': f'{loss.detach().cpu().item():.4f}',
                'train_step': train_step,
                'train_samples': train_samples,
            })
            
            if micro_train_samples >= train_batch:
                train_step += 1
                train_samples += micro_train_samples
                # scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                wandb.log({
                    "train_samples": train_samples,
                    "time": time.time() - t,
                    "policy_loss": accumulated_policy_loss / micro_train_samples,
                    "entropy_loss": accumulated_entropy_loss * self.entropy_coe / micro_train_samples,
                    "accumulated_loss": accumulated_loss / micro_train_samples,
                })
                micro_train_samples = 0
                accumulated_loss = 0
                accumulated_policy_loss = 0
                accumulated_entropy_loss = 0

        # final train step
        if micro_train_samples > 0:
            train_step += 1
            train_samples += micro_train_samples
            torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()
            wandb.log({
                "train_step": train_step,
                "train_samples": train_samples,
                "time": time.time() - t,
                "policy_loss": accumulated_policy_loss / micro_train_samples,
                "entropy_loss": accumulated_entropy_loss * self.entropy_coe / micro_train_samples,
                "accumulated_loss": accumulated_loss / micro_train_samples,
            })
        self.policy_model.eval()

if __name__ == "__main__":
    pass