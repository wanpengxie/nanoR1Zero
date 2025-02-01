import torch.nn

from nanoRL4GPT.conf import Config
from nanoRL4GPT.collector import LMCollector
from nanoRL4GPT.reward import BaseReward, Reward
from nanoRL4GPT.lm_policy import PolicyModel
from torch.utils.tensorboard import SummaryWriter


class GRPO(torch.nn.Module):
    def __init__(self, policy_model: PolicyModel, reward: BaseReward, clip=0.1, logit_post_fn=None):
        super().__init__()
        self.policy_model = policy_model
        self.clip_range = clip
        self.reward_model = reward
        self.gen_steps = 0
        self.train_steps = 0
        self.dist = torch.distributions.Categorical

        if logit_post_fn:
            self.logit_post_fn = logit_post_fn
        else:
            self.logit_post_fn = torch.nn.Softmax(dim=-1)

    # TODO：计算policy loss和KL divergence
    def forward(self, samples):
        # samples: sentence sample (token_ids, label_ids, start_index, gen_model_logits, reward)
        # 为了方便计算，所有sample都被padding到一个长度，不需要计算的部分，将系数0
        # label padding为-100（prompt_id and padding_id)
        # dim: token_ids: (batch, seq_len), label_ids: (batch, seq_len) => with padding -100, 
        # log_gen_probs: (batch, seq_len-1) with padding 0.0, log_ref_probs: (batch, seq_len) with padding 0.0, 
        # rewards: (batch, 1)
        device = self.policy_model.device
        token_ids, label_ids, log_gen_probs, log_ref_probs, start_index, rewards = samples
        token_ids = token_ids.to(device)
        label_ids = label_ids.to(device)
        log_gen_probs = log_gen_probs.to(device)
        log_ref_probs = log_ref_probs.to(device)
        rewards = rewards.to(device)
        
        logits = self.policy_model.forward_policy(token_ids)
        safe_label_ids = label_ids.clamp(min=0)
        mask_ids = label_ids.ne(-100)

        token_probs = self.logit_post_fn(logits)
        token_probs = torch.gather(token_probs, -1, safe_label_ids[:, :, None]).squeeze(-1)
        log_cur_probs = torch.log(token_probs)

        # grpo loss
        prob_ratio = torch.exp(log_cur_probs - log_gen_probs)
        policy_loss = torch.min(prob_ratio, torch.clamp(prob_ratio, 1 - self.clip_range, 1 + self.clip_range))
        policy_loss = torch.sum(policy_loss * mask_ids, dim=-1) / torch.sum(mask_ids, dim=-1)
        policy_loss = torch.sum(policy_loss * rewards)
        
        # kl divergence
        kl_divergence = torch.exp(log_ref_probs - log_cur_probs) - 1 - log_ref_probs + log_cur_probs
        kl_divergence = torch.sum(kl_divergence * mask_ids, dim=-1) / torch.sum(mask_ids, dim=-1)
        kl_divergence = torch.sum(kl_divergence)

        return policy_loss, kl_divergence


    @torch.no_grad()
    def eval(self, input_ids, eos_token, reward_model: BaseReward):
        setences, _, terminal_pos = self.policy_model.generate(input_ids, logit_fn=self.logit_post_fn)
        rewards = reward_model.forward(setences, terminal_pos)
        return rewards


    # 并发生成，一个prompt，生成N个句子
    @torch.no_grad()
    def generate_episode(self,
                         prompt_ids: torch.Tensor,
                         max_sentence_len: int,
                         number_responses: int,
                         eos_token: int = 0,
                         prompt: str = None,
                         ):
        # sequences, gen_logits, start_index = self.policy_model.generate(prompt_ids, max_sentence_len, number_responses, eos_token, self.logit_post_fn)

        # ref_logits = self.policy_model.forward_ref(sequences)[:, start_index-1:-1, :]
        # ref_probs = self.logit_post_fn(ref_logits)
        # token_ref_probs = torch.gather(ref_probs, -1, sequences[:, start_index:, None]).squeeze(-1)

        # gen_probs = self.logit_post_fn(gen_logits)
        # token_gen_probs = torch.gather(gen_probs, -1, sequences[:, start_index:, None]).squeeze(-1)

        # # detach
        # input_ids = sequences.detach().cpu()
        # gen_log_probs = torch.log(token_gen_probs).detach().cpu()
        # ref_log_probs = torch.log(token_ref_probs).detach().cpu()
        

        ## using vllm server
        input_ids, gen_log_probs, ref_log_probs, start_index = self.policy_model.generate_vllm(prompt, max_sentence_len, number_responses, eos_token)

        return input_ids, gen_log_probs, ref_log_probs, start_index


def softmax_fn(temp=1.0, mask_ids=[]):
    def softmax_with_temper(logits: torch.FloatTensor, do_mask=True):
        if do_mask:
            for x in mask_ids: # mask unk and pad token
                logits[:, x].fill_(-1000.0)
        logits = logits/temp
        return torch.nn.Softmax(dim=-1)(logits)
    return softmax_with_temper


if __name__ == "__main__":
    pass