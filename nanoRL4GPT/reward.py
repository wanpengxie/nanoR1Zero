from transformers import GPT2PreTrainedModel
import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import re
class BaseReward(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def calc_reward(self, input_ids, *args):
        raise Exception("not impl")

    def rule_reward(self, ans, pred):
        raise Exception("not impl")


class MathReward(BaseReward):
    def __init__(self):
        super().__init__()

    def parse_answer(self, answer):
        box_ans = self.match_box(answer)
        if box_ans is None:
            return None
        
        try:
            res = int(box_ans)
            return [res, box_ans]
        except Exception as e:
            pass
        
        try:
            res = float(answer)
            return [res, answer]
        except Exception as e:
            pass
        
        ans_expr = parse(
                    answer,
                    extraction_config=[
                        LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                )
        return ans_expr

    def parse_ground_truth(self, ground_truth):
        try:
            res = int(ground_truth)
            return [res, ground_truth]
        except Exception as e:
            pass
        
        try:
            res = float(ground_truth)
            return [res, ground_truth]
        except Exception as e:
            pass

        gold_expr = parse(ground_truth, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
        if len(gold_expr) == 0:
            return None
        return gold_expr

    def match_box(self, response):
        # re match box in response
        pattern = r'\\boxed{(.*)}'
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        return None

    def rule_reward(self, gen_ans, ground_truth):
        reward = 0
        gold_expr = self.parse_ground_truth(ground_truth)
        ans_expr = self.parse_answer(gen_ans)
        print (f'gold_expr: {ground_truth} => {gold_expr}, ans_expr: {gen_ans} => {ans_expr}')
        if ans_expr is None:
                reward = 0
        else:
                reward = float(verify(gold_expr, ans_expr))
        return reward


class Reward(BaseReward):
    def __init__(self, model: GPT2PreTrainedModel):
        super().__init__()
        self.model = model
        self.header = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, eos_indices=None, attention_mask=None):
        hidden_state = self.model(input_ids, attention_mask).last_hidden_state
        eos_indices = eos_indices[:, None, None]
        hidden_state = torch.take_along_dim(hidden_state, eos_indices, dim=1)
        hidden_state = torch.squeeze(hidden_state, dim=1)
        scores = torch.nn.Sigmoid()(self.header(hidden_state))
        return torch.squeeze(scores, dim=-1)

    @torch.no_grad()
    def calc_reward(self, input_ids, eos_indices=None):
        return self.forward(input_ids, eos_indices)


class CounterReward(BaseReward):
    def __init__(self, shortest=True):
        super().__init__()
        self.shortest = shortest

    @torch.no_grad()
    def calc_reward(self, input_ids, eos_indices=None):
        batch = input_ids.shape[0]
        size = input_ids.shape[1]
        if self.shortest:
            return 512 - torch.clone(eos_indices.float())
        else:
            return torch.clone(eos_indices.float())


def train_reward():
    pass


def gen_samples():
    pass


if __name__ == "__main__":
    # test reward of GPT2
    pass