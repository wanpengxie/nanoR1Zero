from transformers import GPT2PreTrainedModel
import torch
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


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
        # find the last \boxed{...} in response
        start = response.rfind('\\boxed{')
        if start == -1:
            return None
        
        start += 7  # length of '\boxed{'
        count = 1
        for i in range(start, len(response)):
            if response[i] == '{':
                count += 1
            elif response[i] == '}':
                count -= 1
                if count == 0:
                    return response[start:i]
        return None
    
    def rule_reward(self, gen_ans, ground_truth):
        reward = 0
        gold_expr = self.parse_ground_truth(ground_truth)
        ans_expr = self.parse_answer(gen_ans)
        box_ans = self.match_box(gen_ans)
        if ans_expr is None:
                reward = 0
        else:
                reward = float(verify(gold_expr, ans_expr))
        # print (f'reward: {reward}, gold_expr: {ground_truth} => {gold_expr}, ans_expr: {box_ans} => {ans_expr}')            
        return reward


if __name__ == "__main__":
    pass