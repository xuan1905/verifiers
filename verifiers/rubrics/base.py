from typing import List
from trl.trainer.grpo_trainer import RewardFunc

def equals_reward_func(completions, answer, **kwargs) -> List[float]:
    responses = [c[0]['content'] for c in completions]
    return [1.0 if r == a else 0.0 for r, a in zip(responses, answer)]

class BaseRubric:
    def __init__(self, **kwargs):
        pass

    def get_reward_funcs(self) -> List[RewardFunc]:
        return [equals_reward_func]