from typing import List, Dict, Any, Tuple

from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams, RequestOutput # type: ignore

from verifiers.envs.base import BaseEnv

class CodeEnv(BaseEnv):
    def __init__(self, system_prompt: str = "", few_shot: List[Dict[str, str]] = []):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot)

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return []

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        return [], []