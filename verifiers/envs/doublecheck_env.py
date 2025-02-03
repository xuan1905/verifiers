from typing import List, Callable, Dict, Any
from verifiers.envs.base import BaseEnv
from vllm import LLM, SamplingParams, RequestOutput

class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        pass

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams
        ) -> list[RequestOutput]:
        return []