from typing import List, Dict, Any, Callable
from verifiers.envs.base import BaseEnv
from vllm import LLM, SamplingParams, RequestOutput

class DummyEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    def generate(self, prompts: List[List[Dict[str, Any]]]) -> list[RequestOutput]:
        outputs = self.llm.chat(prompts, sampling_params=self.sampling_params, use_tqdm=False) # type: ignore
        return outputs
    

    