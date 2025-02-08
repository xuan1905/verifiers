from typing import List, Dict, Sequence, Any, Callable
from verifiers.envs.base import BaseEnv
from vllm import LLM, SamplingParams

class DummyEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    def generate(self, prompts: List[List[Dict[str, Any]]], llm: LLM, sampling_params: SamplingParams) -> List[Sequence[int]]:
        completions = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False) # type: ignore
        return [completion.outputs[0].token_ids for completion in completions]
    

    