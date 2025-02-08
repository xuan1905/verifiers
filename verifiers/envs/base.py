from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Sequence

from vllm import LLM, SamplingParams


class BaseEnv(ABC):

    def __init__(self):
        pass    

    @abstractmethod
    def get_rubric(self) -> List[Callable[..., list[float]]]:
        pass
    
    @abstractmethod
    def generate(self,
                 prompts: List[List[Dict[str, Any]]], llm: LLM,
                 sampling_params: SamplingParams) -> List[Sequence[int]]:
        pass