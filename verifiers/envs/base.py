from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List
from vllm import LLM, SamplingParams, RequestOutput

class BaseEnv(ABC):

    def __init__(self):
        self.llm = None
        self.processing_class = None
        self.sampling_params = None

    @abstractmethod
    def get_rubric(self) -> List[Callable[..., list[float]]]:
        pass
    
    @abstractmethod
    def generate(self, prompts: List[List[Dict[str, Any]]]) -> list[RequestOutput]:
        pass
