from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union
import logging

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams # type: ignore


class BaseEnv(ABC):

    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [], **kwargs: Any):
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")

    def format_prompt(self, prompt: str) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt and len(self.system_prompt) > 0:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and len(self.few_shot) > 0:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    @abstractmethod
    def get_dataset(self, **kwargs: Any) -> Dataset:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass
    
    @abstractmethod
    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str]]:
        pass
