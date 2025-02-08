from abc import ABC, abstractmethod
import asyncio
from typing import Any, Callable, Dict, List, Sequence
from vllm import LLM, SamplingParams, RequestOutput


async def async_llm_chat(messages: List[Dict[str, Any]],
                         llm: LLM,
                         sampling_params: SamplingParams) -> RequestOutput:
    outputs = await asyncio.to_thread(
        lambda: llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
    )
    return outputs[0]

class BaseEnv(ABC):

    def __init__(self):
        pass    

    @abstractmethod
    def get_rubric(self) -> List[Callable[..., list[float]]]:
        pass
    
    @abstractmethod
    def generate(self, prompts: List[List[Dict[str, Any]]], llm: Any, sampling_params: Any) -> List[Sequence[int]]:
        pass
