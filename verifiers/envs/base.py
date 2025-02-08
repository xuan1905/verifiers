from abc import ABC, abstractmethod
import asyncio
from typing import Any, Callable, Dict, List, Sequence
import uuid 

from vllm import AsyncLLMEngine, SamplingParams, RequestOutput

async def async_llm_chat(messages: List[Dict[str, Any]],
                         llm: AsyncLLMEngine,
                         sampling_params: SamplingParams) -> RequestOutput:
    tokenizer = await llm.get_tokenizer()
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    request_id = str(uuid.uuid4())
    gen = llm.generate(text, sampling_params=sampling_params, request_id=request_id)
    return await anext(gen)


class BaseEnv(ABC):

    def __init__(self):
        pass    

    @abstractmethod
    def get_rubric(self) -> List[Callable[..., list[float]]]:
        pass
    
    @abstractmethod
    def generate(self, prompts: List[List[Dict[str, Any]]], llm: Any, sampling_params: Any) -> List[Sequence[int]]:
        pass
