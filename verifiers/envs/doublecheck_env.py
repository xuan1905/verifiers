import asyncio
from typing import List, Callable, Dict, Any, Sequence
from verifiers.envs.base import BaseEnv
from vllm import LLM, SamplingParams


class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        super().__init__()


    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    async def run(self,
                  prompt: List[Dict[str, Any]],
                  llm: LLM,
                  sampling_params: SamplingParams) -> Sequence[int]:
        # first pass
        messages = [m for m in prompt]
        output = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)[0] # type: ignore
        len_prompt = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0

        # double-check step
        messages.append({'role': 'assistant', 'content': output.outputs[0].text})
        messages.append({'role': 'user', 'content': 'Are you sure?'})
        output = self.llm.chat(messages, sampling_params=self.sampling_params, use_tqdm=False)[0] # type: ignore

        # combine outputs
        all_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
        return all_ids[len_prompt:]

    def generate(self, prompts: List[List[Dict[str, Any]]], llm: LLM, sampling_params: SamplingParams) -> List[Sequence[int]]:
        async def run_all():
            tasks = [self.run(prompt, llm, sampling_params) for prompt in prompts]
            outputs = await asyncio.gather(*tasks)
            return outputs
        return asyncio.run(run_all())
