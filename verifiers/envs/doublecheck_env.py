import asyncio
from typing import List, Callable, Dict, Any
from verifiers.envs.base import BaseEnv
from vllm import LLM, SamplingParams, RequestOutput

class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        pass

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    async def run(self,
                  prompt: List[Dict[str, Any]],
                  llm: LLM,
                  sampling_params: SamplingParams
    ) -> RequestOutput:
        output = llm.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0] # type: ignore
        input_text = output.prompt
        input_ids = output.prompt_token_ids
        output_text = output.outputs[0].text
        output_ids = output.outputs[0].token_ids
        prompt.append({'role': 'assistant', 'content': output_text})
        prompt.append({'role': 'user', 'content': 'Are you sure?'})

        output = llm.chat(prompt, sampling_params=sampling_params, use_tqdm=False)[0] # type: ignore 
        # modify so all responses treated as output
        print(output.outputs[0].text)
        print(output.outputs[0].token_ids)
        return output

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams
        ) -> list[RequestOutput]:
        async def a_generate():
            tasks = [self.run(prompt, llm, sampling_params) for prompt in prompts]
            outputs = await asyncio.gather(*tasks)
            return outputs
        return asyncio.run(a_generate())
