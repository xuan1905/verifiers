import asyncio
from typing import List, Callable, Dict, Any
from verifiers.envs.base import BaseEnv
from vllm import LLM, SamplingParams, RequestOutput

class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    async def run(self,
                  prompt: List[Dict[str, Any]]
    ) -> RequestOutput:
        output = self.llm.chat(prompt, sampling_params=self.sampling_params, use_tqdm=False)[0] # type: ignore
        input_text = output.prompt
        input_ids = output.prompt_token_ids
        output_text = output.outputs[0].text
        #output_ids = output.outputs[0].token_ids
        prompt.append({'role': 'assistant', 'content': output_text})
        prompt.append({'role': 'user', 'content': 'Are you sure?'})

        output = self.llm.chat(prompt, sampling_params=self.sampling_params, use_tqdm=False)[0] # type: ignore

        # modify so all responses treated as output
        combined_output_text = output.prompt.removeprefix(input_text) + output.outputs[0].text
        combined_output_ids = list(output.prompt_token_ids)[len(input_ids):] + list(output.outputs[0].token_ids)
        output.prompt = input_text
        output.prompt_token_ids = input_ids
        output.outputs[0].text = combined_output_text
        output.outputs[0].token_ids = combined_output_ids
        return output

    def generate(self, prompts: List[List[Dict[str, Any]]]) -> list[RequestOutput]:
        async def run_all():
            tasks = [self.run(prompt) for prompt in prompts]
            outputs = await asyncio.gather(*tasks)
            return outputs
        return asyncio.run(run_all())
