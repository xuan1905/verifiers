from typing import List, Callable, Dict, Any, Sequence, Tuple

from verifiers.envs.base import BaseEnv

from vllm import LLM, SamplingParams, RequestOutput


class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    def env_step(self,
                 state: Dict[str, Any],
                 llm: LLM,
                 sampling_params: SamplingParams) -> Tuple[Dict[str, Any], RequestOutput]:
        state["messages"].append({'role': 'user', 'content': 'Are you sure?'})
        output = llm.chat(state["messages"], sampling_params=sampling_params) # type: ignore
        state["messages"].append({'role': 'assistant', 'content': output.outputs[0].text})
        state["completed"] = True
        return state, output

    def run(self,
                  prompt: List[Dict[str, Any]],
                  llm: LLM,
                  sampling_params: SamplingParams) -> Sequence[int]:
        # first pass
        messages = [m for m in prompt]
        output = llm.chat(messages, llm, sampling_params=sampling_params) # type: ignore
        len_prompt = len(output.prompt_token_ids) if output.prompt_token_ids is not None else 0
        messages.append({'role': 'assistant', 'content': output.outputs[0].text})
        
        # initialize env state
        state = {"completed": False, "messages": messages}

        # env step -- main loop
        while not state["completed"]:
            state, output = await self.env_step(state, llm, sampling_params)

        # combine outputs
        all_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
        return all_ids[len_prompt:]

    def generate(self, prompts: List[List[Dict[str, Any]]], llm: LLM, sampling_params: SamplingParams) -> List[Sequence[int]]:
        outputs = [self.run(prompt, llm, sampling_params) for prompt in prompts]
        return outputs