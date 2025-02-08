from typing import List, Callable, Dict, Any, Sequence, Tuple

from verifiers.envs.base import BaseEnv

from vllm import LLM, SamplingParams, RequestOutput


class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def get_rubric(self) -> List[Callable[..., list[float]]]:
        return []

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Tuple[Dict[str, Any], RequestOutput]]:
        
        outputs = llm.chat([state["messages"] for state in states], sampling_params=sampling_params) # type: ignore
        for i, state in enumerate(states):
            state["messages"].append({'role': 'assistant', 'content': outputs[i].outputs[0].text})
            state["messages"].append({'role': 'user', 'content': 'Are you sure?'})
            state["prompt_tokens"] = len(outputs[i].prompt_token_ids)

        outputs = llm.chat([state["messages"] for state in states], sampling_params=sampling_params) # type: ignore

        for i, state in enumerate(states):
            state["messages"].append({'role': 'assistant', 'content': outputs[i].outputs[0].text})
            state["completed"] = True
        return states, outputs

    def generate(self, prompts: List[List[Dict[str, Any]]], llm: LLM, sampling_params: SamplingParams) -> List[Sequence[int]]:
        all_completed = False
        states = [{"messages": m, "completed": False, "prompt_tokens": -1} for m in prompts]
        outputs = [None] * len(prompts)
        while not all_completed:
            states, outputs = self.step(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)
        all_ids = [list(output.prompt_token_ids) + list(output.outputs[0].token_ids) for output in outputs]
        completion_ids = [all_ids[states[i]["prompt_tokens"]:] for i, output in enumerate(outputs)]
        return completion_ids
