from typing import List, Dict, Any, Sequence, Tuple, Union

from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams, RequestOutput # type: ignore

from verifiers.envs.base import BaseEnv


class DoubleCheckEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return []

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> Tuple[List[Dict[str, Any]], List[RequestOutput]]:
        
        responses = llm.chat([state["messages"] for state in states], sampling_params=sampling_params) # type: ignore
        for i, state in enumerate(states):
            state["messages"].append({'role': 'assistant', 'content': responses[i].outputs[0].text})
            state["messages"].append({'role': 'user', 'content': 'Are you sure?'})
            state["prompt_tokens"] = len(responses[i].prompt_token_ids)

        responses = llm.chat([state["messages"] for state in states], sampling_params=sampling_params) # type: ignore
        for i, state in enumerate(states):
            state["messages"].append({'role': 'assistant', 'content': responses[i].outputs[0].text})
            state["completed"] = True
        return states, responses

    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str]]:
        all_completed = False
        states = [{"messages": m, "completed": False, "prompt_tokens": -1} for m in prompts]
        responses = []
        while not all_completed:
            states, responses = self.step(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)
        all_ids = [list(r.prompt_token_ids) + list(r.outputs[0].token_ids) for r in responses]
        completion_ids = [a[states[idx]["prompt_tokens"]:] for idx, a in enumerate(all_ids)]
        return completion_ids
