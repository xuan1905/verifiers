from typing import List, Dict, Any, Sequence, Tuple, Union

from vllm import LLM, SamplingParams, RequestOutput # type: ignore

from verifiers.envs.math_env import MathEnv
from verifiers.prompts import SIMPLE_PROMPT, DOUBLECHECK_FEW_SHOT


class DoubleCheckEnv(MathEnv):
    def __init__(self, 
                 dataset: str = "gsm8k",
                 system_prompt: str = SIMPLE_PROMPT,
                 few_shot: List[Dict[str, str]] = DOUBLECHECK_FEW_SHOT[0],
                 fields: List[str | Tuple[str, ...]] = ["reasoning", "answer"],
                 **kwargs):
        super().__init__(dataset=dataset, system_prompt=system_prompt, few_shot=few_shot, fields=fields, **kwargs)

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
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        all_completed = False
        states = [{"messages": m, "completed": False, "prompt_tokens": -1, "prompt_messages": len(m)} for m in prompts]
        responses = []
        while not all_completed:
            states, responses = self.step(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)
        all_ids = [list(r.prompt_token_ids) + list(r.outputs[0].token_ids) for r in responses]
        completion_ids = [a[states[idx]["prompt_tokens"]:] for idx, a in enumerate(all_ids)]

        self.logger.info(f"First completion: {str(states[0]['messages'])}")
        if output_type == "ids":
            return completion_ids # type: ignore
        elif output_type == "messages":
            return [[{"role": "assistant", "content": m.outputs[0].text} for m in s["messages"]] for s in states]
        else:
            raise ValueError(f"Invalid output type: {output_type}")
