import json
import random
from typing import List, Dict, Sequence, Any, Union

from vllm import LLM, SamplingParams # type: ignore

from verifiers.envs.base import BaseEnv


class SimpleEnv(BaseEnv):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = sampling_args

    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        
        custom_sp = sampling_params.clone()
        custom_sp.update(self.sampling_args)

        # get completions
        completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False) # type: ignore

        self.logger.info(
            "Example completion:\n" +
            json.dumps({"role": "assistant", "content": completions[0].outputs[0].text}, indent=4)
        )
        if output_type == "ids":
            return [completion.outputs[0].token_ids for completion in completions]
        elif output_type == "text":
            return [completion.outputs[0].text for completion in completions]
        elif output_type == "messages":
            return [[{"role": "assistant", "content": c.outputs[0].text}] for c in completions]
        else:
            raise ValueError(f"Invalid output type: {output_type}")    