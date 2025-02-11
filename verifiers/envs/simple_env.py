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
        sampling_args = {
            #"include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = sampling_args
        self.tokenizer = None

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
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        states = [{
            "messages": m,
            "prompt_messages": len(m),
            "prompt_tokens": -1,
            "prompt_ids": [],
            "completed": False,
            "completion_ids": []
        } for m in prompts]

        # get completions
        completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False) # type: ignore

        for i, completion in enumerate(completions):
            states[i]["prompt_ids"] = completion.prompt_token_ids
            states[i]["prompt_ids_tk"] = self.tokenizer.apply_chat_template(states[i]["messages"][:states[i]["prompt_messages"]], tokenize=True, add_generation_prompt=True)
            states[i]["completion_ids"] = completion.outputs[0].token_ids
            states[i]["completion_ids_tk"] = self.tokenizer.apply_chat_template(states[i]["messages"][states[i]["prompt_messages"]:], tokenize=True, add_generation_prompt=False)
            states[i]["messages"].append({"role": "assistant", "content": completion.outputs[0].text})

        
        self.logger.info(f"Prompt IDs (tk): {states[0]['prompt_ids_tk']} \nlen: {len(states[0]['prompt_ids_tk'])}")
        self.logger.info(f"Prompt IDs (vllm): {states[0]['prompt_ids']} \nlen: {len(states[0]['prompt_ids'])}")
        self.logger.info(f"Completion IDs (tk): {states[0]['completion_ids_tk']} \nlen: {len(states[0]['completion_ids_tk'])}")    
        self.logger.info(f"Completion IDs (vllm): {states[0]['completion_ids']} \nlen: {len(states[0]['completion_ids'])}")
        self.logger.info(f"All (vllm): {states[0]['prompt_ids'] + states[0]['completion_ids']} \nlen: {len(states[0]['completion_ids']) + len(states[0]['prompt_ids'])}")
        self.logger.info(f"All (tk): {states[0]['prompt_ids_tk'] + states[0]['completion_ids_tk']} \nlen: {len(states[0]['completion_ids_tk']) + len(states[0]['prompt_ids_tk'])}")
        self.logger.info(f"All (tokenized): {self.tokenizer.apply_chat_template(states[0]['messages'], tokenize=False)}")
        
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