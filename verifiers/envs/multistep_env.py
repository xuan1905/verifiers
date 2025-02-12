from abc import abstractmethod
import json
from typing import List, Dict, Sequence, Any, Union

from vllm import LLM, SamplingParams # type: ignore

from verifiers.envs.base import BaseEnv


class MultiStepEnv(BaseEnv):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = sampling_args
        self.tokenizer = None

    @abstractmethod
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass

    @abstractmethod
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass
    
    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        for i, j in enumerate(live_indices):
            if states[j]["prompt_tokens"] == -1:
                states[j]["prompt_tokens"] = len(llm_responses[i].prompt_token_ids)
                states[j]["prompt_ids"] = llm_responses[i].prompt_token_ids
            states[j]["messages"].append({"role": "assistant", "content": llm_responses[i].outputs[0].text})
        
            # update completion ids
            states[j]["completion_ids"] = list(llm_responses[i].prompt_token_ids)
            states[j]["completion_ids"].extend(list(llm_responses[i].outputs[0].token_ids))
            states[j]["completion_ids"] = states[j]["completion_ids"][states[j]["prompt_tokens"]:]
            

            if self.is_completed(states[j]["messages"]) or len(states[j]["completion_ids"]) > sampling_params.max_tokens:
                states[j]["completed"] = True
                states[j]["completion_ids"] = states[j]["completion_ids"][:sampling_params.max_tokens]

                all_ids = self.tokenizer.apply_chat_template(states[j]["messages"], tokenize=True)
                prompt_ids = self.tokenizer.apply_chat_template(states[j]["messages"][:states[j]["prompt_messages"]], tokenize=True, add_generation_prompt=True)
                states[j]["prompt_tokens_tk"] = len(prompt_ids)
                states[j]["prompt_ids_tk"] = prompt_ids
                states[j]["completion_tokens_tk"] = len(all_ids) - len(prompt_ids)
                states[j]["completion_ids_tk"] = all_ids[len(prompt_ids):][:sampling_params.max_tokens]
            
            else:
                states[j]["messages"].append(self.env_response(states[j]["messages"]))

        return states

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [{
            "messages": m,
            "prompt_messages": len(m),
            "prompt_tokens": -1,
            "prompt_ids": [],
            "completed": False,
            "completion_ids": []
        } for m in prompts]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        self.logger.info(
            "Example prompt:\n" +
            json.dumps(states[0]["messages"][:states[0]["prompt_messages"]], indent=4) +
            "\n\nExample completion:\n" +
            json.dumps(states[0]["messages"][states[0]["prompt_messages"]:], indent=4)
        )

        self.logger.info(f"Prompt IDs: {states[0]['prompt_ids']} \nlen: {len(states[0]['prompt_ids'])}")
        self.logger.info(f"Completion IDs (vllm): {states[0]['completion_ids']} \nlen: {len(states[0]['completion_ids'])}")
        self.logger.info(f"Completion IDs (tk): {states[0]['completion_ids_tk']} \nlen: {len(states[0]['completion_ids_tk'])}")    
        self.logger.info(f"All (vllm): {states[0]['prompt_ids'] + states[0]['completion_ids']} \nlen: {len(states[0]['completion_ids']) + len(states[0]['prompt_ids'])}")
        self.logger.info(f"All (tk): {states[0]['prompt_ids_tk'] + states[0]['completion_ids_tk']} \nlen: {len(states[0]['completion_ids_tk']) + len(states[0]['prompt_ids_tk'])}")
        self.logger.info(f"All (tokenized): {self.tokenizer.apply_chat_template(states[0]['messages'], tokenize=False)}")
        
        if output_type == "ids":
            return [s["completion_ids"] for s in states]
        elif output_type == "messages":
            return [s["messages"][s["prompt_messages"]:] for s in states]
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    

    