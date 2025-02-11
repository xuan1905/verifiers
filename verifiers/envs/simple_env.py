import random
from typing import List, Dict, Sequence, Any, Union

from datasets import load_dataset, Dataset
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams # type: ignore

from verifiers.envs.base import BaseEnv

class SimpleEnv(BaseEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [], **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset
        self.system_prompt = system_prompt
        self.few_shot = few_shot

    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def get_dataset(self, **kwargs: Any) -> Dataset:
        dataset: Dataset = load_dataset(self.dataset_name)['train'] # type: ignore
        return dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return []

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 output_type: str = "ids",
                 **kwargs: Any) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        completions = llm.chat(prompts, sampling_params=sampling_params, use_tqdm=False) # type: ignore
        self.logger.info(f"First completion: {completions[0].outputs[0].text}")
        if output_type == "ids":
            return [completion.outputs[0].token_ids for completion in completions]
        elif output_type == "text":
            return [completion.outputs[0].text for completion in completions]
        elif output_type == "messages":
            return [[{"role": "assistant", "content": c.outputs[0].text}] for c in completions]
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    

    