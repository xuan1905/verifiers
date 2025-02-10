from typing import List, Dict, Any, Tuple

from datasets import load_dataset, Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import MathRubric
from verifiers.prompts import SIMPLE_PROMPT, MATH_FEW_SHOT
from verifiers.utils import extract_boxed_answer, extract_hash_answer

class MathEnv(SimpleEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 system_prompt: str = SIMPLE_PROMPT,    
                 few_shot: List[Dict[str, str]] = MATH_FEW_SHOT[0],
                 fields: List[str | Tuple[str, ...]] = ["reasoning", "answer"],
                 **kwargs):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = self.preprocess_dataset(dataset_name=dataset) 
        self.eval_dataset = None
        self.rubric = MathRubric()
    
    def preprocess_dataset(self, dataset_name: str = "gsm8k", split: str = "train") -> Dataset:
        if dataset_name == "gsm8k":
            dataset: Dataset = load_dataset("openai/gsm8k", "main")[split] # type: ignore
            dataset = dataset.map(lambda x: {
                "prompt": self.format_prompt(x["question"]),
                "answer": extract_hash_answer(x["answer"])
            })
            return dataset
        elif dataset_name == "math":
            dataset: Dataset = load_dataset("chiayewken/competition_math")[split] # type: ignore
            dataset = dataset.map(lambda x: {
                "prompt": self.format_prompt(x["question"]),
                "answer": extract_boxed_answer(x["solution"])
            })
            return dataset
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for MathEnv.")
    
    def get_dataset(self, **kwargs: Any):
        return self.dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def eval(self, batch_size: int = 10, **kwargs: Any):
        # TODO: Implement evaluation step
        if self.eval_dataset is None:
            self.eval_dataset = self.preprocess_dataset(dataset_name=self.dataset_name, split="test")    