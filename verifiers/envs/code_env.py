from typing import List, Dict, Any

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import MathRubric
from verifiers.utils import preprocess_dataset

class CodeEnv(MultiStepEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = []):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot)
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot
        )
        self.eval_dataset = None
        self.rubric = MathRubric()
        self.llm_parser = XMLParser(fields=["reasoning", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        return self.llm_parser.parse(messages[-1]["content"]).get("answer") is not None

    def run_code(self, code: str, **kwargs: Any) -> str:
        return "Hello, world!"

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        code = self.llm_parser.parse(messages[-1]["content"]).get("code")
        if code:
            output = self.run_code(code)
            return {"role": "user", "content": self.env_parser.format(output=output)}
        else:
            return {"role": "user", "content": "Error: Code not found, ensure correct formatting."}
