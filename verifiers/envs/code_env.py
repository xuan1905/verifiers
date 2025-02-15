import subprocess
from typing import List, Dict, Any

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import CODE_FEW_SHOT, CODE_PROMPT
from verifiers.rubrics import CodeRubric
from verifiers.utils import preprocess_dataset

class CodeEnv(MultiStepEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 system_prompt: str = CODE_PROMPT,
                 few_shot: List[Dict[str, str]] = CODE_FEW_SHOT[0]):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot)
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot
        )
        self.eval_dataset = None
        self.rubric = CodeRubric()
        self.llm_parser = XMLParser(fields=["reasoning", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def run_code(self, code: str, **kwargs: Any) -> str:
        try:
            # Run the code block in subprocess with 10-second timeout
            result = subprocess.run(
                ['python', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True  # Automatically decodes stdout/stderr to str
            )
            if result.stderr:
                return f"Error: {result.stderr.strip()}"
            return result.stdout.strip() if result.stdout else ""
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 10 seconds"

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid code field (not just None from failed parsing)
            if hasattr(parsed, 'code') and parsed.code is not None:
                output = self.run_code(parsed.code)
                return {"role": "user", "content": self.env_parser.format(output=output)}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Code not found or invalid XML format. Please ensure correct formatting."}
