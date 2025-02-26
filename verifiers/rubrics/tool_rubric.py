import re
from typing import List, Callable, Dict, Any
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.parsers import XMLParser


class ToolRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        
        # Compile regex patterns once at initialization
        self.tool_pattern = re.compile(r"<reasoning>[\s\S]*?</reasoning>[\s\S]*?<tool>[\s\S]*?</tool>")
        self.answer_pattern = re.compile(r"<reasoning>[\s\S]*?</reasoning>[\s\S]*?<answer>[\s\S]*?</answer>")
        
        # Initialize reward functions
        self.reward_funcs = [
            self._correctness_reward_func,
            self._xml_reward_func,
            self._format_reward_func,
            self._tool_execution_reward_func
        ]

    def _get_assistant_messages(self, trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a trajectory."""
        return [msg for msg in trajectory if msg['role'] == 'assistant']

    def _get_last_answer(self, trajectory: List[Dict[str, str]]) -> str | None:
        """Extract the last answer from a trajectory."""
        for msg in reversed(trajectory):
            try:
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'answer') and parsed.answer is not None:
                        return parsed.answer
            except Exception:
                continue
        return None

    def _correctness_reward_func(self, completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""
        responses = [self._get_last_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def _xml_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward function that checks for proper XML tag usage."""
        def count_xml(trajectory) -> float:
            model_messages = self._get_assistant_messages(trajectory)
            if not model_messages:
                return 0.0
            
            # Check each assistant message
            xml_scores = []
            for msg in model_messages:
                text = msg['content']
                count = 0
                # Check for reasoning tag (required)
                count += 1 - abs(text.count("<reasoning>") - 1)
                count += 1 - abs(text.count("</reasoning>") - 1)
                
                # Check for either tool or answer tag (one must be present)
                has_tool = text.count("<tool>") == 1 and text.count("</tool>") == 1
                has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
                count += 1.0 if has_tool or has_answer else 0.0
                
                xml_scores.append(0.1 * count)
            
            # Average the XML scores for each message
            return sum(xml_scores) / len(xml_scores) if xml_scores else 0.0

        return [count_xml(c) for c in completions]

    def _format_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward function that checks if each step follows the expected format."""
        def check_format(trajectory):
            model_messages = self._get_assistant_messages(trajectory)
            if not model_messages:
                return 0.0
            
            format_scores = []
            for msg in model_messages:
                content = msg['content']
                matches_format = (self.tool_pattern.search(content) or 
                                 self.answer_pattern.search(content))
                format_scores.append(1.0 if matches_format else 0.0)
            
            if not format_scores:
                return 0.0
            return 0.2 * (sum(format_scores) / len(format_scores))
        
        return [check_format(c) for c in completions]

    def _tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward function that checks tool execution success at each step."""
        def check_execution(trajectory):
            total_tool_steps = 0
            successful_executions = 0
            
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    try:
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            total_tool_steps += 1
                            # Look for the next user message (environment response)
                            if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                env_response = trajectory[i + 1]['content']
                                # First, try to parse with the environment parser
                                try:
                                    parsed_response = self.env_parser.parse(env_response)
                                    if hasattr(parsed_response, 'result'):
                                        result = parsed_response.result
                                        if len(result) > 0 and not result.startswith('Error:'):
                                            successful_executions += 1
                                            continue
                                except Exception:
                                    pass
                                
                                # Fallback: check if the response doesn't mention errors
                                if len(env_response) > 0 and not env_response.lower().startswith('error'):
                                    successful_executions += 1
                    except Exception:
                        continue
            
            if total_tool_steps == 0:
                return 0.0
            return 0.2 * (successful_executions / total_tool_steps)
        
        return [check_execution(c) for c in completions]

    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return the list of reward functions for use in the trainer."""
        return self.reward_funcs  # type: ignore 