import re
from typing import List, Callable, Dict, Any
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.parsers import XMLParser


class ToolRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        
        # Initialize reward functions
        self.reward_funcs = [
            self.correctness_reward_func,
            self.xml_reward_func,
            self.format_reward_func,
            self.tool_execution_reward_func
        ]

    def get_assistant_messages(self, trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a trajectory."""
        return [msg for msg in trajectory if msg['role'] == 'assistant']

    def get_last_answer(self, trajectory: List[Dict[str, str]]) -> str | None:
        """Extract the last answer from a trajectory."""
        for msg in reversed(trajectory):
            if msg['role'] == 'assistant':
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'answer') and parsed.answer is not None:
                    return parsed.answer
        return None

    def correctness_reward_func(self, completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""
        responses = [self.get_last_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def xml_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward function that checks for proper XML tag usage."""
        def count_xml(trajectory) -> float:
            # Get assistant messages
            model_messages = self.get_assistant_messages(trajectory)
            if not model_messages:
                return 0.0
            
            # Calculate XML tag usage scores for each message
            xml_scores = []
            for msg in model_messages:
                content = msg['content']
                score = 0
                
                # Check for reasoning open tag
                score += 1 - abs(content.count("<reasoning>") - 1)
                score += 1 - abs(content.count("</reasoning>") - 1)
                
                # Check for either tool or answer tags
                if content.count("<tool>") > 0 or content.count("</tool>") > 0:
                    score += 1 - abs(content.count("<tool>") - 1)
                    score += 1 - abs(content.count("</tool>") - 1)
                else:
                    score += 1 - abs(content.count("<answer>") - 1)
                    score += 1 - abs(content.count("</answer>") - 1)
                
                xml_scores.append(score)
            
            # Return average XML score
            if not xml_scores:
                return 0.0
            return 0.2 * (sum(xml_scores) / len(xml_scores))

        return [count_xml(c) for c in completions]

    def format_reward_func(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if each step follows the expected format."""
        def check_format(trajectory):
            model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
            if not model_messages:
                return 0.0
            
            # Calculate format adherence for each message
            format_scores = []
            for msg in model_messages:
                content = msg['content']
                parsed = self.parser.parse(content)
                parsed_no_strip = self.parser.parse(content, strip=False)
                starts_with_reasoning = content.strip().startswith("<reasoning>")
                ends_with_answer = content.strip().endswith("</answer>")
                ends_with_tool = content.strip().endswith("</tool>")
                # Message has correct format if it has reasoning and either tool or answer
                has_correct_format = (
                    hasattr(parsed, 'reasoning') and parsed.reasoning is not None and
                    ((hasattr(parsed, 'tool') and parsed.tool is not None) or 
                        (hasattr(parsed, 'answer') and parsed.answer is not None))
                )
                has_correct_spacing = (
                    has_correct_format and
                    (
                        (hasattr(parsed_no_strip, 'reasoning') and parsed_no_strip.reasoning is not None and
                            (hasattr(parsed_no_strip, 'tool') and parsed_no_strip.tool is not None or
                            hasattr(parsed_no_strip, 'answer') and parsed_no_strip.answer is not None))
                    )
                )
                format_score = 0.4 if has_correct_format else 0.0
                if has_correct_spacing:
                    format_score += 0.2
                if starts_with_reasoning:
                    format_score += 0.2
                if ends_with_answer or ends_with_tool:
                    format_score += 0.2
                format_scores.append(format_score)
            
            # Return average format adherence, weighted by number of steps
            if not format_scores:
                return 0.0
            return 0.2 * (sum(format_scores) / len(format_scores))
        
        return [check_format(c) for c in completions]

    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.
        
        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            # Check response with env_parser
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                successful_executions += 1
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return 0.2 * (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]

    def get_reward_funcs(self) -> List[RewardFunc]:
        """Return the list of reward functions for use in the trainer."""
        return self.reward_funcs  # type: ignore 