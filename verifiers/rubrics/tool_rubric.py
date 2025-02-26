import re
from typing import List
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.parsers import XMLParser


class ToolRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])

        def get_last_answer(trajectory):
            for msg in reversed(trajectory):
                try:
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'answer') and parsed.answer is not None:
                        return parsed.answer
                except Exception:
                    continue
            return None

        def correctness_reward_func(completions, answer, **kwargs) -> List[float]:
            """Reward function that checks if the final answer matches the expected answer."""
            responses = [get_last_answer(c) for c in completions]
            return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

        def xml_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks for proper XML tag usage."""
            def count_xml(trajectory) -> float:
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                text = model_messages[-1]['content']
                count = 0
                # Check for reasoning tag (required)
                count += 1 - abs(text.count("<reasoning>") - 1)
                count += 1 - abs(text.count("</reasoning>") - 1)
                
                # Check for either tool or answer tag (one must be present)
                has_tool = text.count("<tool>") == 1 and text.count("</tool>") == 1
                has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
                count += 1.0 if has_tool or has_answer else 0.0
                
                return 0.1 * count

            return [count_xml(c) for c in completions]

        def format_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks if each step follows the expected format."""
            tool_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<tool>\n.*?\n</tool>\n$"
            answer_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
            
            def check_format(trajectory):
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                format_scores = []
                for msg in model_messages:
                    content = msg['content']
                    matches_format = (re.match(tool_pattern, content, re.DOTALL) or 
                                    re.match(answer_pattern, content, re.DOTALL))
                    format_scores.append(1.0 if matches_format else 0.0)
                
                if not format_scores:
                    return 0.0
                return 0.2 * (sum(format_scores) / len(format_scores))
            
            return [check_format(c) for c in completions]

        def tool_execution_reward_func(completions, **kwargs) -> List[float]:
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
                                    parsed_response = self.env_parser.parse(env_response)
                                    if hasattr(parsed_response, 'result'):
                                        result = parsed_response.result
                                        if len(result) > 0 and not result.startswith('Error:'):
                                            successful_executions += 1
                        except Exception:
                            continue
                
                if total_tool_steps == 0:
                    return 0.0
                return 0.2 * (successful_executions / total_tool_steps)
            
            return [check_execution(c) for c in completions]

        self.reward_funcs = [
            correctness_reward_func,
            xml_reward_func,
            format_reward_func,
            tool_execution_reward_func
        ]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs  # type: ignore 