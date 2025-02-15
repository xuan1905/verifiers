import re
from typing import List
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.parsers import XMLParser


class CodeRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", ("code", "answer")])

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

        def int_answer_reward_func(completions, answer, **kwargs) -> List[float]:
            """Reward function that checks if the final answer is an integer."""
            responses = [get_last_answer(c) for c in completions]
            return [1.0 if str(r).isdigit() else 0.0 for r in responses]

        def xml_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks for proper XML tag usage."""
            def count_xml(trajectory) -> float:
                # Get the last message from the model
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                text = model_messages[-1]['content']
                count = 0
                # Check for reasoning tag (required)
                count += 1 - abs(text.count("<reasoning>") - 1)
                count += 1 - abs(text.count("</reasoning>") - 1)
                
                # Check for either code or answer tag (one must be present)
                has_code = text.count("<code>") == 1 and text.count("</code>") == 1
                has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
                count += 1.0 if has_code or has_answer else 0.0
                
                return 0.1 * count
            
            return [count_xml(c) for c in completions]

        def format_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks if each step follows the expected format."""
            # Pattern matches either code step or final answer step
            code_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<code>\n.*?\n</code>\n$"
            answer_pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
            
            def check_format(trajectory):
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                # Calculate format adherence for each message
                format_scores = []
                for msg in model_messages:
                    content = msg['content']
                    matches_format = (re.match(code_pattern, content, re.DOTALL) or 
                                    re.match(answer_pattern, content, re.DOTALL))
                    format_scores.append(1.0 if matches_format else 0.0)
                
                # Return average format adherence, weighted by number of steps
                if not format_scores:
                    return 0.0
                return 0.2 * (sum(format_scores) / len(format_scores))
            
            return [check_format(c) for c in completions]

        def code_execution_reward_func(completions, **kwargs) -> List[float]:
            """Reward function that checks code execution success at each step."""
            def check_execution(trajectory):
                total_code_steps = 0
                successful_executions = 0
                
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        try:
                            parsed = self.parser.parse(msg['content'])
                            if hasattr(parsed, 'code') and parsed.code is not None:
                                total_code_steps += 1
                                # Look for the next user message (environment response)
                                if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                    try:
                                        env_response = self.parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(env_response, 'output'):
                                            output = env_response.output
                                            if not (output and output.startswith('Error:')):
                                                successful_executions += 1
                                    except Exception:
                                        continue
                        except Exception:
                            continue
                
                # Return proportional reward based on successful executions
                if total_code_steps == 0:
                    return 0.0
                return 0.2 * (successful_executions / total_code_steps)
            
            return [check_execution(c) for c in completions]

        self.reward_funcs = [
            correctness_reward_func,
            int_answer_reward_func,
            xml_reward_func, 
            format_reward_func,
            code_execution_reward_func
        ]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs  # type: ignore
