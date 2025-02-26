import re
from typing import List
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.parsers import XMLParser


class CodeRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])

        def get_last_answer(trajectory):
            for msg in reversed(trajectory):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'answer') and parsed.answer is not None:
                        return parsed.answer
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
                # Get all messages from the model
                model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
                if not model_messages:
                    return 0.0
                
                # Calculate XML tag usage scores for each message
                xml_scores = []
                for msg in model_messages:
                    content = msg['content']
                    score = 0
                    
                    # Check for reasoning open tag
                    score += 1 - abs(content.count("<reasoning>") - 1)
                    # Check for reasoning close tag
                    score += 1 - abs(content.count("</reasoning>") - 1)
                    
                    # Check for either code or answer tags
                    if content.count("<code>") > 0 or content.count("</code>") > 0:
                        score += 1 - abs(content.count("<code>") - 1)
                        score += 1 - abs(content.count("</code>") - 1)
                    else:
                        score += 1 - abs(content.count("<answer>") - 1)
                        score += 1 - abs(content.count("</answer>") - 1)
                    
                    xml_scores.append(score)
                
                # Return average XML score
                if not xml_scores:
                    return 0.0
                return 0.1 * (sum(xml_scores) / len(xml_scores))
            
            return [count_xml(c) for c in completions]

        def format_reward_func(completions, **kwargs) -> List[float]:
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
                    # Message has correct format if it has reasoning and either code or answer
                    has_correct_format = (
                        hasattr(parsed, 'reasoning') and parsed.reasoning is not None and
                        ((hasattr(parsed, 'code') and parsed.code is not None) or 
                         (hasattr(parsed, 'answer') and parsed.answer is not None))
                    )
                    format_scores.append(1.0 if has_correct_format else 0.0)
                
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
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'code') and parsed.code is not None:
                            total_code_steps += 1
                            # Look for the next user message (environment response)
                            if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                env_response = trajectory[i + 1]['content']
                                parsed_response = self.env_parser.parse(env_response)
                                if hasattr(parsed_response, 'output') and parsed_response.output:
                                    output = parsed_response.output
                                    if len(output) > 0 and not output.startswith('Error:'):
                                        successful_executions += 1
                
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
