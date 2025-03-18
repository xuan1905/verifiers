from typing import Callable, Optional, Union, Any, List
import time
import json
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.bfcl_env import BfclEnv
from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.tools.bfcl_tools import INVOLVED_CLASS_TO_FUNC_DOC_PATH
from huanzhi_utils import load_file

if is_peft_available():
    from peft import PeftConfig # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            debug_generate: bool = False,
            debug_rewards: bool = False,
            **kwargs,
    ):
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.debug_generate = debug_generate
        self.debug_rewards = debug_rewards
        self._eval_started = False
        self._train_started = False

    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        # NOTE: Each row in inputs is each row of the dataset
        prompts = [x["prompt"] for x in inputs] # type: ignore
        #Construct the prompt to be displayed in the console
        prompts_to_display = []
        for x in inputs:
            prompt_to_display = f"First User Request: {x['prompt'][-1]['content']}\n\n"
            prompt_to_display += f"User Question Bank: {json.loads(x['user_question_bank'])}\n" if len(json.loads(x['user_question_bank'])) > 0 else ""
            prompt_to_display += f"Involved Classes: {x['involved_classes']}\n\n"
            prompt_to_display += f"Function Documentation:\n\n"
            for class_name in x['involved_classes']:
                tools = []
                func_doc = load_file(INVOLVED_CLASS_TO_FUNC_DOC_PATH[class_name])
                for func in func_doc:
                    tools.append(func['name'])
                prompt_to_display += f"Tools for {class_name}: {tools}\n\n"
            for i, answer in enumerate(json.loads(x['answer'])):
                prompt_to_display += f"Ground Truth for Turn {i}: {answer}\n"
            prompts_to_display.append(prompt_to_display)
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        all_inputs = gather_object(inputs)

        if self.accelerator.is_main_process:
            if isinstance(self.env, BfclEnv):
                env_result = self.env.generate(
                    prompts=all_prompts,
                    llm=self.llm,
                    sampling_params=self.sampling_params,
                    dataset_rows=all_inputs,
                    debug=self.debug_generate
                )
            else:
                env_result = self.env.generate(
                    prompts=all_prompts,
                    llm=self.llm,
                    sampling_params=self.sampling_params,
                )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
            states = env_result['states']
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            states = [None] * len(all_prompts)
        # raise ValueError("Stop Before Computing Rewards")
        
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        states = broadcast_object_list(states, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        states = states[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)
        
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            if isinstance(self.env, BfclEnv):
                output_reward_func = reward_func(completions=completions, states=states, debug=self.debug_rewards)
            else:
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        # NOTE: self.global_step == 0 is because we do eval at the start of training
        mode = "eval" if self.control.should_evaluate else "train"

        # if not hasattr(self, '_correctness_values'):
        #     self._correctness_values = {
        #         'train': [],
        #         'eval': []
        #     }

        # # Initialize or reset correctness tracking at the start of evaluation or training
        # if mode == "eval" and not self._eval_started:
        #     if self._train_started:
        #         self._train_started = False
        #         if self.accelerator.is_main_process:    
        #             print(f"{['=']*20}Training Stopped at step {self.state.global_step}{['=']*20}")
        #     self._eval_started = True
        #     if self.accelerator.is_main_process:
        #         print(f"{'='*20}Evaluation started at step {self.state.global_step}{'='*20}")
        #     if hasattr(self, '_correctness_values'):
        #         self._correctness_values['eval'] = []
        #     else:
        #         self._correctness_values = {
        #             'train': [],
        #             'eval': []
        #         }

        # if mode == "train" and not self._train_started:
        #     if self._eval_started:
        #         if self.accelerator.is_main_process:
        #             print(f"{['=']*20}Evaluation Stopped at step {self.state.global_step}{['=']*20}")
        #         self._eval_started = False
        #         if not hasattr(self, 'eval_set_accuracy'):
        #             self._metrics[mode]["eval_set_accuracy"] = []
        #         if hasattr(self, '_correctness_values'):
        #             curr_eval_set_accuracy = sum(self._correctness_values[mode]) / len(self._correctness_values[mode])
        #             self._metrics[mode]["eval_set_accuracy"].append(curr_eval_set_accuracy)
        #             if self.accelerator.is_main_process:
        #                 print(f"Length of correctness values: {len(self._correctness_values[mode])}")
        #                 print(f"Eval Set Accuracy: {curr_eval_set_accuracy}")
        #                 print(f"All Eval Set Accuracies: {self._metrics[mode]['eval_set_accuracy']}")
        #     self._train_started = True
        #     if self.accelerator.is_main_process:
        #         print(f"{['=']*20}Training started/resumed at step {self.state.global_step}{['=']*20}")
        #     if hasattr(self, '_correctness_values'):
        #         self._correctness_values['train'] = []
        #     else:
        #         self._correctness_values = {
        #             'train': [],
        #             'eval': []
        #         }

        # Calculate correctness based on unified_reward_func
        # Find the index of unified_reward_func
        unified_idx = None
        for i, reward_func in enumerate(self.reward_funcs):
            if reward_func.__name__ == "unified_reward_func": # type: ignore
                unified_idx = i
                break
    
        # Calculate correctness values for each sample if unified_reward_func exists
        if unified_idx is not None:
            correctness = (rewards_per_func[:, unified_idx] >= 1.0).float()
            # if self.accelerator.is_main_process:
            #     print(f"Correctness: {correctness}")
        
        # # If just start training, calculate the accuracy for the previous eval set
        # if mode == "train":
        #     if self._eval_started:
        #         self._eval_started = False
        #         if self.accelerator.is_main_process:
        #             print(f"{'='*40}Evaluation Stopped at step {self.state.global_step}{'='*40}")
        #     if not self._train_started:
        #         self._train_started = True
        #         if self.accelerator.is_main_process:
        #             print(f"{'='*40}Training started/resumed at step {self.state.global_step}{'='*40}")
        #         if self.state.global_step != 0:
        #             # Record the accuracy for the previous eval set (not for the first eval set)
        #             curr_eval_set_accuracy = sum(self._correctness_values['eval']) / len(self._correctness_values['eval'])
        #             if 'eval_set_accuracy' not in self._metrics["eval"]:
        #                 self._metrics["eval"]["eval_set_accuracy"] = []
        #             self._metrics["eval"]["eval_set_accuracy"].append(curr_eval_set_accuracy)
        #             if self.accelerator.is_main_process:
        #                 print(f"Eval Set Accuracy: {curr_eval_set_accuracy}")
        #                 print(f"All Eval Set Accuracies: {self._metrics['eval']['eval_set_accuracy']}")
        #     # Reset the correctness values for the train set (Don't need to store accuracy for the whole training set)
        #     self._correctness_values['train'] = []
        # if mode == "eval":
        #     if self._train_started:
        #         self._train_started = False
        #         if self.accelerator.is_main_process:
        #             print(f"{'='*40}Training Stopped at step {self.state.global_step}{'='*40}")
        #     if not self._eval_started:
        #         self._eval_started = True
        #         if self.accelerator.is_main_process:
        #             print(f"{'='*40}Evaluation started at step {self.state.global_step}{'='*40}")
        #         self._correctness_values['eval'] = [] # Reset the correctness values for the eval set

        # self._correctness_values[mode].extend(correctness.cpu().tolist())
        curr_batch_accuracy = sum(correctness.cpu().tolist()) / len(correctness.cpu().tolist()) if len(correctness.cpu().tolist()) > 0 else 0
        if 'batch_accuracy' not in self._metrics[mode]:
            self._metrics[mode]["batch_accuracy"] = []
        self._metrics[mode]["batch_accuracy"].append(curr_batch_accuracy)
        # if self.accelerator.is_main_process:    
        #     print(f"{mode.upper()} Batch Accuracy: {curr_batch_accuracy}")
        #     print(f"All {mode.upper()} Batch Accuracies: {self._metrics[mode]['batch_accuracy']}")

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)
        # if self.accelerator.is_main_process:
        #     print(f"Completion Length: {completion_length}")
        #     print(f"All Completion Lengths: {self._metrics[mode]['completion_length']}")
            # raise ValueError("Stop")

        # NOTE: Already Gathered Rewards
        reward_per_func_mean = rewards_per_func.mean(0) # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func_mean[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # prompts_to_log = gather_object(prompts)
            prompts_to_log = gather_object(prompts_to_display)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            # Gather individual reward components for logging
            rewards_per_func_to_log = rewards_per_func.tolist()
            # if self.accelerator.is_main_process:
            #     print(f"Rewards Per Func to Log: {rewards_per_func_to_log}")
            # time.sleep(3)

            if self.accelerator.is_main_process:
                # if is_rich_available():
                #     print_prompt_completions_sample(
                #         # [str(prompts_to_log[0][-1]["content"])],
                #         [prompts_to_log[0]],
                #         [completions_to_log[0]],
                #         [rewards_to_log[0]],
                #         # self.control.should_evaluate,
                #         self.state.global_step,
                #     )
                    # print("Dummy print completion")
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    # Add individual reward components to the table
                    for i, reward_func in enumerate(self.reward_funcs):
                        reward_func_name = reward_func.__name__ # type: ignore
                        table[f"{reward_func_name}"] = [row[i] for row in rewards_per_func_to_log]
                    
                    # Add correctness to the table if unified_reward_func exists
                    if unified_idx is not None:
                        table["correctness"] = [(row[unified_idx] >= 1.0) for row in rewards_per_func_to_log]

                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore
                
        # Clean up any existing instances before new generation
        if self.env.env_instances != {}:
            if self.debug_rewards or self.debug_generate:
                print(f"Found existing instances: {self.env.env_instances}")
                print("Cleaning up any existing instances before new generation")
                time.sleep(3)
            self.env.cleanup_instances()
            self.env.env_instances = {}

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }