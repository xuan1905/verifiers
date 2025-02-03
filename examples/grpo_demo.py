# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from verifiers import DummyEnv, DoubleCheckEnv

# Load and prep dataset

SYSTEM_PROMPT = """\
Respond in the following format, using careful step-by-step thinking:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_reasoning(text: str) -> str:
    reason = text.split("<reasoning>")[-1]
    reason = reason.split("</reasoning>")[0]
    return reason.strip()

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                answer="7"
            )},
            {'role': 'user', 'content': 'Are you sure?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                reasoning="I'm sure, yes.",
                answer="7"
            )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def close_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    def ratio(r, a):
        if r.isdigit() and a.isdigit():
            min_ans = min(abs(int(r)), abs(int(a)))
            max_ans = max(abs(int(r)), abs(int(a)))
            return (min_ans / max_ans) ** 2
        return 0.0

    return [ratio(r, a) for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def word_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_reasoning(r) for r in responses]

    def reasoning_value(r: str):
        words = [
            " hmm",
            " wait",
            " let's",
            " seems",
            " possibly",
            " however",
            " careful",
            " maybe",
            " sure",
            " see ",
            " wonder",
            " not",
            " actually",
            " but",
            " perhaps",
            " interesting",
        ]
        value = 0.0
        for w in words:
            if w in r.lower(): 
                value += 0.1
        return min(value, 1.0)
    return [reasoning_value(r) for r in extracted_responses]

def length_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_reasoning(r) for r in responses]
    return [0.00002 * len(r) for r in extracted_responses]

def line_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_reasoning(r) for r in responses]
    return [min(0.05 * len(r.split('.\n\n')), 0.5) for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?</answer>\n"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001

    count -= 2 *abs(0.5 - count)
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-0.5B-GRPO"
    run_name="Qwen-0.5B-GRPO-gsm8k"
    
#model_name = "meta-llama/Llama-3.2-3B-Instruct"
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=2e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    #weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=1024,
    num_train_epochs=3,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    use_vllm=True,
    vllm_gpu_memory_utilization=0.5,
    #vllm_device="cuda:2",
    log_on_each_node=False,
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj"], #, "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        device_map=None
    ).to("cuda")
    model.gradient_checkpointing_enable()
            
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training (works on single-GPU)
    vf_env = DoubleCheckEnv()

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            close_reward_func,
            line_reward_func,
            length_reward_func,
            word_reward_func,
            correctness_reward_func],
        env=vf_env,
        args=training_args,
        train_dataset=dataset,
        #peft_config=peft_config
    )
    trainer.train()
