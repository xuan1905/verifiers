import verifiers as vf
from verifiers.tools import calculate
from verifiers.prompts import SEARCH_FEW_SHOT

model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Initialize tool environment for GSM8K
vf_env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=SEARCH_FEW_SHOT[0],
    tools=[calculate],
    max_steps=3
)

dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(
    run_name="gsm8k_calculator_qwen2.5-m-1.5b",
    num_gpus=8
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)

trainer.train() 