import verifiers as vf
from verifiers.tools import search
from verifiers.prompts import SEARCH_FEW_SHOT

model_name = "Qwen/Qwen2.5-7B-Instruct" 
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.ToolEnv(
    dataset="openbookqa",
    few_shot=SEARCH_FEW_SHOT[0],
    tools=[search],
    max_steps=2
)
train_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(
    run_name="openbookqa_search_qwen2.5-7b",
    num_gpus=8
)
training_args.gradient_accumulation_steps = 4
training_args.per_device_train_batch_size = 4
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train() 


