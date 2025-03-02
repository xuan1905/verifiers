import verifiers as vf
from verifiers.prompts.system_prompts import CODE_PROMPT

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.CodeEnv(dataset="math", few_shot=[], system_prompt=CODE_PROMPT)
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(run_name="math_qwen2.5-7b", num_gpus=8)
# rollouts per prompt
training_args.num_generations = 7
# minibatch size per GPU ( bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
training_args.per_device_train_batch_size = 6
# batches to accumulate (6 prompts * 4 -> 24 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
# no ref model
training_args.beta = 0.0
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()