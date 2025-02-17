from trl import GRPOTrainer
import verifiers as vf
from verifiers.tools import search

model_name = "Qwen/Qwen2.5-Math-1.5B"  # Could also use a model more focused on science/QA
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Initialize tool environment for OpenBookQA
vf_env = vf.ToolEnv(
    dataset="openbookqa",
    tools=[search],
    max_steps=2  # Most questions need 1-2 searches: one for concept, one for verification
)

# Get train dataset and rubric
train_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

# Configure training
training_args = vf.get_default_grpo_config(
    run_name="openbookqa_search_qwen2.5-m-1.5b",
    num_gpus=8,
    eval_steps=100,  # More frequent eval since dataset is smaller
    save_steps=100
)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset
)

# Train with evaluation
trainer.train() 