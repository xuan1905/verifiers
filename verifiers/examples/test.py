import verifiers as vf
from verifiers.tools import calculator, mean
from verifiers.prompts import CALCULATOR_FEW_SHOT
import time
# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "/home/richard/verifiers/outputs/bfcl-qwen2.5-7b-instruct-3-turns/checkpoint-600"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

NUM_GPUS = 8
MAX_STEPS_PER_TURN = 5
CURRICULUM_LEARNING = True
MAX_TURNS_ONLY = True
MAX_NUM_TURNS = 10 # 4 turns is almost max
DEBUG_GENERATE = False
DEBUG_REWARDS = False
PER_DEVICE_BATCH_SIZE = 2
EVAL_STEPS = 100
SAVE_STEPS = 100
# if DEBUG_GENERATE or DEBUG_REWARDS:
#     NUM_GPUS = 2
#     MAX_STEPS_PER_TURN = 5

# Initialize tool environment for GSM8K
vf_env = vf.BfclEnv(
    dataset="bfcl",
    # few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[],
    max_num_turns=MAX_NUM_TURNS,
    max_steps_per_turn=MAX_STEPS_PER_TURN,
    curriculum_learning=CURRICULUM_LEARNING,
)

train_dataset = vf_env.get_dataset(max_num_turns=MAX_NUM_TURNS)
eval_dataset = vf_env.get_eval_dataset(max_num_turns=MAX_NUM_TURNS, max_turn_only=MAX_TURNS_ONLY, n=10)
# train_dataset = train_dataset.select(range(0,10))
# eval_dataset = eval_dataset.select(range(0,20))
print(train_dataset)
print(eval_dataset)
# raise Exception("Stop")
# time.sleep(5)

rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "bfcl-" + model_name.split("/")[-1].lower() + f"-{MAX_NUM_TURNS}-turns"
# if DEBUG_GENERATE or DEBUG_REWARDS:
#     run_name += "-debug"
# run_name = "bfcl-3B-test-run"
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=NUM_GPUS
)
training_args.num_train_epochs = 30
# rollouts per prompt
training_args.num_generations = NUM_GPUS - 1 if NUM_GPUS > 2 else 2
if DEBUG_GENERATE or DEBUG_REWARDS:
    # training_args.num_generations = 2
    training_args.report_to = "none"
# minibatch size per GPU ( bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
training_args.per_device_train_batch_size = PER_DEVICE_BATCH_SIZE
# batches to accumulate (6 prompts * 4 -> 24 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
# ref model configs
training_args.beta = 0.04
training_args.max_grad_norm = 0.2
# evals
training_args.eval_strategy = "steps"
training_args.eval_on_start = True
training_args.eval_steps = EVAL_STEPS
training_args.save_strategy = "steps"
training_args.save_steps = SAVE_STEPS
training_args.per_device_eval_batch_size = PER_DEVICE_BATCH_SIZE
training_args.eval_accumulation_steps = 1
training_args.data_seed = 42

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    debug_generate=DEBUG_GENERATE,
    debug_rewards=DEBUG_REWARDS,
)

trainer.train() 