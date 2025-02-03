# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments.

For now, it supports the TRL implementation of the GRPO algorithm via a [fork](git@github.com:willccbb/trl.git), and requires [vLLM](https://github.com/vllm-project/vllm/tree/main) for inference.

## Installation

We recommend installing via `uv`:

```bash
uv add verifiers
```

Or, if you're old-school:
```bash
pip install verifiers
```

## Usage

```python
from trl import GRPOTrainer, GRPOConfig
from verifiers import DoubleCheckEnv

model_name = "meta-llama/Llama-3.2-1B-Instruct"
training_args = GRPOConfig(use_vllm=True)

vf_env = DoubleCheckEnv()
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=vf_env.get_rubric(),
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```
See `examples/doublecheck.py` for a complete example.


## Citation

If you use this code in your research, please cite:

```bibtex
@article{williamson2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, Will},
  journal={arXiv preprint arXiv:2502.01234},
  year={2025}
}
```
