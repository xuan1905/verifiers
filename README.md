
# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

This repo is a fork of the [verifiers](https://github.com/willccbb/verifiers) repo. The main changes are constructed using an older version of the verifiers repo. Notice that we are using a specific snapshot of the TRL library for the development of this specific fork. We will update this repo to use the latest version of the verifiers repo soon. 

# Installation
To install the dependencies, run the following commands:

```
git clone https://github.com/bespokelabsai/verifiers.git
cd verifiers
uv sync
source .venv/bin/activate
uv pip install flash-attn --no-build-isolation
```

# Reproducing the BFCL Training Result

To reproduce our result, run the following command:
```
accelerate launch --config-file configs/zero3.yaml --num-processes 3 verifiers/examples/bfcl_agent.py
```

The configurations/hyperparameters used are specified in `verifiers/examples/bfcl_agent.py` as global variables. 