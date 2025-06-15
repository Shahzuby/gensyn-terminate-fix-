# RL-Swarm Hivemind Experiment

This repository contains configuration and utility scripts for the RL-Swarm project using Hivemind.

## Configuration

Edit the experiment configuration file:

```bash
cd $HOME/rl-swarm/hivemind_exp/configs/mac/
nano grpo-qwen-2.5-0.5b-deepseek-r1.yaml
```

Example content:

```yaml
torch_dtype: float32
gradient_checkpointing: false
per_device_train_batch_size: 1
```

## Debug Utilities

Located in:

```bash
nano ~/rl-swarm/hivemind_exp/debug_utils.py
```

Includes:

- System info: Python, CPU, Memory, Disk, GPU (NVIDIA/AMD/Apple Silicon)
- Logging: Colored console output + file logs
- Print redirection to logs
