#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, flash

# Two GPUs - 1.3b
run_experiment_single 1 2 llama 1.3b 2048 32  2  2 1 1 True

# Four GPUs - 2.7b
run_experiment_single 1 4 llama 2.7b 2048 64  16  4 1 1 True

# Eight GPUs - 7b
run_experiment_single 1 8 llama 7b   2048 128 64  2 1 4 True