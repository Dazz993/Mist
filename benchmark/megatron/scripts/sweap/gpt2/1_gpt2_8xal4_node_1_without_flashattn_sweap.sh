#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model, global_batch_size, seq_length, max_tp_size, use_flash_attn
run_experiment_sweap 1 2 gpt2-1.3b 32  2048 1 false
run_experiment_sweap 1 4 gpt2-2.7b 64  2048 1 false
run_experiment_sweap 1 8 gpt2-7b   128 2048 1 false