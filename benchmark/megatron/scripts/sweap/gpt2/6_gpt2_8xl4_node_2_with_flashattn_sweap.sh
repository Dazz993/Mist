#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model, global_batch_size, seq_length, max_tp_size, use_flash_attn
run_multinode_experiment_sweap 2 8 gpt2-13b 256 2048 1 true