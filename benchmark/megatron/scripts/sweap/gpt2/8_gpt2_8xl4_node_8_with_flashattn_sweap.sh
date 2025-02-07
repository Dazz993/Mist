#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model, global_batch_size, seq_length, max_tp_size, use_flash_attn
run_multinode_experiment_sweap 8 8 gpt2-40b 1024 2048 8 true