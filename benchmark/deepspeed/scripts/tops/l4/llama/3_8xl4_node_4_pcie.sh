#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, ds_stage, flash

# DS_STAGE=1
# run_multinode_experiment_single 4 8 llama 22b 2048 512 512 1 2 16 "1" False
run_multinode_experiment_single 4 8 llama 22b 2048 512 512 1 2 16 "1" True