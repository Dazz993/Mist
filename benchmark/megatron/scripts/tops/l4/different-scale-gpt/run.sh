#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, flash
# run_multinode_experiment_single 4 8 gpt2 H5120-L32 2048 512 32  8 1 4  True
# run_multinode_experiment_single 4 8 gpt2 H5120-L48 2048 512 64  4 1 8  True
# run_multinode_experiment_single 4 8 gpt2 H5120-L64 2048 512 128 4 1 8  True
run_multinode_experiment_single 4 8 gpt2 H5120-L80 2048 512 512 1 2 16 True