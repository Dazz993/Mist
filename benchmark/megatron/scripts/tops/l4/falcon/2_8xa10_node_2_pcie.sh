#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, flash
# run_multinode_experiment_single 2 8 falcon 13b 2048 256 128 2 1 8 False
run_multinode_experiment_single 2 8 falcon 13b 2048 256 128 2 1 8 True