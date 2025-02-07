#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, flash
run_multinode_experiment_single 2 8 llama 13b 4096 256 32 4 1 4 True
run_multinode_experiment_single 2 8 falcon 13b 4096 256 32 4 1 4 True