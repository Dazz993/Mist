#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, flash
run_experiment_single 1 2 llama  1.3b 4096 32   4     2 1 1 True 
run_experiment_single 1 2 falcon 1.3b 4096 32   4     2 1 1 True 

run_experiment_single 1 4 llama  2.7b 4096 64   4     4 1 1 True
run_experiment_single 1 4 falcon 2.7b 4096 64   4     4 1 1 True

run_experiment_single 1 8 llama  7b   4096 128  8     4 2 1 True
run_experiment_single 1 8 falcon 7b   4096 128  8     4 2 1 True