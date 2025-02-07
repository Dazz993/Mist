#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, ds_stage, flash

# Two Nodes - GPT2-1.3b
# run_experiment_single 1 2 falcon 1.3b 2048 32  32  1 1 2 "1" False
run_experiment_single 1 2 falcon 1.3b 2048 32  32  1 1 2 "1" True
# run_experiment_single 1 2 falcon 1.3b 2048 32  4   2 1 1 "1" False
run_experiment_single 1 2 falcon 1.3b 2048 32  1   2 1 1 "1" True

# # Four Nodes - GPT2-2.7b
# run_experiment_single 1 4 falcon 2.7b 2048 64  64  1 1 4 "1" False
run_experiment_single 1 4 falcon 2.7b 2048 64  64  1 1 4 "1" True

# Eight Nodes - GPT2-7b
# OOM
# run_experiment_single 1 8 falcon 7b   2048 128 128 1 1 8 "1" False
# run_experiment_single 1 8 falcon 7b   2048 128 128 1 1 8 "1" True
# # ZeRO 1-3
# run_experiment_single 1 8 falcon 7b   2048 128 128 1 8 1 "1" False
run_experiment_single 1 8 falcon 7b   2048 128 128 1 8 1 "1" True
# # ZeRO-Offload
# run_experiment_single 1 8 falcon 7b   2048 128 8 8 1 1 "offload" False
run_experiment_single 1 8 falcon 7b   2048 128 4 8 1 1 "offload" True