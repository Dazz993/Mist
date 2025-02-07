#!/bin/bash

source ./scripts/base_sweap.sh

# nnodes, nproc_per_node, model_type, model_size, seq_length, global_batch_size, grad_accumu, dp, tp, pp, ds_stage, flash

# DS_STAGE=1
run_multinode_experiment_single 4 8 gpt2 22b 4096 512 128 4 2 4 "1" False
run_multinode_experiment_single 4 8 gpt2 22b 4096 512 128 4 2 4 "1" True