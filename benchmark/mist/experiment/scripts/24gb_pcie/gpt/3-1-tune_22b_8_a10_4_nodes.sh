#!/bin/bash
source ./scripts/base.sh

#   nnodes, nproc_per_node, model,  model_size, seq_len, global_batch_size, use_flash_attention, extra_args
run 4       8               "gpt2"  "22b"      2048     512                 true                 "--skip-exec"
run 4       8               "gpt2"  "22b"      2048     512                 false                "--skip-exec"