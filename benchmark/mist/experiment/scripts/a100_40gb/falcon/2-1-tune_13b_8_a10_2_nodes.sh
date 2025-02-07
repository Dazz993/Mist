#!/bin/bash
source ./scripts/base.sh

#   nnodes, nproc_per_node, model,  model_size, seq_len, global_batch_size, use_flash_attention, extra_args
run 2       8               "falcon"  "13b"      4096     256                 true                 "--skip-exec"
# run 2       8               "falcon"  "13b"      4096     256                 false                "--skip-exec"