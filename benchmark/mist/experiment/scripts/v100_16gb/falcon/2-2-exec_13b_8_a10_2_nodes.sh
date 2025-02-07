#!/bin/bash
source ./scripts/base.sh

#             nnodes, nproc_per_node, model,  model_size, seq_len, global_batch_size, use_flash_attention, extra_args
run_multinode 2       8               "falcon"  "13b"      2048     256                 true                 "--skip-tune"
# run_multinode 2       8               "falcon"  "13b"      2048     256                 false                "--skip-tune"