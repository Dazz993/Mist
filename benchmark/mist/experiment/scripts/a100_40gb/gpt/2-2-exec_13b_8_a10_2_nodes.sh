#!/bin/bash
source ./scripts/base.sh

#             nnodes, nproc_per_node, model,  model_size, seq_len, global_batch_size, use_flash_attention, extra_args
run_multinode 2       8               "gpt2"  "22b"      4096     256                 true                 "--skip-tune"
run_multinode 2       8               "gpt2"  "22b"      4096     256                 false                "--skip-tune"
