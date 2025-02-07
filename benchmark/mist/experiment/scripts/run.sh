#!/bin/bash
source ./scripts/base.sh

# Step 1: OP profiling
# (
# export MIST_OP_PROFILING=1

# CUDA_VISIBLE_DEVICES=4  run 2       8               "gpt2"  "7b"      2048     256                 true                 "--skip-exec"  &
# CUDA_VISIBLE_DEVICES=5  run 2       8               "gpt2"  "7b"      2048     256                 false                "--skip-exec"  &
# CUDA_VISIBLE_DEVICES=6  run 4       8               "gpt2"  "13b"      2048     512                 true                 "--skip-exec"  &
# CUDA_VISIBLE_DEVICES=7  run 4       8               "gpt2"  "13b"      2048     512                 false                "--skip-exec"  &
# )

# Step 2: Tuning
# (
# export FAKE_DEVICE_NAME="Tesla V100-SXM2-16GB"

# # CUDA_VISIBLE_DEVICES=0  run 2       8               "gpt2"  "7b"      2048     256                 true                 "--skip-exec"
# # CUDA_VISIBLE_DEVICES=0  run 2       8               "gpt2"  "7b"      2048     256                 false                "--skip-exec"
# CUDA_VISIBLE_DEVICES=0  run 4       8               "gpt2"  "13b"      2048     512                 true                 "--skip-exec"
# # CUDA_VISIBLE_DEVICES=0  run 4       8               "gpt2"  "13b"      2048     512                 false                "--skip-exec"
# )


# In one step:
# (
# CUDA_VISIBLE_DEVICES=0  run 1       2               "gpt2"  "1.3b"      4096     32                 true                 "--skip-exec" &
# CUDA_VISIBLE_DEVICES=1  run 1       2               "gpt2"  "1.3b"      4096     32                 false                "--skip-exec" &
# CUDA_VISIBLE_DEVICES=2  run 1       4               "gpt2"  "2.7b"      4096     64                 true                 "--skip-exec" &
# CUDA_VISIBLE_DEVICES=3  run 1       4               "gpt2"  "2.7b"      4096     64                 false                "--skip-exec" &
# )

# wait

# (
# CUDA_VISIBLE_DEVICES=0 run 1       8               "gpt2"  "7b"        4096     128                true                 "--skip-exec" &
# CUDA_VISIBLE_DEVICES=1 run 1       8               "gpt2"  "7b"        4096     128                false                "--skip-exec" &
# )

# wait

# (
# run 1       2               "gpt2"  "1.3b"      4096     32                 true                 "--skip-tune"
# run 1       2               "gpt2"  "1.3b"      4096     32                 false                "--skip-tune"
# run 1       4               "gpt2"  "2.7b"      4096     64                 true                 "--skip-tune"
# run 1       4               "gpt2"  "2.7b"      4096     64                 false                "--skip-tune"
run 1       8               "gpt2"  "7b"        4096     128                true                 "--skip-tune"
run 1       8               "gpt2"  "7b"        4096     128                false                "--skip-tune"
# )

wait