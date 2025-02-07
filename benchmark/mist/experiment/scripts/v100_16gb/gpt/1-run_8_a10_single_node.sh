#!/bin/bash
source ./scripts/base.sh

#   nnodes, nproc_per_node, model,  model_size, seq_len, global_batch_size, use_flash_attention, extra_args
# run 1       2               "gpt2"  "1.3b"      2048     32                 true                 "--skip-exec"
# run 1       2               "gpt2"  "1.3b"      2048     32                 false                "--skip-exec"
# run 1       2               "gpt2"  "1.3b"      2048     32                 true                 "--skip-tune"
# run 1       2               "gpt2"  "1.3b"      2048     32                 false                "--skip-tune"

# run 1       4               "gpt2"  "1.3b"      2048     64                 true                 "--skip-exec"
# run 1       4               "gpt2"  "1.3b"      2048     64                 false                "--skip-exec"
# run 1       4               "gpt2"  "1.3b"      2048     64                 true                 "--skip-tune"
# run 1       4               "gpt2"  "1.3b"      2048     64                 false                "--skip-tune"

# run 1       8               "gpt2"  "2.7b"        2048     128                true                 "--skip-exec"
run 1       8               "gpt2"  "2.7b"        2048     128                false                "--skip-exec"
# run 1       8               "gpt2"  "2.7b"        2048     128                true                 "--skip-tune"
# run 1       8               "gpt2"  "2.7b"        2048     128                false                "--skip-tune"