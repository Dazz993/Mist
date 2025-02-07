#!/bin/bash
source ./scripts/base.sh

#   nnodes, nproc_per_node, model,  model_size,       seq_len, global_batch_size, use_flash_attention, extra_args
# run 4       8               "gpt2"  "_H6144_L48"      4096     512                 true                 "--skip-exec"
# run 4       8               "gpt2"  "_H6144_L64"      4096     512                 true                 "--skip-exec"
# run 4       8               "gpt2"  "_H6144_L80"      4096     512                 true                 "--skip-exec"
# run 4       8               "gpt2"  "_H6144_L96"      4096     512                 true                 "--skip-exec"

<<<<<<< Updated upstream
# run 4       8               "gpt2"  "_H6144_L48"      4096     512                 false                "--skip-exec"
# run 4       8               "gpt2"  "_H6144_L64"      4096     512                 false                "--skip-exec"
# run 4       8               "gpt2"  "_H6144_L80"      4096     512                 false                "--skip-exec"
# run 4       8               "gpt2"  "_H6144_L96"      4096     512                 false                "--skip-exec"
=======
run 4       8               "gpt2"  "_H6144_L48"      4096     512                 false                "--skip-exec"
run 4       8               "gpt2"  "_H6144_L64"      4096     512                 false                "--skip-exec"
run 4       8               "gpt2"  "_H6144_L80"      4096     512                 false                "--skip-exec"
run 4       8               "gpt2"  "_H6144_L96"      4096     512                 false                "--skip-exec"
>>>>>>> Stashed changes
