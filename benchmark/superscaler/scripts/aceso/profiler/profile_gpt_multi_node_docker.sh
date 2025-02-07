#! /bin/bash
MODEL_SIZE=${1:-"all"}
NNODES=${2:-1}
NODE_RANK=${3:-"%n"}
MASTER_ADDR="localhost"
MASTER_PORT=7000

MAX_NUM_GPUS=$(nvidia-smi -L | wc -l)
MODEL_NAME=gpt

ROOT_PATH=$(pwd)

pdsh -f 1024 -R ssh -w worker-[1-$NNODES] \
    "docker restart aceso && "  \
    "docker exec -i aceso bash -c " \
    "'cd $ROOT_PATH && bash scripts/aceso/profiler/profile_gpt_base.sh $NNODES $NODE_RANK $MASTER_ADDR $MASTER_PORT $MODEL_SIZE'"