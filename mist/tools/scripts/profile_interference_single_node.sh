#!/bin/bash

DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')

# Set extra NCCL and other performance related environment variables here
if [ $DEVICE_NAME = "l4" ] || [ $DEVICE_NAME = "v100-sxm2-16gb" ]; then
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4 CUDA_VISIBLE_DEVICES=0,1,3,2,7,6,4,5"
else
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4"
fi

INIT_STR="${OPT_ENV_STR}"

run_profiling() {
    NUM_HOSTS=$1
    NUM_DEVICES_PER_HOST=$2
    INTRA_GROUP_SIZE=${3:-$NUM_DEVICES_PER_HOST}
    INTER_GROUP_SIZE=${4:-$NUM_HOSTS}
    EXTRA_ARGS=${5:-""}

    ${INIT_STR} && \
    torchrun --nproc_per_node $NUM_DEVICES_PER_HOST \
        profile_interference.py \
        --intra-group-size $INTRA_GROUP_SIZE --inter-group-size $INTER_GROUP_SIZE \
        ${EXTRA_ARGS} >> stdout.log 2>&1
}

# run_profiling 1 8 8 1
# run_profiling 1 8 4 1
run_profiling 1 8 2 1

