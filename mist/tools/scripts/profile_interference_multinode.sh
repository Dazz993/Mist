#!/bin/bash

ROOT_PATH=$(pwd)
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')

# CONDA_ENV="dazzle"
# CONDA_STR=". ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
PATH_STR="export PATH=${PATH}"
CUDA_STR="export CUDA_HOME=${CUDA_HOME} && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
PYTHONPATH_STR="export PYTHONPATH=${PYTHONPATH}"

# Set extra NCCL and other performance related environment variables here
if [ $DEVICE_NAME = "l4" ] || [ $DEVICE_NAME = "v100-sxm2-16gb" ]; then
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4 CUDA_VISIBLE_DEVICES=0,1,3,2,7,6,4,5"
else
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4"
fi

INIT_STR="${CONDA_STR} && ${PATH_STR} && ${CUDA_STR} && ${PYTHONPATH_STR} && ${OPT_ENV_STR}"

run_multinode_profiling() {
    NUM_HOSTS=$1
    NUM_DEVICES_PER_HOST=$2
    INTRA_GROUP_SIZE=${3:-$NUM_DEVICES_PER_HOST}
    INTER_GROUP_SIZE=${4:-$NUM_HOSTS}
    EXTRA_ARGS=${5:-""}

    echo "--- Running multi-node profiling with N=${NUM_HOSTS}, M=${NUM_DEVICES_PER_HOST}, INTRA=${INTRA_GROUP_SIZE}, INTER=${INTER_GROUP_SIZE} ---"
    pdsh -f 1024 -R ssh -w worker-[1-$NUM_HOSTS] \
        "${INIT_STR} && \
        cd ${ROOT_PATH} && \
        torchrun --nnodes $NUM_HOSTS --nproc_per_node $NUM_DEVICES_PER_HOST --node_rank %n \
            --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
            profile_interference.py \
            --intra-group-size $INTRA_GROUP_SIZE --inter-group-size $INTER_GROUP_SIZE \
            ${EXTRA_ARGS} >> stdout.log 2>&1"
}

# run_multinode_profiling 4 8 8 4
# run_multinode_profiling 4 8 4 4
# run_multinode_profiling 4 8 2 4
# run_multinode_profiling 4 8 1 4

# run_multinode_profiling 2 8 8 2
# run_multinode_profiling 2 8 4 2
# run_multinode_profiling 2 8 2 2
# run_multinode_profiling 2 8 1 2

run_multinode_profiling 1 8 8 1
run_multinode_profiling 1 8 4 1
run_multinode_profiling 1 8 2 1

