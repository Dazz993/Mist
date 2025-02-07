#! /bin/bash
cd SuperScaler/profiler/
ROOT_PATH=$(pwd)
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
PROFILING_PATH=profiled-time-mist-all-${DEVICE_NAME}/
mkdir -p ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_inter_node.csv

MASTER_ADDR=$(hostname -I | awk '{print $1}')
NODE_RANK="%n"

pdsh -f 1024 -R ssh -w worker-[1-2] \
    "docker restart aceso && "  \
    "docker exec -i aceso bash -c " \
    "'cd $ROOT_PATH && " \
    "set -x && " \
    "MASTER_ADDR=$MASTER_ADDR " \
    "MASTER_PORT=7000 " \
    "NNODES=2 " \
    "GPUS_PER_NODE=1 " \
    "NODE_RANK=%n " \
    "FILE_NAME=$FILE_NAME " \
    "python3 p2p_band_profiler.py'"