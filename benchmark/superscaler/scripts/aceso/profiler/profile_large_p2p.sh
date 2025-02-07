#! /bin/bash
cd SuperScaler/profiler/
RUNTIME_PATH=$(pwd)
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
PROFILING_PATH=profiled-time-mist-all-${DEVICE_NAME}/
mkdir -p ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_intra_node.csv

docker restart aceso && docker exec -i aceso bash -c \
    "cd ${RUNTIME_PATH} && \
        set -x && \
        MASTER_ADDR=localhost \
        MASTER_PORT=7000 \
        NNODES=1 \
        GPUS_PER_NODE=2 \
        NODE_RANK=0 \
        FILE_NAME=${FILE_NAME} \
        python3 p2p_band_profiler.py"