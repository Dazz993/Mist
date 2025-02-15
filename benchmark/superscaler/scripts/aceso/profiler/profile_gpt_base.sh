#! /bin/bash
NNODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_ADDR=${3:-"localhost"}
MASTER_PORT=${4:-7000}

MAX_NUM_GPUS=$(nvidia-smi -L | wc -l)
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
MODEL_NAME=gpt
MODEL_SIZE=${5:-"all"}

cd SuperScaler/profiler/

RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}profiled-time-mist-${MODEL_SIZE}-${DEVICE_NAME}/

mkdir -p ${PROFILING_PATH}

echo "[LOCAL IP] $(hostname -I | awk '{print $1}') " \
        "[NNODES] $NNODES " \
        "[NODE_RANK] $NODE_RANK " \
        "[MASTER_ADDR] $MASTER_ADDR " \
        "[MASTER_PORT] $MASTER_PORT "

for ((tp_size=1; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
do
GPUS_PER_NODE=${tp_size}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

torchrun $DISTRIBUTED_ARGS \
    op_profiler.py \
    --prof-tp-size $tp_size \
    --prof-path $PROFILING_PATH \
    --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_op_profile.pkl \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-repeat-times 40 10 \
    --prof-repeat-threshold 5000 \
    --prof-warmup-times 10 \
    --prof-warmup-threshold 100000 \
    --prof-num-nodes $NNODES \
    --prof-node-rank $NODE_RANK \
    --prof-ref-data ${RUNTIME_PATH}profiled-time-mist-${MODEL_SIZE}-${DEVICE_NAME}/${MODEL_NAME}_op_profile.pkl \
    2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_op_tp${tp_size}.log

echo [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
done

for ((num_gpus=2; num_gpus<=$MAX_NUM_GPUS; num_gpus=num_gpus*2))
do
echo [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

python3 comm_profiler.py \
    --prof-path $PROFILING_PATH \
    --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_comm_profile.pkl \
    --prof-op-time-path $PROFILING_PATH \
    --prof-tp-size $num_gpus \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-warmup-times 5 \
    --prof-repeat-times 20 \
    --max-data-size 4096 \
    2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${num_gpus}gpus.log

echo [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

done