#!/bin/bash

# Set up environment
ROOT_PATH=$(pwd)
RESULTS_DIR="results"
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
DATE_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

# Add Megatron-LM to PYTHONPATH
export PYTHONPATH=$(pwd)/Megatron-LM/:\$PYTHONPATH

# MultiNode settings
# Change the environment variables below to match your system
CONDA_ENV="dazzle"
CONDA_STR=". ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
PATH_STR="export PATH=${PATH}"
CUDA_STR="export CUDA_HOME=${CUDA_HOME} && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
PYTHONPATH_STR="export PYTHONPATH=${PYTHONPATH}"

# Set extra NCCL and other performance related environment variables here
if [ $DEVICE_NAME = "l4" ] || [ $DEVICE_NAME = "v100-sxm2-16gb" ]; then
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4  CUDA_VISIBLE_DEVICES=0,1,3,2,7,6,4,5"
else
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4 "
fi

INIT_STR="${CONDA_STR} && ${PATH_STR} && ${CUDA_STR} && ${PYTHONPATH_STR} && ${OPT_ENV_STR}"

run_experiment_sweap() {
    NUM_HOSTS=$1
    NUM_DEVICES_PER_HOST=$2
    MODEL=$3
    GLOBAL_BATCH_SIZE=$4
    SEQUENCE_LENGTH=$5
    MAX_TP_SIZE=$6
    USE_FLASHATTN=$7
    MASTER_ADDR=${8:-"localhost"}
    MASTER_PORT=${9:-"28888"}
    NODE_RANK=${10:-"0"}

    CURR_RESULTS_DIR=${ROOT_PATH}/${RESULTS_DIR}/${DEVICE_NAME}/${MODEL}/n_${NUM_HOSTS}-m_${NUM_DEVICES_PER_HOST}/gbs_${GLOBAL_BATCH_SIZE}-s_${SEQUENCE_LENGTH}-flashattn_${USE_FLASHATTN}
    CURR_EXP_NAME=${MODEL}-n_${NUM_HOSTS}-m_${NUM_DEVICES_PER_HOST}-gbs_${GLOBAL_BATCH_SIZE}-s_${SEQUENCE_LENGTH}-flashattn_${USE_FLASHATTN}

    # Check num hosts
    if [ "$NUM_HOSTS" -eq "1" ]; then
        MASTER_ADDR="localhost"
    else
        if [ "$MASTER_ADDR" = "localhost" ]; then
            echo "ERROR: Master address cannot be localhost for multi-node runs"
            exit 1
        fi
    fi

    # Setup flashattn flag
    if [ "$USE_FLASHATTN" = "true" ]; then
        FLASHATTN="--use_flash_attn"
    else
        FLASHATTN=""
    fi

    mkdir -p $CURR_RESULTS_DIR

    python -u benchmark_gpt_bert.py \
        -n $NUM_HOSTS \
        -m $NUM_DEVICES_PER_HOST \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --model $MODEL \
        --global_batch_sizes $GLOBAL_BATCH_SIZE \
        --seq_length $SEQUENCE_LENGTH \
        --max_tp_size $MAX_TP_SIZE \
        --results_dir $CURR_RESULTS_DIR \
        --exp_name $CURR_EXP_NAME \
        ${FLASHATTN}
}

run_multinode_experiment_sweap() {
    NUM_HOSTS=$1
    NUM_DEVICES_PER_HOST=$2
    MODEL=$3
    GLOBAL_BATCH_SIZE=$4
    SEQUENCE_LENGTH=$5
    MAX_TP_SIZE=$6
    USE_FLASHATTN=$7
    MASTER_ADDR=${8:-"auto"}
    MASTER_PORT=${9:-"28888"}

    if [ "$MASTER_ADDR" = "auto" ]; then
        MASTER_ADDR=$(hostname -I | awk '{print $1}')
    fi

    echo "--- Running multi-node experiment (Master: ${MASTER_ADDR}:${MASTER_PORT}) ---"

    pdsh -f 1024 -R ssh -w worker-[1-$NUM_HOSTS] \
        "${INIT_STR} && \
        cd ${ROOT_PATH} && \
        source ./scripts/base_sweap.sh && \
        run_experiment_sweap $NUM_HOSTS $NUM_DEVICES_PER_HOST \
            $MODEL $GLOBAL_BATCH_SIZE $SEQUENCE_LENGTH \
            $MAX_TP_SIZE $USE_FLASHATTN \
            $MASTER_ADDR $MASTER_PORT %n"
}

get_case() {
    MODEL_TYPE=$1
    MODEL_SIZE=$2
    SEQUENCE_LENGTH=$3
    GLOBAL_BATCH_SIZE=$4
    GRADIENT_ACCUMULATION_STEPS=$5
    DP_SIZE=$6
    TP_SIZE=$7
    PP_SIZE=$8
    USE_FLASHATTN=$9

    VOCAB_SIZE=50304

    if [ "$MODEL_SIZE" = "1.3b" ]; then
        MODEL_CONFIG="2048, 24, 16"
    elif [ "$MODEL_SIZE" = "2.7b" ]; then
        MODEL_CONFIG="2560, 32, 32"
    elif [ "$MODEL_SIZE" = "7b" ]; then
        MODEL_CONFIG="4096, 32, 32"
    elif [ "$MODEL_SIZE" = "13b" ]; then
        MODEL_CONFIG="5120, 40, 40"
    elif [ "$MODEL_SIZE" = "22b" ]; then
        MODEL_CONFIG="6144, 48, 64"
    elif [ "$MODEL_SIZE" == "H5120-L32" ]; then
        MODEL_CONFIG="5120, 32, 40"
    elif [ "$MODEL_SIZE" == "H5120-L48" ]; then
        MODEL_CONFIG="5120, 48, 40"
    elif [ "$MODEL_SIZE" == "H5120-L64" ]; then
        MODEL_CONFIG="5120, 64, 40"
    elif [ "$MODEL_SIZE" == "H5120-L80" ]; then
        MODEL_CONFIG="5120, 80, 40"
    else
        echo "ERROR: Invalid model size"
        exit 1
    fi

    CASE="('$MODEL_TYPE', $GLOBAL_BATCH_SIZE, ($SEQUENCE_LENGTH, $MODEL_CONFIG, $VOCAB_SIZE), $GRADIENT_ACCUMULATION_STEPS, ($DP_SIZE, $TP_SIZE, $PP_SIZE, True, True, $USE_FLASHATTN), False)"
}

run_experiment_single() {
    NUM_HOSTS=$1
    NUM_DEVICES_PER_HOST=$2
    MODEL_TYPE=$3
    MODEL_SIZE=$4
    SEQUENCE_LENGTH=$5
    GLOBAL_BATCH_SIZE=$6
    GRADIENT_ACCUMULATION_STEPS=$7
    DP_SIZE=$8
    TP_SIZE=$9
    PP_SIZE=${10}
    USE_FLASHATTN=${11}
    MASTER_ADDR=${12:-"localhost"}
    MASTER_PORT=${13:-"28888"}
    NODE_RANK=${14:-"0"}

    MODEL=$MODEL_TYPE-$MODEL_SIZE
    CURR_RESULTS_DIR=${ROOT_PATH}/${RESULTS_DIR}/${DEVICE_NAME}/${MODEL}/n_${NUM_HOSTS}-m_${NUM_DEVICES_PER_HOST}/gbs_${GLOBAL_BATCH_SIZE}-s_${SEQUENCE_LENGTH}-flashattn_${USE_FLASHATTN}
    CURR_EXP_NAME=${MODEL}-n_${NUM_HOSTS}-m_${NUM_DEVICES_PER_HOST}-gbs_${GLOBAL_BATCH_SIZE}-s_${SEQUENCE_LENGTH}-flashattn_${USE_FLASHATTN}-${DATE_TIME}

    # Check num hosts
    if [ "$NUM_HOSTS" -eq "1" ]; then
        MASTER_ADDR="localhost"
    else
        if [ "$MASTER_ADDR" = "localhost" ]; then
            echo "ERROR: Master address cannot be localhost for multi-node runs"
            exit 1
        fi
    fi

    mkdir -p $CURR_RESULTS_DIR

    get_case $MODEL_TYPE $MODEL_SIZE $SEQUENCE_LENGTH $GLOBAL_BATCH_SIZE $GRADIENT_ACCUMULATION_STEPS $DP_SIZE $TP_SIZE $PP_SIZE $USE_FLASHATTN

    set -x

    torchrun --nnodes $NUM_HOSTS --nproc_per_node $NUM_DEVICES_PER_HOST \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        benchmark_gpt_bert_one_case.py \
        "$CASE" \
        $CURR_RESULTS_DIR \
        $CURR_EXP_NAME >> ${CURR_RESULTS_DIR}/${CURR_EXP_NAME}.log 2>&1
}

run_multinode_experiment_single() {
    NUM_HOSTS=$1
    NUM_DEVICES_PER_HOST=$2
    MODEL_TYPE=$3
    MODEL_SIZE=$4
    SEQUENCE_LENGTH=$5
    GLOBAL_BATCH_SIZE=$6
    GRADIENT_ACCUMULATION_STEPS=$7
    DP_SIZE=$8
    TP_SIZE=$9
    PP_SIZE=${10}
    USE_FLASHATTN=${11}
    MASTER_ADDR=${12:-"auto"}
    MASTER_PORT=${13:-"28888"}

    if [ "$MASTER_ADDR" = "auto" ]; then
        MASTER_ADDR=$(hostname -I | awk '{print $1}')
    fi

    echo "--- Running multi-node experiment (Master: ${MASTER_ADDR}:${MASTER_PORT}) ---"

    pdsh -f 1024 -R ssh -w worker-[1-$NUM_HOSTS] \
        "${INIT_STR} && \
        cd ${ROOT_PATH} && \
        source ./scripts/base_sweap.sh && \
        run_experiment_single $NUM_HOSTS $NUM_DEVICES_PER_HOST \
            $MODEL_TYPE $MODEL_SIZE $SEQUENCE_LENGTH \
            $GLOBAL_BATCH_SIZE $GRADIENT_ACCUMULATION_STEPS \
            $DP_SIZE $TP_SIZE $PP_SIZE $USE_FLASHATTN \
            $MASTER_ADDR $MASTER_PORT %n"
}