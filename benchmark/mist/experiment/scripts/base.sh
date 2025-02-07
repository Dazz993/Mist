#!/bin/bash
ROOT_PATH=$(pwd)

DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
VOCAB_SIZE=50304
LOG_DIR="logs"
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')

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

run () {
  NUM_HOSTS=$1
  NUM_DEVICES_PER_HOST=$2
  MODEL_NAME=$3
  MODEL_SIZE=$4
  SEQ_LEN=$5
  GLOBAL_BATCH_SIZE=$6
  USE_FLASH_ATTENTION=$7
  EXTRA_ARGS=${8:-""}

  NUM_GPUS=$((NUM_HOSTS * NUM_DEVICES_PER_HOST))
  CASE_NAME=${MODEL_NAME}/${MODEL_SIZE}/N_${NUM_GPUS}-S_${SEQ_LEN}-B_${GLOBAL_BATCH_SIZE}-F_${USE_FLASH_ATTENTION}
  LOG_PATH=${ROOT_PATH}/${LOG_DIR}/tune/${DEVICE_NAME}/${CASE_NAME}/
  RUNTIME_LOG_PATH=${ROOT_PATH}/${LOG_DIR}/runtime/${DEVICE_NAME}/${CASE_NAME}/

  if [ "$USE_FLASH_ATTENTION" = false ] ; then
    flash_attn="--disable-flash-attn"
  else
    flash_attn=""
  fi

  mkdir -p $LOG_PATH && mkdir -p $RUNTIME_LOG_PATH

  echo "--- Running experiment with $NUM_HOSTS hosts and $NUM_DEVICES_PER_HOST devices per host ---"
  echo [TIME] before running ${MODEL_NAME}/${MODEL_SIZE}_${NUM_GPUS}_gpus: $(date '+%Y-%m-%d-%H-%M-%S') >> ${ROOT_PATH}/${LOG_DIR}/full_log.log

  eval $OPT_ENV_STR

  LOG_PATH=$LOG_PATH \
  RUNTIME_LOG_PATH=$RUNTIME_LOG_PATH \

  python run.py --model ${MODEL_NAME}/${MODEL_SIZE} \
    --seq-len ${SEQ_LEN} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --nnodes ${NUM_HOSTS} \
    --nproc-per-node ${NUM_DEVICES_PER_HOST} \
    --vocab-size ${VOCAB_SIZE} \
    ${flash_attn} \
    ${EXTRA_ARGS} \
    |& tee -a ${LOG_PATH}${MODEL_NAME}_${MODEL_SIZE}_${NUM_GPUS}_gpus_${CURRENT_TIME}.log

  echo [TIME] after running ${MODEL_NAME}/${MODEL_SIZE}_${NUM_GPUS}_gpus: $(date '+%Y-%m-%d-%H-%M-%S') >> ${ROOT_PATH}/logs/full_log.log
  sleep 0.1 # for ctrl+c to work
}

run_multinode () {
  NUM_HOSTS=$1
  NUM_DEVICES_PER_HOST=$2
  MODEL_NAME=$3
  MODEL_SIZE=$4
  SEQ_LEN=$5
  GLOBAL_BATCH_SIZE=$6
  USE_FLASH_ATTENTION=$7
  EXTRA_ARGS=${8:-""}
  MASTER_ADDR=${9:-"auto"}
  MASTER_PORT=${10:-"28888"}

  if [ "$MASTER_ADDR" = "auto" ]; then
      MASTER_ADDR=$(hostname -I | awk '{print $1}')
  fi

  NUM_GPUS=$((NUM_HOSTS * NUM_DEVICES_PER_HOST))
  CASE_NAME=${MODEL_NAME}/${MODEL_SIZE}/N_${NUM_GPUS}-S_${SEQ_LEN}-B_${GLOBAL_BATCH_SIZE}-F_${USE_FLASH_ATTENTION}
  LOG_PATH=${ROOT_PATH}/${LOG_DIR}/tune/${DEVICE_NAME}/${CASE_NAME}/
  RUNTIME_LOG_PATH=${ROOT_PATH}/${LOG_DIR}/runtime/${DEVICE_NAME}/${CASE_NAME}/

  if [ "$USE_FLASH_ATTENTION" = false ] ; then
    flash_attn="--disable-flash-attn"
  else
    flash_attn=""
  fi

  mkdir -p $LOG_PATH && mkdir -p $RUNTIME_LOG_PATH

  echo "--- Running experiment with $NUM_HOSTS hosts and $NUM_DEVICES_PER_HOST devices per host ---"
  echo [TIME] before running ${MODEL_NAME}/${MODEL_SIZE}_${NUM_GPUS}_gpus: $(date '+%Y-%m-%d-%H-%M-%S') >> ${ROOT_PATH}/${LOG_DIR}/full_log.log

  LOG_PATH=$LOG_PATH \
  RUNTIME_LOG_PATH=$RUNTIME_LOG_PATH \

  # Sync the results directory among all nodes
  full_results_path=$(realpath ./results)
  parentdir="$(dirname "$full_results_path")"
  pdcp -f 1024 -R ssh -w worker-[1-$NUM_HOSTS] -r $full_results_path $parentdir

  python run.py --model ${MODEL_NAME}/${MODEL_SIZE} \
    --seq-len ${SEQ_LEN} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --nnodes ${NUM_HOSTS} \
    --nproc-per-node ${NUM_DEVICES_PER_HOST} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    --node-rank "%n" \
    --pdsh-init-cmd "${INIT_STR}" \
    --vocab-size ${VOCAB_SIZE} \
    ${flash_attn} \
    ${EXTRA_ARGS} \
    |& tee -a ${LOG_PATH}${MODEL_NAME}_${MODEL_SIZE}_${NUM_GPUS}_gpus_${CURRENT_TIME}.log

  echo [TIME] after running ${MODEL_NAME}/${MODEL_SIZE}_${NUM_GPUS}_gpus: $(date '+%Y-%m-%d-%H-%M-%S') >> ${ROOT_PATH}/logs/full_log.log
  sleep 0.1 # for ctrl+c to work
}
