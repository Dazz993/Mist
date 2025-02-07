#! /bin/bash

ROOT_PATH=$(pwd)
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')

# Set extra NCCL and other performance related environment variables here
if [ $DEVICE_NAME = "l4" ] || [ $DEVICE_NAME = "v100-sxm2-16gb" ]; then
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4  CUDA_VISIBLE_DEVICES=0,1,3,2,7,6,4,5"
else
  OPT_ENV_STR="export NCCL_SOCKET_NTHREADS=4 "
fi

eval $OPT_ENV_STR

cd SuperScaler/runtime

###### 2GPU ######
#### Model info ####
model_name=gpt
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
model_size=1_3B

#### Hardware info ####
NNODES=1
GPUS_PER_NODE=2
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000
NCCL_SOCKET_IFNAME=${1:-"ens7"}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#### Paths ####
RESULT_PATH=../logs-mist-all-${DEVICE_NAME}/aceso/
LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
mkdir -p ${LOG_PATH}csv

for file_name in $(ls $CONFIG_SAVE_PATH)
do
config_name=`basename $file_name .json`
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][RUNTIME]($CURRENT_TIME) start executing cofnig: $config_name ." >> ${RESULT_PATH}full_log.log

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
       --train-iters 3 \
       --eval-iters 0 \
       --lr-decay-iters 320000 \
       --vocab-file vocabs/gpt2-vocab.json \
       --merge-file vocabs/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --DDP-impl local \
       --fp16 \
       --loss-scale 2048 \
       --log-path $LOG_PATH \
       2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}  

echo "[LOG][RUNTIME]($CURRENT_TIME) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

done 

###### 4GPUs ######
#### Model info ####
model_name=gpt
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
model_size=2_6B

#### Hardware info ####
NNODES=1
GPUS_PER_NODE=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#### Paths ####
RESULT_PATH=../logs-mist-all-${DEVICE_NAME}/aceso/
LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
mkdir -p ${LOG_PATH}csv

for file_name in $(ls $CONFIG_SAVE_PATH)
do
config_name=`basename $file_name .json`
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][RUNTIME]($CURRENT_TIME) start executing cofnig: $config_name ." >> ${RESULT_PATH}full_log.log

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
       --train-iters 3 \
       --eval-iters 0 \
       --lr-decay-iters 320000 \
       --vocab-file vocabs/gpt2-vocab.json \
       --merge-file vocabs/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --DDP-impl local \
       --fp16 \
       --loss-scale 2048 \
       --log-path $LOG_PATH \
       2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}  

echo "[LOG][RUNTIME]($CURRENT_TIME) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

done 

###### 8GPUs ######
#### Model info ####
model_name=gpt
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
model_size=6_7B

#### Hardware info ####
NNODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=7000
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#### Paths ####
RESULT_PATH=../logs-mist-all-${DEVICE_NAME}/aceso/
LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
mkdir -p ${LOG_PATH}csv

for file_name in $(ls $CONFIG_SAVE_PATH)
do
config_name=`basename $file_name .json`
CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
echo "[LOG][RUNTIME]($CURRENT_TIME) start executing cofnig: $config_name ." >> ${RESULT_PATH}full_log.log

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
       --train-iters 3 \
       --eval-iters 0 \
       --lr-decay-iters 320000 \
       --vocab-file vocabs/gpt2-vocab.json \
       --merge-file vocabs/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --DDP-impl local \
       --fp16 \
       --loss-scale 2048 \
       --log-path $LOG_PATH \
       2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}  

echo "[LOG][RUNTIME]($CURRENT_TIME) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

done 