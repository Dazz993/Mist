
###### 8GPUs * 4nodes ######
#### Model info ####
model_name=gpt
model_size=40B

#### Hardware info ####
NNODES=8
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')

#### Distributed info ####
## Modify this for distributed training
NODE_RANK=${1:-0}
MASTER_ADDR=${2:-localhost}
MASTER_PORT=${3:-7000}
NCCL_SOCKET_IFNAME=${4:-"ens5"}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

#### Paths ####
cd SuperScaler/runtime
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
       --loss-scale 8192 \
       --log-path $LOG_PATH \
       2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}  

echo "[LOG][RUNTIME]($CURRENT_TIME) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

done 