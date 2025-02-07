#! /bin/bash
cd SuperScaler/
ROOT_PATH=$(pwd)
cd $ROOT_PATH/search

exp_setting=$1
search_budget=200

#### Paths ####
DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | awk '{print tolower($NF)}')
DATABASE_PATH=${ROOT_PATH}/profiler/profiled-time-mist-all-${DEVICE_NAME}/
RESULT_PATH=${ROOT_PATH}/logs-mist-all-${DEVICE_NAME}/aceso/

## Settings used in Aceso Paper
## model_size    num_nodes   gpus_per_node  global_batch_size
## 1_3B          1           2              32
## 2_6B          1           4              64
## 6_7B          1           8              128
## 13B           2           8              256
## 20B           4           8              512
## 40B           8           8              1024

#### skip the search of 1-GPU case, align the config with baseline system:
# config_name=gpt_350M_mbs8_recomp
# config_path=$ROOT_PATH/logs-large/aceso/configs/gpt/350M/
# mkdir -p ${config_path}csv && mkdir mkdir -p ${config_path}top_configs
# cp single_gpu_configs/$config_name.json $ROOT_PATH/logs-large/aceso/configs/gpt/350M/top_configs/
# python3 aceso_cost_model.py \
#     --initial-point single_gpu_configs/$config_name.json \
#     --profiled-time-path $DATABASE_PATH \
#     --num-gpus-per-node 1 \
#     --num-nodes 1 \
#     --save-to-csv ${config_path}csv/info_$config_name.csv

#### Model info ####
model_name=gpt

#### Hardware info ####
# memory_limit=28000    # For V100-32GB
memory_limit=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
memory_limit=$(($memory_limit*7/8))    # 87.5% of the total memory

#### Search algo parameters ####
budget=$search_budget
max_num_hops=7
init_config=balance

global_batch_sizes=(32 64 128 256 512)
model_sizes=("1_3B" "2_6B" "6_7B" "13B" "22B")
num_nodes_list=(1 1 1 2 4)
gpus_per_node_list=(2 4 8 8 8)

for ((index=0; index<5; index=index+1))
do
    global_batch_size=${global_batch_sizes[$index]}
    model_size=${model_sizes[$index]}
    num_nodes=${num_nodes_list[$index]}
    gpus_per_node=${gpus_per_node_list[$index]}

    LOG_PATH=${RESULT_PATH}search/${model_name}/${model_size}/
    CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/
    mkdir -p ${LOG_PATH}trends && mkdir -p ${CONFIG_SAVE_PATH}top_configs && mkdir -p ${CONFIG_SAVE_PATH}csv

    CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) start searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
        
    python3 aceso_search.py \
        --model-name $model_name \
        --model-size $model_size \
        --global-batch-size $global_batch_size \
        --micro-batch-size 1 2 4 8 \
        --num-nodes $num_nodes \
        --num-gpus-per-node $gpus_per_node \
        --memory-limit $memory_limit \
        --log-path $LOG_PATH \
        --profiled-time-path $DATABASE_PATH \
        --config-save-path $CONFIG_SAVE_PATH \
        --config-suffix $CURRENT_TIME \
        --max-num-hops $max_num_hops \
        --time-budget-total $budget \
        --initial-point $init_config \
        --num-of-saved-configs 3 \
        2>&1 | tee ${LOG_PATH}log_${model_name}_${model_size}_budget${budget}_${CURRENT_TIME}.log
    echo "[LOG][SEARCH]($(date '+%Y-%m-%d-%H-%M-%S')) end searching for $model_name, $model_size, $num_nodes nodes * $gpus_per_node GPUs." >> ${RESULT_PATH}full_log.log
done


 
