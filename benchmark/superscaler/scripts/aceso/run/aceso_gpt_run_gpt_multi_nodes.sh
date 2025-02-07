#! /bin/bash
NNODES=${1:-2}
NCCL_SOCKET_IFNAME=${2:-"ens7"}
NODE_RANK="%n"
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=7000

MAX_NUM_GPUS=$(nvidia-smi -L | wc -l)

ROOT_PATH=$(pwd)

pdsh -f 1024 -R ssh -w worker-[1-$NNODES] \
    "docker restart aceso && "  \
    "docker exec -i aceso bash -c " \
    "'cd $ROOT_PATH && bash scripts/aceso/run/run_gpt_${NNODES}nodes_docker.sh $NODE_RANK $MASTER_ADDR $MASTER_PORT $NCCL_SOCKET_IFNAME'"