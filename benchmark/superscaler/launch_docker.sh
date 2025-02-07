#!/bin/bash

# Assume you are under superscaler root directory
# Double check the current directory by check whether "SuperScaler" directory exists
if [ ! -d "SuperScaler" ]; then
    echo "Please run this script under the root directory of SuperScaler"
    exit 1
fi

launch_docker_cmd="docker run -it -d --name=aceso --gpus=all --privileged \
                   --net=host --ipc=host --shm-size=1g --ulimit memlock=-1 \
                   -v $(pwd):$(pwd) aceso-image bash"

NNODES=${1:-1}
if [ $NNODES -eq 1 ]; then
    eval $launch_docker_cmd
    exit 0
else
    pdsh -f 1024 -R ssh -w worker-[1-$NNODES] bash -c "cd $(pwd) && $launch_docker_cmd"
fi