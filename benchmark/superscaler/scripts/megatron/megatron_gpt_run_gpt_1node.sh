ROOT_PATH=$(pwd)
docker restart aceso
docker exec -i aceso bash -c "cd $(pwd) && bash scripts/mist/megatron/run_gpt_1node.sh"