ROOT_PATH=$(pwd)
docker restart aceso
docker exec -i aceso bash -c "cd $(pwd) && bash scripts/aceso/run/run_gpt_1node.sh"