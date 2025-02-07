# Benchmark for SuperScaler

## Preparation

Clone the project and switch to the correct branch.
```bash
git clone https://github.com/microsoft/SuperScaler.git
cd SuperScaler
git switch EuroSys24AE
git checkout 53d003681662a817e23c4b1aa47308cf17768d33
git submodule update --init --recursive
```

Patch Megatron-LM.
```bash
cd external/Megatron-LM
cp ../aceso_ae_megatron.patch ./
git apply --whitespace=nowarn aceso_ae_megatron.patch
cd ../..
```

Patch SuperScaler for benchmarking the configurations of Mist.
```bash
cp ../super_scaler.patch ./
git apply --whitespace=nowarn super_scaler.patch
cd ..
```


Make sure you are under `/path/to/ProjectRoot/benchmark/superscaler` instead of `/path/to/ProjectRoot/benchmark/superscaler/SuperScaler/`.

Build the docker image.
```bash
docker build -t aceso-image:latest -f Dockerfile.cu121 .
```

Launch a contrainer.
```bash
docker run -it -d --name=aceso --gpus=all --privileged --net=host --ipc=host --shm-size=1g --ulimit memlock=-1 -v $(pwd):$(pwd) aceso-image bash
```

For the multi-node case, launch containers on all nodes.

With a single cmd:
```bash
bash scripts/setup.sh
```

## (Optional) Step 1: Profile (40 minutes)

The profile step can be skipped in the artifact as we provided a pre-profiled database `xxxx`. But you can also profile on your own by executing:

```bash
# Intra node profiler p2p
bash scripts/aceso/profiler/profile_large_p2p.sh
# Inter node profiler p2p (need two nodes)
bash scripts/aceso/profiler/profile_large_dist_p2p_multi_node_docker.sh
# GPT profiler
bash scripts/aceso/profiler/profile_gpt_multi_node_docker.sh "all" 1
# Copy the profiled data
```

## Step 2: Search

```bash
bash scripts/aceso/search/aceso_gpt_search.sh
```

## Step 3: Evaluate

Single Node: 
```bash
bash scripts/aceso/run/aceso_gpt_run_gpt_1node.sh
```

Multi Node:
1. First sync the configs for all nodes (for instance, for l4 gpus)
```bash
sudo chown -R zhanda SuperScaler/logs-mist-all-l4/
bash ../../tools/bash/syncdir.sh SuperScaler/logs-mist-all-l4/
```
2. Run the evaluation script
```bash
bash scripts/aceso/run/aceso_gpt_run_gpt_multi_nodes.sh 2
bash scripts/aceso/run/aceso_gpt_run_gpt_multi_nodes.sh 4
```

## Troubleshooting

If you find issues with packaging, perhaps it's because the version is too new.
```bash
pip uninstall -y setuptools packaging && pip install setuptools==59.6.0 packaging==21.3
```