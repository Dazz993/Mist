# Mist Artifact for EuroSys 25

In this repository, we provide the artifact for the paper **Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization** 🚀.

Mist is an advanced automatic distributed training configuration optimizing system designed to optimize large language model (LLM) training by co-optimizing parallelism techniques (data, tensor, and pipeline parallelism) alongside memory footprint reduction strategies (activation checkpointing, redundancy elimination, and offloading).

Key Features:
- 🚀 Optimized Performance: Achieves up to 2.04× speedup compared to state-of-the-art automatic systems.
- ⚡ Smart Parallelism & Memory Optimization: Dynamically balances memory usage and compute efficiency.
- 🔍 Symbolic Performance Analysis: Rapidly explores optimization configurations using symbolic expressions.
- 🔄 Overlap-Centric Scheduling: Maximizes computation-communication overlap for efficient GPU utilization.


Non Goals ⚠️:
- Production: Mist is a research prototype built on PyTorch to explore distributed training optimizations. Certain production features like dynamic gradient scaling, gradient clipping, and training monitoring are intentionally omitted. For production use, we recommend applying Mist’s optimized strategies in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/) and [DeepSpeed](https://github.com/microsoft/deepspeed). We also disabled these features for baselines for fair performance comparison.
- Numeric Stability: even though we tried our best to make sure the execution is correct and tested the correctness for several base cases, numerical instabilities may arise due to complex overlap scheduling and data race conditions in complicated configurations. We are happy to further improve it once we spot these cases.

**We provide one GCP L4 machine with 8 NVIDIA L4 GPUs for the AE reviewers to reproduce our results. Please contact the authors for server access.**

## Overall Workflow

## Prerequisite (Skip for AE Reviewers)

We recommend to use Docker Engine for building the artifact to fully control all software dependencies. Please follow the instructions to Install [Docker Engine](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) first. Note that if the current user is not in the docker user group, all following docker-related commands requires root privilege (i.e. with sudo) to run. 

For convenience, we also provide the installation script below (extracted from official guide):

```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Step 1: Set-up

Clone the repository and build the docker container. NOTE: for users with GPUs different from L4 GPUs (sm_89), you may have to change the environment variable `TORCH_CUDA_ARCH_LIST` in the Dockerfile. You can find details [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list).

```bash
git clone https://github.com/Dazz993/Mist.git

cd Mist
docker build -t mist -f Dockerfile .
```

## Step 2: Kick-the-Tires (Functionality Test)

### Step 2.1: Run the docker container
```bash
docker run --gpus all -it --rm --privileged --ipc=host --shm-size=20G --ulimit memlock=-1 --name "mist" -v $(pwd):/workspace/ mist
```

### Step 2.2: Set up GPU frequencies

To get consistent and stable results especially for machines like L4, fix the gpu frequency
```bash
# Check supported frequencies
nvidia-smi -q -d SUPPORTED_CLOCKS
# Set the frequencies (e.g. for L4 GPUs)
nvidia-smi -ac 6251,1050
```

### Step 2.3: Analyze the small case
Mist can analyze the model execution time (including the breakdown for pipeline parallelism) and the memory usage efficiently. We use `test-small-base` config as an example, which is a *GPT-1.3B model running on 2 GPUs with BSZ=8, DP=2, FlashAttn=False. This is the best configuration that Megatron-LM can achieve. The corresponding YAML file is `/benchmark/mist/configs/test-small-base.yaml`. This YAML file contains network and memory parameters for GCP L4 GPUs. For other GPUs, this setup can be used as a functionality test.
```bash
cd /workspace/benchmark/mist/analysis/
python run.py --config-name test-small-base
```
Expected results:
```bash
# ... Breakdowns ...
# ..................
Total latency: 10.659405381925026
Peak fwd memory: [19255.25]
Peak bwd memory: [19503.125005722046]
```

### Step 2.4: Exec the small case:
Mist can directly run the configurations in an efficient way.
```bash
cd /workspace/benchmark/mist/exec/
torchrun --nproc-per-node 2 benchmark_one_case.py --config-name test-small-base
```
Expected results:
```bash
[Total]   Median: 11.3290 s, Mean: 11.3290 s, Std: 0.05885070
Total Latency: 11.3290
[Stage Peak]      Allocated memories: [19273.83] MB
[Stage Peak]      Reserved  memories: [20862.00, 21096.00] MB
```
Mist provides highly accurate memory estimation, ensuring reliable resource planning. However, performance estimation may have slight deviations, as Mist primarily focuses on comparing performance across different configurations rather than absolute runtime. Some constant overheads, like optimizer step time, are omitted since they remain the same across configurations. We will cover that later.

### Step 2.5: Tune the small case
Then let's use Mist to tune the best configuration:
```bash
cd /workspace/benchmark/mist/tune/
python tune_one_case.py --config-name test-small-base +output_path=/workspace/benchmark/mist/tune/results/test-small-mist
```

Expected results:
```bash
Best cost: 9.26892465
Best solution: [16,
 [(((0, 11), (1, 1), 5, 0), (2, 1, 1, 0, 0, 1, 0.0, 0.0, 0.0, 0.0)),
  (((12, 23), (1, 1), 0, 0), (2, 1, 1, 0, 0, 1, 0.0, 0.0, 0.0, 0.0))]
]
Saved the best solution to /workspace/benchmark/mist/tune/results/test-small-mist.yaml
```
The outputs can be interpreted as:
```
Gradient Accumulation Steps: 16. Two pipeline stages.
- ----------------------------------------------------
- (0, 11): (layer_idx_start, layer_idx_end)
- (1, 1) : (nnodes, nprocs_per_node)
- 5: number of checkpointed layers in a single stage
- ----------------------------------------------------
- (2, 1, 1): (Batch size, DP, TP)
- (0, 0, 1): (WeightsSharding, GradsSharding, OptSharding)
- (0.0, 0.0, 0.0, 0.0): (W, G, O, A). where they map to weights, grads, 
                        optimizer states, and activation offloading ratio.
```

Execute the tuned configurations:
```bash
cd /workspace/benchmark/mist/exec/
torchrun --nproc-per-node 2 \
    benchmark_one_case.py \
    --config-path /workspace/benchmark/mist/tune/results/ \
    --config-name test-small-mist
```
Expected results:
```
Total Latency: 9.9345
```
Therefore, the speedup is roughly ~14%. This is the datapoint in (Figure 11, (a) - 1).


## Step 3: Run Single-Node Performance Evaluation [Specifically for GCP L4 GPUs]

For L4 GPUs, we directly provide the configurations that are tuned by us that can be used to quickly test the speedup of Mist compared to baselines. We also provide a general process for evaluating on a brand new cluster. See the next section.

### Step 3.1: Evaluate Mist

We provide the best configurations that is found in our used L4 clusters under `/workspace/benchmark/mist/tuned_configs/`.

```bash
cd /workspace/benchmark/mist/tuned_configs/
bash run_single_node.sh
```

Results are summarized in `/workspace/benchmark/mist/tuned_configs/l4-24gb/gpt/summary.json` and corresponding llama file.

### Step 3.2: Evaluate Megatron-LM

Then we evaluate the performance of Megatron-LM. The best configurations of Megatron-LM are manually found by us and mostly match our searching results from our baseline search space.

```bash
cd /workspace/benchmark/megatron/
bash scripts/tops/l4/gpt2/1_8xl4_node_1_pcie.sh
bash scripts/tops/l4/llama/1_8xl4_node_1_pcie.sh
```

Results are under `/workspace/benchmark/megatron/results`.

### Step 3.3: Evaluate DeepSpeed

Similarly, we evaluate the performance of DeepSpeed.

```bash
cd /workspace/benchmark/deepspeed/
bash scripts/tops/l4/gpt2/1_8xl4_node_1_pcie.sh
bash scripts/tops/l4/llama/1_8xl4_node_1_pcie.sh
```

Results are under `/workspace/benchmark/deepspeed/results`.

### Step 3.4: Collect Results

We provide a python file to collect the results for easy comparison.

```bash
cd /workspace/benchmark/
python scripts/collect_single_node_results_v1.py
```

Expected Results (for clearity we ignore absolute numbers)
```bash
+----------------------+-----------------------+------------------------+
| SpeedUp              | SpeedUp vs Megatron   | SpeedUp vs DeepSpeed   |
+======================+=======================+========================+
| gpt2-1.3b-flash_True | 1.279X                | 1.473X                 |
+----------------------+-----------------------+------------------------+
| gpt2-2.7b-flash_True | 1.193X                | 1.488X                 |
+----------------------+-----------------------+------------------------+
| gpt2-7b-flash_True   | 1.191X                | 1.709X                 |
+----------------------+-----------------------+------------------------+
+-----------------------+-----------------------+------------------------+
| SpeedUp               | SpeedUp vs Megatron   | SpeedUp vs DeepSpeed   |
+=======================+=======================+========================+
| llama-1.3b-flash_True | 1.557X                | 1.498X                 |
+-----------------------+-----------------------+------------------------+
| llama-2.7b-flash_True | 1.374X                | 1.535X                 |
+-----------------------+-----------------------+------------------------+
| llama-7b-flash_True   | 1.325X                | 1.730X                 |
+-----------------------+-----------------------+------------------------+
+-----------------------+-----------------------+------------------------+
| SpeedUp               | SpeedUp vs Megatron   | SpeedUp vs DeepSpeed   |
+=======================+=======================+========================+
| gpt2-1.3b-flash_False | 1.175X                | 1.418X                 |
+-----------------------+-----------------------+------------------------+
| gpt2-2.7b-flash_False | 1.141X                | 1.384X                 |
+-----------------------+-----------------------+------------------------+
| gpt2-7b-flash_False   | 1.222X                | 2.053X                 |
+-----------------------+-----------------------+------------------------+
```

## Step 4 Benchmark the Tuning Time

We show how tuning time changes as the search space is increased step by step. To test that, we use a GPT 22B model to be run on 4 $\times$ 8 GPUs.
Beyond the tuning time exploration, this also gives us opportunities to understand the performance for larger scale distribtued training.

```bash
cd /workspace/benchmark/mist/benchmark-tuning-time
python run.py --model=gpt2/22b -n 4 -m 8
```

## Run Single-Node Performance Evaluation for General Cases [Not for AE]

### Profile Networking and Overlap Params [Est. Time: 50mins for 8xL4s]

We first profile the networking and overlapping params. 
```bash
cd /workspace/mist/tools/
bash scripts/profile_interference_single_node.sh
```

The outputs of it should be similar to:
```bash
GPU to GPU params: [4.5, 0.0, 2.995, 0.0004, 4.1604, 0.0003, 4.6325, 0.0005, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0, 4.5, 0.0]
CPU to GPU params: [4.1353, 0.0003]
GPU to CPU params: [3.9305, 0.0006]
Inference params: [10.3312, 7.6785, 4.0921, 2.1695, 2.6942, 3.1015, 1.9612, 2.2752, 2.7452, 2.3535, 6.6221, 2.5708, 2.8219, 4.4773, 2.7195, 3.0124, 1.2946, 1.0224, 1.973, 1.982, 1.9532, 1.9653, 2.4663, 1.5854, 1.3876, 2.8578, 1.9576, 1.9656]
```

Steps:
1. Create a new experiment template under `/workspace/benchmark/mist/configs/experiment/`. For example, if I am using A10 GPUs, I should do:
    ```bash
    cd /workspace/benchmark/mist/configs/experiment/
    cp template-l4.yaml template-a10g.yaml
    ```
    Please make sure the `template-xxx` -> `xxx` is the simplified device name, which can be got by
    ```bash
    python3 -c "import torch; print(torch.cuda.get_device_name(0).split(' ')[-1].lower())"
    ```
2. Update `template-a10g.yaml`:
    - Copy the GPU-GPU, CPU-GPU, GPU-CPU, and Interence params into the `hardware` block in the template.
    - Update the `nvlink` and `memory_capacity`, where `memory_capacity` is the size of device memories (in GB).
3. Update `/workspace/benchmark/mist/experiment/run.py`.
    - Add the entry in the TEMPLATE dict.

### Run the experiments

The entry python file of the experiments is `/workspace/benchmark/mist/experiment/run.py`.

You can create your own shell file or change the existing ones as it lists all the configurations to be run. 

```bash
#   nnodes, nproc_per_node, model,  model_size, seq_len, global_batch_size, use_flash_attention, extra_args
run 1       2               "gpt2"  "1.3b"      2048     32                 true                 "--skip-exec"
run 1       2               "gpt2"  "1.3b"      2048     32                 false                "--skip-exec"
run 1       2               "gpt2"  "1.3b"      2048     32                 true                 "--skip-tune"
run 1       2               "gpt2"  "1.3b"      2048     32                 false                "--skip-tune"`
```

```bash
cd /workspace/benchmark/mist/experiment/
bash scripts/24gb_pcie/gpt/1-run_8_a10_single_node.sh
```

## Run Multi-node Performance Evaluation for General Cases [Not for AE]

Due to the limit of large-scale compute resources, we haven't had a chance to polish the instructions of multi-node performance evaluation in a container manner. You may find in many places we are using `miniconda`. Here we provide general instructions and key files which enables the benchmarking in a similar way.

### Preparation

To easily test the multi-node performance, we use `pdsh` for fast execution.
Specifically, we use `pdsh -f 1024 -R ssh -w worker-[1-$NUM_HOSTS]`
Assume your cluster contains 4 nodes, each with 8 GPUs.
You have to number them from 1 to 4, e.g. the first machine as `worker-1`, the last machine as `worker-4`.
And each machine should be able to directly ssh to another one through `worker-%idx`.

### Profile Networking and Overlap Params

Similarly, we need to first profile the network and overlap params. Please refer to `/workspace/mist/tools/scripts/profile_interference_multinode.sh` for details. [This may take a long time as we apply a relatively fine-grained network bandwidth estimation method.]

After getting the networking and overlap param, repeat the steps above to update the `template-a10g.yaml` and `/workspace/benchmark/mist/experiment/run.py` on all machines.

### Run the experiments

Take a look at `3-1-tune_22b_8_a10_4_nodes.sh` and `3-2-exec_22b_8_a10_4_nodes.sh` under `/workspace/benchmark/mist/experiment/scripts/24gb_pcie/gpt/`.

### Known Benchmarking Issues

- It is annoying that during the multi-node testing, there may be cases where only one or several machines are down because of OOM errors while others are still running. Although we have tried our best to cover this for smooth benchmarking, there may still be idling situation and it requires manual monitoring.