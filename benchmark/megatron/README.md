# Benchmark Megatron-LM

This folder contains the scripts to run the benchmarking experiments on Megatron-LM. It is mainly based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). The benchmarking scripts are significantly modified based on [Alpa's benchmarking](https://github.com/alpa-projects/alpa/tree/main/benchmark/megatron) to match the updated version and what we need to benchmark.

## Folder Structure

``` bash
.
├── Megatron-LM                                 # The Megatron-LM, you need to clone it manually
├── benchmark_transformer_layer_one_case.py     # The script to run one case of the benchmarking
├── benchmark_transformer_layer.py              # The script to run all the cases of the benchmarking
├── README.md
├── result_trans-0.tsv                          # The result of the benchmarking (outputed by the script)
└── outputs-analyze                             # You may or may not have this
```

There are possibly other files in the folder. But they are not tested and may not work.

## Requirements

### Prerequisites

Before everything, make sure you have already installed CMake, CUDA toolkit, and cuDNN, and have set the environment variables `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`, and `CUDNN_PATH` correctly.

To set the environment variables, for example,
```bash
export CUDA_HOME=/usr/local/cuda-12/
export PATH=/usr/local/cuda-12/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
```

If you haven't installed cuDNN, you can either download it from the [official websit - cuDNN-v8.9.7](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz/). Or install it with `pip`:
```bash
pip install nvidia-cudnn-cu12==8.9.7.29
```

Remember to add the CUDNN path to the enviroment:
```bash
find / -name "libcudnn.so*" 2>/dev/null
export CUDNN_PATH=/home/zhanda/miniconda3/envs/dazzle/lib/python3.9/site-packages/nvidia/cudnn/
```

Install TransformerEngine:
```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v1.3
```


### Install Megatron-LM

Create a new conda environment and install the following packages.
```bash
conda create -y -n megatron python==3.9
conda activate megatron
```

We use the latest stable pytorch version at the time of benchmarking torch==2.1.1 and the CUDA version is 11.8.
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

Install Megatron-LM. We use the latest release at the time of benchmarking which is 'core_r0.4.0' (38879f8).
```bash
# Install Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 38879f8
# Apply the patches
git apply --whitespace=nowarn ../megatron.patch

echo "export PYTHONPATH=$(pwd):$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
conda activate megatron

# Install requirements
# NOTE: TransformerEngine may not be easy to install. Make sure you have set cuda toolkit and cudnn correctly.
pip install -r requirements.txt

# Install Apex
git clone https://github.com/NVIDIA/apex
cd apex
git checkout bae1f93d033716dc9115a0baf7bcda328addabe9
# NOTE: Comment out the raised RuntimeError in setup.py if you get errors running the following command.
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# [Optional] Update some CUDA flag envs
export CUDA_DEVICE_MAX_CONNECTIONS=1
# echo "export CUDA_DEVICE_MAX_CONNECTIONS=1" >> ~/.bashrc
```

## Patches

To make sure the comparison is fair, during the benchmarking we disable some components in Megatron-LM. The most parts are in `Megatron-LM/megatron/optimizer/optimizer.py`.
1. Comment out the all_reduce in function `_unscale_main_grads_and_check_for_nan`. Line 279-281.
2. Update `_multi_tensor_copy_this_to_that` to only use `copy_` beginning from Line 35.

```bash
# Assume you are under /path/to/benchmark/megatron
cp patches/optimizer.py Megatron-LM/megatron/optimizer/optimizer.py
```


## Instructions

Temp:
```bash
# (model_name, global_batch_size, (seq_len, hidden_size, num_layers, num_heads, vocab_size), #microbatches, (dp, tp, pp, remat, zero-1, flash-attn), profile
torchrun --nproc_per_node 8 benchmark_gpt_bert_one_case.py "('gpt2', 32, (2048, 2048, 24, 16, 50304), 4, (8, 1, 1, True, True, True), False)" results result
torchrun --nproc_per_node 2 benchmark_gpt_bert_one_case.py "('gpt2', 32, (2048, 2048, 24, 16, 50304), 2, (2, 1, 1, True, True, True), True)" results result
```

If you want to use nsys profile, do
```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --force-overwrite true -o my_profile torchrun --nproc_per_node 4 benchmark_gpt_bert_one_case.py "(16, (2048, 2048, 16, 16, 50304), 4, (4, 1, 1, True))" result
```

Sweap over all the cases:
```bash
python benchmark_gpt_bert.py --nnodes 1 --nproc_per_node 8 --model "gpt2-7b" --global_batch_sizes "64"
```

### Single Node

```bash
python benchmark_transformer_layer.py --nproc_per_node 1
# optional arguments:
#   -h, --help            show this help message and exit
#   --nproc_per_node NPROC_PER_NODE
#   --nnodes NNODES
#   --node_rank NODE_RANK
#   --master_addr MASTER_ADDR
#   --master_port MASTER_PORT
```

## Troubleshooting

1. **`np.float` issue**: `np.float` is deprecated and is be removed in NumPy 1.20. Use `float` instead. The files you may change: `megatron/data/indexed_dataset.py`.
2. **Env issue**: set `CUDA_DEVICE_MAX_CONNECTIONS=1` to enable async gradient reduction.
3. **Pybind11 not found issue**: install pybind11 using `pip install pybind11`.
4. **Fused kernel loading issue**: comment out *megatron/fused_kernels/__init__.py:#L26-28* to support `compute_90`.
