# DeepSpeed Benchmarking [Deprecated]

This folder contains the scripts to run the benchmarking experiments on DeepSpeed-Megatron-LM. It is mainly based on [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) and [DeepSpeedExamples/megatron/Megatron-LM-v1.1.5-ZeRO3](https://github.com/microsoft/DeepSpeedExamples/tree/master/megatron/Megatron-LM-v1.1.5-ZeRO3). The benchmarking scripts are significantly modified based on [Alpa's benchmarking](https://github.com/alpa-projects/alpa/tree/main/benchmark/megatron) to match the updated version and what we need to benchmark.


## Notes of DeepSpeed's Memory Optimization

Useful links:
- [Getting Started](https://www.deepspeed.ai/getting-started/)
- [Megatron-LM GPT2](https://www.deepspeed.ai/tutorials/megatron/)
- [Zero Redundancy Optimizer](https://www.deepspeed.ai/tutorials/zero/)
- [Zero-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
- [Full Configuration](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)
- [`aio` param when using `zero-infinity-nvme`](https://github.com/microsoft/DeepSpeed/issues/998). Also see [this](https://github.com/microsoft/DeepSpeed/issues/998#issuecomment-827027186) for pruned grid search space.
- [`offload_*` param](https://github.com/microsoft/DeepSpeed/issues/1005)

## Folder Structure

```bash
.
├── Megatron-DeepSpeed                          # The DS version of Megatron-LM, you need to clone it manually
├── patch                                       # The patch files we need to replace the original files in Megatron-DeepSpeed
├── tools                                       # Some tools to facilitate the benchmarking
├── benchmark_transformer_layer_one_case.py     # The script to run one case of the benchmarking
├── benchmark_transformer_layer.py              # The script to run all the cases of the benchmarking
├── profile_transformer_layer_one_case.py       # The script to run one case of the profiling
├── profile_transformer_layer.py                # The script to run all the cases of the profiling
├── ds_config_zero_infinity_cpu.json
├── ds_config_zero_infinity_nvme.json
├── ds_config_zero_offload.json
├── ds_config_zero_stage_1.json
├── ds_config_zero_stage_2.json
├── ds_config_zero_stage_3.json
├── result-deepspeed-transformers-0.tsv         # The result of the benchmarking (outputed by the script)
├── results-profile                             # The result of the profiling (outputed by the script)
├── README.md
└── backup
```

## Requirements

Install dependencies:
```bash
sudo apt-get install -y pdsh
pip install ninja six regex pybind11 dill
# Set the correct PATH and CUDA_HOME, e.g.
# export PATH=$PATH:/usr/local/cuda-11.7/bin/
# export CUDA_HOME=/usr/local/cuda-11.7
```

Install DeepSpeed and Megatron-DeepSpeed. Be careful about the 
```bash
pip install deepspeed==0.12.6    # This will also install the dependencies (e.g. torch)
git clone https://github.com/microsoft/Megatron-DeepSpeed
cd Megatron-DeepSpeed
git checkout a9856ce0e75dbe69c96d4e241e8a191b344118d7
git apply --whitespace=nowarn ../megatron-deepspeed.patch
echo "export DEEPSPEED_PATH=$(pwd)" >> ~/.bashrc
source ~/.bashrc
cd ..
```

Install apex.
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# Comment out the raised RuntimeError in setup.py if you get errors running the following command.
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Replace some files in Megatron-DeepSpeed to support running the reversible model. Alternatively, you can also delete the original files and create symbolic links for them.
```bash
# Assume if you are in the /path/to/PROJECT_ROOT/benchmark/deepspeed/
python tools/patch.py -m patch2pkg
# Other notes:
# transformer.py will also be copied to the corresponding path automatically 
# if you use the scripts in `backup/scripts/` although we do not recomment that
```

You may also need to set the ssh key for the `pdsh` command. Modify [hostfile](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) and setup the ssh keys. For example, if I am using gcp, I can do the following in `~/.ssh/config`:
```bash
Host worker-1
  Hostname localhost
  AddKeysToAgent yes
  IdentityFile ~/.ssh/google_compute_engine
```

Download the dataset and the vocabulary if needed.
```bash
cd Megatron-DeepSpeed/dataset  # => PROJECT_ROOT/benchmark/deepspeed/Megatron-DeepSpeed/dataset
bash download_vocab.sh
bash download_books.sh
cd ../..
```


## Instructions

Temp:
```bash
# (model_name, global_batch_size, (seq_len, hidden_size, num_layers, num_heads, vocab_size), #microbatches, (dp, tp, pp, remat, ds_config_file), profile
torchrun --nproc_per_node 4 benchmark_gpt_bert_one_case.py "('gpt', 16, (2048, 2048, 24, 16, 50304), 4, (2, 2, 1, True, 'ds_config_zero_stage_1'), False)" result
```

If you want to use nsys profile, do
```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true -x true --force-overwrite true -o my_profile torchrun --nproc_per_node 4 benchmark_gpt_bert_one_case.py "(16, (2048, 2048, 16, 16, 50304), 4, (4, 1, 1, True))" result
```

Sweap over all the cases:
```bash
python benchmark_gpt_bert.py --nnodes 1 --nproc_per_node 8 --model "gpt2-7b" --global_batch_sizes "64"
```

### Optional if you want to run directly use the shell

Link the `scripts/` to the `Megatron-DeepSpeed/` folder. In this case, you need to go to the `Megatron-DeepSpeed/` folder to run the scripts.
```bash
# Assume if you are in the /path/to/PROJECT_ROOT/benchmark/deepspeed/
ln -s $(pwd)/backup/scripts Megatron-DeepSpeed/scripts
```

To run that, first prepare the dataset and the 

```bash
cd Megatron-DeepSpeed/dataset  # => PROJECT_ROOT/benchmark/deepspeed/Megatron-DeepSpeed/dataset
bash download_books.sh
bash download_vocab.sh
```

## Usage

### Set the `aio` parameters based on the grid search script

Run the grid search based on [`aio` param when using `zero-infinity-nvme`](https://github.com/microsoft/DeepSpeed/issues/998).

```bash
('write', 'single', 'sequential', 4, 1, 8, 1048576) = 2.0834750695752744
('read', 'single', 'sequential', 4, 1, 8, 1048576) = 6.2765769976324925
```

### Benchmarking

Modify the benchmarking configurations in `benchmark_transformer_layer.py`. Run the benchmarking script.
```bash
python benchmark_transformer_layer.py -g 1
# optional arguments:
#   -h, --help            show this help message and exit
#   -g --nproc_per_node NPROC_PER_NODE
#   --nnodes NNODES
#   --node_rank NODE_RANK
#   --master_addr MASTER_ADDR
#   --master_port MASTER_PORT
```

#### Where are the results recorded and what is the meaning?

|      | PP-disabled | PP-Enabled |
| -- | :----------- | :---------- |
| Peak Mem Fwd | schedules.py |            |
| Peak Mem Bwd | engine.py |            |
| Peak Mem Grad Allreduce | engine.py |            |
| Peak Mem Opt | training.py |            |
| Step Latency | training.py |            |
| Time Load Batch | benchmarking_one_case.py |            |
| Time Fwd | benchmarking_one_case.py |            |
| Time Fwd All (load + fwd +loss) | schedules.py |            |
| Time Bwd | engine.py |            |
| Time Grad Allreduce | engine.py |            |
| Time Bwd All (bwd + sync) | schedules.py |            |
| Time Opt | training.py |            |
| Time Fwd-Bwd ending | x |            |
| Time Grad-allreduce ending | x |            |

### Profiling

The basic command to run the profiling is the similar as running the benchmarking.

```bash
pip install torch_tb_profiler
```

Mofidy the profiling configurations in `profile_transformer_layer.py`. Run the profiling script.
```bash
python profile_transformer_layer.py
```

Run the following command to visualize the profiling results.
```bash
tensorboard --logdir results-profile/your-output-folder/
```

#### How to read the profiling results?

Basically, there are three different stages: forward, backward, and optimizer step. For each stage, we need to figure out the compute/GPU kernel occupancy.

My strategy to analyze the profiling results is:
1. know the upbound/best cases using small number of layers to figure out the compute time of each stage.
2. manually check the cpu time of each stage to figure out the cpu time of each stage.

## Troubleshooting

One major problem source is the `pdsh` command. It is used to run commands on multiple nodes. However, by using `pdsh`, it will not do `source ~/.bashrc` for you. So the settings on that file will not be applied. To solve this problem, you may need to manually change some files or modify the `deepspeed`.

1. **Dependency issues like `pybind11 not found` but actually you have `pybind11` installed.** Verify whether this error is caused related to `benchmark/deepspeed/Megatron-DeepSpeed/megatron/data/`. Modify the Makefile in this folder to set the correct python3 path. E.g.,
    ```bash
    # Change from
    # ```
    # CPPFLAGS += $(shell python3 -m pybind11 --includes)
    # LIBEXT = $(shell python3-config --extension-suffix)
    # ```
    # to
    CPPFLAGS += $(shell /home/zhanda/anaconda3/envs/revnn/bin/python3 -m pybind11 --includes)
    LIBEXT = $(shell /home/zhanda/anaconda3/envs/revnn/bin/python3-config --extension-suffix)
    ```

2. **`CUDA_HOME` and nvcc version issue when compiling the fused kernels. This is also caused by `PATH` or `CUDA_HOME` issue when using `pdsh`.** If you have changed the cuda path in these two files, you may need to manually change the `deepspeed` to make it work. To solve that, we add the `export`(envs) in deepspeed's `MultiNodeRunning`(the default runner should be `PDSHRunner`).
    ```bash
    # In file deepspeed/launcher/runner.py ~line 500+
    # --------------------------------------------
    # args.launcher = args.launcher.lower()
    # if args.launcher == PDSH_LAUNCHER:
    #     runner = PDSHRunner(args, world_info_base64)
    # elif args.launcher == OPENMPI_LAUNCHER:
    #     runner = OpenMPIRunner(args, world_info_base64, resource_pool)
    # elif args.launcher == MPICH_LAUNCHER:
    #     runner = MPICHRunner(args, world_info_base64, resource_pool)
    # elif args.launcher == MVAPICH_LAUNCHER:
    #     runner = MVAPICHRunner(args, world_info_base64, resource_pool)
    # elif args.launcher == SLURM_LAUNCHER:
    #     runner = SlurmRunner(args, world_info_base64, resource_pool)
    # else:
    #     raise NotImplementedError(f"Unknown launcher {args.launcher}")
    # ========== add the following lines ==========
    # you may also any other envs you want, e.g. `CUDA_HOME`, `LD_LIBRARY_PATH`, etc.
    runner.add_export("PATH", os.environ['PATH'])
    # =============================================
    ```

3. **Fused Adam issue.** When running DS-Megatron-LM with the original ZeRO-stage-3, we may encounter the problem that `group['param']` is an empty list in `device = group['params'][0].device` in FusedAdam. To solve this, we can add the following line to `apex/optimizers/fused_adam.py#L121`.
    ```python
    for group in self.param_groups:
        # ========== add the following lines ==========
        if group['params'] == []:
            continue
        # =============================================
        device = group['params'][0].device
        bias_correction = 1 if group['bias_correction'] else 0
        beta1, beta2 = group['betas']
    ```

4. **Torch Version issue**. If you encounter the error in `from torch._six import inf`, change to `from torch import inf`.

## Patches for debugging

I also backup some patches I used for debugging. You can find them in `patches/debugging/` folder. Be careful when using them!

```bash
python patch/debugging/_package2patch.py
python patch/debugging/_patch2package.py
```

```bash
python tools/show_debugging_file_diff.py
```