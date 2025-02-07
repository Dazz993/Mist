import argparse
import gc
import os
import sys
import time
import psutil
from functools import partial
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# from megatron.utils import average_losses_across_data_parallel_group
from megatron.model import BertModel, GPTModel
from megatron import (
    initialize_megatron,
    get_args,
    get_timers,
    update_num_microbatches,
    get_num_microbatches,
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core import mpu
from megatron.core.distributed import (
    finalize_model_grads,
    DistributedDataParallel as DDP,
)
from megatron.core.utils import get_model_config
from megatron.model.enums import ModelType
from megatron.initialize import set_jit_fusion_options
from megatron.training import train_step, setup_model_and_optimizer
from common import write_tsv, benchmark_func_walltime, process_benchmarking_results

GB = 1024**3
SLEEP_TIME = 1

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Get rank info and set gpu device
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def find_and_kill_other_processes():
    matching_processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["cmdline"] is None:
                continue
            if "benchmark_gpt_bert_one_case.py" in proc.info["cmdline"]:
                matching_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    for proc in matching_processes:
        if proc.pid != os.getpid():
            print(f"Killing process {proc.pid}")
            proc.kill()

    # For multi-nodes, we need to kill the processes on other nodes
    # TODO(zhanda): Improve this logic
    num_nodes = world_size // torch.cuda.device_count()
    if num_nodes > 1:
        subprocess.run(
            f"pdsh -f 1024 -R ssh -w ~/hosts 'pkill -f benchmark_gpt_bert_one_case.py && sleep 1'",
        )


def get_gpt_functions():
    args = get_args()
    config = core_transformer_config_from_args(args)
    micro_batch_size = args.micro_batch_size
    seq_len = args.seq_length

    def model_provider(pre_process=True, post_process=True):
        model = GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Reduce loss for logging.
        # averaged_loss = average_losses_across_data_parallel_group([loss])
        averaged_loss = [0]
        return loss, {"lm loss": averaged_loss[0]}

    tokens = torch.randint(0, 30000, (micro_batch_size, seq_len), device=device)
    # labels = torch.randint(0, 30000, (micro_batch_size, seq_len), device=device)
    labels = torch.randint(0, 10, (micro_batch_size, seq_len), device=device)
    loss_mask = torch.ones((micro_batch_size, seq_len), dtype=torch.int, device=device)
    attention_mask = torch.ones(
        (micro_batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=device
    )
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(
        micro_batch_size, -1
    )

    def forward_step(data_iterator, model: GPTModel):
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        return output_tensor, partial(loss_func, loss_mask)

    return model_provider, loss_func, forward_step


def get_bert_functions():
    args = get_args()
    config = core_transformer_config_from_args(args)
    micro_batch_size = args.micro_batch_size
    seq_len = args.encoder_seq_length
    num_tokentypes = 2 if args.bert_binary_head else 0

    def model_provider(pre_process=True, post_process=True):
        model = BertModel(
            config=config,
            num_tokentypes=num_tokentypes,
            add_binary_head=args.bert_binary_head,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

        return model

    def loss_func(loss_mask, sentence_order, output_tensor):
        lm_loss_, sop_logits = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        if sop_logits is not None:
            sop_loss = F.cross_entropy(
                sop_logits.view(-1, 2).float(), sentence_order.view(-1), ignore_index=-1
            )
            sop_loss = sop_loss.float()
            loss = lm_loss + sop_loss
            # averaged_losses = average_losses_across_data_parallel_group(
            #    [lm_loss, sop_loss])
            averaged_losses = [0, 0]
            return loss, {"lm loss": averaged_losses[0], "sop loss": averaged_losses[1]}
        else:
            loss = lm_loss
            # averaged_losses = average_losses_across_data_parallel_group(
            #    [lm_loss])
            averaged_losses = [0]
            return loss, {"lm loss": averaged_losses[0]}

    tokens = torch.ones((micro_batch_size, seq_len)).cuda().long()
    types = torch.ones((micro_batch_size, seq_len)).cuda().long()
    sentence_order = None
    loss_mask = torch.ones((micro_batch_size, seq_len)).cuda().float()
    lm_labels = torch.ones((micro_batch_size, seq_len)).cuda().long()
    padding_mask = torch.ones(micro_batch_size, seq_len).cuda().long()

    def forward_step(data_iterator, model):
        if not args.bert_binary_head:
            types = None

        output_tensor = model(
            tokens, padding_mask, tokentype_ids=types, lm_labels=lm_labels
        )
        return output_tensor, partial(loss_func, loss_mask, sentence_order)

    return model_provider, loss_func, forward_step


def benchmark_gpt_bert_one_case(benchmark_case, results_dir, output_file_name):
    # Model configs
    (
        model_name,
        global_batch_size,
        model_config,
        num_micro_batches,
        parallel_args,
        profile,
    ) = benchmark_case
    (seq_len, hidden_size, num_layers, num_heads, vocab_size) = model_config
    (
        dp,
        op,
        pp,
        use_remat,
        zero,
        use_flash_attn,
    ) = parallel_args

    # Set the model type and different model specific arguments
    model_specific_args = []
    model_name = model_name.lower()
    if model_name == "gpt2":
        ffn_hidden_size = hidden_size * 4
        tokenizer_type = "GPT2BPETokenizer"
        model_specific_args.append(["--max-position-embeddings", str(seq_len)])
        model_specific_args.append(["--untie-embeddings-and-output-weights"])
        model_specific_args.append(["--vocab-file", "vocab_files/gpt2-vocab.json"])
        model_specific_args.append(["--merge-file", "vocab_files/gpt2-merges.txt"])

    elif model_name == "llama":
        multiple_of = 256
        ffn_hidden_size = int(hidden_size * 8 / 3)
        ffn_hidden_size = int(
            (ffn_hidden_size + multiple_of - 1) // multiple_of * multiple_of
        )
        tokenizer_type = "GPT2BPETokenizer"
        model_specific_args.append(["--no-position-embedding"])
        model_specific_args.append(["--max-position-embeddings", str(seq_len)])
        model_specific_args.append(["--untie-embeddings-and-output-weights"])
        model_specific_args.append(["--vocab-file", "vocab_files/gpt2-vocab.json"])
        model_specific_args.append(["--merge-file", "vocab_files/gpt2-merges.txt"])
        model_specific_args.append(["--swiglu"])
        model_specific_args.append(["--use-rotary-position-embeddings"])
        model_specific_args.append(["--normalization", "RMSNorm"])

    elif model_name == "falcon":
        ffn_hidden_size = hidden_size * 4
        tokenizer_type = "GPT2BPETokenizer"
        model_specific_args.append(["--max-position-embeddings", str(seq_len)])
        model_specific_args.append(["--untie-embeddings-and-output-weights"])
        model_specific_args.append(["--vocab-file", "vocab_files/gpt2-vocab.json"])
        model_specific_args.append(["--merge-file", "vocab_files/gpt2-merges.txt"])
        model_specific_args.append(["--use-rotary-position-embeddings"])
        model_specific_args.append(["--parallel-block"])

    elif model_name == "bert":
        ffn_hidden_size = hidden_size * 4
        tokenizer_type = "BertWordPieceLowerCase"
        model_specific_args.append(["--max-position-embeddings", str(seq_len)])
        model_specific_args.append(["--vocab-file", "vocab_files/bert-vocab.txt"])

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Parallelism configs
    dp_size, tensor_mp_size, pipeline_mp_size = dp, op, pp
    checkpoint_activations = use_remat

    num_gpus = dp_size * tensor_mp_size * pipeline_mp_size
    assert global_batch_size % (dp_size * num_micro_batches) == 0
    micro_batch_size = global_batch_size // dp_size // num_micro_batches

    device_name = torch.cuda.get_device_name(device).split(" ")[-1]
    nproc_per_node = torch.cuda.device_count()
    nnodes = world_size // nproc_per_node

    # Parallel configs
    # Initialize megatron
    # Model specific arguments
    sys.argv += ["--micro-batch-size", str(micro_batch_size)]
    sys.argv += ["--tensor-model-parallel-size", str(tensor_mp_size)]
    sys.argv += ["--pipeline-model-parallel-size", str(pipeline_mp_size)]
    sys.argv += ["--global-batch-size", str(global_batch_size)]
    sys.argv += ["--num-layers", str(num_layers)]
    sys.argv += ["--hidden-size", str(hidden_size)]
    sys.argv += ["--ffn-hidden-size", str(ffn_hidden_size)]
    sys.argv += ["--num-attention-heads", str(num_heads)]
    sys.argv += ["--seq-length", str(seq_len)]
    for arg in model_specific_args:
        sys.argv += arg
    # Dropouts
    sys.argv += ["--attention-dropout", "0.0"]
    sys.argv += ["--hidden-dropout", "0.0"]
    # Bias: disable bias and thus there is no need for bias fusion
    sys.argv += ["--disable-bias-linear"]
    sys.argv += ["--no-bias-dropout-fusion"]
    # Loss in fp16
    sys.argv += ["--fp16-lm-cross-entropy"]
    sys.argv += ["--optimizer", "adam"]
    sys.argv += ["--train-iters", "100"]
    sys.argv += ["--lr", "0.0000001"]
    sys.argv += ["--bert-no-binary-head"]
    sys.argv += ["--fp16"]
    # Flash attention
    if use_flash_attn:
        sys.argv += ["--use-flash-attn"]
    # Enable overlapping grad sync
    sys.argv += ["--overlap-grad-reduce"]
    # sys.argv += ["--no-delay-grad-reduce"]
    # Constant grad scaler
    # sys.argv += ["--loss-scale", "8192"]
    sys.argv += ["--loss-scale", str(2**14)]
    # Grad clip disabled
    sys.argv += ["--clip-grad", "0.0"]
    # ZeRO 1
    if zero:
        sys.argv += ["--use-distributed-optimizer"]
        sys.argv += ["--overlap-param-gather"]
    if checkpoint_activations:
        # It will directly use 'selective' method
        # sys.argv += ["--recompute-activations"]
        # Or we can choose to use more fine-grained method
        sys.argv += ["--recompute-granularity", "full"]
        sys.argv += ["--recompute-method", "uniform"]
        sys.argv += ["--recompute-num-layers", "1"]
        if tensor_mp_size > 1:
            sys.argv += ["--distribute-saved-activations"]
    # sys.argv += ["--no-masked-softmax-fusion"]
    # sys.argv += ["--no-async-tensor-model-parallel-allreduce"]
    # sys.argv += ["--no-scatter-gather-tensors-in-pipeline"]

    def write_results(mean, std, throughput, peak_mem, _exit=False):
        assert isinstance(mean, str)
        assert isinstance(std, str)
        assert isinstance(throughput, str)
        assert isinstance(peak_mem, str)
        heads = [
            "Rank",
            "Model Name",
            "Hidden Size",
            "Num Layers",
            "Num Heads",
            "Vocab Size",
            "Global Batch Size",
            "Micro Batch Size",
            "#Microbatch",
            "Seq Length",
            "DP-Size",
            "TP-Size",
            "PP-Size",
            "Remat",
            "ZeRO",
            "Device Type",
            "Number Nodes",
            "Number GPUs per Node",
            "Mean Time",
            "Std Time",
            "Throughput",
            "Peak Mem",
        ]
        values = [
            str(local_rank),
            model_name,
            str(hidden_size),
            str(num_layers),
            str(num_heads),
            str(vocab_size),
            str(global_batch_size),
            str(micro_batch_size),
            str(num_micro_batches),
            str(seq_len),
            str(dp_size),
            str(tensor_mp_size),
            str(pipeline_mp_size),
            str(checkpoint_activations),
            str(zero),
            device_name,
            str(nnodes),
            str(nproc_per_node),
            mean,
            std,
            throughput,
            peak_mem,
        ]
        folder = results_dir
        file = f"{model_name}_megatron_{output_file_name}.tsv"
        if not os.path.exists(folder):
            os.makedirs(folder)
        write_tsv(
            heads,
            values,
            os.path.join(folder, file),
            logging_fn=print,
        )

        if _exit:
            print("Exiting by accident.")
            find_and_kill_other_processes()
            print(f"Sleeping for {SLEEP_TIME} seconds before starting the next case. ")
            time.sleep(SLEEP_TIME)
            exit()

    # Initialize megatron
    initialize_megatron(args_defaults={"tokenizer_type": tokenizer_type})
    args = get_args()

    # Check initialization
    assert dp_size == mpu.get_data_parallel_world_size(), (
        f"Expected {dp_size} data parallel size, "
        f"but got {mpu.get_data_parallel_world_size()}"
    )
    assert tensor_mp_size == mpu.get_tensor_model_parallel_world_size(), (
        f"Expected {tensor_mp_size} tensor parallel size, "
        f"but got {mpu.get_tensor_model_parallel_world_size()}"
    )
    assert pipeline_mp_size == mpu.get_pipeline_model_parallel_world_size(), (
        f"Expected {pipeline_mp_size} pipeline parallel size, "
        f"but got {mpu.get_pipeline_model_parallel_world_size()}"
    )

    # Build model
    if model_name in ["gpt2", "llama", "falcon"]:
        model_provider, loss_func, forward_step = get_gpt_functions()
    elif model_name == "bert":
        model_provider, loss_func, forward_step = get_bert_functions()
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    oom = torch.tensor([0], device=device)
    try:
        set_jit_fusion_options()
        model, optimizer, lr_scheduler = setup_model_and_optimizer(
            model_provider, model_type=ModelType.encoder_or_decoder
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        oom = torch.tensor([1], device=device)

    # Check oom of all processes
    dist.all_reduce(oom, op=dist.ReduceOp.SUM)
    if oom.item() > 0:
        write_results("OOM", "OOM", "OOM", "OOM", _exit=True)

    # These logics are under megatron/training.py:train
    for model_module in model:
        model_module.train()
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.delay_grad_reduce:
            config.grad_sync_func = [
                model_chunk.start_grad_sync for model_chunk in model
            ]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.delay_param_gather:
        config.param_sync_func = [
            lambda x: optimizer.finish_param_sync(model_index, x)
            for model_index in range(len(model))
        ]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    def run_func():
        update_num_microbatches(args.consumed_train_samples)
        train_step(forward_step, None, model, optimizer, lr_scheduler, config)
        args.consumed_train_samples += (
            mpu.get_data_parallel_world_size()
            * args.micro_batch_size
            * get_num_microbatches()
        )

    def sync_func():
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

    # Warmup and reset timers
    oom = torch.tensor([0], device=device)
    try:
        # run_func()
        timers = get_timers()
        names = list(timers._timers.keys())
        for name in names:
            timers(name).reset()

        # Benchmark step time
        warmup = 1
        number = 2 if world_size <= 2 else 1
        costs, memories = benchmark_func_walltime(
            run_func,
            warmup=warmup,
            number=number,
            sync_func=sync_func,
            enable_tqdm=False,
        )
        timers.log(names, normalizer=warmup + number)
    except RuntimeError as e:
        print(f"Error: {e}")
        oom = torch.tensor([1], device=device)

    # Check oom of all processes
    if oom.item() > 0:
        write_results("OOM", "OOM", "OOM", "OOM", _exit=True)

    # Stats
    torch.cuda.synchronize()
    dist.barrier()
    mean = np.mean(costs)
    std = np.std(costs)
    throughput = global_batch_size / mean

    # Print results
    # if local_rank == 0:
    peak_mem = torch.cuda.max_memory_allocated()
    write_results(
        mean=f"{mean:.3f}",
        std=f"{std:.3f}",
        throughput=f"{throughput:.3f}",
        peak_mem=f"{peak_mem / GB:.3f}",
    )

    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
            record_shapes=True,
            # with_stack=True,
            # with_modules=True,
        ) as profiler:
            for i in range(4):
                run_func()
                torch.cuda.synchronize()
                profiler.step()

    print(f"Sleeping for {SLEEP_TIME} seconds before starting the next case. ")
    time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    # Three arguments: case, results_dir, output_file_name
    case, results_dir, output_file_name = sys.argv[-3:]
    case = eval(case)
    del sys.argv[-1]
    del sys.argv[-1]
    del sys.argv[-1]

    benchmark_gpt_bert_one_case(case, results_dir, output_file_name)
