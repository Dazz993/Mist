import os
import argparse
from datetime import datetime
from numbers import Number
import time

POWER_OF_TWO = [2**i for i in range(12)]
BATCH_SIZE_CANDIDATES = [1, 2, 4]
SEQ_LENGTH = 2048

GPT_MODELS = {
    "test": (512, 4, 8, 12800),
    "1.3b": (2048, 24, 16, 50304),
    "2.7b": (2560, 32, 32, 50304),
    "7b": (4096, 32, 32, 50304),
    "13b": (5120, 40, 40, 50304),
    "20b": (6144, 44, 64, 50304),
    "22b": (6144, 48, 64, 50304),
    "40b": (8192, 48, 64, 50304),
}


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def _get_power_of_two_smaller_than_or_equal_to(x):
    return [i for i in POWER_OF_TWO if i <= x]


def _get_3d_parallelism_cases(num_gpus):
    pairs = []
    for dp in _get_power_of_two_smaller_than_or_equal_to(num_gpus):
        if num_gpus % dp != 0:
            continue
        remaining = num_gpus // dp
        for tp in _get_power_of_two_smaller_than_or_equal_to(remaining):
            if remaining % tp != 0:
                continue
            pp = remaining // tp
            pairs.append((dp, tp, pp))
    return list(set(pairs))


def _generate_parallelism_cases(
    num_nodes, num_gpus_per_node, global_batch_size, num_layers
):
    cases = []
    num_gpus = num_nodes * num_gpus_per_node
    _3d_parallelism_cases = _get_3d_parallelism_cases(num_gpus)
    for dp, tp, pp in _3d_parallelism_cases:
        # Constraint 1: If num_layers is smaller than the pp, we can't do pipeline parallelism
        if num_layers < pp:
            continue
        # Constraint 2: dp * bsz * #microbatches = global_batch_size
        remaining_batch_size = global_batch_size // dp
        # Heuristic 1: Set batch size to 1 if DP and TP are both 1
        if dp == 1 and tp == 1:
            cases.append((dp, tp, pp, 1, remaining_batch_size))
            continue
        for batch_size in _get_power_of_two_smaller_than_or_equal_to(
            remaining_batch_size
        ):
            if batch_size not in BATCH_SIZE_CANDIDATES:
                continue
            if remaining_batch_size % batch_size != 0:
                continue
            gradient_accumulation_steps = remaining_batch_size // batch_size
            cases.append((dp, tp, pp, batch_size, gradient_accumulation_steps))
    return cases


def generate_cases(
    model_name, global_batch_size_list, seq_length, num_nodes, num_gpus_per_node, args
):
    standard_model_name, model_size = model_name.split("-")
    if standard_model_name in ["gpt2", "bert", "llama", "falcon"]:
        MODELS = GPT_MODELS
    else:
        raise ValueError(f"Unsupported model name: {standard_model_name}")

    hidden_size, num_layers, num_heads, vocab_size = MODELS[model_size]
    use_flash_attn = args.use_flash_attn

    cases = []
    for global_batch_size in global_batch_size_list:
        parallelism_cases = _generate_parallelism_cases(
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            global_batch_size=global_batch_size,
            num_layers=num_layers,
        )
        for parallelism_case in parallelism_cases:
            dp, tp, pp, batch_size, microbatches = parallelism_case
            if args.max_tp_size is not None and tp > args.max_tp_size:
                continue
            model_config = (seq_length, hidden_size, num_layers, num_heads, vocab_size)
            # zero_choices = [True, False] if dp > 1 else [False]
            zero_choices = [True]
            for zero in zero_choices:
                case = (
                    standard_model_name,
                    global_batch_size,
                    model_config,
                    microbatches,
                    (dp, tp, pp, True, zero, use_flash_attn),
                    False,  # No profiling
                )  # The last True means we do the gradient checkpointing
                case_str = str(case)
                cases.append(case_str)
    cases = list(set(cases))
    cases = sorted(cases)
    print("Total number of cases:", len(cases), flush=True)
    return cases


def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes
    results_dir = args.results_dir
    output_name = args.exp_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cases = generate_cases(
        args.model,
        args.global_batch_sizes,
        args.seq_length,
        args.nnodes,
        args.nproc_per_node,
        args=args,
    )

    for i, case_str in enumerate(cases):
        print(f"Running case {i+1}/{len(cases)}: {case_str}", flush=True)
        if args.nnodes == 1:
            # Single node
            ret = run_cmd(
                "torchrun "
                f"--nproc_per_node {args.nproc_per_node} "
                "benchmark_gpt_bert_one_case.py "
                f'"{case_str}" '
                f"{results_dir} "
                f"{output_name} "
                f">> {os.path.join(results_dir, output_name)}.log 2>&1"
            )
        else:
            # Multiple nodes
            ret = run_cmd(
                "torchrun "
                f"--nproc_per_node {args.nproc_per_node} "
                f"--nnodes {args.nnodes} "
                f"--node_rank {args.node_rank} "
                f"--master_addr {args.master_addr} "
                f"--master_port {args.master_port} "
                "benchmark_gpt_bert_one_case.py "
                f'"{case_str}" '
                f"{results_dir} "
                f"{output_name} "
                f">> {os.path.join(results_dir, output_name)}.log 2>&1"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", "-m", type=int, required=True)
    parser.add_argument("--nnodes", "-n", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--global_batch_sizes", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_tp_size", type=int, default=None)
    parser.add_argument("--use_flash_attn", action="store_true")
    args = parser.parse_args()

    # Update the global batch size list
    args.global_batch_sizes = eval(args.global_batch_sizes)
    if isinstance(args.global_batch_sizes, Number):
        args.global_batch_sizes = [int(args.global_batch_sizes)]
    else:
        assert isinstance(args.global_batch_sizes, (tuple, list))
        assert all(isinstance(x, Number) for x in args.global_batch_sizes)
        args.global_batch_sizes = [int(x) for x in args.global_batch_sizes]

    start_time = time.time()
    benchmark_all(args)
    end_time = time.time()
    with open(
        os.path.join(args.results_dir, f"{args.exp_name}_benchmark_time.txt"), "w"
    ) as f:
        f.write(f"Total time: {end_time - start_time:.2f} seconds")
