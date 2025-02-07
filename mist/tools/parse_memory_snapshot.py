import json
import pickle
import argparse
import sys

from functools import partial

from mist.utils.common import load_pickle, save_json, print_to_file
from mist.utils.memory import _format_size

"""
Usage: python analyze_mem.py [-d] -f <filename>
"""

FRAME_DEPTH_TO_ANALYZE = 5

mem = {}
mem["unknown"] = [0, []]


def parse_each_record(record):
    # Example record:
    # {'device': 0, 'address': 139657371189248, 'total_size': 16777216, 'allocated_size': 16777216, 'active_size': 16777216, 'stream': 0, 'segment_type': 'large', 'blocks': [{'size': 16777216, 'state': 'active_allocated', 'history': [{'addr': 139657371189248, 'real_size': 16777216, 'frames': [{'filename': '/home/zhanda/anaconda3/envs/revnn/lib/python3.8/site-packages/apex/optimizers/fused_adam.py', 'name': 'step', 'line': 147}, {'filename': '/home/zhanda/anaconda3/envs/revnn/lib/python3.8/site-packages/torch/optim/optimizer.py', 'name': 'wrapper', 'line': 263}, {'filename': '/home/zhanda/workspace/Reverse-NN/benchmark/megatron/Megatron-LM/megatron/optimizer/optimizer.py', 'name': 'step', 'line': 451}, {'filename': '/home/zhanda/anaconda3/envs/revnn/lib/python3.8/site-packages/torch/utils/_contextlib.py', 'name': 'decorate_context', 'line': 115}, {'filename': '/home/zhanda/workspace/Reverse-NN/benchmark/megatron/Megatron-LM/megatron/training.py', 'name': 'train_step', 'line': 435}, {'filename': 'benchmark_transformer_layer_one_case.py', 'name': 'benchmark_transformer_layer_one_case', 'line': 258}, {'filename': 'benchmark_transformer_layer_one_case.py', 'name': '<module>', 'line': 394}]}]}]}

    device = record["device"]
    address = record["address"]
    total_size = record["total_size"]
    allocated_size = record["allocated_size"]
    active_size = record["active_size"]
    stream = record["stream"]
    segment_type = record["segment_type"]
    blocks = record["blocks"]

    active_blocks = [block for block in blocks if block["state"] == "active_allocated"]
    inactive_blocks = [block for block in blocks if block["state"] == "inactive"]
    # TODO(zhanda): handle active_pending_free_blocks
    active_pending_free_blocks = [
        block for block in blocks if block["state"] == "active_pending_free"
    ]
    assert len(active_blocks) + len(inactive_blocks) + len(
        active_pending_free_blocks
    ) == len(
        blocks
    ), f"len(active_blocks) + len(inactive_blocks) + len(active_pending_free_blocks) != len(blocks)\n{json.dumps(record, indent=4)}"
    print(
        f"There are {len(active_blocks)} active blocks, {len(inactive_blocks)} inactive blocks, and {len(active_pending_free_blocks)} active_pending_free blocks."
    )

    for block in active_blocks:
        if "history" not in block:
            mem["unknown"][0] += block["size"]
            mem["unknown"][1].append(block["size"])
            continue

        history = block["history"]
        for history_entry in history:
            frames = history_entry["frames"]
            if len(frames) == 0:
                mem["unknown"][0] += block["size"]
                mem["unknown"][1].append(block["size"])
                continue

            # Name the tensor allocation with the deepest frame
            frame_depth = len(frames)
            key = []
            for i in range(min(frame_depth, FRAME_DEPTH_TO_ANALYZE)):
                frame = frames[i]
                key.append(
                    f"{frame['filename']}:{frame['line']} ===> [{frame['name']}]"
                )

            key = tuple(key)

            if key not in mem:
                mem[key] = [0, []]
            mem[key][0] += block["size"]
            mem[key][1].append(block["size"])

    inactive_mem = 0
    for block in inactive_blocks:
        inactive_mem += block["size"]

    print(f"Inactive memory usage: {_format_size(inactive_mem)}")


def sort_by_size_and_print(mem):
    for i, (key, value) in enumerate(
        sorted(mem.items(), key=lambda item: item[1][0], reverse=True)
    ):
        msg = ""

        # total memory usage
        msg += _format_size(value[0])

        # allocation sizes
        sorted_each_memory_list = [
            _format_size(size) for size in reversed(sorted(value[1]))
        ]
        sorted_each_memory_set = [
            _format_size(size) for size in reversed(sorted(set(value[1])))
        ]
        sorted_mem_count_pairs = [
            f"{mem} * {sorted_each_memory_list.count(mem)}"
            for mem in sorted_each_memory_set
        ]

        msg += " (" + ", ".join(sorted_mem_count_pairs) + ")"

        # print
        if key == "unknown":
            key = [key]

        print(f"\n[{i + 1:02}] - {msg}")
        for frame in key:
            print(f"     - {frame}")


def cal_mem_sum(mem):
    ret = 0
    for key, value in mem.items():
        ret += value[0]
    return ret


def main(args):
    # Monkey patch print
    # if the file exists, it will be overwritten
    setattr(
        sys.modules["__main__"],
        "print",
        partial(
            print_to_file,
            filename=args.filename.replace(".pkl", ".log"),
            to_screen=True,
        ),
    )

    # Load the memory usage data
    mem_snapshots = load_pickle(args.filename)

    # Dump the original memory usage data if requested
    if args.dump:
        save_json(mem_snapshots, args.filename.replace(".pkl", ".json"))

    # Load and parse the memory usage data
    for mem_snapshot in mem_snapshots:
        parse_each_record(mem_snapshot)

    mem_sorted = sort_by_size_and_print(mem)

    print(f"Total memory usage that can be parsed: {_format_size(cal_mem_sum(mem))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyze memory usage")
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument(
        "-d", "--dump", action="store_true", help="Dump the original memory usage data"
    )
    args = parser.parse_args()

    main(args)
