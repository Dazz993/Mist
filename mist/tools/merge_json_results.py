import argparse
import json
import os
import torch

device_name = torch.cuda.get_device_name(0).split(" ")[-1].lower()


def merge_json_results(inputs, output):
    output_results = {device_name: {}}
    for input_ in inputs:
        with open(input_, "r") as f:
            input_result = json.load(f)
            output_results[device_name].update(input_result[device_name])

    # for device_name_, inner_results in output_results.items():
    #     to_delete = set()
    #     for config_str, info in inner_results.items():
    #         if int(info["intra_group_size"]) in (1, 2, 4):
    #             to_delete.add(config_str)
    #         if int(info["inter_group_size"]) in (2,):
    #             to_delete.add(config_str)
    #     for config_str in to_delete:
    #         del inner_results[config_str]

    with open(output, "w") as f:
        json.dump(output_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=f"results/bandwidth-merged-{device_name}.json",
    )
    parser.add_argument("inputs", type=str, nargs="+")
    args = parser.parse_args()
    merge_json_results(args.inputs, args.output)
