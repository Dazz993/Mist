import argparse
import json
import os
import re
from typing import Dict, Any, List, Tuple

import numpy as np

def collect_data_series(data_dir):
    """
    Collects and loads all data under the given directory.

    Parameters
    ----------
    data_dir : str
        The directory containing the data files.
    """
    data_series : Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename == "summary.json":
                full_path = os.path.join(dirpath, filename)
                with open(full_path, "r") as f:
                    data_series[full_path] = json.load(f)
                print(f"Loaded {full_path}")
    return data_series


def extract_numbers_from_string(s):
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', s)
    return [float(num) for num in numbers]

def calculate_accuracy_for_data(data: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    processed_analyzed_costs = []
    processed_exec_costs = []
    processed_analyzed_memories = []
    processed_exec_memories = []
    for config_name, results in data.items():
        if "analyzed_total_cost" not in results or "analyzed_stage_peak_memories" not in results:
            # print(f"Skipping {config_name} because it does not have the required fields.")
            continue
        if "exec_total_cost" not in results or "exec_stage_peak_allocated_memories" not in results:
            # print(f"Skipping {config_name} because it does not have the required fields.")
            continue
        analyzed_total_cost = results["analyzed_total_cost"]
        analyzed_stage_peak_memories = results["analyzed_stage_peak_memories"]
        exec_total_cost = results["exec_total_cost"]
        exec_stage_peak_allocated_memories = results["exec_stage_peak_allocated_memories"]

        analyzed_total_cost = float(analyzed_total_cost)
        exec_total_cost = float(exec_total_cost)
        analyzed_stage_peak_memories = extract_numbers_from_string(analyzed_stage_peak_memories)
        exec_stage_peak_allocated_memories = extract_numbers_from_string(exec_stage_peak_allocated_memories)
        assert len(analyzed_stage_peak_memories) == len(exec_stage_peak_allocated_memories)

        processed_analyzed_costs.append(analyzed_total_cost)
        processed_exec_costs.append(exec_total_cost)
        processed_analyzed_memories.extend(analyzed_stage_peak_memories)
        processed_exec_memories.extend(exec_stage_peak_allocated_memories)

    processed_analyzed_costs = np.array(processed_analyzed_costs)
    processed_exec_costs = np.array(processed_exec_costs)
    processed_analyzed_memories = np.array(processed_analyzed_memories)
    processed_exec_memories = np.array(processed_exec_memories)

    # Offset the analyzed latency by the difference between the average analyzed latency and the average exec latency
    average_analyzed_latency = np.mean(processed_analyzed_costs)
    average_exec_latency = np.mean(processed_exec_costs)
    processed_analyzed_costs += (average_exec_latency - average_analyzed_latency)
    latency_accuracy = np.mean(np.abs(processed_analyzed_costs - processed_exec_costs) / processed_exec_costs)

    # Memory accuracy
    memory_accuracy = np.mean(np.abs(processed_analyzed_memories - processed_exec_memories) / processed_exec_memories)

    return latency_accuracy, memory_accuracy

def main(data_dir: str):
    """
    Main function to run the script.

    Parameters
    ----------
    data_dir : str
        The directory containing the data files.
    """
    data_series = collect_data_series(data_dir)
    data_series = dict(sorted(data_series.items(), key=lambda x: x[0]))
    for file_path, data in data_series.items():
        latency_accuracy, memory_accuracy = calculate_accuracy_for_data(data)
        print(f"File: {file_path}\tLatency Error Ratio:{latency_accuracy * 100:.2f}%\tMemory Error Ratio: {memory_accuracy * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, required=True, help="Data file")
    args = parser.parse_args()
    main(args.data_dir)