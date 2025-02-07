import os
import pandas as pd
import json
from collections import defaultdict
import tabulate

SIZES = ["1.3b", "2.7b", "7b"]

def extract_ds_megatron_model_info(path):
    """
    Extracts model type, size, and FlashAttn status from the given file path.
    """
    parts = path.split(os.sep)
    if len(parts) < 5:
        return None  # Not a valid path
    
    model_full = parts[-4]  # e.g., 'gpt2-1.3b'
    model_type = "gpt2" if "gpt2" in model_full else "llama"
    model_size = model_full.split('-')[1]  # e.g., '1.3b'
    flashattn = "True" if "flashattn_True" in parts[-2] else "False"
    
    return (model_type, model_size, flashattn)


def collect_summary_files(base_dir):
    """
    Recursively collects all summary.json files in the given directory.
    """
    summary_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "summary.json":
                summary_files.append(os.path.join(root, file))
    return summary_files

def gather_summary_data(base_dir):
    """
    Traverses the directory structure, reads summary.json files, and extracts relevant throughput data.
    """
    summary_data = {}
    summary_files = collect_summary_files(base_dir)

    for file_path in summary_files:
        try:
            with open(file_path, 'r') as f:
                if "gpt" in file_path:
                    model_type = "gpt2"
                elif "llama" in file_path:
                    model_type = "llama"
                else:
                    raise ValueError("Unknown model type")
                summary = json.load(f)
                for key, value in summary.items():
                    model_size = key.split('-')[0]
                    batch_size = int(key.split('-')[3][2:])  # Extract b_ value
                    flashattn = "True" if "f_True" in key else "False"
                    exec_total_cost = float(value["exec_total_cost"])
                    throughput = batch_size / exec_total_cost  # Compute throughput
                    model_info = (model_type, model_size, flashattn)
                    summary_data[model_info] = throughput
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return summary_data


def collect_tsv_files(base_dir, implementation):
    """
    Recursively collects all .tsv files in the given directory.
    """
    tsv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".tsv") and implementation in file:
                tsv_files.append(os.path.join(root, file))
    return tsv_files

def gather_tsv_data(base_dir, implementation):
    """
    Traverses the directory structure, reads .tsv files, and extracts relevant data.
    """
    data_summary = defaultdict(float)
    tsv_files = collect_tsv_files(base_dir, implementation)
    
    for file_path in tsv_files:
        model_info = extract_ds_megatron_model_info(file_path)
        
        if model_info:
            try:
                df = pd.read_csv(file_path, sep='\t', header=None, dtype=str)
                df = df[~df.iloc[:, -2].str.contains("OOM", na=False)]  # Ensure 'OOM' rows are ignored
                df.iloc[:, -2] = pd.to_numeric(df.iloc[:, -2], errors='coerce')  # Convert to numeric
                df = df.dropna(subset=[df.columns[-2]])  # Remove NaN values
                if not df.empty:
                    max_value = df.iloc[:, -2].max()  # Second last column
                    data_summary[model_info] = max(data_summary[model_info], max_value)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return data_summary


def print_results(megatron_data, deepspeed_data, mist_data):
    """
    Prints a tabulated comparison of throughput data across implementations.
    """
    model_type_with_flash = {(model_info[0], model_info[2]) for model_info in 
                             set(megatron_data.keys()) | set(deepspeed_data.keys()) | set(mist_data.keys())}
    
    list_of_data = []
    list_of_ratios = []
    
    for model_type, flashattn in model_type_with_flash:
        data = []
        ratios = []
        for size_ in SIZES:
            megatron_number = megatron_data.get((model_type, size_, flashattn), "")
            deepspeed_number = deepspeed_data.get((model_type, size_, flashattn), "")
            mist_number = mist_data.get((model_type, size_, flashattn), "")

            data.append([
                f"{model_type}-{size_}-flash_{flashattn}",
                megatron_data.get((model_type, size_, flashattn), ""),
                deepspeed_data.get((model_type, size_, flashattn), ""),
                mist_data.get((model_type, size_, flashattn), "")
            ])
            if megatron_number and deepspeed_number and mist_number:
                ratios.append([
                    f"{model_type}-{size_}-flash_{flashattn}",
                    f"{mist_number / megatron_number:.3f}X",
                    f"{mist_number / deepspeed_number:.3f}X",
                ])
        list_of_data.append(data)
        list_of_ratios.append(ratios)

    for data in list_of_data:
        print(tabulate.tabulate(data, headers=["Thoughput", "Megatron", "DeepSpeed", "Mist"], tablefmt="grid"))
    for ratios in list_of_ratios:
        print(tabulate.tabulate(ratios, headers=["SpeedUp", "SpeedUp vs\nMegatron", "SpeedUp vs\nDeepSpeed"], tablefmt="grid"))
    

# Example usage
base_deepspeed_directory = "deepspeed/results/l4"
base_megatron_directory = "megatron/results/l4"
deepspeed_data = gather_tsv_data(base_deepspeed_directory, "deepspeed")
megatron_data = gather_tsv_data(base_megatron_directory, "megatron")

base_mist_directory = "mist/tuned_configs/l4-24gb"
mist_data = gather_summary_data(base_mist_directory)

print(deepspeed_data)
print(megatron_data)
print(mist_data)

print_results(megatron_data, deepspeed_data, mist_data)