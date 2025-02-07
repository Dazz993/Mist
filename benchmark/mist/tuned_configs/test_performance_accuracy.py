import json
import ast
import argparse

def calculate_average_accuracies(data):
    total_perf_accuracy = 0
    total_mem_accuracy = 0
    num_entries = len(data)
    
    for key, values in data.items():
        exec_total_cost = float(values["exec_total_cost"])
        analyzed_total_cost = float(values["analyzed_total_cost"])
        
        # Performance accuracy calculation
        perf_accuracy = (analyzed_total_cost / exec_total_cost) * 100
        total_perf_accuracy += perf_accuracy

        # Remove MB from the string and convert to float
        values["exec_stage_peak_allocated_memories"] = values["exec_stage_peak_allocated_memories"].replace("MB", "")
        values["analyzed_stage_peak_memories"] = values["analyzed_stage_peak_memories"].replace("MB", "")
        
        # Memory accuracy calculation
        exec_mem = ast.literal_eval(values["exec_stage_peak_allocated_memories"])  # Convert string list to actual list
        analyzed_mem = ast.literal_eval(values["analyzed_stage_peak_memories"])  # Convert string list to actual list
        
        mem_accuracies = [(analyzed / exec) * 100 for analyzed, exec in zip(analyzed_mem, exec_mem)]
        avg_mem_accuracy = sum(mem_accuracies) / len(mem_accuracies)
        total_mem_accuracy += avg_mem_accuracy
    
    avg_perf_accuracy = total_perf_accuracy / num_entries
    avg_mem_accuracy = total_mem_accuracy / num_entries
    
    return {
        "average_performance_accuracy": avg_perf_accuracy,
        "average_memory_prediction_accuracy": avg_mem_accuracy
    }

# Example usage with a JSON file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, nargs="+")

    all_results = []

    for data_file in parser.parse_args().data:
        with open(data_file, "r") as file:
            data = json.load(file)        
            results = calculate_average_accuracies(data)
            all_results.append((results, len(data)))
    
    print(all_results)