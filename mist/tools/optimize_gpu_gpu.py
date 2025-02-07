import numpy as np
from scipy.optimize import minimize
from mist.utils.common import load_json, save_json


def preprocess_data():
    data = load_json("results/single_node_bandwidth.json")
    intra_node_data = data["intra_node"]
    latencies = []
    bytes_ = []
    gpus = []
    for value in intra_node_data.values():
        curr_latency = float(value["latency"])
        curr_numel = int(value["numel"])
        curr_element_size = int(value["element_size"])
        curr_gpus = int(value["gpus"])
        latencies.append(curr_latency)
        bytes_.append(
            curr_numel * curr_element_size * 2 / 1024**3
        )  # because it's All-reduce, we need to multiply by 2
        gpus.append(curr_gpus)
    return np.array(latencies), np.array(bytes_), np.array(gpus)


latencies, bytes_, gpus = preprocess_data()


# Objective function to minimize
def objective_function(params):
    bandwidth, bias, dbias = params
    predicted_latencies = (bytes_ / bandwidth) * (
        (gpus - 1 + dbias) / (gpus + dbias)
    ) + bias
    return np.sum((latencies - predicted_latencies) ** 2)


# Initial guesses for bandwidth and bias
initial_guess = [6.0, 0.0, 0.0]

# Perform the minimization
result = minimize(objective_function, initial_guess)

if result.success:
    fitted_params = result.x
    print(f"Optimal Bandwidth: {fitted_params[0]}")
    print(f"Optimal Bias: {fitted_params[1]}")
    print(f"Optimal dBias: {fitted_params[2]}")
else:
    print("Optimization was not successful. Try different initial guesses.")
