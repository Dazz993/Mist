import numpy as np
from scipy.optimize import minimize
from mist.utils.common import load_json, save_json

from typing import List, Tuple, Union, Any

import torch
from copy import deepcopy


def preprocess_data():
    data = load_json("results/bandwidth.json")
    device_name = torch.cuda.get_device_name().lower().replace(" ", "-")
    data = data[device_name]
    gpu_to_gpu_bytes = []
    cpu_to_gpu_bytes = []
    gpu_to_cpu_bytes = []
    gpus = []
    latencies = []
    for value in data.values():
        curr_latency = float(value["latency"])
        curr_gpu_to_gpu_bytes = int(value["gpu_to_gpu_bytes"])
        curr_cpu_to_gpu_bytes = int(value["cpu_to_gpu_bytes"])
        curr_gpu_to_cpu_bytes = int(value["gpu_to_cpu_bytes"])
        curr_gpus = int(value["gpus"])

        latencies.append(curr_latency)
        gpu_to_gpu_bytes.append(curr_gpu_to_gpu_bytes / 1024**3)
        cpu_to_gpu_bytes.append(curr_cpu_to_gpu_bytes / 1024**3)
        gpu_to_cpu_bytes.append(curr_gpu_to_cpu_bytes / 1024**3)
        gpus.append(curr_gpus)
    return (
        np.array(latencies),
        np.array(gpu_to_gpu_bytes),
        np.array(cpu_to_gpu_bytes),
        np.array(gpu_to_cpu_bytes),
        np.array(gpus),
    )


def _select_gpu_to_gpu_only_data(data):
    latencies, gg, cg, gc, gpus = data
    assert (
        len(latencies) == len(gg) == len(cg) == len(gc) == len(gpus)
    ), f"Length of latencies {len(latencies)}, gg {len(gg)}, cg {len(cg)}, gc {len(gc)}, and gpus {len(gpus)} should be the same"
    ret_latencies = []
    ret_bytes = []
    ret_gpus = []
    for i, (l_, gg_, cg_, gc_, gpu_) in enumerate(zip(latencies, gg, cg, gc, gpus)):
        if cg_ == 0 and gc_ == 0:
            ret_latencies.append(l_)
            ret_bytes.append(gg_)
            ret_gpus.append(gpu_)
    return np.array(ret_latencies), np.array(ret_bytes), np.array(ret_gpus)


def _select_cpu_to_gpu_only_data(data):
    latencies, gg, cg, gc, gpus = data
    assert (
        len(latencies) == len(gg) == len(cg) == len(gc) == len(gpus)
    ), f"Length of latencies {len(latencies)}, gg {len(gg)}, cg {len(cg)}, gc {len(gc)}, and gpus {len(gpus)} should be the same"
    ret_latencies = []
    ret_bytes = []
    ret_gpus = []
    for i, (l_, gg_, cg_, gc_, gpu_) in enumerate(zip(latencies, gg, cg, gc, gpus)):
        if gg_ == 0 and gc_ == 0:
            ret_latencies.append(l_)
            ret_bytes.append(cg_)
            ret_gpus.append(gpu_)
        elif gg_ == 0 and cg_ == 0:
            ret_latencies.append(l_)
            ret_bytes.append(gc_)
            ret_gpus.append(gpu_)
    return np.array(ret_latencies), np.array(ret_bytes), np.array(ret_gpus)


def _gpu_to_gpu_latency_model(bytes_, gpus_, gg_bw_k, gg_bw_b, gg_bw_db):
    return (bytes_ / gg_bw_k) * (gpus_ - 1 + gg_bw_db) / (gpus_ + gg_bw_db) + gg_bw_b


def _cpu_to_gpu_latency_model(bytes_, gpus_, cg_bw_k, cg_bw_b):
    return (bytes_ / cg_bw_k) + cg_bw_b


def get_gpu_to_gpu_bandwidth(data):
    latencies, bytes_, gpus = _select_gpu_to_gpu_only_data(data)
    print(f"*** Length of gpu to gpu only data: {len(latencies)}")

    def objective_function(params):
        gg_bw_k, gg_bw_b, gg_bw_db = params
        predicted_latencies = _gpu_to_gpu_latency_model(
            bytes_, gpus, gg_bw_k, gg_bw_b, gg_bw_db
        )
        return np.sum((latencies - predicted_latencies) ** 2)
        # return np.sum(np.abs(latencies - predicted_latencies) / latencies)

    initial_guess = [4.5, 0.0, 0.5]
    result = minimize(objective_function, initial_guess)

    if result.success:
        fitted_params = result.x
        return fitted_params

    raise ValueError("Optimization was not successful. Try different initial guesses.")


def get_cpu_to_gpu_bandwidth(data):
    latencies, bytes_, gpus = _select_cpu_to_gpu_only_data(data)
    print(f"*** Length of cpu to gpu only data: {len(latencies)}")

    def objective_function(params):
        cg_bw_k, cg_bw_b = params
        predicted_latencies = _cpu_to_gpu_latency_model(bytes_, gpus, cg_bw_k, cg_bw_b)
        return np.sum((latencies - predicted_latencies) ** 2)
        # return np.sum(np.abs(latencies - predicted_latencies) / latencies)

    initial_guess = [12.0, 0.0]
    result = minimize(objective_function, initial_guess)

    if result.success:
        fitted_params = result.x
        return fitted_params

    raise ValueError("Optimization was not successful. Try different initial guesses.")


# num_choices = 6
# gg = gg[:num_choices]
# cg = cg[:num_choices]
# gc = gc[:num_choices]
# latencies = latencies[:num_choices]
# gpus = gpus[:num_choices]
# gg[-1] += 1
# cg[-1] += 1
# gc[-1] += 1
# indices = [23, 26, 32, 34, 57, 63, 91, 93]
# gg = gg[indices]
# cg = cg[indices]
# gc = gc[indices]
# latencies = latencies[indices]
# gpus = gpus[indices]

EPS = 1e-6


def _select_multi_comm_data(data):
    latencies, gg, cg, gc, gpus = data
    assert (
        len(latencies) == len(gg) == len(cg) == len(gc) == len(gpus)
    ), f"Length of latencies {len(latencies)}, gg {len(gg)}, cg {len(cg)}, gc {len(gc)}, and gpus {len(gpus)} should be the same"
    ret_latencies = []
    ret_gg = []
    ret_cg = []
    ret_gc = []
    ret_gpus = []
    for i, (l_, gg_, cg_, gc_, gpu_) in enumerate(zip(latencies, gg, cg, gc, gpus)):
        if int(gg_ > 0) + int(cg_ > 0) + (gc_ > 0) >= 2:
            ret_latencies.append(l_)
            ret_gg.append(gg_)
            ret_cg.append(cg_)
            ret_gc.append(gc_)
            ret_gpus.append(gpu_)
    return (
        np.array(ret_latencies),
        np.array(ret_gg),
        np.array(ret_cg),
        np.array(ret_gc),
        np.array(ret_gpus),
    )


def calculate_overlap_and_remain(
    raws: List[Any], factors: List[float]
) -> Tuple[Any, List[Any]]:
    assert len(raws) == len(factors), (
        f"Length of raws {len(raws)} and factors "
        f"({len(factors)}) should be the same"
    )
    scaled = [raw * factor for raw, factor in zip(raws, factors)]
    overlap = deepcopy(scaled[0])
    for scale in scaled:
        overlap = np.minimum(overlap, scale)
    remainings = [
        np.maximum(raw - overlap / factor, 0) for raw, factor in zip(raws, factors)
    ]
    return overlap, remainings


def _interference_model(
    latencies,
    gg,
    cg,
    gc,
    gpus,
    gg_bw_k,
    gg_bw_b,
    gg_bw_db,
    cg_bw_k,
    cg_bw_b,
    g_factor_for_three,
    c_factor_for_three,
    g_factor_for_gg_cg,
    c_factor_for_gg_cg,
    c_factor_for_cg_gc,
):
    gg_latencies = _gpu_to_gpu_latency_model(gg, gpus, gg_bw_k, gg_bw_b, gg_bw_db)
    cg_latencies = _cpu_to_gpu_latency_model(cg, gpus, cg_bw_k, cg_bw_b)
    gc_latencies = _cpu_to_gpu_latency_model(gc, gpus, cg_bw_k, cg_bw_b)

    predicted_latencies = np.zeros_like(latencies)
    updated = np.zeros_like(latencies)
    num_zeros = (
        np.isclose(gg_latencies, 0, atol=EPS).astype(int)
        + np.isclose(cg_latencies, 0, atol=EPS).astype(int)
        + np.isclose(gc_latencies, 0, atol=EPS).astype(int)
    )
    # Update the predicted latencies if two of them are zero
    preliminary_indices = num_zeros >= 2
    predicted_latencies += np.where(
        preliminary_indices,
        gg_latencies + cg_latencies + gc_latencies,
        0,
    )

    # Calculate the overlap and remainings
    overlap, (gg_, cg_, gc_) = calculate_overlap_and_remain(
        [gg_latencies, cg_latencies, gc_latencies],
        [g_factor_for_three, c_factor_for_three, c_factor_for_three],
    )
    num_zeros = (
        np.isclose(gg_, 0, atol=EPS).astype(int)
        + np.isclose(cg_, 0, atol=EPS).astype(int)
        + np.isclose(gc_, 0, atol=EPS).astype(int)
    )
    assert np.all(num_zeros > 0), (
        f"num_zeros should be at least 1 after the first overlap calculation, but got {num_zeros}. "
        f"Indices: {np.where(num_zeros == 0)}"
    )
    first_round_indices = np.logical_and(
        np.logical_not(preliminary_indices), num_zeros == 2
    )
    predicted_latencies += np.where(
        first_round_indices,
        overlap + gg_ + cg_ + gc_,
        0,
    )

    # If gc is zero
    overlap_1, (gg_1, cg_1) = calculate_overlap_and_remain(
        [gg_, cg_], [g_factor_for_gg_cg, c_factor_for_gg_cg]
    )
    assert np.all(
        np.logical_or(np.isclose(gg_1, 0, atol=EPS), np.isclose(cg_1, 0, atol=EPS))
    )
    indices_1 = np.logical_and(num_zeros == 1, np.isclose(gc_, 0, atol=EPS))
    predicted_latencies += np.where(
        indices_1,
        overlap + overlap_1 + gg_1 + cg_1,
        0,
    )

    # If cg is zero
    overlap_2, (gg_2, gc_2) = calculate_overlap_and_remain(
        [gg_, gc_], [g_factor_for_gg_cg, c_factor_for_gg_cg]
    )
    assert np.all(
        np.logical_or(np.isclose(gg_2, 0, atol=EPS), np.isclose(gc_2, 0, atol=EPS))
    )
    indices_2 = np.logical_and(num_zeros == 1, np.isclose(cg_, 0, atol=EPS))
    predicted_latencies += np.where(
        indices_2,
        overlap + overlap_2 + gg_2 + gc_2,
        0,
    )

    # If gg is zero
    overlap_3, (cg_3, gc_3) = calculate_overlap_and_remain(
        [cg_, gc_], [c_factor_for_cg_gc, c_factor_for_cg_gc]
    )
    assert np.all(
        np.logical_or(np.isclose(cg_3, 0, atol=EPS), np.isclose(gc_3, 0, atol=EPS))
    )

    indices_3 = np.logical_and(num_zeros == 1, np.isclose(gg_, 0, atol=EPS))
    predicted_latencies += np.where(
        indices_3,
        overlap + overlap_3 + cg_3 + gc_3,
        0,
    )

    # Double check
    updated = (
        preliminary_indices.astype(int)
        + first_round_indices.astype(int)
        + indices_1.astype(int)
        + indices_2.astype(int)
        + indices_3.astype(int)
    )
    assert all(
        updated == 1
    ), f"updated should be all 1, but got {updated}, indices: {np.where(updated != 1)}"

    return predicted_latencies


def get_interference_factors(data, gg_bw_k, gg_bw_b, gg_bw_db, cg_bw_k, cg_bw_b):
    latencies, gg, cg, gc, gpus = _select_multi_comm_data(data)
    print(f"*** Length of multi comm data: {len(latencies)}")

    def objective_function(params):
        (
            g_factor_for_three,
            c_factor_for_three,
            g_factor_for_gg_cg,
            c_factor_for_gg_cg,
            c_factor_for_cg_gc,
        ) = params
        predicted_latencies = _interference_model(
            latencies,
            gg,
            cg,
            gc,
            gpus,
            gg_bw_k,
            gg_bw_b,
            gg_bw_db,
            cg_bw_k,
            cg_bw_b,
            g_factor_for_three,
            c_factor_for_three,
            g_factor_for_gg_cg,
            c_factor_for_gg_cg,
            c_factor_for_cg_gc,
        )
        return np.sum((latencies - predicted_latencies) ** 2)
        # return np.sum(np.abs(latencies - predicted_latencies) / latencies)

    initial_guess = [2.0, 5.5, 1.7, 8.5, 1.2]
    result = minimize(objective_function, initial_guess, method="Nelder-Mead")

    if result.success:
        fitted_params = result.x
        return fitted_params

    else:
        raise ValueError(
            "Optimization was not successful. Try different initial guesses."
        )
        print(
            f"Optimization was not successful. Try different initial guesses. Return the initial guess."
        )
        return initial_guess


# # Objective function to minimize
# def objective_function(params):
#     predicted_latencies = func(params)
#     return np.sum((latencies - predicted_latencies) ** 2)


# # Initial guesses for bandwidth and bias
# initial_guess = [4.5, 0.5, 12.0, 3.0, 3.0, 2.0, 2.0, 1.0]

# # Perform the minimization
# result = minimize(objective_function, initial_guess)

# if result.success:
#     # First print all the differences between the predicted and actual latencies
#     _latencies = latencies * 1000
#     _predicted_latencies = func(result.x) * 1000
#     for i, (l, p) in enumerate(zip(_latencies, _predicted_latencies)):
#         print(f"Case {i + 1}: {l:.2f} ms, {p:.2f} ms, {abs(l - p):.2f} ms")
#     # Then print the optimal parameters
#     fitted_params = result.x
#     print(f"Optimal gg_bw_k: {fitted_params[0]}")
#     print(f"Optimal gg_bw_db: {fitted_params[1]}")
#     print(f"Optimal cg_bw_k: {fitted_params[2]}")
#     print(f"Optimal g_factor_for_three: {fitted_params[3]}")
#     print(f"Optimal c_factor_for_three: {fitted_params[4]}")
#     print(f"Optimal g_factor_for_gg_cg: {fitted_params[5]}")
#     print(f"Optimal c_factor_for_gg_cg: {fitted_params[6]}")
#     print(f"Optimal c_factor_for_cg_gc: {fitted_params[7]}")
# else:
#     print("Optimization was not successful. Try different initial guesses.")


if __name__ == "__main__":
    data = preprocess_data()

    print(f"*** Begins to optimize the bandwidth")
    gg_bw_k, gg_bw_b, gg_bw_db = get_gpu_to_gpu_bandwidth(data)
    print(f"gg_bw_k: {gg_bw_k}")
    print(f"gg_bw_b: {gg_bw_b}")
    print(f"gg_bw_db: {gg_bw_db}")
    print()

    print(f"*** Begins to optimize the bandwidth")
    cg_bw_k, cg_bw_b = get_cpu_to_gpu_bandwidth(data)
    print(f"cg_bw_k: {cg_bw_k}")
    print(f"cg_bw_b: {cg_bw_b}")
    print()

    print(f"*** Begins to optimize the interference factors")
    (
        g_factor_for_three,
        c_factor_for_three,
        g_factor_for_gg_cg,
        c_factor_for_gg_cg,
        c_factor_for_cg_gc,
    ) = get_interference_factors(data, gg_bw_k, gg_bw_b, gg_bw_db, cg_bw_k, cg_bw_b)
    print(f"g_factor_for_three: {g_factor_for_three}")
    print(f"c_factor_for_three: {c_factor_for_three}")
    print(f"g_factor_for_gg_cg: {g_factor_for_gg_cg}")
    print(f"c_factor_for_gg_cg: {c_factor_for_gg_cg}")
    print(f"c_factor_for_cg_gc: {c_factor_for_cg_gc}")
    print()

    # Print the differences between the predicted and actual latencies
    latencies, gg, cg, gc, gpus = data
    predicted_latencies = _interference_model(
        latencies,
        gg,
        cg,
        gc,
        gpus,
        gg_bw_k,
        gg_bw_b,
        gg_bw_db,
        cg_bw_k,
        cg_bw_b,
        g_factor_for_three,
        c_factor_for_three,
        g_factor_for_gg_cg,
        c_factor_for_gg_cg,
        c_factor_for_cg_gc,
    )
    # for i, (l, p) in enumerate(zip(latencies, predicted_latencies)):
    #     l *= 1000
    #     p *= 1000
    #     print(
    #         f"[Case {i + 1}] - (gg: {gg[i]:.2f} GB, cg: {cg[i]:.2f} GB, gc: {gc[i]:.2f} GB, gpus: {gpus[i]}) "
    #         f"Actual: {l:.2f} ms\t Predicted: {p:.2f} ms\t Difference: {abs(l - p):.2f} ms"
    #     )
