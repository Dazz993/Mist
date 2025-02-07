import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Any
from numbers import Number
import argparse
from functools import partial

import torch
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from numba import njit, prange
from mist.utils.common import load_json, save_json
from mist.utils.device import get_simplified_device_name

minimize = partial(minimize, method="TNC", options={"maxfun": 100000})

UP_BOUND = 50

num_gpus_per_device = torch.cuda.device_count()
COMPUTE_M = 4096
COMPUTE_K = 4096


def _latency_str_to_float(latency: str) -> float:
    if isinstance(latency, Number):
        return float(latency)
    if "ms" in latency:
        return float(latency.replace("ms", "")) / 1000
    elif "s" in latency:
        return float(latency.replace("s", ""))
    return float(latency)


def preprocess_data(args):
    data = load_json(args.data_path)
    device_name = get_simplified_device_name(lower=True)
    data = data[device_name]
    compute_n_s = []
    compute_latencies = []
    gpu_to_gpu_gbytes = []
    cpu_to_gpu_gbytes = []
    gpu_to_cpu_gbytes = []
    intra_sizes = []
    inter_sizes = []
    latencies = []
    for value in data.values():
        if "ms" in value["latency"]:
            value["latency"] = _latency_str_to_float(value["latency"])
        if "ms" in value["std"]:
            value["std"] = _latency_str_to_float(value["std"])
        curr_latency = float(value["latency"])
        curr_latency = curr_latency if curr_latency > 1e-5 else 0
        curr_std = float(value["std"])
        curr_std_div_mean = float(value["std/mean"])
        curr_compute_n = int(value["compute_n"])
        curr_gpu_to_gpu_mbytes = int(value["gpu_to_gpu_mbytes"])
        curr_cpu_to_gpu_mbytes = int(value["cpu_to_gpu_mbytes"])
        curr_gpu_to_cpu_mbytes = int(value["gpu_to_cpu_mbytes"])
        curr_world_size = int(value["world_size"])
        curr_intra_size = int(value["intra_group_size"])
        curr_inter_size = int(value["inter_group_size"])

        if curr_std_div_mean > 0.1:
            print(
                f"WARNING: Entry {curr_compute_n}-{curr_world_size}-{curr_intra_size}-{curr_inter_size} has std/mean: {curr_std_div_mean}. Skipping..."
            )
            continue

        # We actually don't need compute batch size, we only need the compute latency
        entry_name = f"N_{curr_compute_n}-M_{COMPUTE_M}-K_{COMPUTE_K}-G2G_0-C2G_0-G2C_0-W_{curr_world_size}-Intra_{curr_intra_size}-Inter_{curr_inter_size}"
        if entry_name in data:
            curr_compute_latency = _latency_str_to_float(data[entry_name]["latency"])
            curr_compute_latency = (
                curr_compute_latency if curr_compute_latency > 1e-5 else 0
            )
        elif curr_compute_n == 0:
            curr_compute_latency = 0
        else:
            raise ValueError(f"Entry {entry_name} not found in the data")
            # print(f"WARN: Entry {entry_name} not found in the data")
            # continue

        latencies.append(curr_latency)
        compute_n_s.append(curr_compute_n)
        compute_latencies.append(curr_compute_latency)
        gpu_to_gpu_gbytes.append(curr_gpu_to_gpu_mbytes / 1024)
        cpu_to_gpu_gbytes.append(curr_cpu_to_gpu_mbytes / 1024)
        gpu_to_cpu_gbytes.append(curr_gpu_to_cpu_mbytes / 1024)
        intra_sizes.append(curr_intra_size)
        inter_sizes.append(curr_inter_size)
    return (
        np.array(latencies),
        np.array(compute_n_s),
        np.array(compute_latencies),
        np.array(gpu_to_gpu_gbytes),
        np.array(cpu_to_gpu_gbytes),
        np.array(gpu_to_cpu_gbytes),
        np.array(intra_sizes),
        np.array(inter_sizes),
    )


def preprocess_interference_fitting_data(
    data: List[np.ndarray],
    g2g_params: List[Tuple[float, float]],
    c2g_params: Tuple[float, float],
    g2c_params: Tuple[float, float],
    intra_size: Optional[int] = None,
    remove_outliers: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L, B, C, G2G, C2G, G2C, INTRA, INTER = data
    assert (
        len(L)
        == len(B)
        == len(C)
        == len(G2G)
        == len(C2G)
        == len(G2C)
        == len(INTRA)
        == len(INTER)
    ), f"Lengths: {len(L)}, {len(B)}, {len(C)}, {len(G2G)}, {len(C2G)}, {len(G2C)}, {len(INTRA)}, {len(INTER)}"

    # Only select data when INTRA == 8
    # if intra_size is not None:
    #     indices = np.where(INTRA == intra_size)[0]
    #     L = L[indices]
    #     B = B[indices]
    #     C = C[indices]
    #     G2G = G2G[indices]
    #     C2G = C2G[indices]
    #     G2C = G2C[indices]
    #     INTRA = INTRA[indices]
    #     INTER = INTER[indices]

    G2G_LATENCIES = predict_gpu_gpu_comm_latency(
        op_name="all_gather",
        gbytes=G2G,
        intra_size=INTRA,
        inter_size=INTER,
        gpu_gpu_comm_params=g2g_params,
    )

    C2G_LATENCIES = cpu_to_gpu_only_latency_model(C2G, None, c2g_params)
    G2C_LATENCIES = cpu_to_gpu_only_latency_model(G2C, None, g2c_params)

    num_zeros = (
        np.isclose(C, 0).astype(int)
        + np.isclose(G2G_LATENCIES, 0).astype(int)
        + np.isclose(C2G_LATENCIES, 0).astype(int)
        + np.isclose(G2C_LATENCIES, 0).astype(int)
    )
    assert np.all(num_zeros <= 3), f"Num zeros: {num_zeros=}"
    indices = np.where(num_zeros != 3)
    print(f"Number of data points in interference: {len(indices[0])}")
    print(f"Number of data points excluded: {len(L) - len(indices[0])}")

    L = L[indices]
    C = C[indices]
    G2G_LATENCIES = G2G_LATENCIES[indices]
    C2G_LATENCIES = C2G_LATENCIES[indices]
    G2C_LATENCIES = G2C_LATENCIES[indices]
    INTRA = INTRA[indices]
    INTER = INTER[indices]

    if remove_outliers:
        indices = L <= C + G2G_LATENCIES + C2G_LATENCIES + G2C_LATENCIES
        print(
            f"Number of data points in interference after removing outliers: {np.sum(indices)}"
        )
        L = L[indices]
        C = C[indices]
        G2G_LATENCIES = G2G_LATENCIES[indices]
        C2G_LATENCIES = C2G_LATENCIES[indices]
        G2C_LATENCIES = G2C_LATENCIES[indices]
        INTRA = INTRA[indices]
        INTER = INTER[indices]

    return L, C, G2G_LATENCIES, C2G_LATENCIES, G2C_LATENCIES, INTRA, INTER


def select_data_with_different_combination(
    data: List[np.ndarray],
    compute: bool,
    g2g: bool,
    c2g: bool,
    g2c: bool,
):
    L, C, G2G, C2G, G2C, INTRA, INTER = data
    # Construct the mask
    mask = np.array([compute, g2g, c2g, g2c])
    X = np.vstack((C, G2G, C2G, G2C)).T
    is_non_zero = np.logical_not(np.isclose(X, 0))
    indices = np.where(np.all(is_non_zero == mask, axis=-1))[0]
    return L[indices], C[indices], G2G[indices], C2G[indices], G2C[indices]


# =============================================================================
# Single Case
def predict_gpu_gpu_comm_latency(
    op_name: str,
    gbytes: np.ndarray,
    intra_size: np.ndarray,
    inter_size: np.ndarray,
    gpu_gpu_comm_params: List[Tuple[float, float]],
):
    # GPU GPU comm params are structured as follows:
    # N: inter [1, 2, 4, 8] -> outer dimension
    # M: intra [1, 2, 4, 8] -> inner dimension
    # Combination: 4 * 4 * 2 = 32
    # So for i-th index, intra = i % 4, inter = i // 4
    # [N1M1, N1M2, N1M4, N1M8],
    # ...
    if op_name not in ["p2p", "all_reduce", "all_gather", "reduce_scatter", "none"]:
        raise ValueError(f"Invalid op_name: {op_name}")

    gpu_gpu_comm_params = np.array(gpu_gpu_comm_params)
    if gpu_gpu_comm_params.size != 32:
        raise ValueError(
            f"Invalid size of gpu_gpu_comm_params: {gpu_gpu_comm_params.size}"
        )
    gpu_gpu_comm_params = gpu_gpu_comm_params.reshape((4, 4, 2))

    if op_name == "p2p":
        assert inter_size is None or intra_size is None
        if inter_size is not None:
            # Choose inter=2, intra=1
            _size = inter_size
            _bandwidth = gpu_gpu_comm_params[1, 0][0]
        else:
            # Choose inter=1, intra=2
            _size = intra_size
            _bandwidth = gpu_gpu_comm_params[0, 1][0]
        assert np.all(_size == 1), f"_size: {_size}"
        latency = gbytes / _bandwidth
        return latency

    comm = gpu_to_gpu_only_latency_model(
        gbytes, intra_size=intra_size, inter_size=inter_size, params=gpu_gpu_comm_params
    )

    if op_name == "all_reduce":
        factor = 2.1
    elif op_name == "all_gather":
        factor = 1.0
    elif op_name == "reduce_scatter":
        factor = 1.1
    elif op_name == "none":
        factor = 0.0
    else:
        raise ValueError(f"Invalid op_name: {op_name}")

    return comm * factor


@njit(parallel=True, cache=True)
def _compute_comm_latency_impl(
    bytes_: np.ndarray,
    intra_size: np.ndarray,
    inter_size: np.ndarray,
    params: np.ndarray,
):
    nsamples = len(bytes_)
    latency = np.empty(nsamples, dtype=np.float64)
    for i in prange(nsamples):
        byte_ = bytes_[i]
        intra = intra_size[i]
        inter = inter_size[i]
        ngpu = intra * inter

        # Corner case handling
        if byte_ < 1e-5 or ngpu == 1:
            latency[i] = 0
            continue

        if intra == 1:
            inner_index = 0
        elif intra == 2:
            inner_index = 1
        elif intra == 4:
            inner_index = 2
        elif intra == 8:
            inner_index = 3
        elif intra > 8:
            inner_index = 3

        if inter == 1:
            outer_index = 0
        elif inter == 2:
            outer_index = 1
        elif inter == 4:
            outer_index = 2
        elif inter == 8:
            outer_index = 3
        elif inter > 8:
            outer_index = 3

        k, b = params[outer_index, inner_index]

        latency[i] = byte_ / k * (ngpu - 1) / ngpu + b

    return latency


def gpu_to_gpu_only_latency_model(
    bytes_: np.ndarray,
    intra_size: np.ndarray,
    inter_size: np.ndarray,
    params: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    if params is None:
        return np.zeros_like(bytes_)

    is_number = (
        isinstance(bytes_, Number)
        and isinstance(intra_size, Number)
        and isinstance(inter_size, Number)
    )

    # Set default number
    intra_size = intra_size if intra_size is not None else 1
    inter_size = inter_size if inter_size is not None else 1
    # Expand the bytes and gpus to the same size
    bytes_ = np.atleast_1d(bytes_)
    intra_size = np.atleast_1d(intra_size)
    inter_size = np.atleast_1d(inter_size)
    # It should be a 1D array
    assert bytes_.ndim == 1, f"{bytes_.ndim}"
    assert intra_size.ndim == 1, f"{intra_size.ndim}"
    assert inter_size.ndim == 1, f"{inter_size.ndim}"
    # Broadcast to the same size
    max_size = max(bytes_.size, intra_size.size, inter_size.size)
    bytes_ = np.broadcast_to(bytes_, (max_size,))
    intra_size = np.broadcast_to(intra_size, (max_size,))
    inter_size = np.broadcast_to(inter_size, (max_size,))

    # Reshape params
    params = np.array(params)
    assert params.size == 32, f"{params.size}"
    params = params.reshape((4, 4, 2))

    # Calculate the latency
    latency = _compute_comm_latency_impl(bytes_, intra_size, inter_size, params)

    if is_number:
        return latency[0]
    return latency


def fit_gpu_to_gpu_only_params(data):
    def collect_data(target_intra_size=None, target_inter_size=None):
        L, B, C, G2G, C2G, G2C, INTRA, INTER = data
        assert (
            len(L)
            == len(B)
            == len(C)
            == len(G2G)
            == len(C2G)
            == len(G2C)
            == len(INTRA)
            == len(INTER)
        ), f"Lengths: {len(L)}, {len(B)}, {len(C)}, {len(G2G)}, {len(C2G)}, {len(G2C)}, {len(INTRA)}, {len(INTER)}"
        latencies = []
        bytes_ = []
        intra_sizes = []
        inter_sizes = []
        for i, (l, b, c, g2g, c2g, g2c, intra, inter) in enumerate(
            zip(L, B, C, G2G, C2G, G2C, INTRA, INTER)
        ):
            if g2g != 0 and b == 0 and c2g == 0 and g2c == 0:
                satisfy_intra = (
                    target_intra_size is None
                    or intra == target_intra_size
                    or (target_intra_size == 8 and intra > 8)
                )
                satisfy_inter = (
                    target_inter_size is None
                    or inter == target_inter_size
                    or (target_inter_size == 8 and inter > 8)
                )
                if satisfy_intra and satisfy_inter:
                    latencies.append(l)
                    bytes_.append(g2g)
                    intra_sizes.append(intra)
                    inter_sizes.append(inter)
        return (
            np.array(latencies),
            np.array(bytes_),
            np.array(intra_sizes),
            np.array(inter_sizes),
        )

    # =============================================================================
    # Intra-node
    # =============================================================================
    # First do the intra-node
    def loss_fn(params, data):
        latencies, bytes_, intra_sizes, inter_sizes = data
        pred = gpu_to_gpu_only_latency_model(bytes_, intra_sizes, inter_sizes, params)
        return np.sum((pred - latencies) ** 2)

    initial_guess = []
    bounds = []
    for i in range(4 * 4):
        initial_guess.extend([4.5, 0.0])
        bounds.extend([(0.1, None), (0, None)])

    ret = deepcopy(initial_guess)
    for intra_index in range(4):
        for inter_index in range(4):
            curr_target_intra_size = pow(2, intra_index)
            curr_target_inter_size = pow(2, inter_index)
            curr_initial_guess = deepcopy(initial_guess)
            curr_bounds = deepcopy(bounds)
            curr_data = collect_data(curr_target_intra_size, curr_target_inter_size)
            if len(curr_data[0]) == 0:
                continue
            curr_result = minimize(
                partial(loss_fn, data=curr_data),
                curr_initial_guess,
                bounds=curr_bounds,
            )
            if not curr_result.success:
                raise ValueError(f"Optimization failed: {curr_result.message}")

            # Update the result
            curr_slice = slice(
                2 * inter_index * 4 + 2 * intra_index,
                2 * inter_index * 4 + 2 * intra_index + 2,
            )
            ret[curr_slice] = curr_result.x[curr_slice]

            # Calculate the average error ratio
            curr_latenices, curr_bytes_, curr_intra_sizes, curr_inter_sizes = curr_data
            curr_pred = gpu_to_gpu_only_latency_model(
                curr_bytes_, curr_intra_sizes, curr_inter_sizes, curr_result.x
            )
            curr_error = np.abs(curr_latenices - curr_pred)
            curr_error_ratio = curr_error / curr_latenices
            print(
                f"[Intra={curr_target_intra_size}, Inter={curr_target_inter_size}] "
                f"Number of data points: {len(curr_latenices)}, "
                f"Params: {[round(p, 4) for p in ret[curr_slice]]}, "
                f"Average GPU <-> GPU error ratio: {np.mean(curr_error_ratio) * 100:.2f} %"
            )

    latenices, bytes_, intra_sizes, inter_sizes = collect_data()
    pred = gpu_to_gpu_only_latency_model(bytes_, intra_sizes, inter_sizes, ret)
    error = np.abs(latenices - pred)
    error_ratio = error / latenices
    print(f"Average GPU <-> GPU error ratio: {np.mean(error_ratio) * 100:.2f} %")

    return ret

    # return result.x


def predict_cpu_gpu_comm_latency(
    gbytes: np.ndarray, params: Tuple[float, float]
) -> np.ndarray:
    return cpu_to_gpu_only_latency_model(gbytes, np.array([1]), params)


def cpu_to_gpu_only_latency_model(
    bytes_: np.ndarray, gpus: np.ndarray, params: Tuple[float, float]
) -> np.ndarray:
    cg_bw_k, cg_bw_b = params
    latencies = (bytes_ / cg_bw_k) + cg_bw_b
    return np.where(bytes_ == 0, 0, latencies)


def fit_cpu_to_gpu_only_params(data):
    def collect_data():
        L, B, C, G2G, C2G, G2C, INTRA, INTER = data
        assert (
            len(L)
            == len(B)
            == len(C)
            == len(G2G)
            == len(C2G)
            == len(G2C)
            == len(INTRA)
            == len(INTER)
        ), f"Lengths: {len(L)}, {len(B)}, {len(C)}, {len(G2G)}, {len(C2G)}, {len(G2C)}, {len(INTRA)}, {len(INTER)}"
        c2g_latencies = []
        c2g_bytes = []
        g2c_latencies = []
        g2c_bytes = []
        for i, (l, b, c, g2g, c2g, g2c, intra, inter) in enumerate(
            zip(L, B, C, G2G, C2G, G2C, INTRA, INTER)
        ):
            if c2g != 0 and b == 0 and g2g == 0 and g2c == 0:
                c2g_latencies.append(l)
                c2g_bytes.append(c2g)
            if g2c != 0 and b == 0 and g2g == 0 and c2g == 0:
                g2c_latencies.append(l)
                g2c_bytes.append(g2c)
        return (
            (np.array(c2g_latencies), np.array(c2g_bytes)),
            (np.array(g2c_latencies), np.array(g2c_bytes)),
        )

    c2g_data, g2c_data = collect_data()
    print(f"Number of data points in C2G only: {len(c2g_data[0])}")
    print(f"Number of data points in G2C only: {len(g2c_data[0])}")

    def loss_fn(data, params):
        latencies, bytes_ = data
        pred = cpu_to_gpu_only_latency_model(bytes_, None, params)
        return np.sum((pred - latencies) ** 2)

    initial_guess = [6.0, 0.0]
    bounds = [(0.1, None), (0, None)]
    c2g_result = minimize(partial(loss_fn, c2g_data), initial_guess, bounds=bounds)
    g2c_result = minimize(partial(loss_fn, g2c_data), initial_guess, bounds=bounds)
    if not c2g_result.success:
        raise ValueError(f"Optimization failed: {c2g_result.message}")
    if not c2g_result.success:
        raise ValueError(f"Optimization failed: {c2g_result.message}")

    # Calculate the average error ratio for c2g
    c2g_pred = cpu_to_gpu_only_latency_model(c2g_data[1], None, c2g_result.x)
    c2g_error = np.abs(c2g_data[0] - c2g_pred)
    c2g_error_ratio = c2g_error / c2g_data[0]
    print(f"Average CPU -> GPU error ratio: {np.mean(c2g_error_ratio) * 100:.2f} %")
    # Calculate the average error ratio for g2c
    g2c_pred = cpu_to_gpu_only_latency_model(g2c_data[1], None, g2c_result.x)
    g2c_error = np.abs(g2c_data[0] - g2c_pred)
    g2c_error_ratio = g2c_error / g2c_data[0]
    print(f"Average GPU -> CPU error ratio: {np.mean(g2c_error_ratio) * 100:.2f} %")
    # for i, (l, b, g) in enumerate(zip(*c2g_data)):
    #     if c2g_error_ratio[i] > 0.1:
    #         print(
    #             f"[CPU -> GPU] Latency: {l:.4f}, Pred: {c2g_pred[i]:.4f}, Bytes: {b:.5f}, GPUs: {g}, Error: {c2g_error_ratio[i] * 100:.2f}% "
    #         )
    # for i, (l, b, g) in enumerate(zip(*g2c_data)):
    #     if g2c_error_ratio[i] > 0.1:
    #         print(
    #             f"[GPU -> CPU] Latency: {l:.4f}, Pred: {g2c_pred[i]:.4f}, Bytes: {b:.5f}, GPUs: {g}, Error: {g2c_error_ratio[i] * 100:.2f}%"
    #         )
    return c2g_result.x, g2c_result.x


# =============================================================================


# =============================================================================
# Overall Function


def calculate_overlap_and_remain(
    values: np.ndarray,
    factors: np.ndarray,
    remaining_factors: np.ndarray,
    feature_indices: np.ndarray = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if remaining_factors is None:
        remaining_factors = np.ones_like(factors)
    # Scale the values to get the overlap,
    # and calculate the remaining values
    assert (
        values.shape[-1] == factors.shape[-1]
    ), f"Values: {values.shape}, Factors: {factors.shape}"
    # Vectorized version
    scaled = values * factors
    overlap = np.min(scaled[..., feature_indices], axis=-1)
    updated = (scaled - overlap[:, None]) / factors * remaining_factors
    # We don't want to update
    values[..., feature_indices] = updated[..., feature_indices]
    return overlap, values


# def interference_estimate_deprecated(
#     C: np.ndarray,
#     G2G: np.ndarray,
#     C2G: np.ndarray,
#     G2C: np.ndarray,
#     params,
# ) -> np.ndarray:
#     if isinstance(C, Number):
#         C = np.array([C]).astype(float)
#     if isinstance(G2G, Number):
#         G2G = np.array([G2G]).astype(float)
#     if isinstance(C2G, Number):
#         C2G = np.array([C2G]).astype(float)
#     if isinstance(G2C, Number):
#         G2C = np.array([G2C]).astype(float)

#     # Order: Compute, G2G, C2G, G2C
#     (
#         # Compute + G2G
#         f_c_when_c_and_g2g,
#         f_g2g_when_c_and_g2g,
#         # r_c_when_c_and_g2g,
#         # r_g2g_when_c_and_g2g,
#         # Compute + C2G
#         f_c_when_c_and_c2g,
#         f_c2g_when_c_and_c2g,
#         # r_c_when_c_and_c2g,
#         # r_c2g_when_c_and_c2g,
#         # Compute + G2C
#         f_c_when_c_and_g2c,
#         f_g2c_when_c_and_g2c,
#         # r_c_when_c_and_g2c,
#         # r_g2c_when_c_and_g2c,
#         # G2G + C2G
#         f_g2g_when_g2g_and_c2g,
#         f_c2g_when_g2g_and_c2g,
#         # r_g2g_when_g2g_and_c2g,
#         # r_c2g_when_g2g_and_c2g,
#         # G2G + G2C
#         f_g2g_when_g2g_and_g2c,
#         f_g2c_when_g2g_and_g2c,
#         # r_g2g_when_g2g_and_g2c,
#         # r_g2c_when_g2g_and_g2c,
#         # C2G + G2C
#         f_c2g_when_c2g_and_g2c,
#         f_g2c_when_c2g_and_g2c,
#         # r_c2g_when_c2g_and_g2c,
#         # r_g2c_when_c2g_and_g2c,
#         # ===============================
#         # G2G + C2G + G2C (no compute)
#         f_g2g_when_no_c,
#         f_c2g_when_no_c,
#         f_g2c_when_no_c,
#         # r_g2g_when_no_c,
#         # r_c2g_when_no_c,
#         # r_g2c_when_no_c,
#         # Compute + C2G + G2C (no G2G)
#         f_c_when_no_g2g,
#         f_c2g_when_no_g2g,
#         f_g2c_when_no_g2g,
#         # r_c_when_no_g2g,
#         # r_c2g_when_no_g2g,
#         # r_g2c_when_no_g2g,
#         # Compute + G2G + G2C (no C2G)
#         f_c_when_no_c2g,
#         f_g2g_when_no_c2g,
#         f_g2c_when_no_c2g,
#         # r_c_when_no_c2g,
#         # r_g2g_when_no_c2g,
#         # r_g2c_when_no_c2g,
#         # Compute + G2G + C2G (no G2C)
#         f_c_when_no_g2c,
#         f_g2g_when_no_g2c,
#         f_c2g_when_no_g2c,
#         # r_c_when_no_g2c,
#         # r_g2g_when_no_g2c,
#         # r_c2g_when_no_g2c,
#         # ===============================
#         # All
#         f_c_when_all,
#         f_g2g_when_all,
#         f_c2g_when_all,
#         f_g2c_when_all,
#         # r_c_when_all,
#         # r_g2g_when_all,
#         # r_c2g_when_all,
#         # r_g2c_when_all,
#     ) = params

#     raw_X = np.vstack((C, G2G, C2G, G2C)).T
#     X = deepcopy(raw_X)
#     ret = np.zeros_like(C)

#     def update(factors, mask, remaining_factors=None):
#         if not isinstance(factors, np.ndarray):
#             factors = np.array(factors)
#         if remaining_factors is not None and not isinstance(
#             remaining_factors, np.ndarray
#         ):
#             remaining_factors = np.array(remaining_factors)
#         if not isinstance(mask, np.ndarray):
#             mask = np.array(mask)

#         is_zeros = np.isclose(X, 0)
#         indices = np.where(np.all(is_zeros == mask, axis=-1))[0]
#         feature_indices = np.where(mask == 0)[0]
#         overlap, remainings = calculate_overlap_and_remain(
#             X[indices], factors, remaining_factors, indices=feature_indices
#         )
#         ret[indices] += overlap
#         # In-place update X[indices][feature_indices]
#         # becaues remaining's last dim is feature_indices
#         # This does not update the original X
#         # X[indices][..., feature_indices] = remainings
#         # This updates the original X
#         X[indices] = remainings
#         # assert np.all(X >= 0), f"X: {X[X < 0]}"

#     # All
#     update(
#         [f_c_when_all, f_g2g_when_all, f_c2g_when_all, f_g2c_when_all],
#         [0, 0, 0, 0],
#         # [r_c_when_all, r_g2g_when_all, r_c2g_when_all, r_g2c_when_all],
#     )
#     # update(
#     #     np.array([f_c_when_all, f_g2g_when_all, f_c2g_when_all, f_c2g_when_all]),
#     #     np.array([r_c_when_all, r_g2g_when_all, r_c2g_when_all, r_c2g_when_all]),
#     #     np.array([0, 0, 0, 0]),
#     # )

#     # ===============================
#     # No Compute
#     update(
#         [1, f_g2g_when_no_c, f_c2g_when_no_c, f_g2c_when_no_c],
#         [1, 0, 0, 0],
#         # [1, r_g2g_when_no_c, r_c2g_when_no_c, r_g2c_when_no_c],
#     )
#     # No G2G
#     update(
#         [f_c_when_no_g2g, 1, f_c2g_when_no_g2g, f_g2c_when_no_g2g],
#         [0, 1, 0, 0],
#         # [r_c_when_no_g2g, 1, r_c2g_when_no_g2g, r_g2c_when_no_g2g],
#     )
#     # No C2G
#     update(
#         [f_c_when_no_c2g, f_g2g_when_no_c2g, 1, f_g2c_when_no_c2g],
#         [0, 0, 1, 0],
#         # [r_c_when_no_c2g, r_g2g_when_no_c2g, 1, r_g2c_when_no_c2g],
#     )
#     # No G2C
#     update(
#         [f_c_when_no_g2c, f_g2g_when_no_g2c, f_c2g_when_no_g2c, 1],
#         [0, 0, 0, 1],
#         # [r_c_when_no_g2c, r_g2g_when_no_g2c, r_c2g_when_no_g2c, 1],
#     )

#     # ===============================
#     # Now only two values are non-zero
#     # Six cases
#     # 1. Compute + G2G
#     update(
#         [f_c_when_c_and_g2g, f_g2g_when_c_and_g2g, 1, 1],
#         [0, 0, 1, 1],
#         # [r_c_when_c_and_g2g, r_g2g_when_c_and_g2g, 1, 1],
#     )
#     # 2. Compute + C2G
#     update(
#         [f_c_when_c_and_c2g, 1, f_c2g_when_c_and_c2g, 1],
#         [0, 1, 0, 1],
#         # [r_c_when_c_and_c2g, 1, r_c2g_when_c_and_c2g, 1],
#     )
#     # 3. Compute + G2C
#     update(
#         [f_c_when_c_and_g2c, 1, 1, f_g2c_when_c_and_g2c],
#         [0, 1, 1, 0],
#         # [r_c_when_c_and_g2c, 1, 1, r_g2c_when_c_and_g2c],
#     )
#     # 4. G2G + C2G
#     update(
#         [1, f_g2g_when_g2g_and_c2g, f_c2g_when_g2g_and_c2g, 1],
#         [1, 0, 1, 0],
#         # [1, r_g2g_when_g2g_and_c2g, r_c2g_when_g2g_and_c2g, 1],
#     )
#     # 5. G2G + G2C
#     update(
#         [1, f_g2g_when_g2g_and_g2c, 1, f_g2c_when_g2g_and_g2c],
#         [1, 0, 0, 1],
#         # [1, r_g2g_when_g2g_and_g2c, 1, r_g2c_when_g2g_and_g2c],
#     )
#     # 6. C2G + G2C
#     update(
#         [1, 1, f_c2g_when_c2g_and_g2c, f_g2c_when_c2g_and_g2c],
#         [1, 1, 0, 0],
#         # [1, 1, r_c2g_when_c2g_and_g2c, r_g2c_when_c2g_and_g2c],
#     )

#     # update(
#     #     np.array([1, f_g2g_when_no_c, f_c2g_when_no_c, f_c2g_when_no_c]),
#     #     np.array([1, r_g2g_when_no_c, r_c2g_when_no_c, r_c2g_when_no_c]),
#     #     np.array([1, 0, 0, 0]),
#     # )

#     # 2. (No G2G): Compute + G2C + C2G
#     # indices_2 = np.all(is_zeros == np.array([0, 1, 0, 0]), axis=-1)
#     # overlap_2, remainings_after_2 = calculate_overlap_and_remain(
#     #     X[indices_2],
#     #     np.array([f_c_when_no_g2g, 1, f_c2g_when_no_g2g, f_c2g_when_no_g2g]),
#     # )
#     # ret[indices_2] += overlap_2
#     # X[indices_2] = remainings_after_2
#     # update(
#     #     np.array([f_c_when_no_g2g, 1, f_c2g_when_no_g2g, f_c2g_when_no_g2g]),
#     #     np.array([r_c_when_no_g2g, 1, r_c2g_when_no_g2g, r_c2g_when_no_g2g]),
#     #     np.array([0, 1, 0, 0]),
#     # )

#     # 3. (No C2G/G2C): Compute + G2G
#     # indices_3 = np.logical_or(
#     #     np.all(is_zeros == np.array([0, 0, 1, 0]), axis=-1),
#     #     np.all(is_zeros == np.array([0, 0, 0, 1]), axis=-1),
#     # )
#     # overlap_3, remainings_after_3 = calculate_overlap_and_remain(
#     #     X[indices_3],
#     #     np.array(
#     #         [f_c_when_no_g2c, f_g2g_when_no_g2c, f_c2g_when_no_g2c, f_c2g_when_no_g2c]
#     #     ),
#     # )
#     # ret[indices_3] += overlap_3
#     # X[indices_3] = remainings_after_3
#     # update(
#     #     np.array(
#     #         [f_c_when_no_g2c, f_g2g_when_no_g2c, f_c2g_when_no_g2c, f_c2g_when_no_g2c]
#     #     ),
#     #     np.array(
#     #         [r_c_when_no_g2c, r_g2g_when_no_g2c, r_c2g_when_no_g2c, r_c2g_when_no_g2c]
#     #     ),
#     #     np.array([0, 0, 1, 0]),
#     #     np.array([0, 0, 0, 1]),
#     # )

#     # ===============================
#     # Now only two values are non-zero
#     # We have four cases
#     # 1. Compute + C2G/G2C
#     # is_zeros = np.isclose(X, 0)
#     # indices_1 = np.logical_or(
#     #     np.all(is_zeros == np.array([0, 1, 1, 0]), axis=-1),
#     #     np.all(is_zeros == np.array([0, 1, 0, 1]), axis=-1),
#     # )
#     # overlap_1, remainings_after_1 = calculate_overlap_and_remain(
#     #     X[indices_1],
#     #     np.array([f_c_when_c_and_c2g, 1, f_c2g_when_c_and_c2g, f_c2g_when_c_and_c2g]),
#     # )
#     # ret[indices_1] += overlap_1
#     # X[indices_1] = remainings_after_1
#     # update(
#     #     np.array([f_c_when_c_and_c2g, 1, f_c2g_when_c_and_c2g, f_c2g_when_c_and_c2g]),
#     #     np.array([r_c_when_c_and_c2g, 1, r_c2g_when_c_and_c2g, r_c2g_when_c_and_c2g]),
#     #     np.array([0, 1, 1, 0]),
#     #     np.array([0, 1, 0, 1]),
#     # )

#     # # 2. Compute + G2G
#     # update(
#     #     np.array([f_c_when_c_and_g2g, f_g2g_when_c_and_g2g, 1, 1]),
#     #     np.array([r_c_when_c_and_g2g, r_g2g_when_c_and_g2g, 1, 1]),
#     #     np.array([0, 0, 1, 1]),
#     # )

#     # # 3. G2G + C2G/G2C
#     # update(
#     #     np.array(
#     #         [1, f_g2g_when_g2g_and_c2g, f_c2g_when_g2g_and_c2g, f_c2g_when_g2g_and_c2g]
#     #     ),
#     #     np.array([1, r_g2g_when_g2g_and_c2g, r_c2g_when_g2g_and_c2g, r_c2g_when_g2g_and_c2g]),
#     #     np.array([1, 0, 1, 0]),
#     #     np.array([1, 0, 0, 1]),
#     # )

#     # # 4. C2G + G2C
#     # update(
#     #     np.array([1, 1, f_c2g_when_c2g_and_g2c, f_c2g_when_c2g_and_g2c]),
#     #     np.array([1, 1, r_c2g_when_c2g_and_g2c, r_c2g_when_c2g_and_g2c]),
#     #     np.array([1, 1, 0, 0]),
#     # )

#     # Now we only have one value non-zero
#     ret += np.sum(X, axis=-1)

#     assert not np.allclose(ret, 0), f"Ret: {ret}"

#     return ret


# def interference_fitting_deprecated(
#     L: np.ndarray,
#     C: np.ndarray,
#     G2G: np.ndarray,
#     C2G: np.ndarray,
#     G2C: np.ndarray,
# ) -> Any:
#     def objective_function(params):
#         # if np.any(params < 0):
#         #     return np.inf
#         pred = interference_estimate(C, G2G, C2G, G2C, params)
#         # All params should be positive
#         return np.sum((L - pred) ** 2)

#     # initial_guess = [1.0 for _ in range(56)]
#     initial_guess = [
#         # Compute + G2G
#         2.0,
#         2.0,
#         # Compute + C2G
#         2.0,
#         2.0,
#         # Compute + G2C
#         2.0,
#         2.0,
#         # G2G + C2G
#         2.0,
#         2.0,
#         # G2G + G2C
#         2.0,
#         2.0,
#         # C2G + G2C
#         2.0,
#         2.0,
#         # ===============================
#         # G2G + C2G + G2C (no compute)
#         3.0,
#         3.0,
#         3.0,
#         # Compute + C2G + G2C (no G2G)
#         3.0,
#         3.0,
#         3.0,
#         # Compute + G2G + G2C (no C2G)
#         3.0,
#         3.0,
#         3.0,
#         # Compute + G2G + C2G (no G2C)
#         3.0,
#         3.0,
#         3.0,
#         # ===============================
#         # All
#         4.0,
#         4.0,
#         4.0,
#         4.0,
#     ]

#     bounds = [(1.0, UP_BOUND) for _ in range(len(initial_guess))]

#     result = minimize(objective_function, initial_guess, bounds=bounds)

#     if not result.success:
#         raise ValueError(f"Optimization failed: {result.message}")

#     else:
#         print(f"Best params: {list(result.x)}")

#     # Calculate the average error ratio
#     pred = interference_estimate(C, G2G, C2G, G2C, result.x)
#     error = np.abs(L - pred)
#     error_ratio = error / L
#     print(f"Average error ratio: {np.mean(error_ratio)}")
#     for i, error_ratio_ in enumerate(error_ratio):
#         if error_ratio_ > 0.1:
#             print(
#                 f"Index: {i:03d}, Error ratio: {error_ratio_:.4f} L: {L[i]:.6f}, Pred: {pred[i]:.6f}, C: {C[i]:.6f}, G2G: {G2G[i]:.6f}, C2G: {C2G[i]:.6f}, G2C: {G2C[i]:.6f}"
#             )

# return result.x


THREE_CASE_MASKS = [
    [True, True, True, False],
    [True, True, False, True],
    [True, False, True, True],
    [False, True, True, True],
]
TWO_CASE_MASKS = [
    [True, True, False, False],
    [True, False, True, False],
    [True, False, False, True],
    [False, True, True, False],
    [False, True, False, True],
    [False, False, True, True],
]


def interference_estimate(
    C: np.ndarray,
    G2G: np.ndarray,
    C2G: np.ndarray,
    G2C: np.ndarray,
    INTRA: np.ndarray,
    params: Union[List[float], np.ndarray],
    check_data_conprehension: bool = False,
):
    if isinstance(C, Number):
        C = np.array([C]).astype(float)
    if isinstance(G2G, Number):
        G2G = np.array([G2G]).astype(float)
    if isinstance(C2G, Number):
        C2G = np.array([C2G]).astype(float)
    if isinstance(G2C, Number):
        G2C = np.array([G2C]).astype(float)
    if isinstance(INTRA, Number):
        INTRA = np.array([INTRA]).astype(int)

    if not len(params) == 28 * 4:
        raise ValueError(f"Length: {len(params)}. It should be 28 * 4.")

    output = np.zeros_like(C, dtype=float)
    visited = np.zeros_like(C, dtype=bool)
    for i, intra_size in enumerate([1, 2, 4, 8]):
        indices = np.where(INTRA == intra_size)[0]
        curr_params = params[i * 28 : (i + 1) * 28]
        output[indices] = interference_estimate_for_one_group(
            C[indices],
            G2G[indices],
            C2G[indices],
            G2C[indices],
            curr_params,
            check_data_conprehension=check_data_conprehension,
        )
        # Update the visited
        visited[indices] = True

    # Make sure all data points are visited
    assert np.all(visited), f"Visited: {visited}"

    return output


def interference_estimate_for_one_group(
    C: np.ndarray,
    G2G: np.ndarray,
    C2G: np.ndarray,
    G2C: np.ndarray,
    params: Union[List[float], np.ndarray],
    check_data_conprehension: bool = False,
):
    if isinstance(C, Number):
        C = np.array([C]).astype(float)
    if isinstance(G2G, Number):
        G2G = np.array([G2G]).astype(float)
    if isinstance(C2G, Number):
        C2G = np.array([C2G]).astype(float)
    if isinstance(G2C, Number):
        G2C = np.array([G2C]).astype(float)

    if not len(params) == 28:
        raise ValueError(f"Length: {len(params)}. It should be 28.")

    four_cases_params = params[:4]
    three_cases_params = params[4:16]
    two_cases_params = params[16:28]

    raw_X = np.vstack((C, G2G, C2G, G2C)).T
    X = deepcopy(raw_X)
    output = np.zeros_like(C, dtype=float)

    # Four cases
    _update(
        X,
        output,
        mask=[True, True, True, True],
        factors=four_cases_params,
        remaining_factors=None,
        check_data_conprehension=check_data_conprehension,
    )

    # Three cases
    for i, mask in enumerate(THREE_CASE_MASKS):
        factors = np.ones(4, dtype=float)
        factors[mask] = three_cases_params[i * 3 : (i + 1) * 3]
        _update(
            X,
            output,
            mask=mask,
            factors=factors,
            remaining_factors=None,
            check_data_conprehension=check_data_conprehension,
        )

    # Two cases
    for i, mask in enumerate(TWO_CASE_MASKS):
        factors = np.ones(4, dtype=float)
        factors[mask] = two_cases_params[i * 2 : (i + 1) * 2]
        _update(
            X,
            output,
            mask=mask,
            factors=factors,
            remaining_factors=None,
            check_data_conprehension=check_data_conprehension,
        )

    # Now only one value is non-zero
    output += np.sum(X, axis=-1)

    return output


def _update(
    X: np.ndarray,
    output: np.ndarray,
    mask: np.ndarray,
    factors: np.ndarray,
    remaining_factors: np.ndarray = None,
    check_data_conprehension: bool = False,
):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if not isinstance(factors, np.ndarray):
        factors = np.array(factors)
    if not factors.shape[-1] == 4:
        raise ValueError(f"Factors: {factors.shape}")
    if remaining_factors is not None and not isinstance(remaining_factors, np.ndarray):
        remaining_factors = np.array(remaining_factors)

    # Calculate the indices
    is_non_zero = np.logical_not(np.isclose(X, 0))
    indices = np.where(np.all(is_non_zero == mask, axis=-1))[0]
    feature_indices = np.where(mask == 1)[0]

    # Calculate the overlap and remaining values
    overlap, remainings = calculate_overlap_and_remain(
        X[indices],
        factors,
        remaining_factors=remaining_factors,
        feature_indices=feature_indices,
    )
    # Update the output
    output[indices] += overlap.astype(float)
    X[indices] = remainings.astype(float)

    if check_data_conprehension:
        # Make sure for all feature indices, at least one of the remaining values is zero
        for i in feature_indices:
            no_zero = np.all(np.isclose(remainings[..., i], 0))
            if no_zero:
                print(
                    f"Warning: No zero for feature index: {i}. This means it is never a smallest value."
                )


def _construct_params(
    compute: bool,
    g2g: bool,
    c2g: bool,
    g2c: bool,
    partial_params: List[float],
    default_two_cases_params: Optional[List[float]] = None,
    default_three_cases_params: Optional[List[float]] = None,
    default_four_cases_params: Optional[List[float]] = None,
):
    mask = np.array([compute, g2g, c2g, g2c])
    num_overlaps = np.sum(mask)
    assert len(partial_params) == num_overlaps, f"Length: {len(partial_params)}"

    params = np.ones(28, dtype=float)

    if default_four_cases_params is not None:
        params[:4] = default_four_cases_params
    if default_three_cases_params is not None:
        params[4:16] = default_three_cases_params
    if default_two_cases_params is not None:
        params[16:28] = default_two_cases_params

    if num_overlaps == 4:
        params[:4] = partial_params
    elif num_overlaps == 3:
        idx = THREE_CASE_MASKS.index(mask.tolist())
        start_idx = 4 + idx * 3
        params[start_idx : start_idx + 3] = partial_params
    elif num_overlaps == 2:
        idx = TWO_CASE_MASKS.index(mask.tolist())
        start_idx = 16 + idx * 2
        params[start_idx : start_idx + 2] = partial_params

    return params


def interference_fitting_step_by_step(data: List[np.ndarray]):
    def fit(
        compute: bool,
        g2g: bool,
        c2g: bool,
        g2c: bool,
        default_two_cases_params=None,
        default_three_cases_params=None,
        default_four_cases_params=None,
    ):
        # Construct the mask
        mask = np.array([compute, g2g, c2g, g2c])
        num_overlaps = np.sum(mask)
        # Select the data
        L, C, G2G, C2G, G2C = select_data_with_different_combination(
            data, compute, g2g, c2g, g2c
        )
        print(f"[{mask.tolist()}] Number of data points: {len(L)}")
        if len(L) == 0:
            print(f"WARNING: No data points for [{mask.tolist()}]")
            return [num_overlaps for _ in range(num_overlaps)]

        def estimate(params, **kwargs):
            full_params = _construct_params(
                compute,
                g2g,
                c2g,
                g2c,
                params,
                default_two_cases_params=default_two_cases_params,
                default_three_cases_params=default_three_cases_params,
                default_four_cases_params=default_four_cases_params,
            )
            output = interference_estimate_for_one_group(
                C, G2G, C2G, G2C, full_params, **kwargs
            )
            return output

        def objective_function(params):
            assert len(params) == num_overlaps, f"Length: {len(params)}"
            predicted = estimate(params)
            return np.sum((L - predicted) ** 2)

        init_guess = [float(num_overlaps) for _ in range(num_overlaps)]
        bounds = [(1.0, UP_BOUND) for _ in range(num_overlaps)]
        result = minimize(
            objective_function, init_guess, bounds=bounds  # , method="L-BFGS-B"
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Calculate the average error ratio
        pred = estimate(result.x, check_data_conprehension=False)
        error = np.abs(L - pred)
        error_ratio = error / L
        print(
            f"[{mask.tolist()}] Best Params: {[round(p, 4) for p in result.x]}, "
            f"Average error ratio: {np.mean(error_ratio) * 100:.2f} %"
        )
        return result.x

    two_case_params = []
    for i, mask in enumerate(TWO_CASE_MASKS):
        two_case_params.extend(fit(*mask))
    print("#" * 90)

    three_case_params = []
    for i, mask in enumerate(THREE_CASE_MASKS):
        three_case_params.extend(fit(*mask))
    print("#" * 90)

    four_case_params = fit(True, True, True, True)
    print("#" * 90)

    # Calculate the average error ratio for all
    full_params = [
        *four_case_params,
        *three_case_params,
        *two_case_params,
    ]
    L, C, G2G, C2G, G2C, INTRA, INTER = data
    full_pred = interference_estimate_for_one_group(C, G2G, C2G, G2C, full_params)
    full_error = np.abs(L - full_pred)
    full_error_ratio = full_error / L
    print(f"Average error ratio: {np.mean(full_error_ratio) * 100:.2f} %")
    return full_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()

    raw_data = preprocess_data(args)
    print("#" * 90)
    g2g_params = fit_gpu_to_gpu_only_params(raw_data)
    print(f"GPU to GPU params: {[round(p, 4) for p in g2g_params]}")
    print("#" * 90)
    c2g_params, g2c_params = fit_cpu_to_gpu_only_params(raw_data)
    print(f"CPU to GPU params: {[round(p, 4) for p in c2g_params.tolist()]}")
    print(f"GPU to CPU params: {[round(p, 4) for p in g2c_params.tolist()]}")
    print("#" * 90)

    interference_data = preprocess_interference_fitting_data(
        raw_data,
        g2g_params,
        c2g_params,
        g2c_params,
    )
    params = interference_fitting_step_by_step(interference_data)

    # all_interference_params = []
    # all_interference_params.extend(
    #     [1.0 for _ in range(28)]
    # )  # For the case when all are zero
    # for intra_size in [2, 4, 8]:
    #     print("#" * 37, f"Intra size: {intra_size}", "#" * 38)
    #     interference_data = preprocess_interference_fitting_data(
    #         raw_data, intra_g2g_params, inter_g2g_params, c2g_params, g2c_params, intra_size=intra_size
    #     )
    #     params = interference_fitting_step_by_step(interference_data)
    #     all_interference_params.extend(params)
    # assert len(all_interference_params) == 28 * 4, f"Length: {len(all_interference_params)}"

    # all_interference_params[:28] = all_interference_params[28:56]

    # Calculate the average error ratio for all
    L, B, C, G2G, C2G, G2C, INTRA, INTER = raw_data
    G2G_L = predict_gpu_gpu_comm_latency(
        op_name="all_gather",
        gbytes=G2G,
        intra_size=INTRA,
        inter_size=INTER,
        gpu_gpu_comm_params=g2g_params,
    )
    C2G_L = cpu_to_gpu_only_latency_model(C2G, None, c2g_params)
    G2C_L = cpu_to_gpu_only_latency_model(G2C, None, g2c_params)
    # full_pred = interference_estimate_(C, G2G_L, C2G_L, G2C_L, INTRA, all_interference_params)
    full_pred = interference_estimate_for_one_group(C, G2G_L, C2G_L, G2C_L, params)
    full_error = np.abs(L - full_pred)
    full_error_ratio = full_error / L
    print("#" * 90)
    print(f"Average error ratio: {np.mean(full_error_ratio) * 100:.2f} %")

    # # Format the params
    # params = list(params)
    # params = [round(p, 4) for p in params]

    # print(f"Best params: {params}")
    # print(f"Best params for [True, True, True, Ture]:\t {params[:4]}")
    # for i, mask in enumerate(THREE_CASE_MASKS):
    #     start_idx = 4 + i * 3
    #     print(f"Best params for {mask}:\t {params[start_idx:start_idx+3]}")
    # for i, mask in enumerate(TWO_CASE_MASKS):
    #     start_idx = 16 + i * 2
    #     print(f"Best params for {mask}:\t {params[start_idx:start_idx+2]}")

    print(f"#" * 90)
    print(f"GPU to GPU params: {[round(p, 4) for p in g2g_params]}")
    print(f"CPU to GPU params: {[round(p, 4) for p in c2g_params.tolist()]}")
    print(f"GPU to CPU params: {[round(p, 4) for p in g2c_params.tolist()]}")
    # print(f"Interference params: {[round(p, 4) for p in all_interference_params]}")
    print(f"Inference params: {[round(p, 4) for p in params]}")
