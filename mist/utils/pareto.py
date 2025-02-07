"""Sample pareto frontier efficiently"""

from typing import Tuple, Optional
import numpy as np
from numba import jit, njit, prange


def sample_pareto_frontier(
    costs_x: np.ndarray,
    costs_y: np.ndarray,
    sample_size: int = 9,
    alpha_based_sample_size: Optional[int] = None,
    max_costs_x_scale: float = 3.0,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample pareto frontier efficiently. This methods first optimizes costs_x as a basis,
    and then choose the cost_x from [min(costs_x), min(costs_x) * max_costs_x_scale] uniformly or normally.
    The cost_y is selected from the pareto frontier of the costs_x.

    Parameters
    ----------
    costs_x : np.ndarray
        The cost of x-axis. The shape is (batch_size, feature_size).
    costs_y : np.ndarray
        The cost of y-axis. The shape is (batch_size, feature_size).
    sample_size : int
        The number of samples to be selected.
    max_costs_x_scale : float, optional
        The scale of the maximum cost of x-axis, by default 2.0.
    distribution : str, optional
        The distribution of the samples, by default "normal". The options are ["normal", "uniform"].
    mask : Optional[np.ndarray], optional
        The mask to be applied to the pareto frontier, by default None. The shape is (batch_size, feature_size).

    Returns
    -------
    indices : np.ndarray
        The indices of the selected samples. The shape is (batch_size, sample_size).
    selected_costs_x : np.ndarray
        The selected costs_x. The shape is (batch_size, sample_size).
    selected_costs_y : np.ndarray
        The selected costs_y. The shape is (batch_size, sample_size).
    """
    alpha_based_sample_size = alpha_based_sample_size or sample_size
    if sample_size < alpha_based_sample_size:
        raise ValueError(f"sample_size should be larger or equal to alpha_based_sample_size. Got {sample_size} and {alpha_based_sample_size}.")

    if mask is None:
        mask = np.ones_like(costs_x, dtype=bool)

    # Get the default results from alpha-based optimization
    batch_size, feature_size = costs_x.shape
    return_indices = np.full((batch_size, sample_size), -1, dtype=np.int64)
    return_indices[:, :alpha_based_sample_size] = optimize_bi_obj_alpha_based(
        costs_x, costs_y, alpha_array=np.linspace(0, 1, alpha_based_sample_size + 2)[1:-1], mask=mask
    )

    if sample_size > alpha_based_sample_size:
        # Get the costs_x range
        cost_x_lower_bound = np.min(
            costs_x + np.where(mask, 0, np.inf), axis=-1, keepdims=True
        )
        cost_x_upper_bound = cost_x_lower_bound * max_costs_x_scale
        cost_x_candidates = np.linspace(
            cost_x_lower_bound, cost_x_upper_bound, sample_size - alpha_based_sample_size
        )

        # Get the minimum cost_y for each cost_x
        for i, cost_x in enumerate(cost_x_candidates):
            mask_x = np.logical_and.reduce([costs_x >= cost_x, mask])
            return_indices[:, i + alpha_based_sample_size] = optimize_bi_obj_alpha_based(
                costs_x, costs_y, alpha=0.0, mask=mask_x
            )

    # Get the selected costs_x and costs_y
    selected_costs_x = costs_x[np.arange(batch_size)[:, None], return_indices]
    selected_costs_y = costs_y[np.arange(batch_size)[:, None], return_indices]

    # Double check the selected indices are with mask == True
    all_mask_false = np.all(~mask, axis=-1)
    return_indices_valid = np.all(mask[np.arange(batch_size)[:, None], return_indices], axis=-1)
    assert np.all(np.logical_or(all_mask_false, return_indices_valid)), f"{all_mask_false=}, {return_indices_valid=}."

    return return_indices, selected_costs_x, selected_costs_y

def optimize_bi_obj_alpha_based_impl_deprecated_1(
    costs_x: np.ndarray,
    costs_y: np.ndarray,
    alpha_array: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Optimize bi-objective function based on alpha. costs = alpha * costs_x + (1 - alpha) * costs_y.

    Parameters
    ----------
    costs_x : np.ndarray
        The cost of x-axis. The shape is (batch_size, feature_size).
    costs_y : np.ndarray
        The cost of y-axis. The shape is (batch_size, feature_size).
    alpha_array : np.ndarray
        The alpha array to be optimized. The shape is (sample_size,).
    mask : Optional[np.ndarray], optional
        The mask to be applied to the pareto frontier, by default None. The shape is (batch_size, feature_size).

    Returns
    -------
    output_indices : np.ndarray
        The indices of the selected samples. The shape is (batch_size, sample_size).
    """
    penalty = np.where(mask, 0, np.inf)
    costs_y_with_penalty = costs_y + penalty
    results = []
    for alpha in alpha_array:
        costs = alpha / (1 - alpha) * costs_x + costs_y_with_penalty 
        indices = np.argmin(costs, axis=-1)
        results.append(indices)

    output_indices = np.stack(results, axis=1)

    return output_indices

@njit(cache=True, parallel=True, fastmath=True)
def compute_indices(costs_x, costs_y, mask, alpha_array):
    batch_size, feature_size = costs_x.shape
    sample_size = alpha_array.shape[0]
    output_indices = np.empty((batch_size, sample_size), dtype=np.int32)
    
    for i in prange(sample_size):
        alpha = alpha_array[i]
        # Calculate cost only for unmasked elements
        for j in prange(batch_size):
            min_index = -1
            min_cost = np.inf
            for k in range(feature_size):
                if mask[j, k]:  # Only compute where mask is True
                    cost = alpha / (1 - alpha) * costs_x[j, k] + costs_y[j, k]
                    if cost < min_cost:
                        min_cost = cost
                        min_index = k
            output_indices[j, i] = min_index

    return output_indices

def optimize_bi_obj_alpha_based(
    costs_x: np.ndarray,
    costs_y: np.ndarray,
    alpha_array: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Optimize bi-objective function based on alpha. costs = alpha * costs_x + (1 - alpha) * costs_y, considering mask.
    """
    output_indices = compute_indices(costs_x, costs_y, mask, alpha_array)
    return output_indices

def fill_redundant_samples(
    costs_x: np.ndarray,
    costs_y: np.ndarray,
    selected_indices: np.ndarray,
    fill_value: float = np.inf,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill the redundant samples in the selected indices. Within a batch,
    if multiple (cost_x, cost_y) pairs are the same, only the first one 
    is selected. Update the selected indices as well.

    Parameters
    ----------
    costs_x : np.ndarray
        The cost of x-axis. The shape is (batch_size, sample_size).
    costs_y : np.ndarray
        The cost of y-axis. The shape is (batch_size, sample_size).
    selected_indices : np.ndarray
        The selected indices. The shape is (batch_size, sample_size).
    fill_value : float, optional
        The value to be filled, by default np.inf.
    
    Returns
    -------
    policy_indices : List[List]
        The selected unique indices. The shape is (batch_size,).
    updated_costs_x : np.ndarray
        The selected costs_x. The shape is (batch_size, sample_size).
    updated_costs_y : np.ndarray
        The selected costs_y. The shape is (batch_size, sample_size).
    updated_indices : np.ndarray
        The updated selected indices. The shape is (batch_size, sample_size).
    """
    if not costs_x.ndim == costs_y.ndim == selected_indices.ndim == 2:
        raise ValueError(f"The input arrays should be 2D. Got {costs_x.ndim}, {costs_y.ndim}, and {selected_indices.ndim}.")

    batch_size, sample_size = selected_indices.shape
    updated_costs_x = costs_x.copy()
    updated_costs_y = costs_y.copy()
    updated_indices = selected_indices.copy()
    unique_indices = []

    for i in range(batch_size):
        stacked_costs = np.stack([costs_x[i], costs_y[i]], axis=-1).reshape(-1, 2)
        _, curr_unique_indices = np.unique(stacked_costs, axis=0, return_index=True)

        # Determine indicies to fill based on the unique indices
        all_indices = np.arange(sample_size)
        fill_indices = np.setdiff1d(all_indices, curr_unique_indices)

        # Update the costs and indices
        updated_costs_x[i, fill_indices] = fill_value
        updated_costs_y[i, fill_indices] = fill_value
        updated_indices[i, fill_indices] = -1
        unique_indices.append(curr_unique_indices.tolist())

    return unique_indices, updated_costs_x, updated_costs_y, updated_indices


if __name__ == "__main__":
    n, m = 64, 80000

    latency_stable = np.random.randint(0, 100, (n, m))
    latency_delta = np.random.randint(0, 100, (n, m))
    mask = True

    results = sample_pareto_frontier(
        latency_stable, latency_delta, sample_size=30, max_costs_x_scale=10.0, mask=mask
    )

    print(f"results: {results}")
