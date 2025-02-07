from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator, Set
from collections import OrderedDict
from copy import deepcopy, copy
from dataclasses import dataclass, field

import sympy as sp
import numpy as np
import torch
from torch import nn, fx
from torch.fx import Interpreter, GraphModule, config
from torch.fx.node import Argument, Node, Target
from torch.hub import tqdm

from mist.node_database.inputs_outputs_spec import TensorSpec
from mist.logger import get_logger
from mist.utils.tensor_entry import tree_to_entries, TensorEntry


class MemoryPool:
    """
    This is actually a list of MemoryEntry.
    """

    def __init__(
        self,
        pool: Dict[str, Dict[TensorEntry, int]] = None,
    ):
        # A memo recording the mapping of base id to the base tensor
        self.memo: Dict[int, TensorEntry] = {}

        if pool is not None:
            self.pool = pool
        else:
            self.pool = {}
            self.pool["params_require_grad"] = OrderedDict()
            self.pool["params_require_no_grad"] = OrderedDict()
            self.pool["buffers"] = OrderedDict()
            self.pool["param_grads"] = OrderedDict()
            self.pool["optimizer_states"] = OrderedDict()
            self.pool["intermediate"] = OrderedDict()
            self.pool["saved_tensors"] = OrderedDict()

    def add(
        self,
        tensor,
        category: str,
        comment: str = "",
    ):
        assert category in self.pool
        if isinstance(tensor, torch.Tensor):
            entry = TensorEntry.from_tensor(tensor, comment=comment)
            # Record the base tensor if it is not in the memo
            # Otherwise, use the base tensor in the memo instead
            if entry.base_id in self.memo:
                entry = self.memo[entry.base_id]
            else:
                self.memo[entry.base_id] = entry
            # Add the base tensor to the pool
            if entry in self.pool[category]:
                self.pool[category][entry] += 1
            else:
                self.pool[category][entry] = 1

    def batch_add(self, *tensors, category: str, comment: str = ""):
        for tensor in tensors:
            assert not isinstance(tensor, (list, tuple)), f"Plase check {tensors}"
            self.add(tensor, category, comment=comment)

    def remove(self, tensor, category: str):
        assert category in self.pool
        if isinstance(tensor, (torch.Tensor, TensorSpec)):
            entry = TensorEntry.from_tensor(tensor)
            entry = self.memo[entry.base_id]  # Use the base tensor in the memo
            assert entry in self.pool[category], f"{entry} not in {category}"
            if entry in self.pool[category]:
                self.pool[category][entry] -= 1
                if self.pool[category][entry] == 0:
                    del self.pool[category][entry]

    def batch_remove(self, *tensors, category: str):
        for tensor in tensors:
            assert not isinstance(tensor, (list, tuple)), f"Plase check {tensors}"
            self.remove(tensor, category)

    def get_category(self, category):
        assert category in self.pool
        return self.pool[category]

    def print(self, header: str = None):
        if header is None:
            print(header)
        for entry in self.pool:
            print(f"==> {entry}")

    def copy(self) -> MemoryPool:
        pool = deepcopy(self.pool)
        return MemoryPool(pool)

    # def summary(self):
    #     memo: Set[MemoryEntry] = set()
    #     summary: Dict[str, Dict[str, Tuple[MemoryEntry, int]]] = OrderedDict()
    #     total = 0
    #     for category, entries in self.pool.items():
    #         summary[category] = OrderedDict()
    #         category_total = 0
    #         for entry in entries:
    #             if entry in memo:
    #                 continue
    #             memo.add(entry)
    #             summary[category][entry.comment] = (entry, entry.nbytes())
    #             category_total += entry.nbytes()
    #         summary[category]["total"] = category_total
    #         total += category_total
    #     summary["total"] = total
    #     return summary

    # def get_flattened(self, categories: List[str] = None):
    #     if categories is None:
    #         categories = list(self.pool.keys())
    #     flattened = set()
    #     for category in categories:
    #         for entry in self.pool[category]:
    #             flattened.add(entry)
    #     return flattened


# def remove_weights_in_saved_tensors_in_pool(memory_pool, module):
#     params_and_buffers = []
#     params_and_buffers.extend([MemoryEntry.from_tensor(p) for p in module.parameters()])
#     params_and_buffers.extend([MemoryEntry.from_tensor(p) for p in module.buffers()])
#     saved_tensors = memory_pool.get_category("saved_tensors")
#     for entry in params_and_buffers:
#         if entry in saved_tensors:
#             del saved_tensors[entry]


# def remove_weights_in_saved_tensors(saved_tensors, module):
#     if not isinstance(saved_tensors, (tuple, list)):
#         raise TypeError(
#             "saved_tensors should be a list or tuple, but got {}".format(
#                 type(saved_tensors)
#             )
#         )

#     new_saved_tensors = []
#     params_and_buffers = set()
#     params_and_buffers |= set(module.parameters())
#     params_and_buffers |= set(module.buffers())
#     for saved_tensor in saved_tensors:
#         if saved_tensor in params_and_buffers:
#             continue
#         new_saved_tensors.append(saved_tensor)

#     return new_saved_tensors


# def peak_memory_among_different_pools(
#     pools: List[MemoryPool],
#     categories: List[str] = None,
#     outer_memory_entries: Set[MemoryEntry] = None,
# ):
#     if categories is None:
#         categories = list(pools[0].pool.keys())
#     if outer_memory_entries is None:
#         outer_memory_entries = set()
#     flattened = [
#         pool.get_flattened(categories) | outer_memory_entries for pool in pools
#     ]
#     nbytes = [sum([entry.nbytes() for entry in pool]) for pool in flattened]
#     # Compare to get the peak memory
#     peak_memory = -1
#     for i in range(len(nbytes)):
#         peak_memory = sp.Max(peak_memory, nbytes[i])

#     return peak_memory


# def compute_memory_for_flattened(flattened):
#     return sum([entry.nbytes() for entry in flattened])


def nbytes(tensor):
    if not hasattr(tensor, "shape") and not hasattr(tensor, "dtype"):
        return 0
    return np.prod(tensor.shape) * torch.empty([], dtype=tensor.dtype).element_size()


def compute_memory_for_set(_set):
    if _set is None:
        return 0
    if not isinstance(_set, set):
        _set = set(_set)
    return sum(nbytes(entry) for entry in _set)


class SavedTensorsManager:
    def __init__(self):
        self.saved_tensors: List[torch.Tensor] = []
        self.recording: bool = False

    def start_recording(self, clear=True):
        self.recording = True
        if clear:
            self.clear()

    def stop_recording(self):
        self.recording = False

    def clear(self):
        self.saved_tensors = []

    def is_empty(self):
        return len(self.saved_tensors) == 0

    def pack_hook(self, tensor):
        return tensor

    def unpack_hook(self, tensor):
        if self.recording:
            if isinstance(tensor, torch.Tensor):
                self.saved_tensors.append(tensor)
        return tensor


saved_tensors_manager = stm = SavedTensorsManager()
