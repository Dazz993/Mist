# Copyright (c) 2023, Tri Dao.

from typing import Optional, Callable, Union, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


import mist.modules.activations as activations
from mist.modules.fused_dense import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class ParallelMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        activation: Optional[Callable] = F.gelu,
        bias1: bool = True,
        bias2: bool = True,
        sequence_parallel: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        self.hidden_features = (
            hidden_features if hidden_features is not None else in_features * 4
        )
        self.process_group = process_group
        self.fc1 = ColumnParallelLinear(
            self.in_features,
            self.hidden_features,
            process_group=process_group,
            bias=bias1,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.activation = activation
        self.fc2 = RowParallelLinear(
            self.hidden_features,
            self.out_features,
            process_group=process_group,
            bias=bias2,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y


class ParallelGatedMlp(nn.Module):
    """Parallel GatedMlp"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        process_group: Optional[ProcessGroup] = None,
        activation: Optional[Callable] = F.sigmoid,
        bias1: bool = True,
        bias2: bool = True,
        multiple_of: int = 256,
        sequence_parallel: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        self.hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        self.process_group = process_group
        self.fc1 = ColumnParallelLinear(
            self.in_features,
            2 * self.hidden_features,
            process_group=process_group,
            bias=bias1,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.activation = activation
        self.fc2 = RowParallelLinear(
            self.hidden_features,
            self.out_features,
            process_group=process_group,
            bias=bias2,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        elif (
            self.activation == F.silu and activations.swiglu is not None
        ):  # Special case for SwiGLU
            y, gate = y.chunk(2, dim=-1)
            y = activations.swiglu(gate, y)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y


if __name__ == "__main__":
    import torch.distributed as dist
    from mist.overrides.base import reset_mist_patcher

    reset_mist_patcher()

    try:
        # Test with parallelism
        dist.init_process_group(backend="gloo")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    except:
        local_rank = 0
        world_size = 1

    device = torch.device("cuda")
    dtype = torch.float16
    factory_kwargs = {"device": device, "dtype": dtype}

    if local_rank == 0:
        mlp = ParallelMLP(10, 20, 30, **factory_kwargs)
        x = torch.randn(2, 10, device=device, dtype=dtype)
        y = mlp(x)
        print(x)
        print(y)
        print(y.shape)

        gated_mlp = ParallelGatedMlp(10, 20, 30, **factory_kwargs)
        y = gated_mlp(x)
        print(y)
        print(y.shape)

    if world_size != 1:
        dist.barrier()
        x = torch.randn(2, 10, device=device, dtype=dtype)
        print(x)
        # dist.all_reduce(x, group=dist.group.WORLD)
        mlp = ParallelMLP(10, 20, 30, process_group=dist.group.WORLD, **factory_kwargs)
        y = mlp(x)
        print(y)
        print(y.shape)

        gated_mlp = ParallelGatedMlp(
            10,
            20,
            30,
            process_group=dist.group.WORLD,
            activation=F.silu,
            **factory_kwargs,
        )
        y = gated_mlp(x)
        print(y)
        print(y.shape)
