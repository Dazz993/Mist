from typing import Optional

import torch
import torch.distributed
import torch.fx
import torch.fx.proxy

from mist.overrides import register_overriden_func, get_ori_torch_op
from mist.logger import get_logger


class MistProcessGroup(metaclass=torch.fx.ProxyableClassMeta):
    def __init__(self, world_size, rank, global_rank=None):
        self._world_size = world_size
        self._rank = rank
        self._global_rank = global_rank or rank
        self._is_mist_process_group = True

    def rank(self):
        return self._rank

    def size(self):
        return self._world_size

    def __repr__(self) -> str:
        return f"MistProcessGroup(world_size={self._world_size}, rank={self._rank})"


class MistDistHandle:
    def wait(self):
        pass


@register_overriden_func(torch.distributed, "get_rank")
def get_rank(group=None) -> int:
    if isinstance(group, MistProcessGroup):
        return group._rank
    else:
        return get_ori_torch_op(torch.distributed, "get_rank")(group)


@register_overriden_func(torch.distributed, "get_global_rank")
def get_global_rank(group, group_rank) -> int:
    if isinstance(group, MistProcessGroup):
        return group._global_rank
    else:
        return get_ori_torch_op(torch.distributed, "get_global_rank")(group, group_rank)


@register_overriden_func(torch.distributed, "get_world_size")
def get_world_size(group=None) -> int:
    if isinstance(group, MistProcessGroup):
        return group._world_size
    else:
        return get_ori_torch_op(torch.distributed, "get_world_size")(group)


@register_overriden_func(torch.distributed, "all_gather_into_tensor")
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if isinstance(group, MistProcessGroup):
        output_tensor.data = output_tensor.data.clone()
        return MistDistHandle()
    else:
        return get_ori_torch_op(torch.distributed, "all_gather_into_tensor")(
            output_tensor, input_tensor, group=group, async_op=async_op
        )


@register_overriden_func(torch.distributed, "reduce_scatter_tensor")
def reduce_scatter_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if isinstance(group, MistProcessGroup):
        output_tensor.data = output_tensor.data.clone()
        return MistDistHandle()
    else:
        return get_ori_torch_op(torch.distributed, "reduce_scatter_tensor")(
            output_tensor, input_tensor, group=group, async_op=async_op
        )


@register_overriden_func(torch.distributed, "all_reduce")
def all_reduce(
    input_tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False
):
    if isinstance(group, MistProcessGroup):
        input_tensor.data = input_tensor.data.clone()
        return MistDistHandle()
    else:
        return get_ori_torch_op(torch.distributed, "all_reduce")(
            input_tensor, op=op, group=group, async_op=async_op
        )


@register_overriden_func(torch.distributed, "broadcast")
def broadcast(tensor, src, group=None, async_op=False):
    if isinstance(group, MistProcessGroup):
        return MistDistHandle()
    else:
        return get_ori_torch_op(torch.distributed, "broadcast")(
            tensor, src=src, group=group, async_op=async_op
        )
