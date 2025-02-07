from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
import torch.distributed

from mist.utils.sympy import indicator
from mist.sym_torch.autograd_func import symbolic_compatible

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 4 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base


# Raw operation, does not support autograd, but does support async
def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    output_shape = (input_.shape[0] * world_size,) + input_.shape[1:]
    output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def reduce_scatter_raw(
    input_: Tensor, process_group: ProcessGroup, async_op: bool = False
):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output_shape = (input_.shape[0] // world_size,) + input_.shape[1:]
    output = torch.empty(output_shape, dtype=input_.dtype, device=input_.device)
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, op=torch.distributed.ReduceOp.SUM, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(
        input_, op=op, group=process_group, async_op=async_op
    )
    return input_, handle


class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    @symbolic_compatible
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
def all_gather(input_: Tensor, process_group: ProcessGroup) -> Tensor:
    """Gather the input from sequence parallel region and concatenate."""
    return AllGatherFunc.apply(input_, process_group)


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    @symbolic_compatible
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None


def reduce_scatter(input_: Tensor, process_group: ProcessGroup) -> Tensor:
    """Reduce scatter the input from the sequence parallel region and concatenate."""
    return ReduceScatterFunc.apply(input_, process_group)


class AllReduceFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    @symbolic_compatible
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


# Supports autograd, but does not support async
def all_reduce(input_: Tensor, process_group: ProcessGroup) -> Tensor:
    """Gather the input from sequence parallel region and concatenate."""
    return AllReduceFunc.apply(input_, process_group)


def sync_shared_params(model: torch.nn.Module, process_group: ProcessGroup):
    # We want to iterate over parameters with _shared_params=True in the same order,
    # as different ranks might have different number of parameters (e.g., only rank 0 has bias).
    pamams_shared = {
        name: p
        for name, p in model.named_parameters()
        if getattr(p, "_shared_params", False)
    }
    for _, p in sorted(pamams_shared.items()):
        with torch.no_grad():
            # Broadcast needs src to be global rank, not group rank
            torch.distributed.broadcast(
                p,
                src=torch.distributed.get_global_rank(process_group, 0),
                group=process_group,
            )


# Ref: https://github.com/NVIDIA/Megatron-LM/blob/52e636888cccc41e931251c417a7181fc36de926/megatron/optimizer/optimizer.py#L256
def allreduce_sequence_parallel_grad(
    model: torch.nn.Module, process_group: ProcessGroup
):
    # We want to iterate over parameters with _sequence_parallel=True in the same order,
    # as different ranks might have different number of parameters (e.g., only rank 0 has bias).
    params_seqparallel = {
        name: p
        for name, p in model.named_parameters()
        if getattr(p, "_sequence_parallel", False)
    }
    grads = [p.grad for _, p in sorted(params_seqparallel.items())]
    if grads:
        with torch.no_grad():
            coalesced = torch._utils._flatten_dense_tensors(grads)
            torch.distributed.all_reduce(coalesced, group=process_group)
            for buf, synced in zip(
                grads, torch._utils._unflatten_dense_tensors(coalesced, grads)
            ):
                print(buf.shape, synced.shape)
                buf.copy_(synced)


def get_dim_for_local_rank(
    dim: int, world_size: int, local_rank: int, multiple_of: int = 1
) -> int:
    """Get the dim for the local rank derived from splitting dim on world_size processes.

    The split may not be even across the world_size processes.
    """
    multiple = dim // multiple_of
    div = multiple // world_size
    mod = multiple % world_size
    # local_multiple = div + int(local_rank < mod)
    local_multiple = div + indicator(local_rank < mod)
    return local_multiple * multiple_of

torch.fx.wrap("all_gather_raw")
torch.fx.wrap("reduce_scatter_raw")
torch.fx.wrap("all_reduce_raw")
torch.fx.wrap("all_gather")
torch.fx.wrap("reduce_scatter")
torch.fx.wrap("all_reduce")
