from numbers import Integral

import numpy as np
import sympy as sp
import torch

from mist import global_symbol_manager as gsm
from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op
from mist.utils.rearrange import infer_unknown_dim
from mist.utils.sympy import ceil, fake_floordiv


@register_symbolic_op(torch, "reshape")
@register_symbolic_op(torch.Tensor, "reshape")
@register_symbolic_op(torch.Tensor, "view")
class SymbolicView(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *size):
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            size = size[0]
        shape = infer_unknown_dim(in_shape=input.shape, to_shape=size)
        return SymbolicTensor(outputs, symbolic_shape=shape)

@register_symbolic_op(torch.Tensor, "view_as")
class SymbolicViewAs(SymbolicOp):
    @staticmethod
    def apply(outputs, input, other):
        return SymbolicTensor(outputs, symbolic_shape=other.shape)


@register_symbolic_op(torch, "flatten")
@register_symbolic_op(torch.Tensor, "flatten")
class SymbolicFlatten(SymbolicOp):
    @staticmethod
    def apply(outputs, input, start_dim=0, end_dim=-1):
        start_dim = start_dim if start_dim >= 0 else input.ndim + start_dim
        end_dim = end_dim if end_dim >= 0 else input.ndim + end_dim
        shape = input.shape
        shape = (
            shape[:start_dim]
            + (np.prod(shape[start_dim : end_dim + 1]),)
            + shape[end_dim + 1 :]
        )
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch._C._nn, "flatten_dense_tensors")
class SymbolicFlattenDenseTensors(SymbolicOp):
    @staticmethod
    def apply(outputs, tensors):
        numel = sum([t.numel() for t in tensors])
        shape = (numel,)
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch._C._nn, "unflatten_dense_tensors")
class SymbolicUnflattenDenseTensors(SymbolicOp):
    @staticmethod
    def apply(outputs, flat, tensors):
        return tuple(
            SymbolicTensor(output, symbolic_shape=t.shape)
            for output, t in zip(outputs, tensors)
        )


@register_symbolic_op(torch, "einsum")
class SymbolicEinsum(SymbolicOp):
    def apply(outputs, equation, *operands):
        assert "->" in equation, f"equation must contain '->', but got {equation}"
        lhs, rhs = equation.split("->")
        lhs = lhs.split(",")
        assert len(lhs) == len(operands), f"mismatched number of operands"

        letter2length = {}
        for i, (shape_str, tensor) in enumerate(zip(lhs, operands)):
            assert (
                "..." not in shape_str
            ), f"ellipsis is not supported yet, support it after finding a use case"
            assert (
                len(shape_str) == tensor.ndim
            ), f"mismatched number of dimensions, {shape_str} vs {tensor.ndim}"
            for letter, length in zip(shape_str, tensor.shape):
                if letter in letter2length:
                    assert (
                        letter2length[letter] == length
                    ), f"mismatched shape, {letter2length[letter]} vs {length}"
                else:
                    letter2length[letter] = length

        shape = tuple(letter2length[letter] for letter in rhs)

        # TODO(zhanda): it's hard to infer saved tensors for einsum.
        # If we have more than one operands, then we just save all of them
        # for now.
        saved_tensors = []
        if len(operands) == 2:
            a, b = operands
            saved_for_a = b if getattr(a, "requires_grad", False) else None
            saved_for_b = a if getattr(b, "requires_grad", False) else None
            saved_tensors = [saved_for_a, saved_for_b]

        ret = SymbolicTensor(outputs, symbolic_shape=shape)
        ret.context = SymbolicOpContext(op=SymbolicEinsum, saved_tensors=saved_tensors)
        return ret


@register_symbolic_op(torch, "permute")
@register_symbolic_op(torch.Tensor, "permute")
class SymbolicPermute(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *dims, **kwargs):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        ret = SymbolicTensor(
            outputs,
            symbolic_shape=tuple(input.shape[i] for i in dims),
        )
        return ret


@register_symbolic_op(torch, "transpose")
@register_symbolic_op(torch.Tensor, "transpose")
class SymbolicTranspose(SymbolicOp):
    @staticmethod
    def apply(outputs, input, dim0, dim1):
        shape = list(input.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        ret = SymbolicTensor(outputs, symbolic_shape=shape)
        return ret


@register_symbolic_op(torch.Tensor, "t")
class SymbolicLowerCaseT(SymbolicOp):
    @staticmethod
    def apply(outputs, input):
        shape = list(input.shape)
        shape[-1], shape[-2] = shape[-2], shape[-1]
        ret = SymbolicTensor(outputs, symbolic_shape=shape)
        return ret


# Override torch.Tensor.T
@property
def T(self):
    dims = list(range(len(self.shape)))
    dims.reverse()
    dims = tuple(dims)
    return torch.permute(self, dims)


SymbolicTensor.T = T


@register_symbolic_op(torch.Tensor, "expand")
class SymbolicExpand(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *size):
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            size = size[0]
        shape = tuple(input.shape[i] if s == -1 else s for i, s in enumerate(size))
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch.Tensor, "repeat")
class SymbolicRepeat(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *size):
        if len(size) == 1 and isinstance(size[0], (list, tuple)):
            size = size[0]
        pre_n_dims = len(size) - input.ndim
        shape = size[:pre_n_dims] + tuple(
            size[pre_n_dims + i] * input.shape[i] for i in range(input.ndim)
        )
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch, "repeat_interleave")
@register_symbolic_op(torch.Tensor, "repeat_interleave")
class SymbolicRepeatInterleave(SymbolicOp):
    @staticmethod
    def apply(outputs, input, repeats, dim=None, **kwargs):
        shape = list(input.shape)
        shape = list(input.shape)
        if dim is None:
            shape = [np.prod(shape)]
            dim = 0
        if isinstance(repeats, int) or torch.numel(repeats) == 1:
            shape[dim] *= int(repeats)
        else:
            shape[dim] = repeats.sum().item()
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch, "squeeze")
@register_symbolic_op(torch.Tensor, "squeeze")
class SymbolicSqueeze(SymbolicOp):
    @staticmethod
    def apply(outputs, input, dim=None):
        if dim is None:
            shape = [s for s in input.shape if s != 1]
        elif isinstance(dim, Integral):
            shape = list(input.shape)
            shape.pop(dim)
        elif isinstance(dim, (list, tuple)):
            shape = list(input.shape)
            for d in dim:
                shape[d] = None
            shape = [s for s in shape if s is not None]
        return SymbolicTensor(outputs, symbolic_shape=tuple(shape))


@register_symbolic_op(torch, "unsqueeze")
@register_symbolic_op(torch.Tensor, "unsqueeze")
class SymbolicUnsqueeze(SymbolicOp):
    @staticmethod
    def apply(outputs, input, dim):
        if dim < 0:
            dim += input.ndim + 1
        shape = input.shape[:dim] + (1,) + input.shape[dim:]
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch, "split")
@register_symbolic_op(torch.Tensor, "split")
class SymbolicSplit(SymbolicOp):
    @staticmethod
    def apply(outputs, input, split_size_or_sections, dim=0):
        if dim < 0:
            dim += input.ndim
        if not isinstance(split_size_or_sections, (list, tuple)):
            num_splits = input.shape[dim] // split_size_or_sections
            remainder = input.shape[dim] % split_size_or_sections
            if isinstance(num_splits, int):
                pass
            elif isinstance(num_splits, sp.Basic):
                assert all(s in gsm for s in num_splits.free_symbols)
                num_splits = gsm.subs(num_splits)
                assert isinstance(num_splits, Integral), f"num_splits: {num_splits}"
            split_size_or_sections = [split_size_or_sections] * num_splits
            if remainder != 0:
                split_size_or_sections.append(remainder)
        else:
            assert bool(
                sp.Eq(input.shape[dim], sum(split_size_or_sections))
            ), f"split_size_or_sections must sum up to {input.shape[dim]}, but got {split_size_or_sections}"

        symbolic_tensors = []
        for i in range(len(split_size_or_sections)):
            shape = (
                *input.shape[:dim],
                split_size_or_sections[i],
                *input.shape[dim + 1 :],
            )
            symbolic_tensors.append(SymbolicTensor(outputs[i], symbolic_shape=shape))

        return symbolic_tensors


@register_symbolic_op(torch, "chunk")
@register_symbolic_op(torch.Tensor, "chunk")
class SymbolicChunk(SymbolicOp):
    @staticmethod
    def apply(outputs, input, chunks, dim=0):
        if dim < 0:
            dim += input.ndim
        assert isinstance(chunks, Integral), f"chunks must be an integer, got {chunks}"
        total_len = input.shape[dim]
        assert (
            total_len % chunks == 0
        ), f"input must be divisible by chunks, got {total_len} and {chunks}"
        split_size_or_sections = fake_floordiv(total_len, chunks)

        symbolic_tensors = []
        for i in range(chunks):
            shape = (
                *input.shape[:dim],
                split_size_or_sections,
                *input.shape[dim + 1 :],
            )
            symbolic_tensors.append(SymbolicTensor(outputs[i], symbolic_shape=shape))

        return symbolic_tensors


@register_symbolic_op(torch, "unbind")
@register_symbolic_op(torch.Tensor, "unbind")
class SymbolicUnbind(SymbolicOp):
    @staticmethod
    def apply(outputs, input, dim=0):
        if dim < 0:
            dim += input.ndim
        shape = list(input.shape)
        shape.pop(dim)
        return tuple(SymbolicTensor(output, symbolic_shape=shape) for output in outputs)


@register_symbolic_op(torch, "cat")
class SymbolicCat(SymbolicOp):
    @staticmethod
    def apply(outputs, inputs, dim=0, **kwargs):
        shape = list(inputs[0].shape)
        shape[dim] = sum([t.shape[dim] for t in inputs])
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch, "stack")
class SymbolicStack(SymbolicOp):
    @staticmethod
    def apply(outputs, inputs, dim=0, **kwargs):
        if dim is None:
            dim = 0
        if dim < 0:
            dim += inputs[0].ndim + 1
        shape = list(inputs[0].shape)
        shape.insert(dim, len(inputs))
        return SymbolicTensor(outputs, symbolic_shape=shape)


@register_symbolic_op(torch, "gather")
@register_symbolic_op(torch.Tensor, "gather")
class SymbolicGather(SymbolicOp):
    @staticmethod
    def apply(outputs, input, dim, index, **kwargs):
        return SymbolicTensor(outputs, symbolic_shape=index.shape)
