import sympy as sp
import torch
from sympy import ceiling

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op
from mist.sym_torch.impl._shape_prop import get_add_shape


@register_symbolic_op(torch, "empty")
@register_symbolic_op(torch, "zeros")
@register_symbolic_op(torch, "ones")
@register_symbolic_op(torch, "rand")
@register_symbolic_op(torch, "randn")
class SymbolicEmpty(SymbolicOp):
    @staticmethod
    def apply(outputs, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args

        return SymbolicTensor(
            outputs,
            symbolic_shape=shape,
        )


@register_symbolic_op(torch, "full")
class SymbolicFull(SymbolicOp):
    @staticmethod
    def apply(outputs, *args, **kwargs):
        shape = args[0]

        return SymbolicTensor(
            outputs,
            symbolic_shape=shape,
        )


@register_symbolic_op(torch, "randint")
class SymbolicRandint(SymbolicOp):
    @staticmethod
    def apply(outputs, *args, **kwargs):
        shape = args[2]

        return SymbolicTensor(
            outputs,
            symbolic_shape=shape,
        )


@register_symbolic_op(torch, "empty_like")
@register_symbolic_op(torch, "zeros_like")
@register_symbolic_op(torch, "ones_like")
@register_symbolic_op(torch, "rand_like")
@register_symbolic_op(torch, "randn_like")
class SymbolicEmptyLike(SymbolicOp):
    @staticmethod
    def apply(outputs, input, **kwargs):
        return SymbolicTensor(
            outputs,
            symbolic_shape=input.shape,
        )


@register_symbolic_op(torch, "arange")
class SymbolicArange(SymbolicOp):
    @staticmethod
    def apply(outputs, start=0, end=None, step=1, **kwargs):
        if end is None:
            end = start
            start = 0

        if (
            isinstance(start, sp.Basic)
            or isinstance(end, sp.Basic)
            or isinstance(step, sp.Basic)
        ):
            shape = (ceiling((end - start) / step),)

        return SymbolicTensor(
            outputs,
            symbolic_shape=shape,
        )


@register_symbolic_op(torch, "where")
class SymbolicWhere(SymbolicOp):
    @staticmethod
    def apply(outputs, condition, input, other, **kwargs):
        condition_shape = getattr(condition, "shape", ())
        input_shape = getattr(input, "shape", ())
        other_shape = getattr(other, "shape", ())
        shape = get_add_shape(get_add_shape(condition_shape, input_shape), other_shape)

        ret = SymbolicTensor(
            outputs,
            symbolic_shape=shape,
        )
        ret.context = SymbolicOpContext(
            op=SymbolicWhere, saved_tensors=[condition.bool(), input, other]
        )
        return ret


@register_symbolic_op(torch, "tril")
class SymbolicTril(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        return SymbolicTensor(
            outputs,
            symbolic_shape=input.shape,
        )


if __name__ == "__main__":
    from mist import global_symbol_manager as gsm

    b, s, h = gsm.symbols("b s h", (5, 6, 7), integer=True, positive=True)
    x = torch.empty((b, s, h), device="meta")
    print(x)
    y = torch.empty(b, s, h, device="meta")
    print(y)
