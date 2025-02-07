import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op
from mist.sym_torch.impl._shape_prop import get_add_shape


@register_symbolic_op(torch.Tensor, "__invert__")
@register_symbolic_op(torch.Tensor, "neg")
@register_symbolic_op(torch.Tensor, "neg_")
@register_symbolic_op(torch.Tensor, "__neg__")
@register_symbolic_op(torch.Tensor, "type_as")
@register_symbolic_op(torch.nn.init, "kaiming_uniform_")
@register_symbolic_op(torch.nn.init, "uniform_")
@register_symbolic_op(torch.Tensor, "to")
@register_symbolic_op(torch.Tensor, "type_as")
@register_symbolic_op(torch, "cumsum")
@register_symbolic_op(torch.Tensor, "cumsum")
@register_symbolic_op(torch.Tensor, "bool")
@register_symbolic_op(torch.Tensor, "half")
@register_symbolic_op(torch.Tensor, "float")
@register_symbolic_op(torch.Tensor, "triu")
@register_symbolic_op(torch, "triu")
class SymbolicUnaryWithoutSaving(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        return SymbolicTensor(outputs, symbolic_shape=input.shape)


@register_symbolic_op(torch, "relu")
@register_symbolic_op(torch.Tensor, "relu")
@register_symbolic_op(torch.nn.functional, "relu")
@register_symbolic_op(torch.nn.functional, "gelu")
@register_symbolic_op(torch.nn.functional, "silu")
@register_symbolic_op(torch.nn.functional, "elu")
@register_symbolic_op(torch.nn.functional, "selu")
@register_symbolic_op(torch.nn.functional, "celu")
@register_symbolic_op(torch, "pow")
@register_symbolic_op(torch.Tensor, "pow")
@register_symbolic_op(torch.Tensor, "__pow__")
@register_symbolic_op(torch, "tanh")
@register_symbolic_op(torch.Tensor, "tanh")
@register_symbolic_op(torch, "abs")
@register_symbolic_op(torch.Tensor, "abs")
@register_symbolic_op(torch, "log")
@register_symbolic_op(torch.Tensor, "log")
@register_symbolic_op(torch, "cos")
@register_symbolic_op(torch.Tensor, "cos")
@register_symbolic_op(torch, "sin")
@register_symbolic_op(torch.Tensor, "sin")
@register_symbolic_op(torch, "erf")
@register_symbolic_op(torch.Tensor, "erf")
class SymbolicUnaryWithInputSaving(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        ret = SymbolicTensor(outputs, symbolic_shape=input.shape)
        if input.requires_grad:
            ret.context = SymbolicOpContext(
                op=SymbolicUnaryWithInputSaving, saved_tensors=[input]
            )
        return ret


@register_symbolic_op(torch, "sigmoid")
@register_symbolic_op(torch.Tensor, "sigmoid")
@register_symbolic_op(torch, "sqrt")
@register_symbolic_op(torch.Tensor, "sqrt")
@register_symbolic_op(torch, "rsqrt")
@register_symbolic_op(torch.Tensor, "rsqrt")
@register_symbolic_op(torch, "exp")
@register_symbolic_op(torch.Tensor, "exp")
@register_symbolic_op(torch, "tan")
@register_symbolic_op(torch.Tensor, "tan")
class SymbolicUnaryWithOutputSaving(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        ret = SymbolicTensor(outputs, symbolic_shape=input.shape)
        if input.requires_grad:
            ret.context = SymbolicOpContext(
                op=SymbolicUnaryWithOutputSaving, saved_tensors=[ret]
            )
        return ret


@register_symbolic_op(torch, "softmax")
@register_symbolic_op(torch.Tensor, "softmax")
@register_symbolic_op(torch.nn.functional, "softmax")
class SymbolicUnaryWithOutputSavingAndInner(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        ret = SymbolicTensor(outputs, symbolic_shape=input.shape)
        if input.requires_grad:
            ret.context = SymbolicOpContext(
                op=SymbolicUnaryWithOutputSaving,
                saved_tensors=[ret],
                extra_inner_for_bwd=[torch.empty_like(ret)],
            )
        return ret


@register_symbolic_op(torch, "add")
@register_symbolic_op(torch.Tensor, "add")
@register_symbolic_op(torch.Tensor, "add_")
@register_symbolic_op(torch.Tensor, "__add__")
@register_symbolic_op(torch.Tensor, "__radd__")
@register_symbolic_op(torch.Tensor, "__iadd__")
@register_symbolic_op(torch, "sub")
@register_symbolic_op(torch.Tensor, "sub")
@register_symbolic_op(torch.Tensor, "sub_")
@register_symbolic_op(torch.Tensor, "__sub__")
@register_symbolic_op(torch.Tensor, "__rsub__")
@register_symbolic_op(torch.Tensor, "__isub__")
@register_symbolic_op(torch.Tensor, "__or__")
@register_symbolic_op(torch.Tensor, "__ror__")
@register_symbolic_op(torch.Tensor, "__ior__")
@register_symbolic_op(torch.Tensor, "__and__")
@register_symbolic_op(torch.Tensor, "__rand__")
@register_symbolic_op(torch.Tensor, "__iand__")
@register_symbolic_op(torch.Tensor, "__xor__")
@register_symbolic_op(torch.Tensor, "__rxor__")
@register_symbolic_op(torch.Tensor, "__ixor__")
@register_symbolic_op(torch, "lt")
@register_symbolic_op(torch.Tensor, "lt")
@register_symbolic_op(torch.Tensor, "__lt__")
@register_symbolic_op(torch, "le")
@register_symbolic_op(torch.Tensor, "le")
@register_symbolic_op(torch.Tensor, "__le__")
@register_symbolic_op(torch, "gt")
@register_symbolic_op(torch.Tensor, "gt")
@register_symbolic_op(torch.Tensor, "__gt__")
@register_symbolic_op(torch, "ge")
@register_symbolic_op(torch.Tensor, "ge")
@register_symbolic_op(torch.Tensor, "__ge__")
@register_symbolic_op(torch, "eq")
@register_symbolic_op(torch.Tensor, "eq")
@register_symbolic_op(torch.Tensor, "__eq__")
@register_symbolic_op(torch, "ne")
@register_symbolic_op(torch.Tensor, "ne")
@register_symbolic_op(torch.Tensor, "__ne__")
class SymbolicBinaryWithoutSaving(SymbolicOp):
    @staticmethod
    def apply(outputs, input, other, *args, **kwargs):
        if isinstance(outputs, torch.Tensor):
            input_shape = getattr(input, "shape", ())
            other_shape = getattr(other, "shape", ())
            outputs = SymbolicTensor(
                outputs, symbolic_shape=get_add_shape(input_shape, other_shape)
            )
        return outputs


@register_symbolic_op(torch, "mul")
@register_symbolic_op(torch.Tensor, "mul")
@register_symbolic_op(torch.Tensor, "__mul__")
@register_symbolic_op(torch.Tensor, "__rmul__")
@register_symbolic_op(torch.Tensor, "__imul__")
@register_symbolic_op(torch, "div")
@register_symbolic_op(torch.Tensor, "div")
@register_symbolic_op(torch.Tensor, "__truediv__")
@register_symbolic_op(torch.Tensor, "__rtruediv__")
@register_symbolic_op(torch.Tensor, "__itruediv__")
class SymbolicBinaryWithInterSaving(SymbolicOp):
    @staticmethod
    def apply(outputs, input, other, *args, **kwargs):
        input_shape = getattr(input, "shape", ())
        other_shape = getattr(other, "shape", ())
        ret = SymbolicTensor(
            outputs, symbolic_shape=get_add_shape(input_shape, other_shape)
        )

        saved_for_other = input if getattr(other, "requires_grad", False) else None
        saved_for_input = other if getattr(input, "requires_grad", False) else None
        ret.context = SymbolicOpContext(
            op=SymbolicBinaryWithInterSaving,
            saved_tensors=[saved_for_other, saved_for_input],
        )

        return ret


@register_symbolic_op(torch.Tensor, "masked_fill")
@register_symbolic_op(torch.Tensor, "masked_fill_")
class SymbolicBinarySaveFirstArg(SymbolicOp):
    @staticmethod
    def apply(outputs, input, other, *args, **kwargs):
        input_shape = getattr(input, "shape", ())
        other_shape = getattr(other, "shape", ())
        ret = SymbolicTensor(
            outputs, symbolic_shape=get_add_shape(input_shape, other_shape)
        )
        ret.context = SymbolicOpContext(
            op=SymbolicBinarySaveFirstArg, saved_tensors=[input]
        )
        return ret


@register_symbolic_op(torch, "max")
@register_symbolic_op(torch.Tensor, "max")
@register_symbolic_op(torch, "min")
@register_symbolic_op(torch.Tensor, "min")
class SymbolicMax(SymbolicOp):
    @staticmethod
    def apply(outputs, input, *args, **kwargs):
        # torch.max(input) -> Tensor
        if len(args) == 0 and len(kwargs) == 0:
            shape = ()
            ret = SymbolicTensor(outputs, symbolic_shape=shape)
            ret.context = SymbolicOpContext(op=SymbolicMax, saved_tensors=[input])
            return ret

        # torch.max(input, other, *, out=None) → Tensor
        elif len(args) == 1 and isinstance(args[0], torch.Tensor):
            return SymbolicMaximum.apply(outputs, input, *args, **kwargs)

        # torch.max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
        else:
            dim = args[0] if len(args) > 0 else kwargs.get("dim", None)
            assert dim is not None, "dim must be specified"
            keepdim = kwargs.get("keepdim", False)
            shape = list(input.shape)
            if keepdim:
                shape[dim] = 1
            else:
                shape.pop(dim)

            max_tensor = SymbolicTensor(outputs[0], symbolic_shape=tuple(shape))
            max_tensor.context = SymbolicOpContext(op=SymbolicMax, saved_tensors=[])
            max_indices = SymbolicTensor(outputs[1], symbolic_shape=tuple(shape))
            return max_tensor, max_indices


@register_symbolic_op(torch, "maximum")
@register_symbolic_op(torch.Tensor, "maximum")
@register_symbolic_op(torch, "minimum")
@register_symbolic_op(torch.Tensor, "minimum")
class SymbolicMaximum(SymbolicOp):
    @staticmethod
    def apply(outputs, input, other, **kwargs):
        input_shape = getattr(input, "shape", ())
        other_shape = getattr(other, "shape", ())
        ret = SymbolicTensor(
            outputs, symbolic_shape=get_add_shape(input_shape, other_shape)
        )
        ret.context = SymbolicOpContext(
            op=SymbolicMaximum, saved_tensors=[input, other]
        )
        return ret
