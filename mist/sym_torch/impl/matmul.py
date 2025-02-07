import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op
from mist.sym_torch.impl._shape_prop import get_matmul_shape


def _matmul(op, outputs, mat1, mat2):
    ret = SymbolicTensor(
        outputs,
        symbolic_shape=get_matmul_shape(mat1.shape, mat2.shape),
    )

    saved_for_mat2 = mat1 if mat2.requires_grad else None
    saved_for_mat1 = mat2 if mat1.requires_grad else None
    ret.context = SymbolicOpContext(
        op=op,
        saved_tensors=[saved_for_mat1, saved_for_mat2],
    )
    return ret


@register_symbolic_op(torch, "matmul")
@register_symbolic_op(torch.Tensor, "matmul")
@register_symbolic_op(torch, "mm")
@register_symbolic_op(torch.Tensor, "mm")
@register_symbolic_op(torch, "bmm")
@register_symbolic_op(torch.Tensor, "bmm")
class SymbolicMatmul(SymbolicOp):
    @staticmethod
    def apply(outputs, mat1, mat2, *args, **kwargs):
        return _matmul(SymbolicMatmul, outputs, mat1, mat2)

@register_symbolic_op(torch, "baddbmm")
class SymbolicAddBMM(SymbolicOp):
    @staticmethod
    def apply(outputs, input, batch1, batch2, *args, **kwargs):
        return _matmul(SymbolicMatmul, outputs, batch1, batch2)


@register_symbolic_op(torch, "addmm")
class SymbolicAddMM(SymbolicOp):
    @staticmethod
    def apply(outputs, input, mat1, mat2, *args, **kwargs):
        return _matmul(SymbolicAddMM, outputs, mat1, mat2)


@register_symbolic_op(torch.nn.functional, "linear")
class SymbolicLinear(SymbolicOp):
    @staticmethod
    def apply(outputs, input, weight, bias=None):
        return _matmul(SymbolicLinear, outputs, input, weight.t())
