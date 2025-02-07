import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


@register_symbolic_op(torch.nn.init, "kaiming_uniform_")
class SymbolicCrossEntropy(SymbolicOp):
    @staticmethod
    def apply(outputs, tensor, *args, **kwargs):
        ret = SymbolicTensor(outputs, symbolic_shape=tensor.shape)
        return ret
