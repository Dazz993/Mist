import sympy as sp
import torch

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.sym_torch.registry import register_symbolic_op


class SymbolicMisc(SymbolicOp):
    @staticmethod
    def apply(outputs, tensor, *args, **kwargs):
        return SymbolicTensor(outputs, symbolic_shape=tensor.shape)
