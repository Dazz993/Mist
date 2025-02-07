import sympy as sp
import torch
from sympy import ceiling

from mist.sym_torch import SymbolicTensor, SymbolicOp, SymbolicOpContext
from mist.overrides import register_overriden_func, get_ori_torch_op
from mist.utils.torch_function import call_ori_torch_function

"""
The registration of torch.autograd.grad is different from other symbolic ops.

The logic for other normal symbolic ops is:
- Convert all arguments to original tensors
- Call the original op
- Convert the output to symbolic tensors

However, for torch.autograd.grad, if we convert all arguments to original tensors,
then the gradient won't flow through the symbolic tensors. Because torch tensor outputs'
final source is symbolic inputs, instead of torch tensor inputs.
    SymbolicInputs --Some Op--> SymbolicOutputs
        |                                |
        |                                |
        V                                V
    TorchTensorInputs            TorchTensorOutputs

The solution is to manually implement the overriden logic here.
- We only need to convert the outputs to symbolic tensors.
"""


@register_overriden_func(torch.autograd, "grad")
def sym_autograd_grad(*args, **kwargs):
    ori_torch_autograd_grad = get_ori_torch_op(torch.autograd, "grad")
    grad_inputs = call_ori_torch_function(ori_torch_autograd_grad, *args, **kwargs)
    inputs = args[1] if len(args) > 1 else kwargs["inputs"]

    assert len(grad_inputs) == len(
        inputs
    ), f"len(grad_inputs)={len(grad_inputs)} != len(inputs)={len(inputs)}"

    ret = []
    for grad_input, input in zip(grad_inputs, inputs):
        if grad_input is None:
            ret.append(None)
        elif isinstance(grad_input, SymbolicTensor):
            ret.append(SymbolicTensor(grad_input, symbolic_shape=input.shape))
        else:
            assert isinstance(input, torch.Tensor)
            if isinstance(input, SymbolicTensor):
                ret.append(SymbolicTensor(grad_input, symbolic_shape=input.shape))
            else:
                ret.append(grad_input)

    return tuple(ret)
