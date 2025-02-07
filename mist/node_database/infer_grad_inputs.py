from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_map

from mist.utils.pytree import tree_flatten_like, tree_zip_map
from mist.sym_torch.symbolic_tensor import SymbolicTensor
from mist.node_database.node_spec import NodeSpec
from mist.node_database.hardware_spec import HardwareSpec, get_hardware_spec
from mist.node_database.inputs_outputs_spec import (
    InputsSpec,
    TensorSpec,
    OutputSpec,
    UndeterminedOutputSpec,
    EmptyOutputSpec,
    map_to_materialized_tensor,
)


def infer_grad_inputs(
    node_spec: NodeSpec,
    inputs_spec: InputsSpec,
    grad_output_spec: OutputSpec,
    device="meta",
):
    device = torch.device(device)

    if node_spec.op == "call_module":
        # Create a module instance
        module = node_spec.instantiate(device=device)

        # Get the fn
        fn = module.forward

    elif node_spec.op == "call_function":
        # Get the fn
        fn = node_spec.target

    # Prepare the inputs
    args = tuple(
        [
            tree_map(map_to_materialized_tensor, spec)
            for idx, spec in enumerate(inputs_spec.args)
        ]
    )
    kwargs = {
        name: tree_map(map_to_materialized_tensor, spec)
        for name, spec in inputs_spec.kwargs.items()
    }

    # Get the output
    output = fn(*args, **kwargs)

    flat_grad_output_spec, _spec_grad_output_spec = tree_flatten(
        grad_output_spec.output
    )
    flat_outputs, _spec_grad_output = tree_flatten_like(output, _spec_grad_output_spec)
    assert len(flat_grad_output_spec) == len(
        flat_outputs
    ), f"len(flat_grad_output_spec)={len(flat_grad_output_spec)} != len(flat_outputs)={len(flat_outputs)}"

    # Retain the inputs gradients
    for input_tensor in tree_flatten(args)[0] + tree_flatten(kwargs)[0]:
        if isinstance(input_tensor, torch.Tensor) and input_tensor.requires_grad:
            input_tensor.retain_grad()

    # Compute the gradients
    for spec, output in zip(flat_grad_output_spec, flat_outputs):
        if isinstance(spec, TensorSpec):
            assert isinstance(output, torch.Tensor)
            output.sum().backward(retain_graph=True)

    # Get the gradients
    def get_grad(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.grad
        else:
            return None

    grad_args = tree_map(get_grad, args)
    grad_kwargs = tree_map(get_grad, kwargs)

    return InputsSpec(node_spec.signature, *grad_args, **grad_kwargs)


def materialize_tensor(tensor):
    if isinstance(tensor, SymbolicTensor):
        return torch.empty(
            tensor.concrete_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            device="meta",
        )
    elif isinstance(tensor, torch.Tensor):
        return torch.empty_like(
            tensor, requires_grad=tensor.requires_grad, device="meta"
        )
    else:
        return tensor


def infer_grad_inputs_for_symop(
    in_args, in_kwargs, outputs, grad_outputs, return_grad_for_params=False
):
    flat_grad_outputs, flat_grad_outputs_spec = tree_flatten(grad_outputs)
    flat_outputs, flat_outputs_spec = tree_flatten_like(outputs, flat_grad_outputs_spec)
    assert len(flat_grad_outputs) == len(
        flat_outputs
    ), f"len(flat_grad_outputs)={len(flat_grad_outputs)} != len(flat_outputs)={len(flat_outputs)}"

    # Retain the inputs gradients
    inputs = []
    for input_tensor in tree_flatten(in_args)[0] + tree_flatten(in_kwargs)[0]:
        if isinstance(input_tensor, torch.Tensor) and input_tensor.requires_grad:
            input_tensor.retain_grad()
            inputs.append(input_tensor)

    # Compute the gradients
    if inputs:
        for grad_output, output in zip(flat_grad_outputs, flat_outputs):
            if grad_output is not None:
                assert isinstance(output, torch.Tensor)
                grad_inputs = torch.autograd.grad(
                    output,
                    inputs,
                    grad_outputs=grad_output,
                    retain_graph=True,
                    allow_unused=True,
                )
    else:
        grad_inputs = []

    # Get the gradients
    mapping = {
        input_tensor: grad_input
        for input_tensor, grad_input in zip(inputs, grad_inputs)
    }

    def get_grad(tensor):
        if not return_grad_for_params and (
            isinstance(tensor, nn.Parameter) or getattr(tensor, "_is_param", False)
        ):
            return None

        grad = mapping.get(tensor, None)
        if isinstance(tensor, SymbolicTensor) and isinstance(grad, SymbolicTensor):
            return grad
        elif isinstance(tensor, SymbolicTensor) and isinstance(grad, torch.Tensor):
            return SymbolicTensor(
                grad,
                symbolic_shape=tensor.shape,
            )
        elif isinstance(tensor, torch.Tensor) and isinstance(grad, SymbolicTensor):
            raise ValueError(
                f"tensor {tensor} has a concrete value but the gradient has a symbolic value"
            )
        elif isinstance(tensor, torch.Tensor) and isinstance(grad, torch.Tensor):
            return grad
        else:
            return None

    grad_args = tree_map(get_grad, in_args)
    grad_kwargs = tree_map(get_grad, in_kwargs)

    return grad_args, grad_kwargs


if __name__ == "__main__":
    from mist.symbols import global_symbol_manager as gsm

    b, s, h = gsm.symbols("b s h", (4, 128, 768))
    m1 = torch.rand(b, s, h, requires_grad=True)

    torch.nn.Linear.reset_parameters = lambda self: None
    linear = torch.nn.Linear(h, h)

    output = linear(m1)
    grad_output = torch.rand_like(output)

    g_args, g_kwargs = infer_grad_inputs_for_symop(linear, grad_output, m1)
