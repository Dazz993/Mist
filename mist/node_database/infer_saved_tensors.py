import torch

from mist.node_database.node_spec import NodeSpec
from mist.node_database.inputs_outputs_spec import TensorSpec, InputsSpec
from mist.node_database.saved_tensors_spec import SavedTensorsSpec
from mist.node_database.hardware_spec import HardwareSpec, get_hardware_spec
from mist.utils.memory import materialize_module, materialize_tensor


def infer_saved_tensors(
    node_spec: NodeSpec,
    inputs_spec: InputsSpec,
    hardware_spec: HardwareSpec = None,
    device=None,
) -> SavedTensorsSpec:
    """
    This function infers the saved tensors of a node and returns the SavedTensorSpec.

    The way to infer the saved tensors is to use the pack_hook and unpack_hook, and compare the
    saved_tensors with the inputs, output, parameters and buffers.
    """

    name = node_spec.target.__name__
    if name in ["getitem", "getattr"]:
        return SavedTensorsSpec()

    hardware_spec = hardware_spec or get_hardware_spec()
    device = device or torch.device(hardware_spec.torch_device_type)

    if node_spec.op == "call_module":
        # Create a module instance
        module = node_spec.instantiate(device=device)

        # Parameters and buffers
        params_and_buffers_data_ptr_to_name = {
            param.data_ptr(): name for name, param in module.named_parameters()
        }
        params_and_buffers_data_ptr_to_name.update(
            {buffer.data_ptr(): name for name, buffer in module.named_buffers()}
        )

        # Get the fn
        fn = module.forward

    elif node_spec.op == "call_function":
        # Parameters and buffers
        params_and_buffers_data_ptr_to_name = set()

        # Get the fn
        fn = node_spec.target

    # Prepare the inputs
    args = {
        idx: spec.instantiate(device) if isinstance(spec, TensorSpec) else spec
        for idx, spec in enumerate(inputs_spec.args)
    }
    kwargs = {
        name: spec.instantiate(device) if isinstance(spec, TensorSpec) else spec
        for name, spec in inputs_spec.kwargs.items()
    }
    # Get all the inputs
    inputs_data_ptr_to_name = {
        value.data_ptr(): name
        for name, value in {**args, **kwargs}.items()
        if isinstance(value, torch.Tensor)
    }

    # Helper functions for packing and unpacking
    saved = set()

    def pack_hook(tensor):
        saved.add(tensor)
        return tensor

    def unpack_hook(tensor):
        return tensor

    # Run the fn
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output = fn(*list(args.values()), **kwargs)
    output.sum().backward()

    # Generate the saved_tensors_spec
    saved_inputs_signature = []
    save_output = False
    saved_params_and_buffers_signature = []
    saved_intermediate_tensors_spec = []

    for saved_tensor in saved:
        cur_data_ptr = saved_tensor.data_ptr()
        if cur_data_ptr in params_and_buffers_data_ptr_to_name:
            saved_params_and_buffers_signature.append(
                params_and_buffers_data_ptr_to_name[cur_data_ptr]
            )
        elif cur_data_ptr in inputs_data_ptr_to_name:
            saved_inputs_signature.append(inputs_data_ptr_to_name[cur_data_ptr])
        elif cur_data_ptr == output.data_ptr():
            save_output = True
        else:
            saved_intermediate_tensors_spec.append(TensorSpec.from_tensor(saved_tensor))

    saved_tensor_spec = SavedTensorsSpec(
        node_spec,
        saved_inputs_signature,
        save_output,
        saved_params_and_buffers_signature,
        saved_intermediate_tensors_spec,
    )

    return saved_tensor_spec
