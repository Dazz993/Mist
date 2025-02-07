import gc
import os
import pickle
import torch
from typing import Sequence, Union

from torch import nn
from pathlib import Path
from datetime import datetime

from mist.utils.tensor_entry import TensorEntry
from mist.logger import get_logger
from mist.utils.device import get_device
from mist.utils.module import deepcopy_module

logger = get_logger()

GB = 1024**3

# mist/logs/memory_snapshots/
CUDA_MEM_DUMPING_PATH = os.path.join(
    Path(__file__).absolute().parent.parent, "logs", "memory_snapshots"
)


def guess_is_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return True
    if "tensor" in type(tensor).__name__.lower():
        return True
    return False


def cuda_empty_cache():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def save_cuda_memory_snapshot(msg=None):
    if not os.path.exists(CUDA_MEM_DUMPING_PATH):
        os.makedirs(CUDA_MEM_DUMPING_PATH)

    # Get the current time stamp -> %m-%d-%H-%M-%S
    time_stamp = datetime.now().strftime("%m-%d-%H-%M-%S")

    # Deal with the message
    logger.info(f"Dumping the cuda memory snapshot. Msg: {msg}")
    msg = f"_{msg.replace(' ', '_')}" if msg else ""

    # Dump the cuda memory snapshot
    with open(f"{CUDA_MEM_DUMPING_PATH}/mem_snapshot_{time_stamp}{msg}.pkl", "wb") as f:
        pickle.dump(torch.cuda.memory_snapshot(), f)

    logger.info(
        f"Successfully dumped the cuda memory snapshot to {CUDA_MEM_DUMPING_PATH}, "
        f"allocated memory: {torch.cuda.memory_allocated() / GB:.2f} GB, "
        f"cached memory: {torch.cuda.memory_cached() / GB:.2f} GB, "
        f"max memory: {torch.cuda.max_memory_allocated() / GB:.2f} GB"
    )


def _format_size(sz, pref_sz=None):
    # Initialize the preferred size
    pref_sz = pref_sz or sz

    prefixes = ["B ", "KB", "MB", "GB", "TB", "PB"]
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if pref_sz < 1.0 * 1000:
            break
        prefix = new_prefix
        sz //= 1024
        pref_sz /= 1024

    pref_sz_str = f"{pref_sz:.2f}"
    output = f"{pref_sz_str:>6} {prefix}"

    return output


def report_memory(msg=""):
    if msg:
        msg = f"[{msg}] \t"

    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    peak_memory_allocated = torch.cuda.max_memory_allocated()
    peak_memory_reserved = torch.cuda.max_memory_reserved()
    print(
        f"{msg}"
        f"Alloc: {_format_size(memory_allocated)}\t | "
        f"Peak Alloc: {_format_size(peak_memory_allocated)}\t | "
        f"Reserved: {_format_size(memory_reserved)}\t | "
        f"Peak Reserved: {_format_size(peak_memory_reserved)}"
    )


def materialize_tensor(tensor, device="cuda", rand=True, bound=1e-2):
    """
    Materialize a tensor from a given Tensor or TensorSpec.

    Parameters
    ----------
    tensor : Optional[Union[torch.Tensor, TensorSpec]]
        Input tensor or tensor spec.
    device : str, optional
        device, by default "cuda"
    """
    # Guess whether it's a tensor
    if not guess_is_tensor(tensor):
        return tensor

    device = device or tensor.device
    device = torch.device(device)
    # If bool
    if tensor.dtype == torch.bool:
        if rand:
            output = torch.randint(
                low=0,
                high=2,
                size=tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
        else:
            output = torch.zeros(
                tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
    # If int-like
    elif tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        if rand:
            output = torch.randint(
                low=0,
                high=2,
                size=tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
        else:
            output = torch.zeros(
                tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
    # If float-like
    elif tensor.dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
        if rand:
            # It's possibly too slow if we use it for many times, e.g. during benchmarking
            output = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
            nelements = output.numel()
            bound = abs(bound)
            with torch.no_grad():
                # output.uniform_(-bound, bound)
                output.normal_(mean=0, std=bound)
            # output = torch.randint(
            #     low=-10,
            #     high=10,
            #     size=tensor.shape,
            #     dtype=tensor.dtype,
            #     # layout=tensor.layout,
            #     device=device,
            #     requires_grad=tensor.requires_grad,
            # )
            # nelements = output.numel()
            # with torch.no_grad():
            #     if nelements >= 1024 * 1024:
            #         output /= 100
            #     else:
            #         output /= 10
        else:
            output = torch.zeros(
                tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            )
    else:
        raise NotImplementedError(f"Unknown dtype: {tensor.dtype}")

    if device.type == "cpu":
        output = output.pin_memory()

    return output


def materialize_module(
    mod: nn.Module,
    device="cuda",
    inplace=False,
    ignored_modules: Sequence[Union[nn.Module, str]] = None,
):
    # TODO(zhanda): check the shared parameters. This is not supported yet.
    if ignored_modules is not None and (
        mod in ignored_modules or (hasattr(mod, "name") and mod.name in ignored_modules)
    ):
        return mod

    device = get_device(device)
    if not inplace:
        # However, all the parameters that are already on the device will be kept on the device
        memo = {}
        for name, param in mod.named_parameters():
            if param.device == device:
                memo[id(param)] = param
        mod = deepcopy_module(mod)

    class_name = None
    if hasattr(mod, "__class__"):
        class_name = str(mod.__class__).lower()

    if class_name is not None:
        if "linear" in class_name:
            _weight = getattr(mod, "weight", None)
            _bias = getattr(mod, "bias", None)
            if _weight is not None:
                new_weight = nn.Parameter(materialize_tensor(_weight, device=device))
                torch.nn.init.kaiming_uniform_(new_weight, a=5 ** 0.5)
                mod._parameters["weight"] = new_weight
            if _bias is not None:
                new_bias = nn.Parameter(materialize_tensor(_bias, device=device))
                torch.nn.init.zeros_(new_bias)
                mod._parameters["bias"] = new_bias
        if "layernorm" in class_name or "layer_norm" in class_name or "norm" in class_name:
            _weight = getattr(mod, "weight", None)
            _bias = getattr(mod, "bias", None)
            if _weight is not None:
                new_weight = nn.Parameter(materialize_tensor(_weight, device=device))
                torch.nn.init.ones_(new_weight)
                mod._parameters["weight"] = new_weight
            if _bias is not None:
                new_bias = nn.Parameter(materialize_tensor(_bias, device=device))
                torch.nn.init.zeros_(new_bias)
                mod._parameters["bias"] = new_bias
        if "embedding" in class_name:
            _weight = getattr(mod, "weight", None)
            if _weight is not None:
                new_weight = nn.Parameter(materialize_tensor(_weight, device=device))
                torch.nn.init.normal_(new_weight, mean=0, std=1)
                mod._parameters["weight"] = new_weight

    for name, param in mod._parameters.items():
        if param is not None and param.device != device:
            mod._parameters[name] = nn.Parameter(
                materialize_tensor(param, device=device)
            )
    for name, buf in mod._buffers.items():
        if buf is not None and buf.device != device:
            mod._buffers[name] = materialize_tensor(buf, device=device)

    for name, submod in mod._modules.items():
        mod._modules[name] = materialize_module(
            submod, device=device, inplace=inplace, ignored_modules=ignored_modules
        )

    # Double check
    for name, param in mod.named_parameters():
        if param is not None and param.device != device:
            raise RuntimeError(
                f"Failed to materialize the module {mod} on device {device}, "
                f"param {name} is on device {param.device}"
            )
    for name, buf in mod.named_buffers():
        if buf is not None and buf.device != device:
            raise RuntimeError(
                f"Failed to materialize the module {mod} on device {device}, "
                f"buffer {name} is on device {buf.device}"
            )

    return mod


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """A wrapper for _make_viewless_tensor for tracing."""
    return _make_viewless_tensor(inp, requires_grad, keep_graph)


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view (i.e., its '._base' field is
    not set)."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor


torch.fx.wrap("_make_viewless_tensor")
