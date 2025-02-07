import inspect
from functools import wraps
from typing import List, Union
from types import MethodType

import torch
import torch.nn as nn
import torch.utils.checkpoint

from mist.utils.inspect import map_args_kwargs_to_args
from mist.logger import get_logger

logger = get_logger()


# Use gradient checkpointing
class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *args, **kwargs):
        ordered_args = map_args_kwargs_to_args(
            inspect.signature(self._module.forward), *args, **kwargs
        )
        return torch.utils.checkpoint.checkpoint(
            self._module, *ordered_args, use_reentrant=True
        )


def wrap_forward_with_gradient_checkpointing(module: nn.Module):
    """
    Note: this wrapping is not the best way. But we have to do this because we
    want to support multiple wrapping.
    """
    orig_forward_method = module.forward
    orig_forward_signature = inspect.signature(orig_forward_method)
    orig_forward_func = type(module).forward

    @wraps(orig_forward_func)
    def wrapped_checkpoint_forward(self, *args, **kwargs):
        # logger.debug(f"Running into checkpointing wrapper.")
        ordered_args = map_args_kwargs_to_args(orig_forward_signature, *args, **kwargs)
        return torch.utils.checkpoint.checkpoint(orig_forward_method, *ordered_args, use_reentrant=True)

    module.forward = MethodType(wrapped_checkpoint_forward, module)
    module._orig_forward_without_ckpt = orig_forward_method


def apply_gradient_checkpointing(
    module: nn.Module, sub_modules: List[Union[nn.Module, str]]
):
    replace_list = []
    for parent in module.modules():
        for name, child in parent._modules.items():
            if child in sub_modules or (
                hasattr(child, "name") and child.name in sub_modules
            ):
                replace_list.append((parent, name, child))

    for parent, name, child in replace_list:
        wrap_forward_with_gradient_checkpointing(child)
