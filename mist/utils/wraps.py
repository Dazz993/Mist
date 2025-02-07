from functools import wraps
from types import MethodType

import torch
import torch.nn as nn


def wraps_nn_module_forward(this: nn.Module, that: nn.Module):
    """Wrap this forward method with that forward method"""
    assert isinstance(this, nn.Module)
    assert isinstance(that, nn.Module)

    @wraps(type(that).forward)
    def wrapped_forward(self, *args, **kwargs):
        return type(this).forward(self, *args, **kwargs)

    this.forward = MethodType(wrapped_forward, this)
