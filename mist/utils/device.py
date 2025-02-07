import contextlib
import os
from typing import Optional
from unittest.mock import patch

import torch


def get_device(device=None):
    if isinstance(device, torch.device):
        ret = device
    elif isinstance(device, str):
        ret = torch.device(device)
    elif isinstance(device, int):
        ret = torch.device(f"cuda:{device}")
    elif device is None:
        ret = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        raise ValueError(
            f"device must be a str or torch.device, but got {type(device)}"
        )

    if ret.type == "cuda" and ret.index is None:
        ret = torch.device(f"cuda:{torch.cuda.current_device()}")

    return ret


def all_params_and_buffers_in_device(module, device, allow_cpu: bool = False):
    ret = True
    for name, param in module.named_parameters():
        if param is not None and param.device != device:
            if allow_cpu and param.device.type == "cpu":
                continue
            ret = False
    for name, buf in module.named_buffers():
        if buf is not None and buf.device != device:
            if allow_cpu and buf.device.type == "cpu":
                continue
            ret = False
    return ret


def get_simplified_device_name(device=None, lower=False):
    device = get_device(device)
    device_name = torch.cuda.get_device_name(device).split(" ")[-1]
    if lower:
        device_name = device_name.lower()
    return device_name


@contextlib.contextmanager
def mock_cuda_device_name(name):
    """A context manager to temporarily change the name of the current CUDA device.

    Parameters
    ----------
    name
        name of the fake device.
    """
    f = lambda *args, **kwargs: name
    with patch("torch.cuda.get_device_name", f):
        yield


@contextlib.contextmanager
def mock_cuda_device_name_if_needed():
    """A context manager that conditionally changes the name of the CUDA device
    based on the 'FAKE_DEVICE_NAME' environment variable."""
    fake_device_name = os.getenv("FAKE_DEVICE_NAME")
    if fake_device_name:
        with mock_cuda_device_name(fake_device_name):
            yield
    else:
        yield  # Do nothing, just pass through this context manager


def stream_synchronize(*streams):
    for stream in streams:
        for other_stream in streams:
            if stream is not other_stream:
                stream.wait_stream(other_stream)
