import inspect
from typing import Dict, Callable

import torch
import torch.overrides


def is_abstractmethod(func):
    return getattr(func, "__isabstractmethod__", False)


def map_args_kwargs_to_kwargs(sig: inspect.Signature, *args, **kwargs):
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def map_args_kwargs_to_args(sig: inspect.Signature, *args, **kwargs):
    return list(map_args_kwargs_to_kwargs(sig, *args, **kwargs).values())


def inspect_torch_function_signature(func):
    overrides = torch.overrides.get_testing_overrides()
    overrides.update({inspect.unwrap(k): v for k, v in overrides.items()})
    if func in overrides:
        return inspect.signature(overrides[func])
    return inspect.signature(func)
