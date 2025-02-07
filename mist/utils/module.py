import re
from typing import Sequence, Optional, Any, List, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed

DEFAULT_ROOT_MODULE_NAME = ""


def getattr_recursive(module, target: str):
    if target == DEFAULT_ROOT_MODULE_NAME:
        return module

    target_atoms = target.split(".")
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target '{'.'.join(target_atoms[:i])}'"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def hasattr_recursive(module, target: str):
    target_atoms = target.split(".")
    attr_itr = module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            return False
        attr_itr = getattr(attr_itr, atom)
    return True


def set_module_name_recursive(root, prefix=""):
    for name, module in root.named_modules():
        module.name = (prefix + "." + name) if prefix else name


def summarize_sub_modules_path(sub_modules_path: Sequence[str]):
    ret = set()
    for path in sub_modules_path:
        # Replace the number with 'N' using regex
        path = re.sub(r"\d+", "N", path)
        ret.add(path)
    # sort the path by the number of '.' in the path
    ret = sorted(ret, key=lambda x: x.count("."))
    return list(ret)


def count_module_parameters(module, requires_grad: Optional[bool] = None):
    if requires_grad is None:
        return sum(p.numel() for p in module.parameters())
    else:
        return sum(
            p.numel() for p in module.parameters() if p.requires_grad == requires_grad
        )


def named_parameters_with_duplicates(
    module: nn.Module, **kwargs: Any
) -> List[Tuple[str, nn.Parameter]]:
    """
    This API is required as some modules overwrite `named_parameters()` but do not support
    `remove_duplicate`.
    """
    assert (
        "remove_duplicate" not in kwargs
    ), "_named_parameters_with_duplicates cannot be used with `remove_duplicate` argument."
    kwargs["remove_duplicate"] = False
    try:
        ret = list(module.named_parameters(**kwargs))
    except AssertionError as e:
        kwargs.pop("remove_duplicate")
        ret = list(module.named_parameters(**kwargs))
    return ret


def deepcopy_module(module: nn.Module, memo=None):
    """
    This is used to deepcopy a nn.Module with some special handling.
    """
    if memo is None:
        memo = {}
    # ProcessGroups are not deepcopyable, so add it to the memo
    for name, sub_module in module.named_modules():
        for attr_name, attr in sub_module.__dict__.items():
            if isinstance(attr, torch.distributed.ProcessGroup):
                memo[id(attr)] = attr

    return deepcopy(module, memo)
