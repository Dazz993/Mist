import inspect
from typing import Dict, Tuple, Any, Set, List, Optional, Union, Callable, NamedTuple
from importlib import import_module
from functools import wraps

import torch
import torch.nn as nn

from mist.logger import get_logger

logger = get_logger(__name__)

# ========================================================================
# Original ops related
# ========================================================================

DOMAINS = {
    torch,
    torch.nn.functional,
    torch.nn.init,
    torch._C._nn,
    torch.Tensor,
    torch._C._TensorBase,
    torch.distributed,
    torch.jit,
    torch.autograd,
}
ORI_TORCH_OPS: Dict[Tuple[Callable, str], Callable] = {}
for domain in DOMAINS:
    if inspect.isclass(domain):
        for cls in reversed(domain.mro()[1:]):
            for name, op in cls.__dict__.items():
                ORI_TORCH_OPS[(domain, name)] = op

    for name, op in domain.__dict__.items():
        ORI_TORCH_OPS[(domain, name)] = op


def get_ori_torch_op(module: Callable, name: str) -> Callable:
    if (module, name) in ORI_TORCH_OPS:
        return ORI_TORCH_OPS[(module, name)]
    else:
        return getattr(module, name)
    # raise ValueError(f"{module}.{name} is not added to ORI_TORCH_OPS, please add it.")


# ========================================================================
# Patchers related
# ========================================================================


PATCHERS = []


def get_root_patcher():
    assert len(PATCHERS) > 0, "No patcher found"
    return PATCHERS[0]


def get_patchers():
    return PATCHERS


class PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        raise NotImplementedError()


class PatchedFnSetItem(PatchedFn):
    def revert(self):
        self.frame_dict[self.fn_name] = self.orig_fn


class PatchedFnDel(PatchedFn):
    def revert(self):
        del self.frame_dict[self.fn_name]


class PatchedFnSetAttr(PatchedFn):
    def revert(self):
        setattr(self.frame_dict, self.fn_name, self.orig_fn)


class PatchedFnDelAttr(PatchedFn):
    def revert(self):
        delattr(self.frame_dict, self.fn_name)


class MistPatcher:
    def __init__(self):
        self.patches_made = []
        self.ori_items: Dict[Tuple[int, str], Any] = {}
        self.ori_attrs: Dict[Tuple[Any, str], Any] = {}

        global PATCHERS
        PATCHERS.append(self)

    def patch_setitem(self, frame_dict: Dict[str, Any], name: str, new_fn: Callable):
        if name not in frame_dict:
            self.patches_made.append(PatchedFnDel(frame_dict, name, None))
        else:
            self.ori_items[(id(frame_dict), name)] = frame_dict[name]
            self.patches_made.append(
                PatchedFnSetItem(frame_dict, name, frame_dict[name])
            )
        frame_dict[name] = new_fn
        # logger.debug(f"Patch dict key {name}, value: {new_fn}")

    def patch_setattr(self, parent: Any, name: str, new_fn: Callable):
        if not hasattr(parent, name):
            self.patches_made.append(PatchedFnDelAttr(parent, name, None))
        else:
            # [2023/09/20 Zhanda]: This is because if we directly use `getattr(parent, name)`,
            # we may get `method of class` instead of `classmethod`. This will cause error
            # because the former is bound to the class, while the latter is not.
            if name in getattr(parent, "__dict__", {}):
                ori_attr = parent.__dict__[name]
            else:
                ori_attr = getattr(parent, name)
            self.ori_attrs[(parent, name)] = ori_attr
            self.patches_made.append(PatchedFnSetAttr(parent, name, ori_attr))
        setattr(parent, name, new_fn)
        # logger.debug(f"Patch {parent}.{name} with {new_fn}")

    def get_ori_item(self, frame_dict: Dict[str, Any], name: str):
        if (id(frame_dict), name) in self.ori_items:
            return self.ori_items[(id(frame_dict), name)]
        else:
            return frame_dict[name]

    def get_ori_attr(self, parent: Any, name: str):
        if (parent, name) in self.ori_attrs:
            return self.ori_attrs[(parent, name)]
        else:
            return getattr(parent, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        while self.patches_made:
            self.patches_made.pop().revert()

        global PATCHERS
        PATCHERS.pop()


patcher = MistPatcher()


# ========================================================================
# RevertPatcher related
# ========================================================================
def _fn_wrapper(orig_fn):
    """A helper function to solve the issue that some builtin can not assign new attribute"""

    @wraps(orig_fn)
    def wrapper(*args, **kwargs):
        return orig_fn(*args, **kwargs)

    return wrapper


class MistRevertPatcher(MistPatcher):
    """
    A patcher that reverts the patches made by a given patcher
    """

    def __init__(self, patcher_to_revert=None):
        super().__init__()

        patcher_to_revert = patcher_to_revert or get_root_patcher()
        self._patcher_to_revert = patcher_to_revert

        for patch in patcher_to_revert.patches_made:
            frame_dict, fn_name, orig_fn = (
                patch.frame_dict,
                patch.fn_name,
                patch.orig_fn,
            )
            # This is not 100% correct because we don't have
            # patch_delitem, patch_delattr, etc.
            if isinstance(patch, (PatchedFnSetItem, PatchedFnDel)):
                try:
                    self.patch_setitem(frame_dict, fn_name, orig_fn)
                except Exception as e:
                    self.patch_setitem(frame_dict, fn_name, _fn_wrapper(orig_fn))
            elif isinstance(patch, (PatchedFnSetAttr, PatchedFnDelAttr)):
                try:
                    self.patch_setattr(frame_dict, fn_name, orig_fn)
                except Exception as e:
                    self.patch_setattr(frame_dict, fn_name, _fn_wrapper(orig_fn))
            else:
                raise ValueError(f"Unknown patch type: {patch}")


def reset_mist_patcher():
    assert len(PATCHERS) in [
        0,
        1,
        2,
    ], f"Found {len(PATCHERS)} patchers, only 0, 1, 2 are allowed."
    if len(PATCHERS) == 0:
        return
    elif len(PATCHERS) == 1:
        revert_patcher = MistRevertPatcher(patcher_to_revert=get_root_patcher())
    else:
        non_root_patcher = PATCHERS[-1]
        assert isinstance(non_root_patcher, MistRevertPatcher), (
            f"Found two patchers where the last one is not a MistRevertPatcher: "
            f"{PATCHERS}"
        )


# ========================================================================
# Overriden torch ops related
# ========================================================================


def override_attr(
    parent: Callable,
    name: str,
    attr: Any,
    allow_duplicate: bool = True,
):
    patcher = get_root_patcher()

    # Check if the attribute has been overriden
    if (parent, name) in patcher.ori_attrs and not allow_duplicate:
        raise ValueError(f"{parent}.{name} has been overriden.")

    # If the attribute exists, try to also wrap it or do post-processing
    if (parent, name) in patcher.ori_attrs or hasattr(parent, name):
        ori_attr = patcher.get_ori_attr(parent, name)
        try:
            attr = wraps(ori_attr)(attr)
        except Exception as e:
            pass

        patcher.patch_setattr(parent, name, attr)

        # Wraps the __module__ to deal with torch.nn.functional and torch._C._nn
        module_name = getattr(ori_attr, "__module__", None)
        if module_name is not None and module_name.startswith("torch"):
            try:
                module = import_module(module_name)
            except Exception as e:
                # Multi-level import
                _parent_module_name, _, _module_main_name = module_name.rpartition(".")
                _parent_module = import_module(_parent_module_name)
                module = import_module(_module_main_name, package=_parent_module)
            if hasattr(module, name) and (module, name) not in patcher.ori_attrs:
                override_attr(module, name, attr, allow_duplicate=allow_duplicate)

    else:
        # If the attribute does not exist, we just patch it
        patcher.patch_setattr(parent, name, attr)

    return attr


def override_item(
    frame_dict: Dict[str, Any],
    name: str,
    item: Any,
    allow_duplicate: bool = True,
):
    patcher = get_root_patcher()

    # Check if the item has been overriden
    if (id(frame_dict), name) in patcher.ori_items and not allow_duplicate:
        raise ValueError(f"{name} has been overriden.")

    # If the item exists, try to also wrap it
    if (id(frame_dict), name) in patcher.ori_items or name in frame_dict:
        ori_item = patcher.get_ori_item(frame_dict, name)
        try:
            item = wraps(ori_item)(item)
        except Exception as e:
            pass

    patcher.patch_setitem(frame_dict, name, item)

    return item


def register_overriden_func(
    parent: Callable,
    name: str,
    allow_duplicate: bool = True,
):
    def decorator(func):
        return override_attr(parent, name, func, allow_duplicate=allow_duplicate)

    return decorator
