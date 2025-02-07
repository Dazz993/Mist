from functools import wraps

import torch
from torch.fx._symbolic_trace import _Patcher

PATCHERS = []


def get_root_patcher():
    assert len(PATCHERS) > 0, "No patcher found"
    return PATCHERS[0]


def get_patchers():
    return PATCHERS


_ori_patcher_init = _Patcher.__init__
_ori_patcher_exit = _Patcher.__exit__


class _Overloaded_Patcher:
    """
    A patcher that overrides the default initialization of the _Patcher class

    The main difference is that
    (1) we keep a global variable of the patcher so that we can easily access
        it and control the patching process.
    """

    def __init__(self):
        self.patches_made = []
        self.visited = set()

        global PATCHERS
        PATCHERS.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ori_patcher_exit(self, exc_type, exc_val, exc_tb)
        global PATCHERS
        PATCHERS.pop()


_Patcher.__init__ = _Overloaded_Patcher.__init__
_Patcher.__exit__ = _Overloaded_Patcher.__exit__


def _fn_wrapper(orig_fn):
    """A helper function to solve the issue that some builtin can not assign new attribute"""

    @wraps(orig_fn)
    def wrapper(*args, **kwargs):
        return orig_fn(*args, **kwargs)

    return wrapper


class RevertPatcher(_Patcher):
    """
    A patcher that reverts the patches made by a given patcher
    """

    def __init__(self, patcher_to_revert):
        super().__init__()

        self._patcher_to_revert = patcher_to_revert

        for patch in patcher_to_revert.patches_made:
            frame_dict, fn_name, orig_fn = (
                patch.frame_dict,
                patch.fn_name,
                patch.orig_fn,
            )
            if isinstance(frame_dict, dict):
                self.patch(frame_dict, fn_name, _fn_wrapper(orig_fn))
            else:
                self.patch_method(frame_dict, fn_name, _fn_wrapper(orig_fn))
