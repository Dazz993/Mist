from functools import wraps

import sympy as sp
import torch
from torch.types import _int

from mist.overrides import override_attr

# This is to fix the issue that torch.Size only accepts int as input
# There are there methods to fix this:
# 1. The best way: change torch/csrc/Size.cpp - THPSize_pynew.
#    However, it requires recompiling torch.
# 2. Workaround 1: override torch.SymInt since torch.Size accepts torch.SymInt.
#    --> override_attr(torch, "SymInt", (sp.Basic, torch.SymInt))
#    However, this may cause some errors for type checking.
# 3. Workaround 2: make sp.Basic a subclass of torch.SymInt.
#    This is not a recommended way and should be forbidden. However, it works.


SymInt = type("SymInt", (), {})
override_attr(torch, "SymInt", SymInt)

new_sp_bases = (sp.Basic.__bases__) + (torch.SymInt,)
# override_attr(sp.Basic, "__bases__", new_sp_bases)
sp.Basic.__bases__ = new_sp_bases
