from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set

import sympy as sp
import torch
import torch.fx

from mist.overrides import override_attr
from mist.distributed.overrides import MistProcessGroup

BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    # sp.Basic,
    # MistProcessGroup,
]
base_types = BaseArgumentTypes.__args__  # type: ignore[attr-defined]

override_attr(torch.fx.node, "base_types", base_types)
override_attr(torch.fx.proxy, "base_types", base_types)
