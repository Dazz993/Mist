from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Sequence, List, Dict, Any, Optional, Union
from numbers import Number
import numpy as np
import sympy as sp

from mist import global_symbol_manager as gsm

import torch
import torch.distributed
import torch.distributed.elastic
import torch.distributed.elastic.rendezvous

import torch.distributed.elastic.rendezvous.c10d_rendezvous_backend

Var = Union[sp.Basic, Number]
VarInt = Union[sp.Basic, int]
VarFloat = Union[sp.Basic, float]
VarBool = Union[sp.Basic, bool]


@dataclass
class MistLayerOptimizationStrategy:
    # Hardware
    nnodes: Optional[VarInt] = gsm.symbols("m", 1, integer=True, positive=True)
    nproc_per_ndoe: Optional[VarInt] = gsm.symbols("n", 1, integer=True, positive=True)

    # Training
    per_device_batch_size: Optional[VarInt] = gsm.symbols(
        "b", 1, integer=True, positive=True
    )

    # Parallelism
    dp_size: Optional[VarInt] = gsm.symbols("dp", 1, integer=True, positive=True)
    tp_size: Optional[VarInt] = gsm.symbols("tp", 1, integer=True, positive=True)
    ws_size: Optional[VarInt] = gsm.symbols("ws", 1, integer=True, positive=True)
    gs_size: Optional[VarInt] = gsm.symbols("gs", 1, integer=True, positive=True)
    os_size: Optional[VarInt] = gsm.symbols("os", 1, integer=True, positive=True)

    # Memory optimization
    ckpt_option: Optional[VarBool] = gsm.symbols("ckpt", 0, integer=True)
    wo_ratio: Optional[VarFloat] = gsm.symbols("wo", 0.0, real=True, nonnegative=True)
    go_ratio: Optional[VarFloat] = gsm.symbols("go", 0.0, real=True, nonnegative=True)
    oo_ratio: Optional[VarFloat] = gsm.symbols("oo", 0.0, real=True, nonnegative=True)
