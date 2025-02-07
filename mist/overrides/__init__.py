import os
from importlib import import_module
from pathlib import Path

from .base import (
    ORI_TORCH_OPS,
    get_ori_torch_op,
    get_root_patcher,
    get_patchers,
    override_attr,
    override_item,
    register_overriden_func,
    MistRevertPatcher,
)

path = Path(__file__).parent.absolute()
files = [
    f
    for f in os.listdir(path)
    if os.path.isfile(os.path.join(path, f)) and f not in {"__init__.py", "base.py"}
]
for file in files:
    mod = import_module(f".{file.split('.')[0]}", package="mist.overrides")
