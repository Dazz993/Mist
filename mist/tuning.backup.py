from dataclasses import dataclass
from typing import Callable, Sequence, List, Dict, Any, Optional
from copy import deepcopy

import numpy as np
import sympy as sp

from mist.symbols import global_symbol_manager as gsm


class Constant:
    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, "value") and name == "value":
            raise AttributeError("Cannot set value of ConstantInt")
        return super().__setattr__(name, value)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        choices_str = f", choices={self.choices}" if self.choices is not None else ""
        return f"{cls}({self.name}, {self.value}{choices_str})"


class ConstantBool(Constant):
    def __init__(self, name, value, choices=None):
        self.name = name
        self.value = value
        self.choices = choices

    def __bool__(self) -> bool:
        return self.value

    def __repr__(self) -> str:
        # Go to Constant.__repr__
        return super().__repr__()


class ConstantInt(int, Constant):
    def __new__(cls, name, value, choices=None):
        obj = super().__new__(cls, value)
        obj.name = name
        obj.value = value
        obj.choices = choices
        return obj

    def __repr__(self) -> str:
        # Go to Constant.__repr__
        return super(int, self).__repr__()


class ConstantFloat(float, Constant):
    def __new__(cls, name, value, choices=None):
        obj = super().__new__(cls, value)
        obj.name = name
        obj.value = value
        obj.choices = choices
        return obj

    def __repr__(self) -> str:
        # Go to Constant.__repr__
        return super(float, self).__repr__()


@dataclass
class Variable:
    name: str
    dtype: type = int
    default: Any = 0
    choices: Sequence[Any] = None

    def __post_init__(self):

        # Check if the inputs are valid
        assert self.choices is not None, "choices must be specified"
        assert isinstance(self.choices, Sequence), "choices must be a sequence"

        # Update symbol creation kwargs
        _kwargs = {}

        if self.dtype == int:
            _kwargs["integer"] = True
        elif self.dtype == float:
            _kwargs["real"] = True
        else:
            raise ValueError(f"Unknown dtype: {self.dtype}")

        if all(value > 0 for value in self.choices):
            _kwargs["positive"] = True
        elif all(value >= 0 for value in self.choices):
            _kwargs["nonnegative"] = True

        # Create symbol and map it in symbol manager
        self.symbol = gsm.symbols(self.name, **_kwargs)
        gsm.map(self.symbol, self.default)

    def _identity(self):
        return self.symbol

    def __hash__(self) -> int:
        return hash(self._identity())


@dataclass
class ListOfVariables:
    name: str
    dtype: type = int
    length: int = 0
    default: Sequence[Any] = None
    choices: Sequence[Any] = None

    def __post_init__(self):

        # Check if the inputs are valid
        assert (
            len(self.default) == self.length
        ), f"length of default must match, {self.default} vs {self.length}"

        # Create all variables
        self.variables: List[Variable] = []
        for i in range(self.length):
            self.variables.append(
                Variable(f"{self.name}_{i}", self.dtype, self.default[i], self.choices)
            )

        # Get all symbols
        self.symbols = [var.symbol for var in self.variables]

    def __getitem__(self, index):
        return self.variables[index]


class MistConfig:
    def __init__(self, name: str):

        # Config Name (e.g. pp_group_1)
        self.name = name

    def register(self, cls, name: str, *args, **kwargs):
        name = f"{self.name}_{name}" if self.name else name
        return cls(name, *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def get(self, name: str):
        v = getattr(self, name)
        if isinstance(v, Constant):
            return v.value
        elif isinstance(v, Variable):
            return v.symbol
        else:
            return v


class MistIntraOpBaseConfig(MistConfig):
    def __init__(self, name: str, parent: MistConfig, n_layers: int = 1):
        super().__init__(name)

        self.parent = parent
        self.load_from_parent()

        self.n_layers = n_layers

        # Parallelism
        self.tp_size = 1
        self.dp_size = 1

        # Parallelism Redundancy
        self.opt_rdn_sharding_size = 1
        self.grad_rdn_sharding_size = 1
        self.wgt_fwd_rdn_sharding_size = 1
        self.wgt_bwd_rdn_sharding_size = 1
        self.tp_act_rdn_sharding_size = 1

        # Memory Optimization
        self.ckpt_block_size = 1
        self.ckpt_block_num = 0
        self.opt_ofd_ratio = 0
        self.grad_ofd_ratio = 0
        self.wgt_fwd_ofd_ratio = 0
        self.wgt_bwd_ofd_ratio = 0

        # Optimizer step
        self.group_size_of_grad_sync_and_opt = 1

        # CPU Compute
        self.cpu_grad_accumu = False
        self.cpu_opt_step = False

    def satisfy_constraints(self, mapping):
        symbol2variable = self.symbol2variable()
        variable2value = {
            symbol2variable[symbol]: value for symbol, value in mapping.items()
        }
        # Batch size constraint
        batch_size = self.batch_size
        dp_size = self.dp_size
        if batch_size in variable2value and dp_size in variable2value:
            batch_size_value = variable2value[batch_size]
            dp_size_value = variable2value[dp_size]
            if (
                batch_size_value % dp_size_value != 0
                or batch_size_value < dp_size_value
            ):
                return False

        return True

    def load_from_parent(self):
        if hasattr(self.parent, "global_batch_size"):
            self.global_batch_size = self.parent.global_batch_size
        if hasattr(self.parent, "batch_size"):
            self.batch_size = self.parent.batch_size
        if hasattr(self.parent, "gpu_cpu_bandwidth"):
            self.gpu_cpu_bandwidth = self.parent.gpu_cpu_bandwidth
        if hasattr(self.parent, "gpu_gpu_bandwidth"):
            self.gpu_gpu_bandwidth = self.parent.gpu_gpu_bandwidth

    def rename(self, name):
        self.name = name
        new_dict = {}
        for k, v in self.__dict__.items():
            if k.startswith(f"{self.name}_"):
                new_k_name = k.replace(f"{self.name}_", f"{name}_")
                new_dict[new_k_name] = v
                self.__dict__.pop(k)
        self.__dict__.update(new_dict)
        return self

    def variables(self):
        ret = []
        for v in self.__dict__.values():
            if isinstance(v, Variable):
                ret.append(v)
        return ret

    def symbol2variable(self):
        ret = {}
        for v in self.variables():
            ret[v.symbol] = v
        return ret


class MistInterOpBaseConfig(MistConfig):
    intra_op_config_cls = MistIntraOpBaseConfig

    def __init__(
        self,
        name: str = None,
        n_gpus=1,
        gpu_cpu_bandwidth=3e9,
        gpu_gpu_bandwidth=3e9,
        global_batch_size=4096,
        batch_size=4,
        n_layers=4,
    ):
        super().__init__(name)

        # Hardware spec
        self.n_gpus = n_gpus
        self.gpu_cpu_bandwidth = gpu_cpu_bandwidth
        self.gpu_gpu_bandwidth = gpu_gpu_bandwidth

        # Train
        self.global_batch_size = global_batch_size
        self.batch_size = batch_size

        # Model
        self.n_layers = n_layers

        # Parallelism
        self.pp_size = 1
        self.layers_per_pp_group = [self.n_layers]
        self.gpus_per_pp_group = [self.n_gpus]

        self.create_configs_for_each_layer()

    def create_configs_for_each_layer(self):

        layers_per_pp_group = self.layers_per_pp_group

        # Config for each layer
        preprocessing_config = self.intra_op_config_cls("preprocessing", self)
        postprocessing_config = self.intra_op_config_cls("postprocessing", self)
        # The same layer in the same group shares the same config
        config_for_each_group = [
            self.intra_op_config_cls(f"pp_group_{i}", self, layers_per_pp_group[i])
            for i in range(len(layers_per_pp_group))
        ]

        ret = []
        # First group contains pre-process
        ret += [preprocessing_config] + [
            config_for_each_group[0]
        ] * layers_per_pp_group[0]

        # Middle groups
        for i in range(1, len(layers_per_pp_group) - 1):
            ret += [config_for_each_group[i]] * layers_per_pp_group[i]

        # Last group contains post-process
        ret += [config_for_each_group[-1]] * layers_per_pp_group[-1] + [
            postprocessing_config
        ]

        self.config_for_each_layer = ret
        return ret


class MistIntraOpConfig(MistIntraOpBaseConfig):
    def __init__(self, name: str, parent: MistConfig, n_layers: int = 1):
        super().__init__(name, parent)

        self.parent = parent
        self.load_from_parent()

        self.n_layers = n_layers

        # Parallelism
        self.tp_size = self.register(
            Variable, "tp_size", int, default=1, choices=[1, 2, 4, 8, 16, 32]
        )
        # self.dp_size = self.register(
        #     Variable, "dp_size", int, default=1, choices=[1, 2, 4, 8, 16, 32]
        # )
        self.dp_size = self.register(Variable, "dp_size", int, default=1, choices=[1])

        # Parallelism Redundancy
        self.opt_rdn_sharding_size = self.register(
            Variable, "opt_rdn_sharding_size", int, default=1, choices=[1, 2, 4, 8, 16]
        )
        self.grad_rdn_sharding_size = self.register(
            Variable, "grad_rdn_sharding_size", int, default=1, choices=[1, 2, 4, 8, 16]
        )
        self.wgt_fwd_rdn_sharding_size = self.register(
            Variable,
            "wgt_fwd_rdn_sharding_size",
            int,
            default=1,
            choices=[1, 2, 4, 8, 16],
        )
        self.wgt_bwd_rdn_sharding_size = self.register(
            Variable,
            "wgt_bwd_rdn_sharding_size",
            int,
            default=1,
            choices=[1, 2, 4, 8, 16],
        )
        self.tp_act_rdn_sharding_size = self.register(
            Variable,
            "tp_act_rdn_sharding_size",
            int,
            default=1,
            choices=[1, 2, 4, 8, 16],
        )

        # Memory Optimization
        # self.ckpt_block_size = self.register(
        #     Variable, "ckpt_block_size", int, default=1, choices=[1, 2, 3, 4]
        # )
        self.ckpt_block_size = self.register(ConstantInt, "ckpt_block_size", 1)
        self.ckpt_block_num = self.register(
            Variable, "ckpt_block_num", int, default=1, choices=range(0, n_layers + 1)
        )
        self.opt_ofd_ratio = self.register(
            Variable,
            "opt_ofd_ratio",
            float,
            default=0.0,
            choices=np.arange(0.0, 1.0 + 0.1, 0.1).tolist(),
        )
        self.grad_ofd_ratio = self.register(
            Variable,
            "grad_ofd_ratio",
            float,
            default=0.0,
            choices=np.arange(0.0, 1.0 + 0.1, 0.1).tolist(),
        )
        self.wgt_fwd_ofd_ratio = self.register(
            Variable,
            "wgt_fwd_ofd_ratio",
            float,
            default=0.0,
            choices=np.arange(0.0, 1.0 + 0.1, 0.1).tolist(),
        )
        self.wgt_bwd_ofd_ratio = self.register(
            Variable,
            "wgt_bwd_ofd_ratio",
            float,
            default=0.0,
            choices=np.arange(0.0, 1.0 + 0.1, 0.1).tolist(),
        )

        # Optimizer step
        self.group_size_of_grad_sync_and_opt = self.register(
            Variable,
            "group_size_of_grad_sync_and_opt",
            int,
            default=1,
            choices=[1, 2, 3, 4],
        )

        # CPU Compute
        self.cpu_grad_accumu = self.register(
            ConstantBool, "cpu_grad_accumu", False, choices=[True, False]
        )
        self.cpu_opt_step = self.register(
            ConstantBool, "cpu_opt_step", False, choices=[True, False]
        )


class MistInterOpConfig(MistInterOpBaseConfig):
    intra_op_config_cls = MistIntraOpConfig

    def __init__(self, name: str = None):
        super().__init__(name)

        # Hardware spec
        self.gpu_cpu_bandwidth = self.register(ConstantFloat, "gpu_cpu_bandwidth", 3e9)
        self.gpu_gpu_bandwidth = self.register(ConstantFloat, "gpu_gpu_bandwidth", 3e9)

        # Model
        self.n_layers = 4

        # Hardware
        self.num_gpus = self.register(ConstantInt, "num_gpus", 32)
        # TODO(zhanda): add bandwidth information

        # Train
        self.global_batch_size = self.register(ConstantInt, "global_batch_size", 4096)
        self.batch_size = self.register(
            Variable, "batch_size", int, default=4, choices=[4]
        )
        # self.batch_size = self.register(
        #     Variable, "batch_size", int, default=32, choices=range(1, 32 + 1)
        # )

        # Parallelism
        self.pp_size = 1
        self.layers_per_pp_group = [self.n_layers]
        self.gpus_per_pp_group = [self.n_gpus]

        self.create_configs_for_each_layer()


if __name__ == "__main__":
    num_gpus = ConstantInt("num_gpus", 128)
    tp_size = Variable("tp_size", int, default=1, choices=[1, 2, 4, 8, 16, 32])

    layers_per_pp_group = ListOfVariables(
        "layers_per_pp_group",
        int,
        length=4,
        default=[1, 1, 1, 1],
        choices=[1, 2, 4, 8, 16, 32],
    )

    config = MistInterOpConfig(f"model", 16)
