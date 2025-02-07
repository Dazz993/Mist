from __future__ import annotations
import argparse
import os
from enum import Enum, auto
from functools import partial, lru_cache
from itertools import accumulate, chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Set,
    Sequence,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
from torch.distributed import ProcessGroup
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import flat_param as torch_fsdp_flat_param
from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream
from torch.distributed.fsdp._runtime_utils import (
    _div_if_needed,
    _check_grad_to_accumulate,
)
from torch.distributed.utils import _p_assert

from mist.logger import get_logger, update_logger_formatter_for_rank
from mist.re_swap_manager.mem_buffer_pool import (
    allocate_memory_buffer,
    get_memory_buffer,
    get_ring_memory_buffer,
    MemoryBuffer,
    RingMemoryBuffer,
)
from mist.re_swap_manager.swap_backend import TensorSwapHandle, swap_
from mist.utils.module import named_parameters_with_duplicates
from mist.utils.shard import shard_tensor, unshard_tensor
from mist.utils.storage import _free_storage

logger = get_logger(__name__)

"""
For parameters:
1. flat_param will be created and be used as the root data.
2. flat_param either points to _full_param or _local_shard.
3. _full_param can be freed. During unsharding, if the buffer is inputted,
   the buffer will be temporarily used as the storage of the unsharded flat_param.
4. _local_shard will always be maintained as the local shard of the flat_param.
   If it has been swapped out, then there will be a buffer to store the cuda data.
5. after FlatParam/Handle is initialized, the module.parameters() will be empty.
   params is still in the original place but as a tensor instead of nn.Parameter.
   This is to say, the params are not in `module._parameters` anymore.
   
For grads:
1. grads will be accumulated to param.grad (and then accumulated to flat_param.grad)
   because params will be view of the flat_param. If the buffer is inputted,
   the buffer will be temporarily used as the storage of the grads.
2. however, after grads are accumulated (which means in the full_backward_hook),
   the grads will be copied to the flat_grad, `_saved_grad`.
3. reduce_scatter will be called on the `_saved_grad`.
4. all other grads (flat_param.grad, and param.grad for param in _params) will be
   deallocated.
"""

DTYPE_RGRAD = Tuple[torch.dtype, bool]
ALIGNED_SIZE = 8  # bytes

_FLATPARAM_USE_UNSAFE_SETATTR = "FLATPARAM_USE_UNSAFE_SETATTR"
_FLATTENED_TO_FLATPARAM = "FLATTENED_TO_FLATPARAM"


class HandleTrainingState(Enum):
    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()


class HandleShardingStrategy(Enum):
    NO_SHARD = auto()
    OPT_ONLY = ZeRO_1 = auto()
    OPT_AND_GRAD = ZeRO_2 = auto()
    FULL_SHARD = ZeRO_3 = auto()

    def from_shard_flags(
        shard_weights: bool, shard_grads: bool, shard_opts: bool
    ) -> HandleShardingStrategy:
        """Return the sharding strategy given the shard flags"""
        if shard_opts and shard_grads and shard_weights:
            return HandleShardingStrategy.FULL_SHARD
        elif shard_opts and shard_grads:
            return HandleShardingStrategy.OPT_AND_GRAD
        elif shard_opts:
            return HandleShardingStrategy.OPT_ONLY
        else:
            return HandleShardingStrategy.NO_SHARD


GRAD_SHARD_HANDLE_STRATEGIES = (
    HandleShardingStrategy.FULL_SHARD,
    HandleShardingStrategy.OPT_AND_GRAD,
)


def _set_flattened_to_flatparam(tensor: torch.Tensor) -> None:
    setattr(tensor, _FLATTENED_TO_FLATPARAM, True)


# NOTE: These are hacks to bypass `nn.Module.__setattr__` checks.
def _unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    module._parameters[param_name] = param
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, param)


def _unsafe_setattr_tensor(
    module: nn.Module, param_name: str, tensor: torch.Tensor
) -> None:
    module._parameters.pop(param_name, None)
    # This bypasses any overrides in case `module` is an instance of an
    # `nn.Module` subclass
    super(nn.Module, module).__setattr__(param_name, tensor)


def _safe_setattr_tensor_or_param(
    module: nn.Module,
    param_name: str,
    tensor_or_param: Union[torch.Tensor, nn.Parameter],
):
    # Call `delattr()` and `setattr()` to go through `nn.Module` checks
    if hasattr(module, param_name):
        delattr(module, param_name)
    setattr(module, param_name, tensor_or_param)


class ParamInfo(NamedTuple):
    """Information for an original parameter."""

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str


class SharedParamInfo(NamedTuple):
    """
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    """

    param_name: str  # unprefixed
    module: nn.Module
    module_name: str
    prim_param_name: str  # unprefixed
    prim_module: nn.Module
    prim_module_name: str


class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""

    in_shard: bool
    # Use to index into the sharded flat parameter, e.g.
    # `flat_param[offset_in_shard : offset_in_shard + numel_in_shard]`
    offset_in_shard: Optional[int]
    numel_in_shard: Optional[int]
    # Use to get part of the parameter in the local shard from a flattened
    # version of the unsharded parameter, e.g.
    # `param.flatten()[intra_param_start_idx : intra_param_end_idx + 1]`
    intra_param_start_idx: Optional[int]
    intra_param_end_idx: Optional[int]  # inclusive


class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flat parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original parameter.
    """

    param_names: Tuple[str, ...]
    param_shapes: Tuple[torch.Size, ...]
    param_numels: Tuple[int, ...]
    param_offsets: Tuple[Tuple[int, int], ...]


class _FlatParameterMeta(_ParameterMeta):
    # Make `isinstance(t, FlatParameter)` return True for custom tensor
    # instances that have the _is_flat_param flag for BC
    def __instancecheck__(self, instance):
        # NB: do NOT test the super implementation
        return isinstance(instance, torch.Tensor) and getattr(
            instance, "_is_flat_param", False
        )


class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    _unsharded_size: torch.Size
    _unsharded_numel: int
    _sharded_size: torch.Size
    _sharded_numel: int
    _num_params: int
    _param_infos: Tuple[ParamInfo, ...]
    _shapes: Tuple[torch.Size, ...]
    _fqns: Tuple[str, ...]
    _numels: Tuple[int, ...]
    _shard_param_infos: Tuple[_ShardParamInfo, ...]
    _shared_param_infos: Tuple[SharedParamInfo, ...]
    _module: nn.Module
    _modules: Set[nn.Module]
    _full_param: torch.Tensor
    _local_shard: torch.Tensor
    _saved_grad: torch.Tensor
    _params: Optional[List[nn.Parameter]]
    _shared_params: Optional[List[nn.Parameter]]

    def __new__(cls, data=None, requires_grad=True):
        assert cls is FlatParameter, "subclasses FlatParameter not supported"
        r = nn.Parameter.__new__(nn.Parameter, data, requires_grad)  # type: ignore[call-arg]
        r._is_flat_param = True  # type: ignore[attr-defined]
        return r

    # NB: This is not a regular method, because FlatParameters are not actually
    # instances of this class (see __new__ above).  So you must indirectly
    # call this directly through the classmethod.
    @classmethod
    def _init_metadata(
        cls,
        self,
        param_infos: List[ParamInfo],
        numels: List[int],
        shapes: List[torch.Size],
        fqns: List[str],
        shared_param_infos: List[SharedParamInfo],
        params: Optional[List[nn.Parameter]],
        shared_params: Optional[List[nn.Parameter]],
    ) -> None:
        """
        Initializes attributes holding metadata about the original parameters
        comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """
        assert len(param_infos) == len(shapes)
        assert len(param_infos) == len(fqns)
        self._num_params = len(param_infos)
        self._param_infos = param_infos
        self._shapes = shapes
        self._fqns = fqns
        self._numels = tuple(numels)
        assert len(self._numels) == self._num_params

        self._shared_param_infos = tuple(shared_param_infos)
        self._modules = {pi.module for pi in self._param_infos}.union(
            {spi.module for spi in self._shared_param_infos}
        )
        assert (params is None) == (shared_params is None)
        # In our implementation, params should be None since we don't
        # enable ``use_orig_params``.
        if params is not None:
            assert shared_params is not None and len(shared_params) == len(
                shared_param_infos
            )
            self._params = list(params)
            self._shared_params = shared_params
            # Mark the original parameters to avoid flattening them into
            # another `FlatParameter` during recursive construction
            for param in chain(self._params, self._shared_params):
                _set_flattened_to_flatparam(param)
        else:
            self._params = None
            self._shared_params = None
        self._unsharded_size = self.size()
        self._unsharded_numel = self.size().numel()
        _set_flattened_to_flatparam(self)


class FlatParamHandle:
    def __init__(
        self,
        params: Sequence[Union[nn.Parameter, torch.Tensor]],
        module: nn.Module,
        device: torch.device,
        sharding_strategy: HandleShardingStrategy,
        process_group: dist.ProcessGroup,
        all_gather_process_group: Optional[dist.ProcessGroup] = None,
        reduce_scatter_process_group: Optional[dist.ProcessGroup] = None,
        param_swap_ratio: float = 0.0,
        grad_swap_ratio: float = 0.0,
    ):
        # logger.debug(f"[0] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")

        params = list(params)
        if len(params) == 0:
            raise ValueError(f"Cannot construct a FlatParamHandle with no parameters.")

        self.module = module
        self.device = device
        self.sharding_strategy = sharding_strategy
        self.training_state = HandleTrainingState.IDLE
        self.process_group = process_group
        self.rank = process_group.rank()
        self.world_size = process_group.size()
        self.all_gather_process_group = all_gather_process_group or process_group
        self.reduce_scatter_process_group = (
            reduce_scatter_process_group or process_group
        )
        if self.all_gather_process_group.size() != self.world_size:
            raise ValueError("all_gather_process_group.size() != process_group.size()")
        if self.reduce_scatter_process_group.size() != self.world_size:
            raise ValueError(
                "reduce_scatter_process_group.size() != process_group.size()"
            )
        self.param_swap_ratio = param_swap_ratio
        self.grad_swap_ratio = grad_swap_ratio
        self._gradient_predivide_factor = (
            default_hooks.DefaultState._get_gradient_predivide_factor(self.world_size)
        )
        self._gradient_postdivide_factor = (
            self.world_size / self._gradient_predivide_factor
        )

        # ##############################
        # Init flat param and metadata #
        # ##############################
        self._init_setattr_fns()
        # logger.debug(f"[0-1] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
        self._init_flat_param_and_metadata(params, module, aligned_numel=0)
        self.flat_param._sharding_strategy = sharding_strategy
        # logger.debug(f"[0-2] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
        self._use_unsharded_views(as_params=False)
        # logger.debug(f"[0-3] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")

        # ##########################
        # Init sharding attributes #
        # ##########################
        flat_param = self.flat_param
        # _full numel is the alias of the _unsharded_numel
        self._full_numel = self._unsharded_numel = flat_param._unsharded_numel
        # _sharded_numel is the numel of the sharded flat param
        self._sharded_numel = self._unsharded_numel // self.world_size
        # _sharded_numel is full or sharded depending on the sharding strategy.
        self._sharded_weight_numel = (
            self._sharded_numel
            if self.uses_weight_sharding_strategy
            else self._unsharded_numel
        )
        self._sharded_grad_numel = (
            self._sharded_numel
            if self.uses_grad_sharding_strategy
            else self._unsharded_numel
        )

        # ##############
        # Init Buffers #
        # ##############
        self._use_memory_buffer = True
        self._partial_weight_buffer: Optional[MemoryBuffer] = None
        self._partial_grad_buffer: Optional[MemoryBuffer] = None
        self._curr_occupied_full_weights_buffer: Optional[MemoryBuffer] = None
        self._curr_occupied_full_grads_buffer: Optional[MemoryBuffer] = None
        if self._use_memory_buffer:
            # logger.debug(f"[1] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
            self._init_partial_memory_buffers()
            # 1. Replace the full params with the buffers (either the full ring buffer or the partial buffer)
            # to avoid extra memory allocation. This function would replace the flat_param._full_param.
            # 2. However, we want the flat_param.data be changed as well. So we need to call the _use_unsharded_views.
            # logger.debug(f"[2] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
            self.alloc_full_weights(force=True)
            # logger.debug(f"[3] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
            self._use_unsharded_views(as_params=False)
            # logger.debug(f"[3-1] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")

        # ###############
        # Init sharding #
        # ###############
        self.shard_and_init_flat_param_attibutes()

        # logger.debug(f"[4] Memory Allocated: {torch.cuda.memory_allocated() // 1024 ** 2} MB")

        # ##############
        # Post process #
        # ##############
        self.dealloc_full_weights()

    def _init_setattr_fns(self):
        use_unsafe_setattr = os.environ.get(_FLATPARAM_USE_UNSAFE_SETATTR, "") == "1"
        self._setattr_tensor: Callable[[nn.Module, str, torch.Tensor], None]
        self._setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
        if use_unsafe_setattr:
            self._setattr_tensor = _unsafe_setattr_tensor
            self._setattr_param = _unsafe_setattr_param
        else:
            self._setattr_tensor = _safe_setattr_tensor_or_param
            self._setattr_param = _safe_setattr_tensor_or_param

    def _init_flat_param_and_metadata(
        self,
        params: List[Union[torch.Tensor, nn.Parameter]],
        module: nn.Module,
        aligned_numel: int,
    ) -> None:
        if len(params) == 0:
            raise ValueError("Expects non-empty `params`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects `aligned_numel` to be non-negative but got {aligned_numel}"
            )
        if aligned_numel != 0:
            raise NotImplementedError(
                "Aligned padding is not yet supported in FlatParamHandle"
            )

        (
            dtype,
            flat_param_requires_grad,
            device,
        ) = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        # Full numel
        _total_numel = sum(p.numel() for p in params)
        _padded_total_numel = _aligned_size(_total_numel, align=ALIGNED_SIZE)
        if _padded_total_numel != _total_numel:
            extra_param = torch.nn.Parameter(
                torch.zeros(
                    _padded_total_numel - _total_numel, dtype=dtype, device=device
                ),
                requires_grad=flat_param_requires_grad,
            )
            module.register_parameter(
                "_flat_param_padding_only_for_alignment", extra_param
            )
            params_set.add(extra_param)
        # For alignment padding, only `numels` gets strictly non-`None`
        # elements, and all other lists get `None` elements for padding.
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        fqns: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[
            Union[torch.Tensor, nn.Parameter], Tuple[nn.Module, str, str]
        ] = {}
        params_to_flatten: List[Union[torch.Tensor, nn.Parameter]] = []
        shared_params: List[Union[torch.Tensor, nn.Parameter]] = []
        total_numel = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param not in params_set:
                    continue
                if param in shared_param_memo:  # shared reference
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
                    shared_params.append(param)
                    shared_param_infos.append(
                        SharedParamInfo(
                            param_name,
                            submodule,
                            submodule_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                else:
                    shared_param_memo[param] = (submodule, submodule_name, param_name)
                    params_to_flatten.append(param)
                    param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                    numels.append(param.numel())
                    shapes.append(param.shape)
                    fqn = (
                        submodule_name + "." + param_name
                        if submodule_name
                        else param_name
                    )
                    fqns.append(fqn)
                    total_numel += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {params}\nmodule: {module}"
            )
        # Pass `aligned_numel=0` since we already included padding tensors
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
            params_to_flatten,
            aligned_numel=0,
            requires_grad=flat_param_requires_grad,
        )
        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            fqns,
            shared_param_infos,
            params=None,
            shared_params=None,
        )

    @torch.no_grad()
    def shard_and_init_flat_param_attibutes(self):
        """
        Shard the flat_param, set the flat_param.data, _local_shard, and _full_param.
        Free the storage if needed.

        Here we create a new sharded tensor for the shard metadata initialization. However,
        we want to use either the full buffer or the partial buffer in practice. If swapping
        is enabled, when swapping out, the storage would be naturally pointed to the partial
        buffer. However, if no swapping but sharding is enabled, we have to point the storage
        manually to the partial buffer.
        """
        flat_param = self.flat_param
        unsharded_flat_param_data = self.flat_param.data
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
            flat_param._local_shard = flat_param.data
            flat_param._full_param = flat_param.data
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            # ===================================================================================
            # Shard the flat_param anyway to init shard metadata
            sharded_flat_param = shard_tensor(
                tensor=flat_param.flatten(),
                num_shards=self.world_size,
                shard_id=self.rank,
                share_storage=False,
            )
            flat_param.set_(sharded_flat_param)
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(0, start_idx, end_idx)
            # ===================================================================================
            if self.uses_weight_sharding_strategy:
                # If no swapping is enabled, then we have to manually point the storage to the partial buffer.
                if self.param_swap_ratio == 0.0:
                    assert (
                        self._partial_weight_buffer.numel == flat_param.numel()
                    ), f"{self._partial_weight_buffer.numel} != {flat_param.numel()}"
                    partial_buffer_tensor = self._partial_weight_buffer.data
                    partial_buffer_tensor[:] = sharded_flat_param.flatten()
                    flat_param.set_(partial_buffer_tensor)
                    # sharded_flat_param = shard_tensor(
                    #     tensor=unsharded_flat_param_data.flatten(),
                    #     num_shards=self.world_size,
                    #     shard_id=self.rank,
                    #     share_storage=False,
                    #     buffer=self._partial_weight_buffer.data
                    # )
                    # flat_param.set_(sharded_flat_param)
                # Set the unsharded data ptr to _full_param and free its storage.
                flat_param._local_shard = flat_param.data
                flat_param._full_param = unsharded_flat_param_data
                _p_assert(
                    flat_param.size() == flat_param._sharded_size,
                    f"Expects {flat_param._sharded_size} but got {flat_param.size()}",
                )
                _p_assert(
                    flat_param._local_shard.size() == flat_param._sharded_size,
                    f"Expects {flat_param._sharded_size} but got {flat_param.size()}",
                )
                _p_assert(
                    flat_param._full_param.size() == flat_param._unsharded_size,
                    f"Expects {flat_param._unsharded_size} but got {flat_param._full_param.size()}",
                )
                if self._curr_occupied_full_weights_buffer is None:
                    # Free the storage only if _full_param is not the full mem buffer
                    _free_storage(flat_param._full_param)
                    _p_assert(
                        unsharded_flat_param_data._typed_storage()._size() == 0,
                        "Double check the storage size of the unsharded flat param",
                    )
            else:
                # Reset the unsharded data to flat_param
                flat_param.set_(unsharded_flat_param_data)
                flat_param._local_shard = flat_param.data
                flat_param._full_param = flat_param.data
                _p_assert(
                    flat_param.size()
                    == flat_param._local_shard.size()
                    == flat_param._full_param.size()
                    == flat_param._unsharded_size,
                    (
                        f"Expects {flat_param._unsharded_size} but got "
                        f"{flat_param.size()}, {flat_param._local_shard.size()}, "
                        f"{flat_param._full_param.size()}, {flat_param._unsharded_size}"
                    ),
                )

    def _init_shard_metadata(
        self,
        numel_padded: int,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> None:
        """
        Initializes shard-related metadata for this rank's shard of the flat
        parameter: ``_sharded_size``, ``_shard_param_infos``, and
        ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
        flat_param = self.flat_param
        flat_param._sharded_size = flat_param.size()
        flat_param._sharded_numel = sharded_flat_param_numel = flat_param.numel()  # type: ignore[attr-defined]
        if self.uses_sharded_strategy:
            _p_assert(
                flat_param._unsharded_numel
                == flat_param._sharded_numel * self.world_size,
                f"Expects {flat_param._sharded_numel * self.world_size} but got {flat_param._unsharded_numel}",
            )
        _p_assert(
            unsharded_start_idx >= 0 and unsharded_start_idx <= unsharded_end_idx,
            f"unsharded_start_idx: {unsharded_start_idx} unsharded_end_idx: {unsharded_end_idx}",
        )
        _p_assert(
            numel_padded <= sharded_flat_param_numel,
            f"numel_padded: {numel_padded} "
            f"sharded_flat_param_numel: {sharded_flat_param_numel}",
        )
        shard_param_infos = self._get_shard_metadata(
            unsharded_start_idx, unsharded_end_idx
        )
        assert (
            len(shard_param_infos) == flat_param._num_params
        ), f"Expects length {flat_param._num_params} but got {len(shard_param_infos)}"
        flat_param._shard_param_infos = shard_param_infos  # type: ignore[attr-defined]
        flat_param._shard_numel_padded = numel_padded  # type: ignore[attr-defined]

    def _get_shard_metadata(
        self,
        unsharded_start_idx: int,
        unsharded_end_idx: int,
    ) -> Tuple[_ShardParamInfo, ...]:
        """
        Computes the shard metadata based on ``unsharded_start_idx`` and
        ``unsharded_end_idx`` (inclusive), which give the interval of the
        unsharded flat parameter specifying the shard.
        """
        flat_param_offsets = self._get_flat_param_offsets()
        assert len(flat_param_offsets) == len(
            self.flat_param._numels
        ), f"Expected {len(self.flat_param._numels)} but got {len(flat_param_offsets)}"
        shard_param_infos: List[_ShardParamInfo] = []
        sharded_flat_param_numel = unsharded_end_idx - unsharded_start_idx + 1
        # `unsharded_param_start_idx` and `unsharded_param_end_idx` are indices
        # into the unsharded flat parameter (inclusive) of the given parameter
        for i, (unsharded_param_start_idx, unsharded_param_end_idx) in enumerate(
            flat_param_offsets
        ):
            in_sharded_flat_param = (
                unsharded_start_idx <= unsharded_param_end_idx
                and unsharded_end_idx >= unsharded_param_start_idx
            )
            if not in_sharded_flat_param:
                shard_param_info = _ShardParamInfo(False, None, None, None, None)
            else:
                if unsharded_start_idx <= unsharded_param_start_idx:
                    # This branch can only happen once since the rank's
                    # unsharded start index can only intersect one parameter
                    intra_param_start_idx = 0
                    offset_in_shard = unsharded_param_start_idx - unsharded_start_idx
                else:
                    intra_param_start_idx = (
                        unsharded_start_idx - unsharded_param_start_idx
                    )
                    offset_in_shard = 0
                assert (
                    offset_in_shard >= 0 and offset_in_shard < sharded_flat_param_numel
                ), (
                    f"Invalid `offset_in_shard` of {offset_in_shard} for "
                    f"sharded flat parameter with {sharded_flat_param_numel} numel"
                )
                intra_param_end_idx = (
                    min(unsharded_param_end_idx, unsharded_end_idx)
                    - unsharded_param_start_idx
                )
                numel_in_shard = intra_param_end_idx - intra_param_start_idx + 1
                shard_param_info = _ShardParamInfo(
                    True,
                    offset_in_shard,
                    numel_in_shard,
                    intra_param_start_idx,
                    intra_param_end_idx,
                )
            shard_param_infos.append(shard_param_info)
        return tuple(shard_param_infos)

    def _get_flat_param_offsets(self) -> List[Tuple[int, int]]:
        """
        Returns [start, end] offsets of each original parameter's flattened
        data in the unsharded flat parameter (without padding).
        NOTE: The returned list includes elements for alignment padding.
        """
        cumulative_sum = list(accumulate(self.flat_param._numels))
        starts = [0] + cumulative_sum[:-1]
        ends = [end - 1 for end in cumulative_sum]  # inclusive
        param_offsets = list(zip(starts, ends))
        return param_offsets

    def _validate_tensors_to_flatten(
        self, tensors: List[Union[torch.Tensor, nn.Parameter]]
    ) -> Tuple:
        """
        Validates the tensors to flatten and returns any necessary metadata.
        """
        dtype: Optional[torch.dtype] = None
        # Return as the logical OR over each tensor's value
        flat_param_requires_grad: Optional[bool] = None
        device: Optional[torch.device] = None
        # For `use_orig_params=True`, permit non-uniform `requires_grad`
        for tensor in tensors:
            if isinstance(tensor, FlatParameter):
                raise ValueError("Cannot flatten a `FlatParameter`")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError(
                    f"Must flatten tensors with uniform dtype but got {dtype} "
                    f"and {tensor.dtype}"
                )
            if (
                flat_param_requires_grad is not None
                and tensor.requires_grad != flat_param_requires_grad
            ):
                raise ValueError(
                    "Must flatten tensors with uniform `requires_grad` when "
                    "`use_orig_params=False`"
                )
            if device is not None and tensor.device != device:
                raise ValueError(
                    "Must flatten tensors on the same device but got both "
                    f"{device} and {tensor.device}"
                )
            dtype = tensor.dtype
            flat_param_requires_grad = flat_param_requires_grad or tensor.requires_grad
            device = tensor.device
        assert flat_param_requires_grad is not None, "Requires non-empty `tensors` list"
        return dtype, flat_param_requires_grad, device

    def flatten_tensors(
        self,
        tensors: List[torch.Tensor],
        aligned_numel: int,
    ) -> torch.Tensor:
        """
        Flattens ``tensors`` into a single flat tensor optionally including
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
        if len(tensors) == 0:
            raise ValueError("Expects non-empty `tensors`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        if aligned_numel != 0:
            raise NotImplementedError(
                "Aligned padding is not yet supported in FlatParamHandle"
            )

        dtype, _, device = self._validate_tensors_to_flatten(tensors)
        flat_tensors: List[torch.Tensor] = []
        flat_tensors = [torch.flatten(_detach_if_needed(tensor)) for tensor in tensors]
        # sum_elem = sum(tensor.numel() for tensor in flat_tensors)
        # if sum_elem % aligned_numel != 0:
        #     pad_elem = aligned_numel - sum_elem % aligned_numel
        #     flat_tensors.append(torch.zeros(pad_elem, dtype=dtype, device=device))
        return torch.cat(flat_tensors, dim=0)

    def flatten_tensors_into_flat_param(
        self,
        tensors: List[torch.Tensor],
        aligned_numel: int,
        requires_grad: bool,
    ) -> FlatParameter:
        flat_param_data = self.flatten_tensors(tensors, aligned_numel)
        return FlatParameter(flat_param_data, requires_grad=requires_grad)

    def _init_partial_memory_buffers(self):
        """
        Create the partial memory buffers. If the dst_numel_in_cuda is the same as the
        current numel, then directly reuse the data for the memory buffer.
        """
        partial_weight_numel = _aligned_size(
            self._sharded_weight_numel * (1 - self.param_swap_ratio), align=ALIGNED_SIZE
        )
        data = (
            self.flat_param if partial_weight_numel == self.flat_param.numel() else None
        )
        self._partial_weight_buffer: MemoryBuffer = allocate_memory_buffer(
            name=f"{self.module.name}.partial_weight_buffer_dtype_{self.flat_param.dtype}_requires_grad_{self.flat_param.requires_grad}",
            numel=partial_weight_numel,
            dtype=self.flat_param.dtype,
            track_usage=True,
            data=data,
        )
        if self.flat_param.requires_grad:
            partial_grad_numel = _aligned_size(
                self._sharded_grad_numel * (1 - self.grad_swap_ratio),
                align=ALIGNED_SIZE,
            )
            self._partial_grad_buffer: MemoryBuffer = allocate_memory_buffer(
                name=f"{self.module.name}.partial_grad_buffer_dtype_{self.flat_param.dtype}",
                numel=partial_grad_numel,
                dtype=self.flat_param.dtype,
                track_usage=True,
            )

    def _get_unflat_views(
        self,
        tensor: Optional[torch.Tensor] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Returns unflattened ``Tensor`` views into ``tensor`` if it is not
        ``None`` or ``flat_param`` otherwise, where the unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
        flat_param = self.flat_param
        if tensor is None:
            tensor = flat_param
        views = (
            subtensor.view(shape)
            for (subtensor, shape) in zip(
                tensor.split(flat_param._numels, dim=0), flat_param._shapes
            )
        )
        return views

    def _check_unsharded(self, tensor: torch.Tensor):
        msg_prefix = "Expects tensor to be unsharded "
        _p_assert(tensor is not None, msg_prefix + "but got `None`")
        unsharded_size = self.flat_param._unsharded_size
        _p_assert(
            tensor.size() == unsharded_size,
            msg_prefix + f"with size {unsharded_size} but got {tensor.size()}",
        )

    def _use_unsharded_views(self, as_params: bool) -> None:
        """
        Unflattens the unsharded flat parameter by setting the original
        parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.
        """
        flat_param = self.flat_param
        self._check_unsharded(flat_param)
        views = self._get_unflat_views()

        for i, (view, (param_name, module, _)) in enumerate(
            zip(views, flat_param._param_infos)
        ):
            if as_params:
                self._setattr_param(module, param_name, nn.Parameter(view))
            else:  # `as_params=False`
                param_var: torch.Tensor = view
                self._setattr_tensor(module, param_name, param_var)
        for i, (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in enumerate(self.flat_param._shared_param_infos):
            prim_param: Union[torch.Tensor, nn.Parameter] = getattr(
                prim_module, prim_param_name
            )
            _p_assert(
                not as_params or isinstance(prim_param, nn.Parameter),
                f"as_params={as_params} type(prim_param)={type(prim_param)}",
            )
            if as_params:
                self._setattr_param(module, param_name, prim_param)
            else:  # `as_params=False`
                self._setattr_tensor(module, param_name, prim_param)

    def set_training_state(self, state: HandleTrainingState) -> None:
        self.training_state = state

    def is_weight_unsharded(self) -> bool:
        return self.flat_param.numel() == self._unsharded_numel

    def needs_full_weights_allocation(self) -> bool:
        if self.is_weight_unsharded():
            return False
        # =============================================================================
        # TODO(zhanda): remove this part later if it's not touched.
        # If the weight is not sharded and the swapping is disabled, then we don't
        # need to allocate the full weights.
        if (
            self.sharding_strategy != HandleShardingStrategy.FULL_SHARD
            and self._dst_numel_in_cuda_for_partial_weights == self.flat_param.numel()
        ):
            raise ValueError(
                "This should not happen. Check the logic and add the comment here."
            )
            return False
        # =============================================================================
        return True

    def needs_full_grads_allocation(self) -> bool:
        if self.flat_param.grad is not None:
            if self.flat_param.grad.numel() > self._unsharded_numel:
                raise ValueError(
                    f"FlatParam grad numel: {self.flat_param.grad.numel()}, "
                    f"Full numel: {self._unsharded_numel}."
                )
            elif self.flat_param.grad.numel() == self._unsharded_numel:
                return False
        # =============================================================================
        # TODO(zhanda): remove this part later if it's not touched.
        if (
            self.sharding_strategy not in GRAD_SHARD_HANDLE_STRATEGIES
            and self._dst_numel_in_cuda_for_partial_grads == self.flat_param.numel()
        ):
            raise ValueError(
                "This should not happen. Check the logic and add the comment here."
            )
            return False
        # =============================================================================
        return True

    @torch.no_grad()
    def alloc_full_weights(self, force: bool = False):
        """
        Overrided to use the buffer and to support swapping.
        """
        if not force and not self.needs_full_weights_allocation():
            return
        flat_param = self.flat_param
        if getattr(flat_param, "_full_param", None) is not None:
            unsharded_flat_param = flat_param._full_param
        else:
            unsharded_flat_param = flat_param
        if not self.uses_weight_sharding_strategy and self.param_swap_ratio == 0.0:
            # No sharding and no swapping. Directly load the partial weight buffer which
            # is actually the full weight buffer.
            buffer: MemoryBuffer = self._partial_weight_buffer
            assert buffer is not None, "buffer is None"
            assert (
                buffer.numel >= self._full_numel
            ), f"Buffer numel: {buffer.numel}, Full numel: {self._full_numel}."
            buffer_tensor = buffer.data[: self._full_numel]
            unsharded_flat_param.data = buffer_tensor.view(unsharded_flat_param.shape)
        else:
            # Otherwise load the full weight buffer from the buffer pool.
            if self._curr_occupied_full_weights_buffer is not None:
                return unsharded_flat_param
            ring_buffer: RingMemoryBuffer = self._full_weights_ring_buffer
            buffer: MemoryBuffer = ring_buffer.get_next_buffer()
            buffer_tensor = buffer.new(self._full_numel)
            if buffer_tensor.numel() == self.flat_param.numel():
                buffer_tensor[:] = self.flat_param.flatten()
            elif self.param_swap_ratio == 0.0:
                start = self.rank * self._sharded_weight_numel
                end = (self.rank + 1) * self._sharded_weight_numel
                assert self.flat_param.numel() == self._sharded_weight_numel, (
                    f"FlatParam numel: {self.flat_param.numel()}, "
                    f"Sharded numel: {self._sharded_weight_numel}."
                )
                buffer_tensor[start:end] = self.flat_param.flatten()
                # logger.error(
                #     f"[{self.module.name}] Allocate full weights. {self.flat_param._local_shard.float().sum().item()}"
                # )
            unsharded_flat_param.data = buffer_tensor.view(unsharded_flat_param.shape)
            self._curr_occupied_full_weights_buffer = buffer
            # logger.debug(
            #     f"[{self.module.name}] Allocate full weights. "
            #     f"Buffer name: {buffer.name}, "
            #     f"FlatParam numel: {self.flat_param.numel()}, "
            #     f"Buffer numel: {buffer.numel}."
            # )
        return unsharded_flat_param

    @torch.no_grad()
    def alloc_full_grads(self):
        if not self.flat_param.requires_grad:
            return

        flat_param = self.flat_param
        unsharded_flat_param = self.flat_param._full_param
        if not self.uses_grad_sharding_strategy and self.grad_swap_ratio == 0.0:
            saved_grad = getattr(self.flat_param, "_saved_grad", None)
            if saved_grad is not None:
                assert saved_grad.size() == self.flat_param._unsharded_size, (
                    f"Saved grad size: {saved_grad.size()}, "
                    f"Unsharded size: {self.flat_param._unsharded_size}."
                )
                # Saved grad is actually also the buffer data. Do a double check.
                assert (
                    saved_grad.data_ptr() == self._partial_grad_buffer.data.data_ptr()
                ), (
                    f"[{self.module.name}] Saved grad data ptr: {saved_grad.data_ptr()}, "
                    f"Buffer data ptr: {self._partial_grad_buffer.data.data_ptr()}. "
                    f"Saved grad size: {saved_grad.size()}, "
                    f"Buffer size: {self._partial_grad_buffer.data.size()}."
                )
                self.flat_param.grad = saved_grad
            else:
                buffer: MemoryBuffer = self._partial_grad_buffer
                assert buffer is not None, "buffer is None"
                assert (
                    buffer.numel >= self._full_numel
                ), f"Buffer numel: {buffer.numel}, Full numel: {self._full_numel}."
                buffer_tensor = buffer.data[: self._full_numel]
                flat_param.grad = buffer_tensor.view(unsharded_flat_param.shape)
                # logger.error(
                #     f"[{self.module.name}] Allocate full grads. {flat_param.grad[:100].sum().item()}"
                # )
        else:
            if self._curr_occupied_full_grads_buffer is not None:
                return flat_param.grad
            ring_buffer: RingMemoryBuffer = self._full_grads_ring_buffer
            buffer: MemoryBuffer = ring_buffer.get_next_buffer()
            # logger.info(f"[{self.module.name}] Allocate full grads. Buffer name: {buffer.name}")
            buffer.data.zero_()
            buffer_tensor = buffer.new(self._full_numel)
            buffer_tensor.zero_()
            # logger.info(
            #     f"[{self.module.name}] - 1 Allocate full grads. "
            #     f"Sum: {buffer_tensor.float().sum()}"
            # )
            # if self.grad_swap_ratio == 0.0:
            #     # If not swapping-in, then we have to manually copy the grad to the buffer.
            #     buffer_tensor.zero_()
            #     start = self.rank * self._sharded_grad_numel
            #     end = (self.rank + 1) * self._sharded_grad_numel
            #     _saved_grad = getattr(self.flat_param, "_saved_grad", None)
            #     if _saved_grad is not None:
            #         assert _saved_grad.numel() == self._sharded_grad_numel, (
            #             f"Saved grad numel: {_saved_grad.numel()}, "
            #             f"Sharded numel: {self._sharded_grad_numel}."
            #         )
            #         buffer_tensor[start:end] = _saved_grad.flatten()
            # logger.info(
            #     f"[{self.module.name}] - 2 Allocate full grads. "
            #     f"Sum: {buffer_tensor.float().sum()}"
            # )
            # Because torch doesn't support directly assigning a tensor to the grad
            # if the shapes don't match. So we use this trick to assign the grad.
            ori_flat_param_data = flat_param.data
            flat_param.data = buffer_tensor.view(unsharded_flat_param.shape)
            flat_param.grad = buffer_tensor.view(unsharded_flat_param.shape)
            flat_param.data = ori_flat_param_data
            self._curr_occupied_full_grads_buffer = buffer
            logger.debug(
                f"[{self.module.name}] Allocate full grads. "
                f"Buffer name: {buffer.name}, "
                f"FlatParam numel: {self.flat_param.numel()}, "
                f"buffer numel: {buffer.numel}."
            )

        # torch.cuda.synchronize()
        # logger.info(
        #     f"[{self.module.name}] - 3 Allocate full grads. "
        #     f"Sum: {flat_param.grad.float().sum()}"
        # )

        return flat_param.grad

    @torch.no_grad()
    def swap_in_weights(self, stream: torch.Stream) -> bool:
        """Swap in the weights (_local_shard) to the full weight buffer."""
        swap_handle = getattr(self.flat_param._local_shard, "_swap_handle", None)
        if swap_handle is None:
            return False
        if swap_handle.numel_in_cuda() >= self._sharded_weight_numel:
            return False
        flat_param = self.flat_param
        buffer_tensor = None
        if self._use_memory_buffer:
            # Get the unsharded flat param which is where the buffer is
            unsharded_flat_param = self.flat_param._full_param
            # Get the correct shard from the unsharded flat param
            buffer_tensor = unsharded_flat_param.data[: self._sharded_weight_numel]
            flat_param.data.record_stream(stream)
            flat_param._local_shard.record_stream(stream)
            flat_param.data = buffer_tensor
            flat_param._local_shard.data = buffer_tensor
        worked = swap_(
            flat_param._local_shard,
            state="cuda",
            stream=stream,
            cache_cpu_data=True,
            cuda_buffer=buffer_tensor,
        )
        return worked

    @torch.no_grad()
    def swap_in_grads(self, stream: torch.Stream, shard: bool = False) -> bool:
        """Swap in the grads (_saved_grad) to the full grad buffer."""
        if not self.flat_param.requires_grad:
            return False
        if getattr(self.flat_param, "_saved_grad", None) is None:
            return False
        swap_handle = getattr(self.flat_param._saved_grad, "_swap_handle", None)
        if swap_handle is None:
            # logger.error(
            #     f"[{self.module.name}] Swap in grads: {self.flat_param._saved_grad.float().abs().sum().item()}"
            # )
            return False
        if swap_handle.numel_in_cuda() >= self._sharded_grad_numel:
            return False
        flat_param = self.flat_param
        buffer_tensor = None
        if self._use_memory_buffer:
            if shard or self.uses_grad_sharding_strategy:
                buffer_tensor = shard_tensor(
                    tensor=flat_param.grad.flatten(),
                    num_shards=self.world_size,
                    shard_id=self.rank,
                    share_storage=True,
                )
            else:
                buffer_tensor = flat_param.grad
            flat_param._saved_grad.data = buffer_tensor

        # Update the dst_numel_in_cuda if shard is True since the total_size changes.
        # dst_numel_in_cuda_for_partial = None
        # if shard:
        #     dst_numel_in_cuda_for_partial =

        # Swap in the grads
        # logger.error(f"{flat_param._saved_grad._cuda_data.shape=}, {flat_param._saved_grad._cpu_data.shape=}")
        # logger.error(f"{flat_param._saved_grad._cuda_data[:3146240].cuda().float().abs().sum().item()=}")
        # logger.error(f"{flat_param._saved_grad._cpu_data[:3146240].cuda().float().abs().sum().item()=}")
        worked = swap_(
            flat_param._saved_grad,
            state="cuda",
            stream=stream,
            cache_cpu_data=False,
            cuda_buffer=buffer_tensor,
        )
        # logger.error(
        #     f"[{self.module.name}] Swap in grads: {flat_param._saved_grad.float().abs().sum().item()}"
        # )
        return worked

    @torch.no_grad()
    def swap_out_weights(
        self, stream: torch.Stream, dst_numel_in_cuda: int = None
    ) -> bool:
        """Swap out the weights (_local_shard)"""
        flat_param = self.flat_param
        if dst_numel_in_cuda is None:
            assert (
                self.param_swap_ratio is not None
            ), "dst_numel_in_cuda is None and self.param_swap_ratio is None"
            dst_numel_in_cuda = int(flat_param.numel() * (1 - self.param_swap_ratio))
        if dst_numel_in_cuda > flat_param.numel():
            raise ValueError(
                f"dst_numel_in_cuda: {dst_numel_in_cuda} flat_param.numel(): {flat_param.numel()}"
            )
        elif dst_numel_in_cuda == flat_param.numel():
            return False
        # Get the partial tensor buffer to swap to
        buffer_tensor = None
        if self._use_memory_buffer and self._partial_weight_buffer is not None:
            self._partial_weight_buffer.reset()
            buffer_tensor = self._partial_weight_buffer.new(dst_numel_in_cuda)

        # Swap out the weights
        worked = swap_(
            flat_param._local_shard,
            state="partial",
            stream=stream,
            dst_numel_in_cuda_for_partial=dst_numel_in_cuda,
            cache_cpu_data=True,
            cuda_buffer=buffer_tensor,
        )
        flat_param.data.record_stream(stream)
        flat_param._local_shard.record_stream(stream)
        flat_param.data = flat_param._local_shard
        return worked

    @torch.no_grad()
    def swap_out_grads(
        self, stream: torch.cuda.Stream, dst_numel_in_cuda: int = None
    ) -> bool:
        """Swap out the grads (_saved_grad)."""
        if not self.flat_param.requires_grad:
            return False
        if getattr(self.flat_param, "_saved_grad", None) is None:
            return False
        flat_param = self.flat_param
        saved_grad = flat_param._saved_grad
        if dst_numel_in_cuda is None:
            assert self.grad_swap_ratio is not None, "dst_numel_in_cuda is None"
            dst_numel_in_cuda = int(saved_grad.numel() * (1 - self.grad_swap_ratio))
        if dst_numel_in_cuda > saved_grad.numel():
            raise ValueError(
                f"dst_numel_in_cuda: {dst_numel_in_cuda} saved_grad.numel(): {saved_grad.numel()}"
            )
        elif dst_numel_in_cuda == saved_grad.numel():
            # logger.error(
            #     f"[{self.module.name}] Swap out grads: {self.flat_param._saved_grad.float().abs().sum().item()}"
            # )
            return False
        # Get the buffer (partial grads)
        buffer_tensor = None
        if self._use_memory_buffer and self._partial_grad_buffer is not None:
            self._partial_grad_buffer.reset()
            buffer_tensor = self._partial_grad_buffer.new(dst_numel_in_cuda)
        # logger.error(
        #     f"[{self.module.name}] Swap out grads: {flat_param._saved_grad.float().abs().sum().item()}"
        # )
        # Swap out the grads
        worked = swap_(
            flat_param._saved_grad,
            state="partial",
            dst_numel_in_cuda_for_partial=dst_numel_in_cuda,
            stream=stream,
            cache_cpu_data=False,
            cuda_buffer=buffer_tensor,
        )
        return worked

    ############
    # Sharding #
    ############
    def unshard(self, stream: Optional[torch.Stream] = None) -> None:
        """
        Runs the unshard logic. This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        stream = torch.cuda.current_stream() if stream is None else stream
        if not self.needs_unshard():
            # Even when not needing an unshard, we should switch to using
            # the unsharded flat parameter
            assert (
                self.flat_param._full_param.data_ptr() == self.flat_param.data_ptr()
            ), (
                f"Expects the same data pointer but got "
                f"{self.flat_param._full_param.data_ptr()} and {self.flat_param.data_ptr()}"
            )
            unsharded_flat_param = self.flat_param._full_param
            self._use_unsharded_flat_param(unsharded_flat_param)
            # Debugging
            # logger.error(
            #     f"[{self.module.name}] Unshard: {self.flat_param.float().sum().item()}"
            # )
            # world_size = 8
            # numel_per_rank = self.flat_param.numel() // world_size
            # for i in range(world_size):
            #     logger.error(
            #         f"[{self.module.name}] Rank {i}: {self.flat_param[i * numel_per_rank : (i + 1) * numel_per_rank].float().sum().item()}"
            #     )
            return
        unsharded_flat_param = self.flat_param._full_param
        padded_unsharded_flat_param = self._all_gather_flat_param(
            unsharded_flat_param, stream=stream
        )
        self._use_unsharded_flat_param(padded_unsharded_flat_param)

    def _all_gather_flat_param(
        self,
        unsharded_flat_param: torch.Tensor,
        stream: Optional[torch.Stream] = None,
    ) -> torch.Tensor:
        """
        All-gathers the handle's flat parameter to the destination
        ``padded_unsharded_flat_param``, and switches to using the all-gathered
        tensor.
        """
        stream = torch.cuda.current_stream() if stream is None else stream
        _p_assert(
            hasattr(self, "process_group") and hasattr(self, "world_size"),
            "Expects a process group and world size to have been set via `shard()`",
        )
        sharded_flat_param = self.flat_param.data
        # logger.error(
        #     f"[{self.module.name}] Before All gather: {sharded_flat_param.float().sum()}"
        # )
        expected_numel = sharded_flat_param.numel() * self.world_size
        _p_assert(
            unsharded_flat_param.numel() == expected_numel,
            f"Expects {expected_numel} numel but got {unsharded_flat_param.numel()}",
        )

        # HACK this should be handled by C10D
        if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
            tensor_list = list(
                torch.chunk(
                    unsharded_flat_param,
                    dist.get_world_size(self.all_gather_process_group),
                )
            )
            work = dist.all_gather(
                tensor_list, sharded_flat_param, group=self.all_gather_process_group
            )
        else:
            with torch.cuda.stream(stream):
                dist.all_gather_into_tensor(
                    unsharded_flat_param,
                    sharded_flat_param,
                    self.all_gather_process_group,
                    # async_op=True,
                )
                # logger.error(
                #     f"[{self.module.name}] After All gather: {unsharded_flat_param.float().sum()}"
                # )
                sharded_flat_param.record_stream(stream)
        return unsharded_flat_param

    def _use_unsharded_flat_param(
        self,
        unsharded_flat_param: torch.Tensor,
    ) -> None:
        """
        Switches to using the *unpadded* unsharded flat parameter, which is a
        view into the *padded* unsharded flat parameter.

        NOTE(zhanda): This function actually doesn't matter since we have reassigned
        the params in the forward function anyway.
        """
        unsharded_size = self.flat_param._unsharded_size
        self.flat_param.data = unsharded_flat_param[: unsharded_size.numel()].view(
            unsharded_size
        )  # this `.view()` is not autograd visible
        in_forward = self.training_state == HandleTrainingState.FORWARD
        in_pre_backward = self.training_state == HandleTrainingState.BACKWARD_PRE
        if in_forward:
            self._use_unsharded_views(as_params=False)

    def needs_unshard(self) -> bool:
        """Returns if the handle's flat parameter needs to be unsharded."""
        if not self.uses_weight_sharding_strategy:
            return False
        if self.is_weight_unsharded():
            return False
        return True

    def reshard(self):
        """
        When flat_param is resharded,
        1. the flat_param will be freed
        2. params will be empty
        3. only the local shard will be kept.
        """
        if not self.uses_weight_sharding_strategy:
            return

        flat_param = self.flat_param
        flat_param.data = flat_param._local_shard

    @property
    def _full_weights_ring_buffer(self) -> RingMemoryBuffer:
        return get_ring_memory_buffer(
            f"full_weights_dtype_{self.flat_param.dtype}_requires_grad_{self.flat_param.requires_grad}"
        )

    @property
    def _full_grads_ring_buffer(self) -> RingMemoryBuffer:
        return get_ring_memory_buffer(f"full_grads_dtype_{self.flat_param.dtype}")

    @torch.no_grad()
    def dealloc_full_weights(self):
        if not self._use_memory_buffer and self.uses_weight_sharding_strategy:
            unsharded_flat_param = self.flat_param._full_param
            _p_assert(unsharded_flat_param._typed_storage()._size() > 0, "size is 0")
            _p_assert(
                unsharded_flat_param.device == self.device,
                f"device: {unsharded_flat_param.device}",
            )
            # _no_dispatch_record_stream(
            #     unsharded_flat_param, self.device.current_stream()
            # )
            _free_storage(unsharded_flat_param)
        if self._curr_occupied_full_weights_buffer is not None:
            self._curr_occupied_full_weights_buffer.reset()
            self._curr_occupied_full_weights_buffer = None
        # if self.param_swap_ratio == 0.0:
        #     # Copy the local shard to local shard if swapping is disabled.
        #     buffer: MemoryBuffer = self._partial_weight_buffer
        #     assert buffer is not None, "buffer is None"
        #     buffer.data.copy_(self.flat_param._local_shard.flatten())
        #     self.flat_param._local_shard.data = buffer.data

    @torch.no_grad()
    def dealloc_full_grads(self):
        assert self._use_memory_buffer, "This should not happen."
        if self._curr_occupied_full_grads_buffer is not None:
            logger.debug(
                f"Deallocate full grads for {self.module.name}. "
                f"Buffer name: {self._curr_occupied_full_grads_buffer.name}, "
                f"FlatParam numel: {self.flat_param.numel()}, "
                # f"FlatParam grad numel: {self.flat_param.grad.numel()}. "
            )
            self._curr_occupied_full_grads_buffer.reset()
            self._curr_occupied_full_grads_buffer = None

    def reduce_grad(self, stream: torch.cuda.Stream, skip=False):
        """The arg `stream` is used for the `no_dispatch_record_stream`."""
        if not self.flat_param.requires_grad:
            return
        flat_param = self.flat_param
        unsharded_grad = flat_param.grad.data
        # logger.error(f"[{self.module.name}] {skip=} Before reducing grad: {unsharded_grad.flatten()[:100].sum():.6f}")
        if skip:
            new_sharded_grad = unsharded_grad
        elif not self.uses_sharded_strategy:
            # If not using sharded strategy and the grad is needed to be reduced,
            # we directly call all_reduce.
            # FIXME(zhanda): fix the bug of the sync of the grad brought by the
            # post div factor.
            # _div_if_needed(unsharded_grad, self._gradient_predivide_factor)
            with torch.cuda.stream(stream):
                dist.all_reduce(unsharded_grad, group=self.reduce_scatter_process_group)
            # _div_if_needed(unsharded_grad, self._gradient_postdivide_factor)
            new_sharded_grad = unsharded_grad
        else:
            # If using sharded strategy, we need to call reduce_scatter.
            # padded_unsharded_grad, new_sharded_grad = self._get_reduce_scatter_tensors(
            #     unsharded_grad
            # )
            new_sharded_grad = shard_tensor(
                tensor=unsharded_grad,
                num_shards=self.world_size,
                shard_id=self.rank,
                share_storage=True,
            )
            # _div_if_needed(
            #     unsharded_grad,
            #     self._gradient_predivide_factor
            #     * self.reduce_scatter_process_group.size(),
            # )
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                dist.reduce_scatter_tensor(
                    new_sharded_grad,
                    unsharded_grad,
                    group=self.reduce_scatter_process_group,
                    # async_op=True,
                )
                # FIXME(zhanda): fix the bug of the sync of the grad brought by the
                # post div factor.
                # _div_if_needed(new_sharded_grad, self._gradient_postdivide_factor)
                new_sharded_grad.record_stream(stream)
                unsharded_grad.record_stream(stream)
        # NOTE(zhanda): Accumulate and post process
        # Instead of accumulating the grad. Since we have already accumulated the grad
        # because the swapped-in grads are in flat_param.grad. So we just need to
        # assign the sharded_grad to the _saved_grad.
        # self._accumulate_sharded_grad(new_sharded_grad)
        if getattr(flat_param, "_saved_grad", None) is None:
            flat_param._saved_grad = new_sharded_grad
        else:
            # flat_param._saved_grad.data = new_sharded_grad
            flat_param._saved_grad = new_sharded_grad
        TensorSwapHandle.clean(flat_param._saved_grad)
        # torch.cuda.current_stream().wait_stream(stream)
        # torch.cuda.synchronize()
        # logger.error(
        #     f"[{self.module.name}] {skip=} After reducing grad: "
        #     f"New Grad: {new_sharded_grad.flatten()[:100].sum():.6f}, Old Grad: {unsharded_grad.flatten()[:100].sum():.6f}, "
        #     f"FlatParam: {flat_param.flatten()[:100].sum():.6f}"
        # )
        self._post_reduce_grad_callback()
        # Since the unsharded gradient is produced in the computation
        # stream and consumed in the post-backward stream, inform the
        # caching allocator (before it goes out of scope)
        _no_dispatch_record_stream(unsharded_grad, stream)

    # TODO(zhanda): Delete this if it's not used.
    # def _get_reduce_scatter_tensors(
    #     self, unsharded_grad: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Returns the input and output tensors to reduce-scatter, respectively.
    #     """
    #     world_size = self.world_size
    #     chunks = list(unsharded_grad.chunk(world_size))
    #     numel_to_pad = world_size * chunks[0].numel() - unsharded_grad.numel()
    #     padded_unsharded_grad = (
    #         F.pad(unsharded_grad, [0, numel_to_pad])
    #         if numel_to_pad > 0
    #         else unsharded_grad
    #     )
    #     # =============================================================
    #     # Original implementation
    #     # new_sharded_grad = torch.empty_like(chunks[0])  # padded
    #     new_sharded_grad = chunks[0].clone().detach()
    #     # =============================================================
    #     return padded_unsharded_grad, new_sharded_grad

    # TODO(zhanda): Delete this if it's not used.
    # def _accumulate_sharded_grad(self, sharded_grad: torch.Tensor):
    #     flat_param = self.flat_param
    #     accumulate_grad = getattr(flat_param, "_saved_grad", None) is not None
    #     if accumulate_grad:
    #         _check_grad_to_accumulate(sharded_grad, flat_param._saved_grad)
    #         flat_param._saved_grad += sharded_grad
    #     else:
    #         flat_param._saved_grad = sharded_grad

    def _post_reduce_grad_callback(self):
        flat_param = self.flat_param
        assert flat_param._saved_grad is not None
        flat_param.grad = None
        # for param in flat_param._params:
        #     assert param.grad is None

    @property
    def uses_sharded_strategy(self) -> bool:
        return self.sharding_strategy != HandleShardingStrategy.NO_SHARD

    @property
    def uses_weight_sharding_strategy(self) -> bool:
        return self.sharding_strategy == HandleShardingStrategy.FULL_SHARD

    @property
    def uses_grad_sharding_strategy(self) -> bool:
        return self.sharding_strategy in GRAD_SHARD_HANDLE_STRATEGIES

    def __repr__(self):
        return f"FlatParamHandle({self.module.name})"


def _aligned_size(size, align):
    ret = (size + align - 1) // align * align
    ret = int(ret)
    return ret


def _get_aligned_numel(unsharded_dtype: torch.dtype):
    # NOTE: This alignment constraint comes from TorchInductor.
    ALIGNMENT = 16  # bytes
    unsharded_dtype_size = _get_dtype_size(unsharded_dtype)
    aligned_numel = ALIGNMENT // unsharded_dtype_size
    return aligned_numel


@lru_cache(8)
def _get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _detach_if_needed(
    param_or_tensor: Union[nn.Parameter, torch.Tensor]
) -> torch.Tensor:
    return (
        param_or_tensor.detach()
        if isinstance(param_or_tensor, nn.Parameter)
        else param_or_tensor
    )
