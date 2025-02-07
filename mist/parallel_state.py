from datetime import timedelta
from typing import List, Tuple, Dict
from pprint import pformat
from dataclasses import dataclass

import torch
import torch.distributed as dist

from mist.config import MistConfig
from mist.logger import get_logger

logger = get_logger()

# Inited
_INITIALIZED = False

# DATA Parallel group
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GLOBAL_RANKS = None
_DATA_PARALLEL_ALL_GATHER_GROUP = None
_DATA_PARALLEL_REDUCE_SCATTER_GROUP = None

# TENSOR Parallel group
_TENSOR_PARALLEL_GROUP = None
_TENSOR_PARALLEL_GLOBAL_RANKS = None

# Pre- and Post- Processing group
_PRE_POST_DATA_PARALLEL_GROUP = None
_PRE_POST_DATA_PARALLEL_GLOBAL_RANKS = None
_PRE_POST_TENSOR_PARALLEL_GROUP = None
_PRE_POST_TENSOR_PARALLEL_GLOBAL_RANKS = None

# Pipeline parallel helper
_RANK_TO_STAGE_IDX = None
_STAGE_IDX_TO_RANKS = None
_STAGE_IDX = None

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_PARALLEL_FORWARD_SEND_NEXT_INFOS = []
_PIPELINE_PARALLEL_FORWARD_RECV_PREV_INFOS = []


@dataclass(frozen=True)
class ShardInfo:
    is_sharded: bool
    shard_idx: int = 0
    num_shards: int = 1

    def __repr__(self):
        if not self.is_sharded or self.num_shards == 1:
            return f"ShardInfo(No-Shard)"
        else:
            return f"ShardInfo(Shard {self.shard_idx} of {self.num_shards})"


# Embedding group
_EMBEDDING_GROUP = None
_EMBEDDING_GLOBAL_RANKS = None
_POSITION_EMBEDDING_GROUP = None
_POSITION_EMBEDDING_GLOBAL_RANKS = None


def _map_stage_idx_to_ranks(
    device_assignment: List[Tuple[int, int]],
    global_device_mesh: Tuple[int, int],
):
    stage_idx_to_ranks = {}
    _fully_spare_nodes = list(range(global_device_mesh[0]))
    _fragmented_nodes = {}

    for stage_idx, (n, m) in enumerate(device_assignment):
        # If the stage uses all gpus in some nodes, then we need to make sure
        # that the nodes for this stage are fully spare.
        if m == global_device_mesh[1]:
            assert len(_fully_spare_nodes) >= n
            node_indices = _fully_spare_nodes[:n]
            _fully_spare_nodes = _fully_spare_nodes[n:]
            start_node_idx = node_indices[0]
            stage_idx_to_ranks[stage_idx] = list(
                range(
                    start_node_idx * global_device_mesh[1],
                    (start_node_idx + n) * global_device_mesh[1],
                )
            )
        # If the stage uses only part of the gpus in some nodes, then we need to
        # find a partially spare node for this stage.
        else:
            # Judge whether there is a partially spare node for this stage.
            suitable_partially_spare_node = None
            for node_idx, num_used_gpus in _fragmented_nodes.items():
                if num_used_gpus + m <= global_device_mesh[1]:
                    suitable_partially_spare_node = node_idx
                    break

            # If there is a partially spare node, use it.
            if suitable_partially_spare_node is not None:
                node_idx = suitable_partially_spare_node
                start_idx = (
                    node_idx * global_device_mesh[1] + _fragmented_nodes[node_idx]
                )
                _fragmented_nodes[node_idx] += m
            # If there is no partially spare node, split a fully spare node.
            else:
                assert len(_fully_spare_nodes) > 0
                node_idx = _fully_spare_nodes.pop(0)
                start_idx = node_idx * global_device_mesh[1]
                _fragmented_nodes[node_idx] = m

            end_idx = start_idx + m
            stage_idx_to_ranks[stage_idx] = list(range(start_idx, end_idx))

    return stage_idx_to_ranks


def initialize_parallel(
    mist_config: MistConfig,
):
    """Initialize the parallel state."""
    global _INITIALIZED
    assert not _INITIALIZED, "parallel state is already initialized"
    _INITIALIZED = True

    # Get world size and rank and make sure they are consistent.
    assert torch.distributed.is_initialized()
    global_rank: int = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()

    # Preparations
    hardware_config = mist_config.hardware
    strategy_config = mist_config.strategy

    global_device_mesh = (hardware_config.num_nodes, hardware_config.num_gpus_per_node)
    num_stages = strategy_config.num_stages
    stage_strategies = strategy_config.stage_strategies
    device_assignment = strategy_config.device_assignment

    # Pipeline parallel
    # Match the stage index to the ranks and the rank to the stage index.
    global _RANK_TO_STAGE_IDX
    global _STAGE_IDX_TO_RANKS
    global _STAGE_IDX
    stage_idx_to_ranks: Dict[int, List[int]] = _map_stage_idx_to_ranks(
        device_assignment=device_assignment,
        global_device_mesh=global_device_mesh,
    )
    rank_to_stage_idx = {}
    for stage_idx, ranks in stage_idx_to_ranks.items():
        for rank in ranks:
            rank_to_stage_idx[rank] = stage_idx
    _RANK_TO_STAGE_IDX = rank_to_stage_idx
    _STAGE_IDX_TO_RANKS = stage_idx_to_ranks
    _STAGE_IDX = rank_to_stage_idx[global_rank]
    if global_rank == 0:
        logger.info(f"Inited pipeline parallel state with {num_stages} stages.")
        logger.info(f"Rank to StageIdx: {rank_to_stage_idx}")
        logger.info(f"StageIdx to Ranks: {stage_idx_to_ranks}")
    torch.distributed.barrier()

    # Because different pipeline stage can have differnt dp_size and tp_size,
    # thus the pipeline transfer mapping can be complicated.
    global _PIPELINE_PARALLEL_FORWARD_SEND_NEXT_INFOS
    global _PIPELINE_PARALLEL_FORWARD_RECV_PREV_INFOS
    for stage_idx in range(num_stages - 1):
        curr_stage_idx = stage_idx
        next_stage_idx = stage_idx + 1
        curr_stage_strategy = stage_strategies[curr_stage_idx]
        next_stage_strategy = stage_strategies[next_stage_idx]
        curr_bsz_per_device = curr_stage_strategy[0]
        curr_dp_size = curr_stage_strategy[1]
        curr_tp_size = curr_stage_strategy[2]
        next_bsz_per_device = next_stage_strategy[0]
        next_dp_size = next_stage_strategy[1]
        next_tp_size = next_stage_strategy[2]

        dp_tp_usage = [[0 for _ in range(curr_tp_size)] for _ in range(curr_dp_size)]
        curr_selected_tp_indices = [0 for _ in range(curr_dp_size)]

        def next_available_tp_idx(curr_dp_idx):
            curr_selected_tp_idx = curr_selected_tp_indices[curr_dp_idx]
            next_selected_tp_idx = (curr_selected_tp_idx + 1) % curr_tp_size
            if (
                dp_tp_usage[curr_dp_idx][next_selected_tp_idx]
                < dp_tp_usage[curr_dp_idx][curr_selected_tp_idx]
            ):
                ret = next_selected_tp_idx
            else:
                ret = curr_selected_tp_idx
            curr_selected_tp_indices[curr_dp_idx] = ret
            dp_tp_usage[curr_dp_idx][ret] += 1
            return ret

        for next_dp_idx in range(next_dp_size):
            if curr_bsz_per_device >= next_bsz_per_device:
                # Next: dp[0, 3], dp[4, 8]
                #        \   /     \   /
                # Curr:   dp0,      dp1
                assert curr_dp_size <= next_dp_size
                dp_factor = curr_bsz_per_device // next_bsz_per_device
                # The rank in a dp group is enough to send data to the next stage.
                curr_dp_indices = [next_dp_idx // dp_factor]
                send_shards = (
                    [ShardInfo(True, next_dp_idx % dp_factor, dp_factor)]
                    if dp_factor > 1
                    else [ShardInfo(False)]
                )
                recv_shards = [ShardInfo(False)]
            else:  # curr_bsz_per_device < next_bsz_per_device
                # Curr: dp[0, 3], dp[4, 8]
                #        \   /     \   /
                # Next:   dp0,      dp1
                dp_factor = next_bsz_per_device // curr_bsz_per_device
                curr_dp_indices = list(
                    range(next_dp_idx * dp_factor, (next_dp_idx + 1) * dp_factor)
                )
                send_shards = [ShardInfo(False) for _ in range(dp_factor)]
                recv_shards = [
                    ShardInfo(True, curr_dp_idx % dp_factor, dp_factor)
                    for curr_dp_idx in curr_dp_indices
                ]
            for next_tp_idx in range(next_tp_size):
                next_rank = _STAGE_IDX_TO_RANKS[next_stage_idx][
                    next_dp_idx * next_tp_size + next_tp_idx
                ]
                for curr_dp_idx, send_shard, recv_shard in zip(
                    curr_dp_indices, send_shards, recv_shards
                ):
                    # To fully utilize the bandwidth
                    curr_tp_idx = next_available_tp_idx(curr_dp_idx)
                    curr_rank = _STAGE_IDX_TO_RANKS[curr_stage_idx][
                        curr_dp_idx * curr_tp_size + curr_tp_idx
                    ]
                    if global_rank == curr_rank and _STAGE_IDX == curr_stage_idx:
                        _PIPELINE_PARALLEL_FORWARD_SEND_NEXT_INFOS.append(
                            (next_rank, send_shard)
                        )
                    elif global_rank == next_rank and _STAGE_IDX == next_stage_idx:
                        _PIPELINE_PARALLEL_FORWARD_RECV_PREV_INFOS.append(
                            (curr_rank, recv_shard)
                        )

    logger.info(
        f"[Rank {global_rank} Pipeline Parallel Forward Send Next Ranks: {_PIPELINE_PARALLEL_FORWARD_SEND_NEXT_INFOS}"
    )
    logger.info(
        f"[Rank {global_rank} Pipeline Parallel Forward Recv Prev Ranks: {_PIPELINE_PARALLEL_FORWARD_RECV_PREV_INFOS}"
    )

    global _TENSOR_PARALLEL_GROUP
    global _TENSOR_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_ALL_GATHER_GROUP
    global _DATA_PARALLEL_REDUCE_SCATTER_GROUP
    for stage_idx in range(num_stages):
        # Build data parallel and tensor parallel groups
        stage_ranks = stage_idx_to_ranks[stage_idx]
        stage_strategy = stage_strategies[stage_idx]
        dp_size = stage_strategy[1]
        tp_size = stage_strategy[2]
        assert dp_size * tp_size == len(stage_ranks), (
            f"Stage {stage_idx} has {len(stage_ranks)} ranks, "
            f"but the product of dp_size ({dp_size}) and "
            f"tp_size ({tp_size}) is {dp_size * tp_size}"
        )

        for dp_idx in range(dp_size):
            start = dp_idx * tp_size
            end = (dp_idx + 1) * tp_size
            tp_ranks = stage_ranks[start:end]
            tp_group = dist.new_group(tp_ranks, timeout=timedelta(seconds=mist_config.nccl_timeout))
            if global_rank in tp_ranks:
                _TENSOR_PARALLEL_GROUP = tp_group
                _TENSOR_PARALLEL_GLOBAL_RANKS = tp_ranks

        for tp_idx in range(tp_size):
            dp_ranks = stage_ranks[tp_idx::tp_size]
            dp_ranks = sorted(list(set(dp_ranks)))
            dp_group = dist.new_group(dp_ranks, timeout=timedelta(seconds=mist_config.nccl_timeout))
            dp_all_gather_group = dist.new_group(dp_ranks, timeout=timedelta(seconds=mist_config.nccl_timeout))
            dp_reduce_scatter_group = dist.new_group(dp_ranks, timeout=timedelta(seconds=mist_config.nccl_timeout))
            if global_rank in dp_ranks:
                _DATA_PARALLEL_GROUP = dp_group
                _DATA_PARALLEL_GLOBAL_RANKS = dp_ranks
                _DATA_PARALLEL_ALL_GATHER_GROUP = dp_all_gather_group
                _DATA_PARALLEL_REDUCE_SCATTER_GROUP = dp_reduce_scatter_group

    logger.info(
        f"[Rank {global_rank}] Tensor Parallel Global Ranks: {_TENSOR_PARALLEL_GLOBAL_RANKS}"
    )
    logger.info(
        f"[Rank {global_rank}] Data Parallel Global Ranks: {_DATA_PARALLEL_GLOBAL_RANKS}"
    )


def get_num_pipeline_stages():
    assert _INITIALIZED, "parallel state is not initialized"
    return len(_STAGE_IDX_TO_RANKS)


def get_global_ranks_of_stage(stage_idx=None):
    assert _INITIALIZED, "parallel state is not initialized"
    if stage_idx is None:
        stage_idx = _STAGE_IDX
    return _STAGE_IDX_TO_RANKS[stage_idx]


def get_pipeline_parallel_stage_idx(global_rank=None):
    """Return the stage index of the caller."""
    assert _INITIALIZED, "parallel state is not initialized"
    if global_rank is None:
        return _STAGE_IDX  # = _RANK_TO_STAGE_IDX[torch.distributed.get_rank()]
    return _RANK_TO_STAGE_IDX[global_rank]


def get_pipeline_parallel_forward_send_next_infos():
    """Return the global rank that follows the caller in the pipeline"""
    assert _INITIALIZED, "parallel state is not initialized"
    return _PIPELINE_PARALLEL_FORWARD_SEND_NEXT_INFOS


def get_pipeline_parallel_forward_recv_prev_infos():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _INITIALIZED, "parallel state is not initialized"
    return _PIPELINE_PARALLEL_FORWARD_RECV_PREV_INFOS


def get_pipeline_parallel_backward_send_prev_ranks():
    """Return the global rank that follows the caller in the pipeline"""
    assert _INITIALIZED, "parallel state is not initialized"
    return _PIPELINE_PARALLEL_FORWARD_RECV_PREV_INFOS


def get_pipeline_parallel_backward_recv_next_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _INITIALIZED, "parallel state is not initialized"
    return _PIPELINE_PARALLEL_FORWARD_SEND_NEXT_INFOS


def is_pipeline_first_stage():
    """Return True if in the first pipeline parallel stage, False otherwise."""
    assert _INITIALIZED, "parallel state is not initialized"
    return get_pipeline_parallel_stage_idx() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline parallel stage, False otherwise."""
    return get_pipeline_parallel_stage_idx() == get_num_pipeline_stages() - 1


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _INITIALIZED, "parallel state is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    assert _INITIALIZED, "parallel state is not initialized"
    if get_data_parallel_group() is None:
        return 1
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_global_ranks():
    """Return global ranks for the data parallel group."""
    assert _INITIALIZED, "parallel state is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS


def get_data_parallel_rank():
    """Return rank for the data parallel group."""
    assert _INITIALIZED, "parallel state is not initialized"
    if get_data_parallel_group() is None:
        return 0
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_data_parallel_all_gather_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _INITIALIZED, "parallel state is not initialized"
    return _DATA_PARALLEL_ALL_GATHER_GROUP


def get_data_parallel_reduce_scatter_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _INITIALIZED, "parallel state is not initialized"
    return _DATA_PARALLEL_REDUCE_SCATTER_GROUP


def get_tensor_parallel_group():
    """Get the tensor parallel group the caller rank belongs to."""
    assert _INITIALIZED, "parallel state is not initialized"
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_world_size():
    """Return world size for the tensor parallel group."""
    assert _INITIALIZED, "parallel state is not initialized"
    if get_tensor_parallel_group() is None:
        return 1
    return torch.distributed.get_world_size(group=get_tensor_parallel_group())


def get_tensor_parallel_global_ranks():
    """Return global ranks for the tensor parallel group."""
    assert _INITIALIZED, "parallel state is not initialized"
    return _TENSOR_PARALLEL_GLOBAL_RANKS


def get_tensor_parallel_rank():
    """Return rank for the tensor parallel group."""
    assert _INITIALIZED, "parallel state is not initialized"
    if get_tensor_parallel_group() is None:
        return 0
    return torch.distributed.get_rank(group=get_tensor_parallel_group())
