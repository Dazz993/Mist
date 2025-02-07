# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch

from dazzle.model_parallel_config import ModelParallelConfig
from dazzle.parallel_state import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    get_pipeline_parallel_group,
    get_pipeline_parallel_next_rank,
    get_pipeline_parallel_prev_rank,
    get_pipeline_parallel_rank,
)
from dazzle.logger import get_logger

logger = get_logger(__name__)

MAX_TENSOR_NDIM = 5

# Types
Shape = Union[List[int], torch.Size]
TensorProperty = Tuple[Shape, torch.dtype, bool]  # (shape, dtype, requires_grad)


def _communicate_shapes(
    tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    def _expand_to_max_ndim(tensor_shape):
        """Expand tensor shape to max ndim by prepending -1s."""
        tensor_shape = list(tensor_shape)
        return [-1] * (MAX_TENSOR_NDIM - len(tensor_shape)) + tensor_shape

    def _shrink_to_real_ndim(tensor_shape):
        """Shrink tensor shape to real ndim by removing leading -1s."""
        ret = [dim for dim in tensor_shape if dim != -1]
        return ret

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (MAX_TENSOR_NDIM), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (MAX_TENSOR_NDIM), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            _expand_to_max_ndim(tensor_send_prev.size()),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            _expand_to_max_ndim(tensor_send_next.size()),
            device=torch.cuda.current_device(),
            dtype=torch.int64,
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0] * MAX_TENSOR_NDIM
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()
        recv_prev_shape = _shrink_to_real_ndim(recv_prev_shape)

    recv_next_shape = [0] * MAX_TENSOR_NDIM
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()
        recv_next_shape = _shrink_to_real_ndim(recv_next_shape)

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            get_pipeline_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            get_pipeline_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            get_pipeline_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            get_pipeline_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
):
    reqs = []
    rank = get_pipeline_parallel_rank()
    if get_pipeline_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=get_pipeline_parallel_next_rank(),
                group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=get_pipeline_parallel_prev_rank(),
                group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=get_pipeline_parallel_prev_rank(),
                group=group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=get_pipeline_parallel_next_rank(),
                group=group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=get_pipeline_parallel_prev_rank(),
                group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=get_pipeline_parallel_next_rank(),
                group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=get_pipeline_parallel_next_rank(),
                group=group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=get_pipeline_parallel_prev_rank(),
                group=group,
            )
            reqs.append(send_prev_req)
    return reqs


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    recv_prev_tensor_property: Optional[TensorProperty],
    recv_next_tensor_property: Optional[TensorProperty],
    config: ModelParallelConfig,
    wait_on_reqs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        recv_prev_tensor_property (TensorProperty, optional):
            property of tensor to receive from previous rank,
            (shape, dtype, requires_grad). If recv_prev_tensor_property
            is None, then recv_prev will be set to False.

        recv_next_tensor_property (TensorProperty, optional):
            property of tensor to receive from next rank,
            (shape, dtype, requires_grad). If recv_next_tensor_property
            is None, then recv_next will be set to False.

        wait_on_reqs (boolean, optional, default=True):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if recv_prev and recv_prev_tensor_property is None:
        recv_prev = False
        logger.debug(
            f"RANK {get_pipeline_parallel_rank()} of WORLD SIZE {torch.distributed.get_world_size()}: "
            f"recv_prev_tensor_property is None, setting recv_prev to False"
        )
    if recv_next and recv_next_tensor_property is None:
        recv_next = False
        logger.debug(
            f"RANK {get_pipeline_parallel_rank()} of WORLD SIZE {torch.distributed.get_world_size()}: "
            f"recv_next_tensor_property is None, setting recv_next to False"
        )

    if recv_prev:
        assert (
            isinstance(recv_prev_tensor_property, tuple)
            and len(recv_prev_tensor_property) == 3
        ), f"recv_prev_tensor_property must be a tuple (shape, dtype, requires_grad), got {recv_prev_tensor_property}"
        (
            recv_prev_shape,
            recv_prev_dtype,
            recv_prev_requires_grad,
        ) = recv_prev_tensor_property
    if recv_next:
        assert (
            isinstance(recv_next_tensor_property, tuple)
            and len(recv_next_tensor_property) == 3
        ), f"recv_next_tensor_property must be a tuple (shape, dtype, requires_grad), got {recv_next_tensor_property}"
        (
            recv_next_shape,
            recv_next_dtype,
            recv_next_requires_grad,
        ) = recv_next_tensor_property

    if config.variable_seq_lengths:
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )

    assert not (
        recv_prev and recv_prev_shape is None
    ), f"recv_prev_shape is None when recv_prev is True"
    assert not (
        recv_next and recv_next_shape is None
    ), f"recv_next_shape is None when recv_next is True"

    if recv_prev:
        tensor_recv_prev = torch.empty(
            recv_prev_shape,
            requires_grad=recv_prev_requires_grad,
            device=torch.cuda.current_device(),
            dtype=recv_prev_dtype,
        )
    if recv_next:
        tensor_recv_next = torch.empty(
            recv_next_shape,
            requires_grad=recv_next_requires_grad,
            device=torch.cuda.current_device(),
            dtype=recv_next_dtype,
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=get_pipeline_parallel_group(),
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def mp_dict_logger(
    preface: str,
    dict_to_log: dict,
):
    """Helper function to log a dictionary.

    Arguments:
        preface (str, required):
            preface to log before the dictionary.

        dict_to_log (dict, required):
            dictionary to log.
    """

    def _process_item(key, value):
        if isinstance(value, torch.Tensor):
            return f"{key}: ({tuple(value.shape)} {value.dtype} {value.requires_grad})"
        else:
            return f"{key}: {value}"

    strings = [
        f"[RANK {get_pipeline_parallel_rank()}] {f'[{preface}]' if preface else ''}"
    ]
    for key, value in dict_to_log.items():
        strings.append(_process_item(key, value))

    logger.debug(", ".join(strings))


def recv_forward(
    tensor_property: TensorProperty, config: ModelParallelConfig
) -> torch.Tensor:
    """Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """

    mp_dict_logger("recv_forward begins", {"tensor_property": tensor_property})

    if is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers("forward-recv", log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            recv_prev_tensor_property=tensor_property,
            recv_next_tensor_property=None,
            config=config,
        )
        if config.timers is not None:
            config.timers("forward-recv").stop()

    mp_dict_logger(
        "recv_forward ends",
        {
            "tensor_property": tensor_property,
            "input_tensor": input_tensor,
        },
    )

    return input_tensor


def recv_backward(
    tensor_property: TensorProperty, config: ModelParallelConfig
) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """

    mp_dict_logger("recv_backward begins", {"tensor_property": tensor_property})

    if is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers("backward-recv", log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            recv_prev_tensor_property=None,
            recv_next_tensor_property=tensor_property,
            config=config,
        )
        if config.timers is not None:
            config.timers("backward-recv").stop()

    mp_dict_logger(
        "recv_backward ends",
        {
            "tensor_property": tensor_property,
            "output_tensor_grad": output_tensor_grad,
        },
    )

    return output_tensor_grad


def send_forward(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_forward begins",
        {
            "output_tensor": output_tensor,
        },
    )

    if not is_pipeline_last_stage():
        if config.timers is not None:
            config.timers("forward-send", log_level=2).start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            recv_prev_tensor_property=None,
            recv_next_tensor_property=None,
            config=config,
        )
        if config.timers is not None:
            config.timers("forward-send").stop()

    mp_dict_logger(
        "send_forward ends",
        {
            "output_tensor": output_tensor,
        },
    )


def send_backward(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_backward begins",
        {
            "input_tensor_grad": input_tensor_grad,
        },
    )

    if not is_pipeline_first_stage():
        if config.timers is not None:
            config.timers("backward-send", log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            recv_prev_tensor_property=None,
            recv_next_tensor_property=None,
            config=config,
        )
        if config.timers is not None:
            config.timers("backward-send").stop()

    mp_dict_logger(
        "send_backward ends",
        {
            "input_tensor_grad": input_tensor_grad,
        },
    )


def send_forward_recv_backward(
    output_tensor: torch.Tensor,
    tensor_property: TensorProperty,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_forward_recv_backward begins",
        {
            "output_tensor": output_tensor,
            "tensor_property": tensor_property,
        },
    )

    if is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers("forward-send-backward-recv", log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            recv_prev_tensor_property=None,
            recv_next_tensor_property=tensor_property,
            config=config,
        )
        if config.timers is not None:
            config.timers("forward-send-backward-recv").stop()

    mp_dict_logger(
        "send_forward_recv_backward ends",
        {
            "output_tensor": output_tensor,
            "tensor_property": tensor_property,
            "output_tensor_grad": output_tensor_grad,
        },
    )

    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor,
    tensor_property: TensorProperty,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_backward_recv_forward begins",
        {
            "input_tensor_grad": input_tensor_grad,
            "tensor_property": tensor_property,
        },
    )

    if is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers("backward-send-forward-recv", log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            recv_prev_tensor_property=tensor_property,
            recv_next_tensor_property=None,
            config=config,
        )
        if config.timers is not None:
            config.timers("backward-send-forward-recv").stop()

    mp_dict_logger(
        "send_backward_recv_forward ends",
        {
            "input_tensor_grad": input_tensor_grad,
            "tensor_property": tensor_property,
            "input_tensor": input_tensor,
        },
    )

    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_property: TensorProperty,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_forward_recv_forward begins",
        {
            "output_tensor": output_tensor,
            "recv_prev": recv_prev,
            "tensor_property": tensor_property,
        },
    )

    if tensor_property is None:
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers("forward-send-forward-recv", log_level=2).start()
        input_tensor, _, wait_handles = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            recv_prev_tensor_property=tensor_property,
            recv_next_tensor_property=None,
            wait_on_reqs=(not overlap_p2p_comm),
            config=config,
        )
        if config.timers is not None:
            config.timers("forward-send-forward-recv").stop()
        if overlap_p2p_comm:
            return input_tensor, wait_handles

    mp_dict_logger(
        "send_forward_recv_forward ends",
        {
            "output_tensor": output_tensor,
            "recv_prev": recv_prev,
            "tensor_property": tensor_property,
            "input_tensor": input_tensor,
        },
    )

    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_property: TensorProperty,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_backward_recv_backward begins",
        {
            "input_tensor_grad": input_tensor_grad,
            "recv_next": recv_next,
            "tensor_property": tensor_property,
        },
    )

    if config.timers is not None:
        config.timers("backward-send-backward-recv", log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        recv_prev_tensor_property=None,
        recv_next_tensor_property=tensor_property,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers("backward-send-backward-recv").stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles

    mp_dict_logger(
        "send_backward_recv_backward ends",
        {
            "input_tensor_grad": input_tensor_grad,
            "recv_next": recv_next,
            "tensor_property": tensor_property,
            "output_tensor_grad": output_tensor_grad,
        },
    )

    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    recv_forward_tensor_property: TensorProperty,
    recv_backward_tensor_property: TensorProperty,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """

    mp_dict_logger(
        "send_forward_backward_recv_forward_backward begins",
        {
            "output_tensor": output_tensor,
            "input_tensor_grad": input_tensor_grad,
            "recv_prev": recv_prev,
            "recv_next": recv_next,
            "recv_forward_tensor_property": recv_forward_tensor_property,
            "recv_backward_tensor_property": recv_backward_tensor_property,
        },
    )

    if config.timers is not None:
        config.timers(
            "forward-backward-send-forward-backward-recv", log_level=2
        ).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        recv_prev_tensor_property=recv_forward_tensor_property,
        recv_next_tensor_property=recv_backward_tensor_property,
        config=config,
    )
    if config.timers is not None:
        config.timers("forward-backward-send-forward-backward-recv").stop()

    mp_dict_logger(
        "send_forward_backward_recv_forward_backward ends",
        {
            "output_tensor": output_tensor,
            "input_tensor_grad": input_tensor_grad,
            "recv_prev": recv_prev,
            "recv_next": recv_next,
            "recv_forward_tensor_property": recv_forward_tensor_property,
            "recv_backward_tensor_property": recv_backward_tensor_property,
            "input_tensor": input_tensor,
            "output_tensor_grad": output_tensor_grad,
        },
    )

    return input_tensor, output_tensor_grad
