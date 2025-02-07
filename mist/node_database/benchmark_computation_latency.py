from __future__ import annotations
from typing import (
    Optional,
    Callable,
    Union,
    Tuple,
    Dict,
    List,
    Sequence,
    Any,
    TYPE_CHECKING,
)
from functools import partial
from tqdm import tqdm
from time import perf_counter

import torch
import numpy as np
import random

from torch.utils._pytree import tree_map

if TYPE_CHECKING:
    from mist.node_database.node_spec import NodeSpec

# Benchmarking settings
# 1. Whether to flash GPU L2 cache before each benchmarking iteration
FLASH_L2 = True
# 2. Whether to zero out the gradiesnt before each benchmarking iteration
ZERO_GRAD = False
# 3. Seconds to benchmark
SECONDS_TO_BENCHMARK = 1
MAX_ITERS_TO_BENCHMARK = 1000
MIN_ITERS_TO_BENCHMARK = 10


def benchmark_node(
    node_spec: NodeSpec,
    forward_only: bool = False,
    device=None,
) -> Tuple[float, float]:
    fn = node_spec.materialize_target(device=device)

    def prepare_func():
        if ZERO_GRAD:
            if isinstance(fn, torch.nn.Module):
                fn.zero_grad()

        if FLASH_L2:
            return node_spec.materialize_inputs(device=device)

        return None

    def run_func(*args, **kwargs):
        ouput = fn(*args, **kwargs)
        return ouput

    if not FLASH_L2:
        args, kwargs = node_spec.materialize_inputs(device=device)
        run_func = partial(run_func, *args, **kwargs)

    reference_latencies = benchmark_func_for_fwd(
        run_func,
        warmup=MIN_ITERS_TO_BENCHMARK,
        number=MIN_ITERS_TO_BENCHMARK,
        prepare_func=prepare_func,
        sync_func=torch.cuda.synchronize,
        enable_tqdm=False,
    )
    n = int(SECONDS_TO_BENCHMARK / np.median(np.array(reference_latencies)).item())
    n = max(min(n, MIN_ITERS_TO_BENCHMARK), MIN_ITERS_TO_BENCHMARK)

    if forward_only:
        fwd_latencies = benchmark_func_for_fwd(
            run_func,
            warmup=n,
            number=n,
            prepare_func=prepare_func,
            sync_func=torch.cuda.synchronize,
            enable_tqdm=False,
        )

        return fwd_latencies, np.array([0.0])

    else:
        try:
            fwd_latencies, bwd_latencies = benchmark_func_for_fwd_bwd(
                run_func,
                warmup=n * 2,
                number=n,
                prepare_func=prepare_func,
                sync_func=torch.cuda.synchronize,
                enable_tqdm=False,
            )
        except torch.cuda.OutOfMemoryError:
            return np.array([np.inf]), np.array([np.inf])

        return fwd_latencies, bwd_latencies


def benchmark_func_for_fwd(
    fwd_func,
    warmup,
    number,
    prepare_func=None,
    sync_func=None,
    enable_tqdm=False,
):
    sync_func = sync_func or (lambda: None)
    _tqdm = partial(tqdm, disable=not enable_tqdm)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]

    def _run_func(step=-1):
        start_event = start_events[step]
        end_event = end_events[step]

        # Generate inputs if prepare_func is provided
        inputs = prepare_func() if prepare_func is not None else None

        if inputs is not None:
            assert isinstance(inputs, tuple) and len(inputs) == 2
            args, kwargs = inputs
            cur_fwd_func = partial(fwd_func, *args, **kwargs)
        else:
            cur_fwd_func = fwd_func

        # torch.cuda._sleep(1000000)

        # Run the function and do synchronization
        # Run the forward
        start_event.record()
        cur_fwd_func()
        end_event.record()

    # Warmup
    for _ in _tqdm(range(warmup)):
        _run_func()

    # Benchmark
    for step in _tqdm(range(number)):
        _run_func(step)

    torch.cuda.synchronize()
    costs = [
        start_events[i].elapsed_time(end_events[i]) / 1000.0 for i in range(number)
    ]

    return np.array(costs)


def benchmark_func_for_fwd_bwd(
    fwd_func,
    warmup,
    number,
    prepare_func=None,
    sync_func=None,
    enable_tqdm=False,
):
    sync_func = sync_func or (lambda: None)
    _tqdm = partial(tqdm, disable=not enable_tqdm)

    fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]
    fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]
    bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]
    bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(number + 1)]

    # if prepare_func is not None and prepare_func() is not None:
    #     raise NotImplementedError(f"prepare_func is not supported yet")
    # assert hasattr(fwd_func, "args") and hasattr(fwd_func, "keywords")
    # inputs = []
    # inputs.extend(
    #     t for t in fwd_func.args if isinstance(t, torch.Tensor) and t.requires_grad
    # )
    # inputs.extend(
    #     t
    #     for t in fwd_func.keywords.values()
    #     if isinstance(t, torch.Tensor) and t.requires_grad
    # )

    def _run_func(step=-1):
        fwd_start_event = fwd_start_events[step]
        fwd_end_event = fwd_end_events[step]
        bwd_start_event = bwd_start_events[step]
        bwd_end_event = bwd_end_events[step]

        if prepare_func is not None and (inputs := prepare_func()) is not None:
            assert isinstance(inputs, tuple) and len(inputs) == 2
            args, kwargs = inputs
            cur_fwd_func = partial(fwd_func, *args, **kwargs)
        else:
            cur_fwd_func = fwd_func

        # Run the forward
        # torch.cuda._sleep(1000000)
        fwd_start_event.record()
        output = cur_fwd_func()
        fwd_end_event.record()

        grad_tensor = torch.ones_like(output) * random.random()

        # Run the backward
        # torch.cuda._sleep(1000000)
        bwd_start_event.record()
        torch.autograd.backward(output, grad_tensors=grad_tensor)
        # torch.autograd.grad(outputs=output, inputs=inputs, grad_outputs=grad_tensor)
        bwd_end_event.record()

    # Warmup
    for _ in _tqdm(range(warmup)):
        _run_func()

    # Benchmark
    for step in _tqdm(range(number)):
        _run_func(step)

    torch.cuda.synchronize()
    costs_fwd = [
        fwd_start_events[i].elapsed_time(fwd_end_events[i]) / 1000.0
        for i in range(number)
    ]
    costs_bwd = [
        bwd_start_events[i].elapsed_time(bwd_end_events[i]) / 1000.0
        for i in range(number)
    ]

    return np.array(costs_fwd), np.array(costs_bwd)
