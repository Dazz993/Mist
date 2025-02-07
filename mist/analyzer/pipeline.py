from __future__ import annotations
from typing import List, Tuple
from time import perf_counter
from numbers import Number

import sympy as sp


class Node:
    def __init__(
        self,
        name: str,
        duration: float,
        prevs: List[Node] = None,
        nexts: List[Node] = None,
    ):
        self.name = name
        self.prevs = prevs or []
        self.nexts = nexts or []
        self.duration = duration

    def __repr__(self):
        return f"Node({self.name}){[n.name for n in self.prevs]}"


def latency_for_pipe(num_stages: int, num_micro_batches: int, latencies: List[float]):
    assert num_micro_batches >= num_stages
    assert isinstance(latencies, list), "latencies must be a list"
    assert (
        len(latencies) == num_stages
    ), "latencies must have the same length as num_stages"
    assert all(
        len(latencies_per_stage) == num_micro_batches * 2
        for latencies_per_stage in latencies
    ), "latencies must have the same length as num_micro_batches * 2"

    _time_graph_creation_start = perf_counter()

    # Create nodes
    nodes = []
    fwd_nodes = []
    bwd_nodes = []
    warmup_nodes = []
    fb_nodes = []
    cooldown_nodes = []
    for p in range(num_stages):
        stage_latencies = latencies[p]
        nodes_in_stage = []
        fwd_in_stage = []
        bwd_in_stage = []
        warmup_in_stage = []
        fb_in_stage = []
        cooldown_in_stage = []
        warmup = num_stages - p - 1
        fwd, bwd = 0, 0

        # Warmup: all nodes are the fwd nodes
        for mb in range(warmup):
            node = Node(f"p{p}_fwd{fwd}", stage_latencies[fwd + bwd])
            fwd += 1
            nodes_in_stage.append(node)
            fwd_in_stage.append(node)
            warmup_in_stage.append(node)

        # 1F1B: a fwd node and a bwd node
        for mb in range(num_micro_batches - warmup):
            fwd_node = Node(f"p{p}_fwd{fwd}", stage_latencies[fwd + bwd])
            fwd += 1
            nodes_in_stage.append(fwd_node)
            fwd_in_stage.append(fwd_node)
            fb_in_stage.append(fwd_node)
            bwd_node = Node(f"p{p}_bwd{bwd}", stage_latencies[fwd + bwd])
            bwd += 1
            nodes_in_stage.append(bwd_node)
            bwd_in_stage.append(bwd_node)
            fb_in_stage.append(bwd_node)

        # Cooldown: all nodes are the bwd nodes
        for mb in range(warmup):
            node = Node(f"p{p}_bwd{bwd}", stage_latencies[fwd + bwd])
            bwd += 1
            nodes_in_stage.append(node)
            bwd_in_stage.append(node)
            cooldown_in_stage.append(node)

        nodes.append(nodes_in_stage)
        fwd_nodes.append(fwd_in_stage)
        bwd_nodes.append(bwd_in_stage)
        warmup_nodes.append(warmup_in_stage)
        fb_nodes.append(fb_in_stage)
        cooldown_nodes.append(cooldown_in_stage)

    # Link nodes in each stage
    for p in range(num_stages):
        for mb in range(num_micro_batches * 2):
            if mb != 0:
                nodes[p][mb].prevs.append(nodes[p][mb - 1])

    # Link nodes with fwd and bwd
    for p in range(num_stages):
        for mb in range(num_micro_batches):
            if p != 0:
                fwd_nodes[p][mb].prevs.append(fwd_nodes[p - 1][mb])
            if p != num_stages - 1:
                bwd_nodes[p][mb].prevs.append(bwd_nodes[p + 1][mb])

    _time_graph_creation_end = perf_counter()

    # Calculate the latency for each node
    node2end_time = {}

    def calculate_end_time(node):
        if node in node2end_time:
            return node2end_time[node]
        if not node.prevs:
            node2end_time[node] = node.duration
            return node.duration
        end_time = (
            sp.Max(*(calculate_end_time(prev) for prev in node.prevs)) + node.duration
        )
        node2end_time[node] = end_time
        return end_time

    end_time = calculate_end_time(nodes[0][-1])

    _time_calculation_end = perf_counter()

    # Calculate time for the processing
    # print(f"Graph creation and linking: {time_2 - time_1:.6f} s")
    # print(f"Latency calculation: {time_3 - time_2:.6f} s")

    return end_time


def latency_for_pipe_with_fixed_time_in_stage(
    num_stages: int, num_micro_batches: int, latencies: List[float]
):
    assert len(latencies) == num_stages
    assert all(isinstance(l, (Number, sp.Basic)) for l in latencies)

    # longest_latency = sp.Max(*latencies)
    longest_latency = max(latencies)
    latency = sum(latencies) + longest_latency * (num_micro_batches - 1)
    return latency


if __name__ == "__main__":
    from tqdm import tqdm

    # latencies = [[1] * 64] * 16
    # for i in tqdm(range(10000)):
    #     latency_for_pipe(num_stages=16, num_micro_batches=32, latencies=latencies)

    time = perf_counter()
    # fwd_latencies = [sp.symbols(f"f{i}") for i in range(16)]
    # bwd_latencies = [sp.symbols(f"b{i}") for i in range(16)]
    fwd_latencies = [1] * 16
    bwd_latencies = [1] * 16
    fwd_end_time = latency_for_pipe_with_fixed_time_in_stage(
        num_stages=16, num_micro_batches=32, latencies=fwd_latencies
    )
    bwd_end_time = latency_for_pipe_with_fixed_time_in_stage(
        num_stages=16, num_micro_batches=32, latencies=bwd_latencies
    )
    print(f"Symbolic Pipe Latency Construction Time: {perf_counter() - time:.6f} s")
