import inspect
from copy import copy
from itertools import chain
from functools import wraps
from types import MethodType
from typing import Optional, Sequence, Dict, Any, List

import torch
from torch import nn, fx
from torch.utils._pytree import tree_map

from mist.logger import get_logger
from mist.tracer.symbolic_tracer import get_default_sub_modules
from mist.utils.hf import fix_hf_model
from mist.utils.fx import create_graph_module, recursively_create_graph_modules
from mist.utils.module import getattr_recursive
from mist.tracer.symbolic_tracer import device_safe_func_exec
from mist.utils.device import get_device
from mist.utils.memory import cuda_empty_cache, make_viewless_tensor
from mist.utils.inspect import map_args_kwargs_to_args
from mist.utils.wraps import wraps_nn_module_forward

logger = get_logger(__name__)


def _extract_node_sequence_for_each_block(
    model, root_graph, sub_modules: Optional[Sequence[str]] = None
):
    sub_modules = sub_modules or get_default_sub_modules(model)
    node_sequence_pre_layer = []
    node_sequence_blocks = [[] for _ in range(len(sub_modules))]
    node_sequence_post_layer = []

    nodes = list(root_graph.nodes)

    def find_potential_getitem_nodes(nodes, idx):
        anchor_node = nodes[idx]
        ret = [anchor_node]
        for i in range(idx + 1, len(nodes)):
            node = nodes[i]
            if (
                node.op == "call_function"
                and node.target.__name__ == "getitem"
                and node.args[0] is anchor_node
            ):
                ret.append(node)
        return ret

    last_block_end_node = None
    for i, node in enumerate(nodes):
        if node.op == "call_module" and node.target in sub_modules:
            follower_getitem_nodes = find_potential_getitem_nodes(nodes, i)
            last_block_end_node = follower_getitem_nodes[-1]

    # Pre layer
    for node in root_graph.nodes:
        if node.op == "call_module" and node.target in sub_modules:
            break
        node_sequence_pre_layer.append(node)

    # Blocks
    block_idx = -1
    for node in root_graph.nodes:
        if node.op == "call_module" and node.target in sub_modules:
            block_idx += 1
        if 0 <= block_idx < len(sub_modules):
            node_sequence_blocks[block_idx].append(node)
        if node is last_block_end_node:
            break

    # Post layer
    valid = False
    for node in root_graph.nodes:
        if node is last_block_end_node:
            valid = True
            continue
        if valid:
            node_sequence_post_layer.append(node)

    assert len(node_sequence_pre_layer) + sum(
        len(node_sequence_block) for node_sequence_block in node_sequence_blocks
    ) + len(node_sequence_post_layer) == len(root_graph.nodes)

    return node_sequence_pre_layer, node_sequence_blocks, node_sequence_post_layer


def _analyze_inputs_outputs_for_node_sequence(node_sequence):
    """
    From a node sequence, analyze the input and output nodes.
    """
    input_nodes = set()
    node2num_users = {node: len(node.users) for node in node_sequence}

    for node in node_sequence:
        for input_node in node._input_nodes:
            if input_node not in node_sequence:
                input_nodes.add(input_node)
            else:
                node2num_users[input_node] -= 1

    output_nodes = [node for node in node_sequence if node2num_users[node] != 0]
    return list(input_nodes), output_nodes


def _analyze_input_output_nodes_liveness(
    input_nodes_for_each_stage, output_nodes_for_each_stage
):
    """
    Liveness analysis for input and output nodes for each stage.
    Returns a dict mapping each node to a tuple of (first_stage_as_output, last_stage_as_input).
    """
    if len(input_nodes_for_each_stage) != len(output_nodes_for_each_stage):
        raise ValueError(
            f"Number of input stages ({len(input_nodes_for_each_stage)}) "
            f"does not match number of output stages ({len(output_nodes_for_each_stage)})"
        )

    num_stages = len(input_nodes_for_each_stage)
    node2last_input = {}
    node2first_output = {}
    node2range = {}
    for i, (input_nodes, output_nodes) in enumerate(
        zip(input_nodes_for_each_stage, output_nodes_for_each_stage)
    ):
        for node in input_nodes:
            node2last_input[node] = max(i, node2last_input.get(node, -1))
        for node in output_nodes:
            node2first_output[node] = min(i, node2first_output.get(node, num_stages))
    for node in chain(node2last_input, node2first_output):
        node2range[node] = (node2first_output[node], node2last_input[node])
    logger.debug(f"Node range: {node2range}")
    return node2range


def _update_input_output_nodes_for_each_stage_according_to_liveness(
    input_nodes_for_each_stage, output_nodes_for_each_stage, node2range
):
    """
    Update the input and output nodes for each stage according to the liveness analysis.
    """
    updated_input_nodes_for_each_stage = copy(input_nodes_for_each_stage)
    updated_output_nodes_for_each_stage = copy(output_nodes_for_each_stage)
    for node, (start, end) in node2range.items():
        for idx in range(start, end):
            if node not in updated_output_nodes_for_each_stage[idx]:
                updated_output_nodes_for_each_stage[idx].append(node)
        for idx in range(start + 1, end + 1):
            if node not in updated_input_nodes_for_each_stage[idx]:
                updated_input_nodes_for_each_stage[idx].append(node)
    logger.debug(
        f"Updated input nodes for each stage: {updated_input_nodes_for_each_stage}"
    )
    logger.debug(
        f"Updated output nodes for each stage: {updated_output_nodes_for_each_stage}"
    )
    return updated_input_nodes_for_each_stage, updated_output_nodes_for_each_stage


def _check_and_sort_inputs_outputs_for_each_stage(
    input_nodes_for_each_stage, output_nodes_for_each_stage
):
    """
    Check matching of outputs of stage i and inputs of stage i+1,
    and update the input and output nodes for each stage in place.
    """
    if len(input_nodes_for_each_stage) != len(output_nodes_for_each_stage):
        raise ValueError(
            f"Number of input stages ({len(input_nodes_for_each_stage)}) "
            f"does not match number of output stages ({len(output_nodes_for_each_stage)})"
        )

    n_stages = len(input_nodes_for_each_stage)

    def sort_nodes(nodes):
        return sorted(nodes, key=lambda x: x.name)

    assert len(input_nodes_for_each_stage[0]) == 0, "Stage 0's inputs should be empty"
    assert (
        len(output_nodes_for_each_stage[-1]) == 0
    ), "Stage -1's outputs should be empty"
    for i in range(n_stages - 1):
        # Check matching
        input_nodes = input_nodes_for_each_stage[i + 1]
        output_nodes = output_nodes_for_each_stage[i]
        assert set(input_nodes) == set(
            output_nodes
        ), f"Stage {i} does not match, {input_nodes} != {output_nodes}"
        assert len(input_nodes) == len(output_nodes), (
            f"Stage {i} does not match, "
            f"number of input nodes ({len(input_nodes)}) != number of output nodes ({len(output_nodes)})"
        )
        # Reorganize the output nodes to match the input nodes
        output_nodes_for_each_stage[i] = input_nodes

        # Let's sort it
        input_nodes_for_each_stage[i + 1] = sort_nodes(
            input_nodes_for_each_stage[i + 1]
        )
        output_nodes_for_each_stage[i] = sort_nodes(output_nodes_for_each_stage[i])


def _check_pipe_modules_runnable(pipe_modules, inputs):
    device = next(pipe_modules[0].parameters()).device
    first_stage_module_signature = inspect.signature(pipe_modules[0].forward)
    first_stage_module_bound_signature = first_stage_module_signature.bind(**inputs)
    cur_inputs = first_stage_module_bound_signature.arguments.values()
    with torch.no_grad():
        for i, pipe_module in enumerate(pipe_modules):
            try:
                # cur_outputs = pipe_module(*cur_inputs)
                cur_outputs = device_safe_func_exec(
                    device,
                    get_device(torch.cuda.current_device()),
                    pipe_module,
                    *cur_inputs,
                )
                cur_inputs = (
                    cur_outputs if isinstance(cur_outputs, tuple) else (cur_outputs,)
                )
            except Exception as e:
                pipe_module.graph.print_tabular()
                raise e

            # TODO(zhanda): fix this later directly in the graph level
            # Deal with different cases of the inputs and outputs:
            # if i < len(pipe_modules) - 1:
            #     next_pipe_module = pipe_modules[i + 1]
            #     next_pipe_module_signature = inspect.signature(next_pipe_module.forward)
            #     if isinstance(cur_outputs, tuple) and len(cur_outputs) == len(
            #         next_pipe_module_signature.parameters
            #     ):
            #         cur_inputs = cur_outputs
            #     else:
            #         pipe_module.forward = wrap_forward_with_tuple_output(
            #             pipe_module.forward
            #         )
            #         cur_inputs = (cur_outputs,)


def create_new_graph_for_node_sequence(
    node_sequence, root_module, input_nodes=None, output_nodes=None
):
    """
    Create a new graph for a node sequence. This will wrap it in a torch.fx.Graph.

    In the meanwhile, ori_input_nodes and ori_output_nodes are returned for further
    processing.
    """
    local_input_nodes, local_output_nodes = _analyze_inputs_outputs_for_node_sequence(
        node_sequence
    )
    input_nodes = input_nodes or local_input_nodes
    output_nodes = output_nodes or local_output_nodes

    graph = torch.fx.Graph()
    mapping = {}

    # Create placeholder nodes
    for node in input_nodes:
        if node.op == "placeholder":
            mapping[node] = graph.node_copy(node)
        else:
            mapping[node] = graph.placeholder(node.name, type_expr=node.type)

    # Create intermediate nodes
    for node in node_sequence:
        mapping[node] = graph.node_copy(node, lambda n: mapping[n])

    # Create output nodes
    # 1. If there is no output node, it means the last node itself is the output node
    # 2. If there is only one output node, it is the output node
    if len(output_nodes) == 0:
        assert (
            node_sequence[-1].op == "output"
        ), f"Last node is not output: {node_sequence[-1]} even though there is no output node"
    elif len(output_nodes) == 1:
        graph.output(mapping[output_nodes[0]])
    else:
        graph.output(tree_map(lambda n: mapping[n], tuple(output_nodes)))

    return graph, input_nodes, output_nodes


def create_pipe_modules_based_on_input_output_nodes_and_sequence(
    model,
    node_sequences_for_each_stage,
    input_nodes_for_each_stage,
    output_nodes_for_each_stage,
    modules_to_graphs=None,
    additional_globals=None,
):
    """
    Create pipe modules based on the given strategy.
    """
    pipe_modules = []
    for i, (input_nodes, output_nodes) in enumerate(
        zip(input_nodes_for_each_stage, output_nodes_for_each_stage)
    ):
        graph, _, _ = create_new_graph_for_node_sequence(
            node_sequence=node_sequences_for_each_stage[i],
            root_module=model,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
        )
        if additional_globals is not None:
            graph._codegen.additional_globals = MethodType(
                lambda _self: list(additional_globals), graph._codegen
            )
        pipe_modules.append(
            recursively_create_graph_modules(model, graph, modules_to_graphs)
        )
    return pipe_modules


def build_pipe_modules(
    model, node_sequence_for_each_stage, modules_to_graphs=None, inputs=None
):
    # Gather all additional globals from the graph for _codegen
    addtionl_globals = set()
    for graph in modules_to_graphs.values():
        addtionl_globals.update(graph._codegen.additional_globals())

    # Get the vanilla inputs and outputs for each stage
    vanilla_inputs_for_each_stage = []
    vanilla_outputs_for_each_stage = []
    for i, stage_node_sequence in enumerate(node_sequence_for_each_stage):
        input_nodes, output_nodes = _analyze_inputs_outputs_for_node_sequence(
            stage_node_sequence
        )
        vanilla_inputs_for_each_stage.append(input_nodes)
        vanilla_outputs_for_each_stage.append(output_nodes)

    # Collect liveness information
    node2range = _analyze_input_output_nodes_liveness(
        vanilla_inputs_for_each_stage, vanilla_outputs_for_each_stage
    )

    # Update the input and output nodes for each stage
    (
        updated_input_nodes_for_each_stage,
        updated_output_nodes_for_each_stage,
    ) = _update_input_output_nodes_for_each_stage_according_to_liveness(
        vanilla_inputs_for_each_stage, vanilla_outputs_for_each_stage, node2range
    )

    # Check matching and sort in place
    _check_and_sort_inputs_outputs_for_each_stage(
        updated_input_nodes_for_each_stage, updated_output_nodes_for_each_stage
    )

    # Create pipe modules
    pipe_modules = create_pipe_modules_based_on_input_output_nodes_and_sequence(
        model,
        node_sequence_for_each_stage,
        updated_input_nodes_for_each_stage,
        updated_output_nodes_for_each_stage,
        modules_to_graphs=modules_to_graphs,
        additional_globals=addtionl_globals,
    )

    # Fix issues for hf models (resetting some pre-saved attributes)
    for pipe_module in pipe_modules:
        fix_hf_model(pipe_module)

    # Check and fix issues for inconsistent input/output formats
    # we want to make sure modules can be run recursively with `inputs = module(*inputs))`
    if inputs is not None:
        _check_pipe_modules_runnable(pipe_modules, inputs)
        # Empty cache because there may be a lot of allocations and deallocations
        cuda_empty_cache()

    return pipe_modules


def build_pipe_modules_for_analyzing(
    model: nn.Module,
    root_graph: fx.Graph,
    modules_to_graphs: Dict[str, fx.Graph],
    sub_modules: Optional[Sequence[str]] = None,
    inputs: Dict[str, Any] = None,
):
    """Build pipe modules for each layer for analyzing.

    Parameters
    ----------
    model
        root module (owning_module)
    root_graph
        root graph of root module
    modules_to_graphs
        a dict mapping sub module names to their graphs
    inputs, optional
        inputs to the model to check the correctness of the pipe modules
    """
    sub_modules = sub_modules or get_default_sub_modules(model)
    assert all(
        m in modules_to_graphs for m in sub_modules
    ), f"Not all sub modules are traced: {sub_modules}"

    # Generate node sequence for each stage
    (
        node_sequence_pre_layer,
        node_sequence_blocks,
        node_sequence_post_layer,
    ) = _extract_node_sequence_for_each_block(model, root_graph, sub_modules)

    # Unify the node sequences
    node_sequence_for_each_stage = [
        node_sequence_pre_layer,
        *node_sequence_blocks,
        node_sequence_post_layer,
    ]

    # Build pipe modules
    pipe_modules = build_pipe_modules(
        model,
        node_sequence_for_each_stage,
        modules_to_graphs=modules_to_graphs,
        inputs=inputs,
    )

    assert len(pipe_modules) == len(sub_modules) + 2, (
        f"Number of pipe modules ({len(pipe_modules)}) "
        f"does not match number of sub modules ({len(sub_modules)})"
    )

    return pipe_modules


def _tree_make_viewless_tensor(x):
    if isinstance(x, torch.Tensor):
        return make_viewless_tensor(x, requires_grad=x.requires_grad, keep_graph=True)

    def _fn(x):
        if isinstance(x, torch.Tensor):
            return make_viewless_tensor(
                x, requires_grad=x.requires_grad, keep_graph=True
            )
        return x

    return tree_map(_fn, x)


class PipeModule(nn.Module):
    def __init__(self, blocks, pre_layer=None, post_layer=None):
        super().__init__()
        self.pre_layer = pre_layer
        self.blocks = blocks  # A single nn.Module containing all blocks for this stage
        self.post_layer = post_layer

        wraps_nn_module_forward(self, self._first_module)

    @property
    def _first_module(self):
        if self.pre_layer is not None:
            return self.pre_layer
        else:
            return self.blocks

    def forward(self, *args, **kwargs):
        # Map args and kwargs to args for the first module
        inputs = map_args_kwargs_to_args(
            inspect.signature(self._first_module.forward),
            *args,
            **kwargs,
        )

        # Run the pre layer if exists
        if self.pre_layer is not None:
            inputs = self.pre_layer(*inputs)
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
            inputs = _tree_make_viewless_tensor(inputs)

        # Run the blocks
        inputs = self.blocks(*inputs)
        inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        inputs = _tree_make_viewless_tensor(inputs)

        # Run the post layer if exists
        if self.post_layer is not None:
            inputs = self.post_layer(*inputs)
            inputs = _tree_make_viewless_tensor(inputs)

        return inputs


def build_pipe_modules_based_on_block_partition(
    model: nn.Module,
    root_graph: fx.Graph,
    modules_to_graphs: Dict[str, fx.Graph],
    block_partition: List[int],
    sub_modules: Optional[Sequence[str]] = None,
    inputs: Dict[str, Any] = None,
    raise_error_if_single_block: bool = True,
):
    """Build pipe modules based on the given block partition.

    Parameters
    ----------
    model
        root module (owning_module)
    root_graph
        root graph of root module
    modules_to_graphs
        a dict mapping sub module names to their graphs
    block_partition
        a list of integers indicating the partition of blocks
    inputs, optional
        inputs to the model to check the correctness of the pipe modules
    """
    if raise_error_if_single_block and len(block_partition) == 1:
        raise ValueError(
            f"No need to build pipe modules for a single block: {block_partition}"
        )

    sub_modules = sub_modules or get_default_sub_modules(model)

    # Generate node sequence for each stage
    (
        node_sequence_pre_layer,
        node_sequence_blocks,
        node_sequence_post_layer,
    ) = _extract_node_sequence_for_each_block(model, root_graph, sub_modules)

    assert len(node_sequence_blocks) == sum(
        block_partition
    ), f"Invalid block partition. Has {len(node_sequence_blocks)} blocks, but {sum(block_partition)} partitions"

    # Get the correct node sequence for each stage
    node_sequence_for_each_stage = []
    for i in range(len(block_partition)):
        curr_node_sequence = []
        for node_sequence in node_sequence_blocks[
            sum(block_partition[:i]) : sum(block_partition[: i + 1])
        ]:
            curr_node_sequence.extend(node_sequence)
        node_sequence_for_each_stage.append(curr_node_sequence)

    # ======================================================================
    # # Deprecated: because we want to keep the pre/post layers individually
    # node_sequence_for_each_stage[0] = [
    #     *node_sequence_pre_layer,
    #     *node_sequence_for_each_stage[0],
    # ]
    # node_sequence_for_each_stage[-1] = [
    #     *node_sequence_for_each_stage[-1],
    #     *node_sequence_post_layer,
    # ]
    # ======================================================================
    node_sequence_for_each_stage.insert(0, node_sequence_pre_layer)
    node_sequence_for_each_stage.append(node_sequence_post_layer)
    assert len(node_sequence_for_each_stage) == len(block_partition) + 2

    # Build pipe modules
    raw_pipe_modules = build_pipe_modules(
        model,
        node_sequence_for_each_stage,
        modules_to_graphs=modules_to_graphs,
        inputs=inputs,
    )
    assert (
        len(raw_pipe_modules) == len(block_partition) + 2
    ), f"Number of pipe modules ({len(pipe_modules)}) does not match number of partitions ({len(block_partition)})"

    # Merge the pipe modules
    if len(block_partition) == 1:
        pipe_module = PipeModule(
            pre_layer=raw_pipe_modules[0],
            blocks=raw_pipe_modules[1],
            post_layer=raw_pipe_modules[-1],
        )
        pipe_modules = [pipe_module]
    else:
        pipe_modules = []
        for i in range(len(block_partition)):
            if i == 0:
                pipe_module = PipeModule(
                    pre_layer=raw_pipe_modules[0],
                    blocks=raw_pipe_modules[i + 1],
                )
            elif i == len(block_partition) - 1:
                pipe_module = PipeModule(
                    blocks=raw_pipe_modules[i + 1],
                    post_layer=raw_pipe_modules[-1],
                )
            else:
                pipe_module = PipeModule(blocks=raw_pipe_modules[i + 1])
            pipe_modules.append(pipe_module)

    assert len(pipe_modules) == len(block_partition), (
        f"Number of pipe modules ({len(pipe_modules)}) "
        f"does not match number of partitions ({len(block_partition)})"
    )

    # Check the runnability of the pipe modules again
    if inputs is not None:
        _check_pipe_modules_runnable(pipe_modules, inputs)

    cuda_empty_cache()

    return pipe_modules


if __name__ == "__main__":
    from mist import gsm
    from mist.sym_torch import SymbolicTensor
    from mist.tracer.symbolic_tracer import mist_trace
    from mist.utils.hf import create_meta_model, create_meta_dummy_inputs

    model_name = "tiiuae/falcon-7b"
    b, s = gsm.symbols("b, s", (4, 128), integer=True, positive=True)
    input_names = ["input_ids", "attention_mask", "labels"]
    model, config = create_meta_model(model_name)
    symbolic_inputs = create_meta_dummy_inputs(
        model=model,
        batch_size=b,
        seq_len=s,
        input_names=input_names,
    )
    graph, modules_to_graphs = mist_trace(model, symbolic_inputs)

    # ----------------------------------------

    pipe_modules = build_pipe_modules_for_analyzing(
        model, graph, inputs=symbolic_inputs
    )
