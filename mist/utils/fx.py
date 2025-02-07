from typing import Optional, List, Sequence, Dict, Any, Tuple, Set
import torch
from torch import fx, nn
from torch.fx import GraphModule
from torch.fx.graph_module import _copy_attr
from torch.utils._pytree import (
    _register_pytree_node,
    Context,
    _odict_flatten,
    _odict_unflatten,
)
import inspect

from transformers.modeling_outputs import ModelOutput

from mist.tracer.hf import HFTracer, _generate_random_int
from mist.logger import get_logger
from mist.utils.module import hasattr_recursive, getattr_recursive

logger = get_logger()


# Utils for tracing HF models
_generate_dummy_input = HFTracer._generate_dummy_input


def _generate_dummy_inputs_for_hf(
    model: nn.Module,
    input_names: Optional[List[str]] = None,
    shape: Optional[Sequence[int]] = None,
):
    """
    Generate dummy inputs for HF models.

    Parameters
    ----------
    model
        A pretrained HF model.
    input_names : list, optional
        the names of inputs that should be generated, by default None
    shape, optional
        the shape of inptu, should be [batch_size, seq_len], by default None
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    if shape is None:
        batch_size = _generate_random_int()
        seq_len = _generate_random_int()
        shape = [batch_size, seq_len]

    dummy_inputs = {}
    for name in input_names:
        dummy_inputs.update(_generate_dummy_input(model, name, shape))

    # Specifically for ModelOutput
    # Deprecated because we have changed the `create_arg` to make it work for ModelOutput
    # if "return_dict" in inspect.signature(model.forward).parameters:
    #     dummy_inputs["return_dict"] = False

    return dummy_inputs


# Utils for creating graph modules
def is_primitive_module(m: nn.Module):
    """
    Check if a module is a primitive module.

    Note: there are two cases that should be manually handled:
    1. torch.nn.Sequential
    2. torch.nn.ModuleList
    """
    return (
        m.__module__.startswith("torch.nn")
        or m.__module__.startswith("torch.ao.nn")
        and not isinstance(m, (nn.Sequential, nn.ModuleList))
    )


def simplify_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    gm.delete_all_unused_submodules()
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def create_graph_module(model: nn.Module, graph: fx.Graph) -> fx.GraphModule:
    """
    Create graph module with some utilities.

    Parameters
    ----------
    model
        The original model.
    graph
        The graph to be used to create the graph module.
    """

    logger.debug(f"Creating graph module for {model.name}")

    gm = fx.GraphModule(model, graph)
    gm = simplify_graph_module(gm)

    if hasattr(model, "name"):
        gm.name = model.name

    return gm


def recursively_create_graph_modules(
    model: nn.Module,
    root_graph: fx.Graph,
    modules_to_graphs: Dict[str, fx.Graph],
) -> fx.GraphModule:
    """
    Create graph module with some utilities.

    Parameters
    ----------
    model
        The owning model.
    root_graph
        The root graph to be used to create the graph module.
    modules_to_graphs
        A dictionary that maps module names to graphs. This is used to recursively create graph modules.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected root to be a nn.Module, but got {type(model)} instead."
        )

    # Create the root graph module
    gm = create_graph_module(model, root_graph)

    # Submodules that are in the GraphModule
    sub_modules: Set[str] = set(getattr(m, "name", None) for m in gm.modules())
    sub_modules.remove(getattr(gm, "name", None))
    sub_modules.discard(None)

    # Sort modules_to_graphs by the number of '.' in the module name
    # to ensure that the parent module is created before the child module.
    modules_to_graphs = dict(
        sorted(modules_to_graphs.items(), key=lambda x: x[0].count("."))
    )

    # Create graph modules for all submodules
    for name, graph in modules_to_graphs.items():
        if name not in sub_modules:
            continue
        mod_parent_name, _, mod_self_name = name.rpartition(".")
        self_mod = getattr_recursive(gm, name)
        parent_mod = getattr_recursive(gm, mod_parent_name)
        parent_mod._modules[mod_self_name] = create_graph_module(self_mod, graph)

    # Run a second pass to make sure that all necessary modules/attributes are copied
    # because we are creating graph modules in a recursive way. The modules/attributes
    # used in the parent module but not in the current module may be missed.
    # See ``torch/fx/graph_module.py:GraphModule.__init__`` for more details.
    for name, graph in modules_to_graphs.items():
        if name not in sub_modules:
            continue
        gm_mod = getattr_recursive(gm, name)
        ori_mod = getattr_recursive(model, name)
        for node in graph.nodes:
            if node.op in ["get_attr", "call_module"]:
                if not hasattr_recursive(gm_mod, node.target):
                    logger.debug(
                        f"Copying {node.target} from {mod_parent_name} to {name} for hierarchical graph module creation"
                    )
                    _copy_attr(ori_mod, gm_mod, node.target)

    return gm


def recursive_print_graph(gm: fx.GraphModule):
    print(f"\n{'='*80}\n")
    graph = gm.graph
    print(
        f'[Module Name] "{gm.name}", [is GraphModule] {isinstance(gm, fx.GraphModule)}'
    )
    graph.print_tabular()

    for node in graph.nodes:
        if node.op == "call_module":
            submodule = gm.get_submodule(node.target)
            if is_primitive_module(submodule):
                continue
            elif not isinstance(submodule, fx.GraphModule):
                print(f"submodule {submodule.name} is not GraphModule, skipping...\n")
            else:
                recursive_print_graph(submodule)


# Utils to register ModelOutput in pytree map
def _model_output_flatten(model_output: ModelOutput) -> Tuple[List[Any], Context]:
    """
    ModelOutput is a subclass of OrderedDict and dataclass.
    """
    cls = type(model_output)
    values, keys = _odict_flatten(model_output)
    return values, (keys, cls)


def _model_output_unflatten(values: List[Any], context: Context) -> ModelOutput:
    keys, cls = context
    return cls(**_odict_unflatten(values, keys))


def register_model_output_in_pytree_map():
    """
    This function is used to register the ModelOutput class in pytree map.
    """
    _register_pytree_node(ModelOutput, _model_output_flatten, _model_output_unflatten)
    for cls in ModelOutput.__subclasses__():
        _register_pytree_node(cls, _model_output_flatten, _model_output_unflatten)


def _slice_flatten(slice: slice) -> Tuple[List[Any], Context]:
    """
    slice is a subclass of tuple.
    """
    cls = type(slice)
    start, stop, step = slice.start, slice.stop, slice.step
    return [start, stop, step], cls


def _slice_unflatten(values: List[Any], context: Context) -> slice:
    cls = context
    return cls(*values)


def register_slice_in_pytree_map():
    """
    This function is used to register the slice class in pytree map.
    """
    _register_pytree_node(slice, _slice_flatten, _slice_unflatten)
