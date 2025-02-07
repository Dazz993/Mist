from typing import Optional, List, Sequence, Dict, Any, Tuple
import torch
from torch import fx, nn
from torch.utils._pytree import (
    _register_pytree_node,
    Context,
    _odict_flatten,
    _odict_unflatten,
)

from transformers.modeling_outputs import ModelOutput

from mist.tracer.hf import HFTracer, _generate_random_int
from mist.logger import get_logger

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
    gm.delete_all_unused_submodules()
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    if hasattr(model, "name"):
        gm.name = model.name

    return gm


def recursively_create_graph_modules(
    root: nn.Module,
    modules_to_graphs: Dict[str, fx.Graph],
) -> fx.GraphModule:
    """
    Create graph module with some utilities.

    Parameters
    ----------
    root
        The original model.
    modules_to_graphs
        A dictionary that maps module names to graphs. This is used to recursively create graph modules.
    """

    # Pop out the root module
    root_graph = modules_to_graphs.pop(root.name)

    # Create the root graph module
    gm = create_graph_module(root, root_graph)

    # Sort modules_to_graphs by the number of '.' in the module name
    # to ensure that the parent module is created before the child module.
    modules_to_graphs = dict(
        sorted(modules_to_graphs.items(), key=lambda x: x[0].count("."))
    )

    for name, graph in modules_to_graphs.items():
        mod_parent_name, _, mod_self_name = name.rpartition(".")
        self_mod = gm.get_submodule(name)
        parent_mod = gm.get_submodule(mod_parent_name)
        parent_mod._modules[mod_self_name] = create_graph_module(self_mod, graph)

    return gm


def recursive_print_graph(gm: fx.GraphModule):

    print(f"\n{'='*80}\n")
    graph = gm.graph
    print(f"module name: '{gm.name}', is GraphModule: {isinstance(gm, fx.GraphModule)}")
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
