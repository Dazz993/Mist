from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from torch.utils._pytree import (
    SUPPORTED_NODES,
    TreeSpec,
    LeafSpec,
    PyTree,
    _is_leaf,
    _get_node_type,
    tree_flatten,
    tree_map,
    tree_unflatten,
)


def tree_flatten_like(pytree: PyTree, spec: TreeSpec) -> List[Any]:
    """Flattens a pytree into a list of values that has the same structure as
    the given spec.
    """
    if _is_leaf(pytree) or spec.type is None:
        return [pytree], LeafSpec()

    # Spec must be a TreeSpec
    assert isinstance(spec, TreeSpec), f"Expected a TreeSpec, got {spec}"

    # Input pytree must have the same node type as the spec
    flatten_fn = SUPPORTED_NODES[spec.type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)
    assert len(child_pytrees) == len(
        spec.children_specs
    ), f"Expected {len(spec.children_specs)} children, got {len(child_pytrees)}"

    # Recursively flatten the children
    result: List[Any] = []
    children_specs: List["TreeSpec"] = []
    for child, ref_child_spec in zip(child_pytrees, spec.children_specs):
        flat, child_spec = tree_flatten_like(child, ref_child_spec)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(spec.type, context, children_specs)


def tree_zip_map(fn: Any, pytree_1, pytree_2):
    """Maps a function which takes two arguments over two pytrees.

    Parameters
    ----------
    fn
        func which takes two arguments and returns a value
    pytree_1
        pytree 1 to map over
    pytree_2
        pytree 2 to map over
    """
    flat_args_1, treedef_1 = tree_flatten(pytree_1)
    flat_args_2, treedef_2 = tree_flatten(pytree_2)
    assert treedef_1 == treedef_2, f"Expected {treedef_1} == {treedef_2}"
    flat_result = [fn(arg_1, arg_2) for arg_1, arg_2 in zip(flat_args_1, flat_args_2)]
    return tree_unflatten(flat_result, treedef_1)
