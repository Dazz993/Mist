import torch
from torch.fx.immutable_collections import immutable_list


def output_first_item_in_output_node_pass(graph):
    """
    By default, the output of the traced graph of the HF model is a ModelOutput object,
    or a dict. However, we want to directly get the loss value. This function replaces
    the output of the traced graph with the loss value.

    Note: Here we assume you use ``return_dict=False`` when calling the HF model.
    """

    output_node = list(graph.nodes)[-1]
    outputs = output_node.args[0]
    assert isinstance(outputs, (tuple, list)), (
        "Expected the output of the traced graph to be a tuple/list "
        f"(which is actually an immutable_list), but got {type(outputs)}. "
        "Perhaps you are using `return_dict=True` when calling the HF model?"
    )

    loss = outputs[0]
    graph.erase_node(output_node)
    graph.create_node(
        op="output", target="output", args=(loss,), name="output", type_expr=None
    )
    graph.lint()

    return graph
