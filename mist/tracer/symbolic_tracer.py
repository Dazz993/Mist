from __future__ import annotations
import inspect
import math
import re
import sys
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)
from types import MethodType
import numpy as np

import sympy as sp
import torch
import torch.nn as nn
import torch.distributed
from torch.fx._compatibility import compatibility
from torch.fx.proxy import TraceError
from torch.fx import Graph, GraphModule, Proxy, Tracer, Node
from torch.utils._pytree import tree_map
from transformers.modeling_outputs import ModelOutput

from mist import gsm
from mist.logger import get_logger
from mist.tracer.hf import (
    HFTracer,
    HFProxy,
    _proxies_to_metas,
    _gen_constructor_wrapper,
    _IS_IN_DEBUG_MODE,
    _MANUAL_META_OVERRIDES,
)
from mist.utils.fx import (
    register_model_output_in_pytree_map,
)
from mist.distributed.overrides import MistProcessGroup
from mist.utils.module import getattr_recursive
from mist.utils.graph_pass import output_first_item_in_output_node_pass
from mist.utils.module import set_module_name_recursive, summarize_sub_modules_path
from mist.utils.memory import materialize_module, materialize_tensor
from mist.utils.tracing_patcher import get_root_patcher, RevertPatcher
from mist.symbols import temporarily_set_sp_eq_ne
from mist.utils.tensor_entry import TensorEntry, tensor_to_entry, tree_to_entries

try:
    import apex
except ImportError:
    apex = None

dict_keys = type({}.keys())

logger = get_logger()

# This is used to register HF/ModelOutput class in pytree_map
register_model_output_in_pytree_map()


def _guess_self_module(func):
    if isinstance(func, nn.Module):
        return True, func
    elif inspect.ismethod(func):
        mod = func.__self__
        if isinstance(mod, nn.Module):
            return True, mod
    else:
        closure = getattr(func, "__closure__", [])
        if closure and isinstance(closure[0].cell_contents, nn.Module):
            return True, closure[0].cell_contents

    return False, None


def _move_to_device(obj, device):
    if not isinstance(obj, (torch.Tensor, nn.Module)):
        return obj
    return obj.to(device)


def device_safe_func_exec(__device, __fallback_device, func, *args, **kwargs):
    """This function is used to execute a function.
    If the function fails, we will try to fallback to another device.
    """
    try:
        outputs = func(*args, **kwargs)
    except (RuntimeError, TypeError) as e:
        logger.debug(f"Error when executing {func} on {__device}. ")
        logger.debug(f"Try to fallback to {__fallback_device}")

        # Try to fallback to self.fallback_device
        # Materialize the module if necessary
        found, ori_module = _guess_self_module(func)
        if found:
            materialized_module = materialize_module(
                ori_module,
                device=__fallback_device,
                inplace=False,
            )
            func = materialized_module.forward

        # Materialize the args and kwargs
        materialized_args = tree_map(materialize_tensor, args)
        materialized_kwargs = tree_map(materialize_tensor, kwargs)
        # Deal with some special cases where kwargs are not in func's signature
        func_signature = inspect.signature(func)
        materialized_kwargs = {
            k: v
            for k, v in materialized_kwargs.items()
            if k in func_signature.parameters
        }
        # Call the materialized module
        outputs = func(*materialized_args, **materialized_kwargs)
        # Move the outputs back to the original device
        outputs = tree_map(partial(_move_to_device, device=__device), outputs)

    return outputs


def is_fx_tracable(mod):
    return not hasattr(mod, "traceable") or mod.traceable


def _guess_is_sequence_of_module(module):
    if not isinstance(module, nn.Module):
        return False
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        return True

    # After tracing, ModuleList will be converted to a raw nn.Module
    # if the keys of the raw nn.Module are of a sequence of str of int,
    # we can guess that this is a ModuleList
    sub_module_names = list(module._modules.keys())
    if not sub_module_names:
        return False
    is_module_list = True
    prev_index = None
    for name in sub_module_names:
        try:
            index = int(name)
        except ValueError:
            is_module_list = False
            break
        if prev_index is None:
            prev_index = index
        elif index != prev_index + 1:
            is_module_list = False
            break
        prev_index = index

    return is_module_list


def get_default_sub_modules(
    model: nn.Module,
    return_names: bool = True,
    allow_multiple_module_list: bool = True,
):
    if not hasattr(model, "name"):
        raise AttributeError(
            f"{model} does not have an attribute called name. "
            "Please set the name recursively for all submodules"
        )

    module_list = []
    for m in model.modules():
        # if isinstance(m, (nn.ModuleList, nn.Sequential)):
        if _guess_is_sequence_of_module(m):
            if not allow_multiple_module_list and module_list is not None:
                raise ValueError(
                    "Only support one ModuleList or Sequential in a module. "
                    "If you want to trace into multiple ModuleList or Sequential, "
                    "please set allow_multiple_module_list=True"
                )
            module_list += list(m.children())

    if return_names:
        return [m.name for m in module_list]
    else:
        return module_list


def complete_args_with_default_values(
    sig: inspect.Signature,
    args: Dict[str, Any] = None,
    except_args: Union[Dict[str, Any], Sequence[str]] = None,
    raise_error_if_no_default_value: bool = True,
):
    """
    This helper function will complete the args with default values.

    Parameters
    ----------
    sig
        The signature of the function.
    args
        Input args provided by the user.
    except_args
        The args that should not be completed.
    raise_error_if_no_default_value
        Whether to raise error if no default value is provided.

    Returns
    -------
    args
        Completed args.
    """

    args = args or {}
    except_args = except_args or {}

    for param in sig.parameters.values():
        if (
            param.kind == inspect.Parameter.VAR_POSITIONAL
            or param.kind == inspect.Parameter.VAR_KEYWORD
        ):
            continue
        if param.name in except_args or param.name in args:
            continue
        if param.default is inspect.Parameter.empty:
            if raise_error_if_no_default_value:
                raise ValueError(
                    f"You need to specify a default value for the parameter {param}"
                )
        else:
            args[param.name] = param.default

    return args


class MistTracer(HFTracer):
    """
    This is our reimplementation of HFTracer which supports hierarchical symbolic tracing.
    Actually since this class has already overrides many methods of HFTracer, it is mostly the
    same as a subclass of PyTorch internal Tracer (the grandparent class).

    Parameters
    ----------
    sub_modules:
    """

    def __init__(
        self,
        sub_modules: Optional[Sequence[str]] = None,
        sub_graph_modules: Optional[Sequence[str]] = None,
        autowrap_modules=(math,),
        autowrap_functions=(),
    ):
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
        )
        if sub_modules is not None and (
            not isinstance(sub_modules, (set, list, tuple))
            or not all(isinstance(m, str) for m in sub_modules)
        ):
            raise ValueError(
                f"sub_modules should be a sequence of str. {sub_modules}. Make sure you have added the name instead of nn.Modules."
            )

        if sub_graph_modules is not None and (
            not isinstance(sub_graph_modules, (set, list, tuple))
            or not all(isinstance(m, str) for m in sub_graph_modules)
        ):
            raise ValueError(
                f"sub_graph_modules should be a sequence of str. {sub_graph_modules}. Make sure you have added the name instead of nn.Modules."
            )

        self.sub_modules = set(sub_modules) if sub_modules is not None else set()
        self.sub_graph_modules = (
            set(sub_graph_modules) if sub_graph_modules is not None else set()
        )
        if not self.sub_graph_modules.issubset(self.sub_modules):
            raise ValueError(
                f"sub_graph_modules should be a subset of sub_modules. {self.sub_graph_modules} vs {self.sub_modules}"
            )

        # During trace, we collect the graph of the sub_modules that are traced into.
        # but we don't convert it to GraphModule immediately because it may lose some attributes
        # that are not used in this graph (but may be used in other graphs).
        # The graph modules will be created in the postprocess stage.
        self.modules_to_graphs: Dict[str, Graph] = OrderedDict()

        # Record the information of the inputs and outputs
        self._input_entires: Dict[Node, TensorEntry] = OrderedDict()
        self._output_entires: Dict[Node, TensorEntry] = OrderedDict()
        # Record the output of the function that is traced.
        self._output = None

        # Additional globals (e.g. Symbols, and MistProcessGroup)
        self._additional_globals: Set[Tuple[str, Any]] = set()

    def _try_adding_symbols_to_additional_globals(self, a: Any):
        if isinstance(a, (sp.Basic)):
            for s in a.free_symbols:
                self._additional_globals.add((repr(s), s))

    def create_arg(self, a: Any):
        # If the last call stack is trace(), then we record the output of the function that is traced.
        if sys._getframe(1).f_code.co_name == "trace":
            self._output = a

        # Special handling for symbols: symbols should be added to globals
        # to be executed during the graph module
        if isinstance(a, sp.Basic):
            self._try_adding_symbols_to_additional_globals(a)
            return a
        elif isinstance(a, torch.Tensor):
            for s in a.shape:
                self._try_adding_symbols_to_additional_globals(s)
        elif isinstance(a, Proxy):
            if isinstance(t := getattr(a, "_metadata", None), torch.Tensor):
                for s in t.shape:
                    self._try_adding_symbols_to_additional_globals(s)

        # Handle other type of wrapped objects
        if isinstance(a, np.integer):
            return int(a)
        # for ModelOutput of transformers
        elif isinstance(a, ModelOutput):
            args = {key: self.create_arg(value) for key, value in a.items()}
            return self.create_node("call_function", a.__class__, (args,), {})
        # for ProcessGroup and distributed related
        elif isinstance(a, torch.distributed.ProcessGroup):
            # Intern this as a constant attribute
            i = 0
            while True:
                qualname = f"_{a.__class__.__name__}_constant_{i}"
                if not hasattr(self.root, qualname):
                    break
                i += 1
            setattr(self.root, qualname, a)
            return self.create_node("get_attr", qualname, (), {})
        elif isinstance(a, torch.distributed.ReduceOp):
            return a
        elif apex is not None and isinstance(a, apex.transformer.AttnMaskType):
            setattr(self.root, f"_{a.__class__.__name__}_constant", a)
            return self.create_node("get_attr", f"_{a.__class__.__name__}_constant", (), {})

        arg = Tracer.create_arg(self, a)
        return arg

    def trace(
        self,
        root: Union[nn.Module, Callable[..., Any]],
        meta_args: Optional[Dict[str, Any]] = None,
        concrete_args: Optional[Dict[str, Any]] = None,
        complete_concrete_args_for_inputs_not_in_meta_args: bool = True,
        device=None,
        fallback_device="cuda",
    ) -> Graph:
        self.device = device
        self.fallback_device = fallback_device
        self.device_safe_func_exec = partial(
            device_safe_func_exec,
            device,
            fallback_device,
        )

        if isinstance(root, nn.Module):
            if not hasattr(root, "name"):
                set_module_name_recursive(root)
            if not self.sub_modules:
                self.sub_modules = get_default_sub_modules(root)
        else:
            root.name = ""

        # Some notes on python signatures:
        # (1) _ParameterKind.POSITIONAL_OR_KEYWORD: a normal argument
        # (2) _ParameterKind.VAR_POSITIONAL: *args
        # (3) _ParameterKind.VAR_KEYWORD: **kwargs
        sig = inspect.signature(root.forward if isinstance(root, nn.Module) else root)
        meta_args = meta_args or {}
        concrete_args = concrete_args or {}

        # DEBUG
        logger.debug("")
        logger.debug(f"Tracing module {root.name if root.name else 'root'}")
        logger.debug(f"--- siganature: {sig}")
        logger.debug(f"--- meta_args: {meta_args}")
        logger.debug(f"--- concrete_args: {concrete_args}")
        if root.name == "":
            logger.debug(
                f"--- sub_modules: {summarize_sub_modules_path(self.sub_modules)}"
            )
            logger.debug(
                f"--- sub_graph_modules: {summarize_sub_modules_path(self.sub_graph_modules)}"
            )

        assert not any(
            isinstance(arg, HFProxy) for arg in meta_args.values()
        ), f"meta_args should not contain HFProxy. {meta_args}"

        # Complete concrete_args with default values
        if complete_concrete_args_for_inputs_not_in_meta_args:
            concrete_args = complete_args_with_default_values(
                sig,
                args=concrete_args,
                except_args=meta_args,
                raise_error_if_no_default_value=True,
            )
            logger.debug(f"--- concrete_args (after completion): {concrete_args}")
        self.concrete_args = concrete_args

        # Setup meta_args
        meta_args = {
            key: value if isinstance(value, torch.Tensor) else value
            for key, value in meta_args.items()
        }
        for param in sig.parameters.values():
            if (
                param.kind == inspect.Parameter.VAR_POSITIONAL
                and param.name not in meta_args
            ):
                meta_args[f"*{param.name}"] = []
            if (
                param.kind == inspect.Parameter.VAR_KEYWORD
                and param.name not in meta_args
            ):
                meta_args[f"**{param.name}"] = {}
        self.meta_args = meta_args

        # Patch torch methods and do tracing
        self.patched_torch_methods = {
            target: _gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            with torch.no_grad():
                # trace function from grandparent class
                self.graph = Tracer.trace(self, root, concrete_args=concrete_args)
        except Exception as e:
            logger.error(
                f"Error when tracing module {root.name if root.name else 'root'}"
            )
            raise e
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)

        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in (sig.parameters.keys() - concrete_args.keys()):
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    to_visit = [node]
                    to_delete = OrderedDict()
                    while to_visit:
                        n = to_visit.pop(0)
                        to_delete[n] = None
                        to_visit += list(n.users.keys())

                    for user in reversed(to_delete.keys()):
                        self.graph.erase_node(user)

        # TODO: solves GraphModule creation.
        # Without this, return type annotation "Tuple" is causing code execution failure.
        if node.op == "output":
            node.type = None

        logger.debug(f"*** finish tracing module {root.name if root.name else 'root'}")
        logger.debug("")

        self.modules_to_graphs[root.name] = self.graph

        # Fix the undefined symbol error in graph: add the addtional globals to the graph
        logger.debug(f"--- additional_globals: {self._additional_globals}")
        self.graph._codegen.additional_globals = MethodType(
            lambda _self: list(self._additional_globals), self.graph._codegen
        )

        return self.graph

    def create_proxy(
        self,
        kind,
        target,
        args,
        kwargs,
        name=None,
        type_expr=None,
        proxy_factory_fn=None,
    ):
        # DEBUG
        logger.debug(
            f"==> create_proxy [BEGIN]: {kind}, {target}, [args] {args}, [kwargs] {kwargs}"
        )

        rv = Tracer.create_proxy(
            self, kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )

        # Deal with the case where the target is a placeholder
        # Logic is somewhat complicated because we need to deal with corner cases
        if kind == "placeholder":
            metadata = None
            if target in self.meta_args:
                metadata = self.meta_args.pop(target)
            elif args:
                metadata = args[0]

            # Deal with corner cases: (1) None, (2) var positional, (3) var keyword
            if metadata is not None:
                if target.startswith("**"):
                    ret = {}
                    for name, value in self.meta_args.items():
                        rv = Tracer.create_proxy(
                            self, kind, name, args, kwargs, type_expr, proxy_factory_fn
                        )
                        rv.install_metadata(value)
                        ret[name] = rv
                elif target.startswith("*"):
                    ret = []
                else:
                    rv.install_metadata(metadata)
                    ret = rv
            else:
                ret = None

            show_value = ret._metadata if isinstance(ret, Proxy) else ret
            logger.debug(
                f"==> create_proxy [END]  : {kind}, {target}, {'[metadata] ' + str(show_value)}"
            )
            return ret

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs and self.device is not None:
                kwargs["device"] = self.device

        # try:
        with nullcontext():
            args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

            if kind == "call_function":
                # meta_out = target(*args_metas, **kwargs_metas)
                meta_out = self.device_safe_func_exec(
                    target, *args_metas, **kwargs_metas
                )
            elif kind == "call_method":
                meta_target = getattr(args_metas[0].__class__, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(
                        f"{self} does not have an attribute called orig_forward"
                    )
                self._disable_module_getattr = True
                try:
                    mod = self.root.get_submodule(target)
                    mod_name = getattr(mod, "name", None)
                    if mod_name in self.sub_graph_modules:
                        # (1) Generate inputs for a new tracing of the submodule
                        #
                        # Note 1: we have updated the code to make sure that if the real metadata is None, we will not wrap the
                        # meta_arg by Proxy. So the only different between meta_args and concrete_args is that the latter won't show in the
                        # function signature (which means it shouldn't be input to the submodule's forward function). However, because for submodule
                        # function call, we don't want to change the function call, so we make them as meta_args instead of concrete_args.
                        #
                        # See ``create_args_for_root`` in ``torch/fx/_symbolic_trace.py`` for the logic of creating args for root
                        # See ``create_proxy`` in this file for the logic of metadata wrapping
                        #
                        # TODO(zhanda): make better comments for dealing with *args and **kwargs.
                        # Note 2: if there is VAR_KEYWORD in the signature, e.g. **kwargs, we need to manually deal with this case
                        # i.e. ``meta_args[f"**{param.name}"] = {}``. Note, during the placeholder creation, *args is indexed by
                        # '*args' (not 'args') and **kwargs (not 'kwargs') is indexed by '**kwargs'. And this is handled quite
                        # differently.

                        sig = inspect.signature(mod.forward)

                        # Generate meta_args for the submodule
                        bound_sig = sig.bind(*args_metas, **kwargs_metas)
                        cur_meta_args = bound_sig.arguments

                        # Special handling for VAR_KEYWORD
                        for i, (name, param) in enumerate(sig.parameters.items()):
                            if param.kind == inspect.Parameter.VAR_KEYWORD:
                                cur_meta_args[f"**{name}"] = {}

                        # Trace the submodule and get the meta_out
                        sub_tracer = MistTracer(
                            sub_modules=self.sub_modules,
                            sub_graph_modules=self.sub_graph_modules,
                        )
                        sub_graph = sub_tracer.trace(
                            mod,
                            meta_args=cur_meta_args,
                            device=self.device,
                        )

                        # Here output can be HFProxy, py_tree (e.g. list/tuple) of HFProxy, or even anything else
                        meta_out = tree_map(_proxies_to_metas, sub_tracer._output)

                        # Update the dict of the mapping from traced module to the traced graph
                        self.modules_to_graphs.update(sub_tracer.modules_to_graphs)

                    # Submod in sub_modules but not in sub_graph_modules (prev case)
                    # we don't trace into these submodules
                    elif mod_name in self.sub_modules:
                        with RevertPatcher(get_root_patcher()):
                            # meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                            meta_out = self.device_safe_func_exec(
                                self.orig_forward, *args_metas, **kwargs_metas
                            )
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                except:
                    raise
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if isinstance(attr_itr, torch.Tensor):
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            if not isinstance(rv, Proxy):
                raise ValueError("Don't support composite output yet")
            rv.install_metadata(meta_out)

            logger.debug(
                f"==> create_proxy [END]  : {kind}, {target}; [meta_args] {tree_to_entries(args_metas)}, [meta_kwargs] {kwargs_metas}; [meta_out] {tree_to_entries(meta_out)}"
            )

        # except Exception as e:
        #     logger.error(f"ERROR: [kind] {kind}, [target] {target}")
        #     logger.error(f"ERROR: [args] {args}")
        #     logger.error(f"ERROR: [meta_args] {args_metas}")
        #     logger.error(f"ERROR: [kwargs] {kwargs}")
        #     logger.error(f"ERROR: [meta_kwargs] {kwargs_metas}")
        #     logger.error(f"ERROR: {e}")
        #     exit(1)
        #     raise e
        #     if _IS_IN_DEBUG_MODE:
        #         logger.warn(
        #             f"Could not compute metadata for {kind} target {target}: {e}"
        #         )

        return rv

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """
        This function is used for the tracer logic (overriding the default is_leaf_module)
        """
        if not is_fx_tracable(m) or (getattr(m, "name", None) in self.sub_modules):
            return True
        return super().is_leaf_module(m, module_qualified_name)

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: "Proxy") -> Iterator:
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        if hasattr(obj, "_metadata"):
            _metadata = obj._metadata
            logger.warning(
                f"Proxy object {obj} is being iterated over. Its metadata is {_metadata}"
            )
            if isinstance(_metadata, (dict, list, tuple, dict_keys)):
                return iter(_metadata)

        raise TraceError(
            "Proxy object cannot be iterated. This can be "
            "attempted when the Proxy is used in a loop or"
            " as a *args or **kwargs function argument. "
            "See the torch.fx docs on pytorch.org for a "
            "more detailed explanation of what types of "
            "control flow can be traced, and check out the"
            " Proxy docstring for help troubleshooting "
            "Proxy iteration errors"
        )


def mist_trace(
    model: nn.Module,
    inputs: Dict[str, Any],
    sub_modules: Optional[Sequence[str]] = None,
    trace_into_submodule: bool = True,
    output_first_item_in_output_node: bool = False,
    device=None,
    fallback_device="cuda",
):
    set_module_name_recursive(model)
    sub_modules = sub_modules or get_default_sub_modules(model)
    sub_graph_modules = sub_modules if trace_into_submodule else None

    tracer = MistTracer(
        sub_modules=sub_modules,
        sub_graph_modules=sub_graph_modules,
    )
    with temporarily_set_sp_eq_ne():
        root_graph = tracer.trace(
            model, meta_args=inputs, device=device, fallback_device=fallback_device
        )

    if output_first_item_in_output_node:
        output_first_item_in_output_node_pass(root_graph)

    for name, graph in tracer.modules_to_graphs.items():
        graph.owning_module = getattr_recursive(model, name)
        graph.eliminate_dead_code()
        graph.lint()

    return root_graph, tracer.modules_to_graphs
