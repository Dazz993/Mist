from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Any, Sequence
from collections import OrderedDict
from copy import deepcopy
import inspect
import functools
import re
import sys

import torch
from torch import nn, fx
from torch.fx import Graph, GraphModule, Proxy, Tracer, Node
from torch.utils._pytree import tree_map

from transformers.modeling_outputs import ModelOutput

from mist.logger import get_logger, patch_output_fn_with_indent, logging
from mist.utils.fx import (
    create_graph_module,
    is_primitive_module,
    register_model_output_in_pytree_map,
)
from mist.tracer.hf import (
    HFTracer,
    HFProxy,
    _proxies_to_metas,
    _gen_constructor_wrapper,
    _IS_IN_DEBUG_MODE,
    _MANUAL_META_OVERRIDES,
)

logger = get_logger()

LOGGING_INDENT = -4
LOGGING_INDENT_STEP = 4

# This is used to register HF/ModelOutput class in pytree_map
register_model_output_in_pytree_map()


def is_fx_tracable(mod):
    return not hasattr(mod, "traceable") or mod.traceable


def _get_leaf_module_for_submodule_tracing(root: nn.Module):
    non_prim_leaf_modules = list(
        set(
            [
                m.name
                for m in root.modules()
                if not is_primitive_module(m)
                and m != root
                and not isinstance(m, nn.ModuleList)
            ]
        )
    )
    return non_prim_leaf_modules


def _better_show_submodules_path(leaf_modules_path: Sequence[str]):
    ret = set()
    for path in leaf_modules_path:
        # Replace the number with 'N' using regex
        path = re.sub(r"\d+", "N", path)
        ret.add(path)
    # sort the path by the number of '.' in the path
    ret = sorted(ret, key=lambda x: x.count("."))
    return list(ret)


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
    non_prim_leaf_modules : List[str]
        List of names of modules that should be treated as leaf modules. These modules will be converted
        to GraphModule with subgraph tracing applied to them.
        Note:
            (1) All the primitive modules (e.g., torch.nn.XXXX) will be treated as leaf modules by default.
                And they will not be traced into.
            (2) If leaf_modules is None (or not provided), all non-primitive child modules will be treated
                as leaf modules. And these modules will be traced into.
            (3) If leaf_modules is [], then the module is flattened and fully traced.
    """

    def __init__(self, **config: Dict[str, Any]) -> None:
        super().__init__()
        self.name = "mist_hf"
        self.non_prim_leaf_modules = config.get("non_prim_leaf_modules", None)

        # During trace, we collect the graph of the submodules that are traced into.
        # but we don't convert it to GraphModule immediately because it may lose some attributes
        # that are not used in this graph (but may be used in other graphs).
        # The graph modules will be created in the postprocess stage.
        self.modules_to_graphs: Dict[str, Graph] = OrderedDict()
        self._fn_output = None

    def create_arg(self, a: Any) -> "Argument":
        if sys._getframe(1).f_code.co_name == "trace":
            self._fn_output = a

        if isinstance(a, ModelOutput):
            args = {key: self.create_arg(value) for key, value in a.items()}
            return self.create_node("call_function", a.__class__, (args,), {})

        arg = Tracer.create_arg(self, a)
        return arg

    def trace(
        self,
        root: Union[nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        meta_args: Optional[Dict[str, Any]] = None,
        complete_concrete_args_with_inputs_not_in_meta_args: bool = True,
    ) -> Graph:

        # global LOGGING_INDENT
        # LOGGING_INDENT += LOGGING_INDENT_STEP
        # _patch_output_fn_with_indent(logger, logger.debug, LOGGING_INDENT)

        # Make preparations for root module
        # (1) Set module name for all submodules and then set leaf modules
        # (2) For subgraph tracing for leaf modules,
        #     we need to set the leaf modules if not assigned (maintain the structure by default)
        if isinstance(root, nn.Module):
            if not hasattr(root, "name"):
                for name, m in root.named_modules():
                    m.name = name
            if self.non_prim_leaf_modules is None:
                self.non_prim_leaf_modules = _get_leaf_module_for_submodule_tracing(
                    root
                )
        else:
            root.name = ""
            self.non_prim_leaf_modules = []

        # Some notes on python signatures:
        # (1) _ParameterKind.POSITIONAL_OR_KEYWORD: a normal argument
        # (2) _ParameterKind.VAR_POSITIONAL: *args
        # (3) _ParameterKind.VAR_KEYWORD: **kwargs
        sig = inspect.signature(root.forward if isinstance(root, nn.Module) else root)
        concrete_args = concrete_args or {}
        meta_args = meta_args or {}

        # DEBUG
        logger.debug("")
        logger.debug(f"Tracing module {root.name if root.name else 'root'}")
        logger.debug(f"--- siganature: {sig}")
        logger.debug(f"--- meta_args: {meta_args}")
        logger.debug(f"--- concrete_args: {concrete_args}")
        # Only show non_prim_leaf_modules when root is the top module for clarity
        if root.name == "":
            logger.debug(
                f"--- non_prim_leaf_modules: {_better_show_submodules_path(self.non_prim_leaf_modules)}"
            )

        assert not any(
            isinstance(arg, HFProxy) for arg in meta_args.values()
        ), f"meta_args should not contain HFProxy. {meta_args}"

        # Complete concrete_args with default values
        if complete_concrete_args_with_inputs_not_in_meta_args:
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
            input_name: input_.to("meta")
            if isinstance(input_, torch.Tensor)
            else input_
            for input_name, input_ in meta_args.items()
        }
        for param in sig.parameters.values():
            if (
                param.kind == inspect.Parameter.VAR_KEYWORD
                and param.name not in meta_args
            ):
                meta_args[f"**{param.name}"] = {}
        self.meta_args = meta_args

        # Get all input_args
        # no use for concrete_args because concrete_args' placeholders are renamed with postfix number
        # self.input_args = {}
        # self.input_args.update(self.concrete_args)
        # self.input_args.update(self.meta_args)

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

        # LOGGING_INDENT -= LOGGING_INDENT_STEP
        # _patch_output_fn_with_indent(logger, logger.debug, LOGGING_INDENT)

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

        if kind == "placeholder":
            # concrete_args (input args that are not in meta_args) are renamed with postfix number
            # so we can't find them in the predefined dict e.g. self.input_args
            if target in self.meta_args or args:
                if target in self.meta_args:
                    rv.install_metadata(self.meta_args[target])
                else:
                    if isinstance(args[0], torch.Tensor):
                        rv.install_metadata(args[0].to("meta"))
                    else:
                        rv.install_metadata(args[0])
            logger.debug(
                f"==> create_proxy [END]  : {kind}, {target}, {'[metadata] ' + str(rv._metadata) if hasattr(rv, '_metadata') else '[no metadata]'}"
            )
            return rv

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, _proxies_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, _proxies_to_metas)

            if kind == "call_function":
                meta_target = _MANUAL_META_OVERRIDES.get(target, target)
                meta_out = meta_target(*args_metas, **kwargs_metas)
                if isinstance(meta_out, torch.Tensor):
                    meta_out = meta_out.to(device="meta")
            elif kind == "call_method":
                method = getattr(args_metas[0].__class__, target)
                meta_target = _MANUAL_META_OVERRIDES.get(method, method)
                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AttributeError(
                        f"{self} does not have an attribute called orig_forward"
                    )
                self._disable_module_getattr = True
                try:
                    # When a call_module is encountered,
                    # (1) if the module can be manually overridden, simply call the overridden one
                    # (2) if the module in self.non_prim_leaf_modules, trace the submodule
                    # (3) otherwise, call the original forward function
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in _MANUAL_META_OVERRIDES:
                        meta_out = _MANUAL_META_OVERRIDES[mod_type](
                            mod, *args_metas, **kwargs_metas
                        )
                    elif mod.name in self.non_prim_leaf_modules:

                        # (1) Generate inputs for a new tracing of the submodule
                        # here we traverse the signature of the submodule's forward function first len(args_metas) args are
                        # the args of the submodule the rest of the args are kwargs of the submodule
                        #
                        # Note 1: for normal arguments the meta_args will be wrapped by Proxy, and its _metadata stores the real
                        # tensor info the concrete_args will not be wrapped by Proxy, and its value is like Tensor, int, float, etc.
                        #
                        # Therefore, if this _arg is None, we should add it to concrete_args (which will not be wrapped by Proxy)
                        # because we may encounter the case like ``if attention_mask is None: ...`` and the value wrapped by Proxy
                        # will not pass the None check because Proxy(None) is not None.
                        #
                        # See ``create_args_for_root`` in ``torch/fx/_symbolic_trace.py`` for the logic of creating args for root
                        #
                        # TODO(zhanda): make better comments for dealing with *args and **kwargs.
                        # Note 2: if there is VAR_KEYWORD in the signature, e.g. **kwargs, we need to manually deal with this case
                        # i.e. ``meta_args[f"**{param.name}"] = {}``. Note, during the placeholder creation, *args is indexed by
                        # '*args' (not 'args') and **kwargs (not 'kwargs') is indexed by '**kwargs'. And this is handled quite
                        # differently.

                        sig = inspect.signature(mod.forward)
                        _meta_args = {}
                        _concrete_args = {}
                        for i, (name, param) in enumerate(sig.parameters.items()):
                            if param.kind == inspect.Parameter.VAR_KEYWORD:
                                # _concrete_args[f"**{name}"] = {}
                                _meta_args[f"**{name}"] = {}
                                pass
                            else:
                                _arg = None
                                if i < len(args_metas):
                                    _arg = args_metas[i]
                                elif name in kwargs_metas:
                                    _arg = kwargs_metas[name]

                                if _arg is not None:
                                    _meta_args[name] = _arg
                                else:
                                    _concrete_args[name] = None

                        # (2) wrap the forward function of the submodule to get the meta_out
                        #     and trace the submodule. After tracing, restore things to normal.
                        #     Note:
                        #     (1) because PyTorch internal tracer get fn using getattr(type(root), self.traced_func_name)
                        #         we need to wrap the forward function of the submodule for the class instead of the instance
                        #     (2) we need to use functools.wraps to copy the original attributes of the forward function
                        #         e.g. function name, signature, etc.

                        # Note: This is deprecated because there are errors if the submodule's root_fn has *args or **kwargs
                        # because when although we wrap the fn with the original signature, when the tracer call `create_args_for_root`
                        # it will unwrap the function, and then make *args and **kwargs as normal arguments. This is not what we want.
                        # What we want is it should call `create_args_for_root` for ori_forward_fn. But to make it, we should rewrite
                        # the create_args_for_root and there will be a lot of hard code.
                        #
                        # But since what we want is to store the output of the forward function, we can make it in another ways.
                        # 1. change `Tracer.trace`: ``self.create_arg(fn(*args))``, we first get ``output = fn(*args)`` and then
                        #    ``self.create_arg(output)``. This requires to change the trace function.
                        # 2. another way is to override the create_arg function. If the father call name is `trace`, we know the last
                        #    call to `create_arg` should contain the output.

                        # ori_forward_fn = getattr(mod_type, "forward")
                        # output = None
                        #
                        # @functools.wraps(ori_forward_fn)
                        # def forward_wrapper(self, *args, **kwargs):
                        #     nonlocal output
                        #     logger.debug(f"DEBUGGING: args: {args}, kwargs: {kwargs}")

                        #     # The first branch is actually quite not intuitive and quite tricky
                        #     # See the comment in ``create_args_for_root`` in ``torch/fx/_symbolic_trace.py``
                        #     # function ``_patch_function``. If the tracer find there is a *args or **kwargs
                        #     # it will treat them as normal arguments.
                        #     #
                        #     # Originally it does not matter but here we do another wrapper for the forward
                        #     # to store the output. Therefore, we need to align the original design.

                        #     output = ori_forward_fn(self, *args, **kwargs)
                        #     return output
                        #
                        # setattr(mod_type, "forward", forward_wrapper)

                        # Trace the submodule and get the meta_out
                        sub_tracer = MistTracer(
                            non_prim_leaf_modules=self.non_prim_leaf_modules
                        )
                        sub_graph = sub_tracer.trace(
                            mod, concrete_args=_concrete_args, meta_args=_meta_args
                        )

                        # Here output can be HFProxy, py_tree (e.g. list/tuple) of HFProxy, or even anything else
                        meta_out = tree_map(_proxies_to_metas, sub_tracer._fn_output)
                        # The following two lines are for debugging, can be removed later
                        # this will be useful
                        # _meta_out = fx.node.map_aggregate(output, _proxies_to_metas)
                        # logger.debug(
                        #     f"DEBUGING FOR META_OUT EXTRACTION: output: {output}, pytree mapping: {meta_out}, fx.node mapping: {_meta_out}"
                        # )

                        # Restore functions and attrs
                        # setattr(mod_type, "forward", ori_forward_fn)

                        # Update the dict of the mapping from traced module to the traced graph
                        self.modules_to_graphs.update(sub_tracer.modules_to_graphs)

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
                        meta_out = attr_itr.to(device="meta")
                    else:
                        meta_out = attr_itr
                finally:
                    self._disable_module_getattr = False
            else:
                return rv

            if not isinstance(rv, Proxy):
                raise ValueError("Don't support composite output yet")
            rv.install_metadata(meta_out)

            logger.debug(
                f"==> create_proxy [END]  : {kind}, {target}; [meta_args] {args_metas}, [meta_kwargs] {kwargs_metas}; [meta_out] {meta_out}"
            )

        except Exception as e:
            logger.error(f"ERROR: [kind] {kind}, [target] {target}")
            logger.error(f"ERROR: [args] {args}")
            logger.error(f"ERROR: [meta_args] {args_metas}")
            logger.error(f"ERROR: [kwargs] {kwargs}")
            logger.error(f"ERROR: [meta_kwargs] {kwargs_metas}")
            logger.error(f"ERROR: {e}")
            exit(1)
            raise e
            if _IS_IN_DEBUG_MODE:
                logger.warn(
                    f"Could not compute metadata for {kind} target {target}: {e}"
                )

        return rv

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """
        This function is used for the tracer logic (overriding the default is_leaf_module)
        """
        if not is_fx_tracable(m) or (
            hasattr(m, "name") and m.name in self.non_prim_leaf_modules
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)
