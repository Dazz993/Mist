from typing import List, Optional, Union, Tuple, Any, Dict, Sequence, Callable
import argparse
from collections import OrderedDict
from copy import deepcopy
from functools import cache, partial
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils._pytree import tree_flatten
from transformers import LlamaConfig, GPT2Config, BertConfig

from mist import gsm
from mist.analyzer.layer_analyzer import LayerInfo, analyze_blocks
from mist.config import MistConfig
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.pipeline_parallel.pipe_module import build_pipe_modules_for_analyzing
from mist.symbols import temporarily_set_sp_eq_ne
from mist.tracer.symbolic_tracer import mist_trace
from mist.utils.memory import materialize_module
from mist.utils.device import get_device

from mist.model.gpt import GPTModel, gpt_inputs_provider
from mist.model.llama import LLamaModel, llama_inputs_provider
from mist.model.falcon import FalconModel, falcon_inputs_provider
from mist.model.bert import BertModel, bert_inputs_provider
from mist.model.t5 import T5Model, t5_inputs_provider

from mist.config import ModelConfig, TrainingConfig
from mist.symbols import temporarily_set_sp_eq_ne

logger = get_logger(__name__)

def get_model_cls(model_name: str) -> Callable:
    model_name = model_name.lower()
    if model_name.startswith("gpt2"):
        return GPTModel
    elif model_name.startswith("llama"):
        return LLamaModel
    elif model_name.startswith("falcon"):
        return FalconModel
    elif model_name.startswith("bert"):
        return BertModel
    elif model_name.startswith("t5"):
        return T5Model
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def get_inputs_provider(model_name: str) -> Callable:
    model_name = model_name.lower()
    if model_name.startswith("gpt2"):
        return gpt_inputs_provider
    elif model_name.startswith("llama"):
        return llama_inputs_provider
    elif model_name.startswith("falcon"):
        return falcon_inputs_provider
    elif model_name.startswith("bert"):
        return bert_inputs_provider
    elif model_name.startswith("t5"):
        return t5_inputs_provider
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def base_model_provider(
    model_config: ModelConfig,
    model_cls: Optional[Callable] = None,
    model_name: Optional[str] = None,
    num_hidden_layers: Optional[int] = None,
    pre_process: bool = True,
    post_process: bool = True,
    process_groups: Optional[Union[List[dist.ProcessGroup], dist.ProcessGroup]] = None,
    pre_post_process_group: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    if model_cls is not None and model_name is not None:
        _model_cls = get_model_cls(model_name)
        assert model_cls == _model_cls, f"model_cls and model_name are inconsistent: {model_cls} != {_model_cls}"
    elif model_cls is None:
        assert model_name is not None, "model_cls or model_name must be provided."
        model_cls = get_model_cls(model_name)

    curr_model_config = model_config
    if num_hidden_layers is not None:
        curr_model_config = deepcopy(model_config)
        curr_model_config.num_hidden_layers = num_hidden_layers

    # Set process groups.
    if not isinstance(process_groups, list):
        process_groups = [process_groups] * curr_model_config.num_hidden_layers
    assert len(process_groups) == curr_model_config.num_hidden_layers

    # Set pre_post_process_group.
    if pre_post_process_group is None:
        pre_post_process_group = process_groups[0]

    # Get device and dtype.
    device = device or get_device()
    dtype = dtype or model_config.params_dtype

    with temporarily_set_sp_eq_ne():
        model = model_cls(
            model_config=curr_model_config,
            process_groups=process_groups,
            pre_post_process_group=pre_post_process_group,
            pre_process=pre_process,
            post_process=post_process,
            device=device,
            dtype=dtype,
        )

    return model
