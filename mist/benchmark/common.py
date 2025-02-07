import os
from omegaconf import DictConfig, OmegaConf
import hydra
from pprint import pprint, pformat
from typing import List, Optional, Union, Tuple, Any, Dict, Callable
from functools import partial

import torch
import torch.nn as nn

from mist import global_symbol_manager as gsm
from mist.analyzer.layer_analyzer import LayerInfo, analyze_blocks
from mist.analyzer.batched_module_analyzer import batched_stage_analyze
from mist.config import (
    MistConfig,
    ModelConfig,
    TrainingConfig,
    HardwareConfig,
    StrategyConfig,
)
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.pipeline_parallel.pipe_module import build_pipe_modules_for_analyzing
from mist.tracer.symbolic_tracer import mist_trace
from mist.tuning.optimization import build_and_tune_optimization_problem
from mist.symbols import temporarily_set_sp_eq_ne
from mist.utils.memory import cuda_empty_cache
from mist.utils.device import get_device
from mist.utils.common import load_pickle, save_pickle
from mist.model_provider import base_model_provider, get_inputs_provider

logger = get_logger(__name__)


def trace_and_analyze(
    model_provider: Callable,
    inputs_provider: Callable,
    symbol_mapping: Dict[str, Any],
    mist_config: MistConfig,
):
    tp_size = symbol_mapping["tp_size"]
    dummy_mist_process_group = MistProcessGroup(tp_size, rank=0, global_rank=8)
    model = model_provider(
        device=get_device(torch.cuda.current_device()),
        process_groups=dummy_mist_process_group,
        pre_post_process_group=dummy_mist_process_group,
    )
    inputs = inputs_provider()

    with torch.no_grad():
        with temporarily_set_sp_eq_ne():
            outputs = model(**inputs)

    # Trace the model
    graph, modules_to_graphs = mist_trace(
        model, inputs, device=get_device(torch.cuda.current_device())
    )

    # Extract pre-layer, block-layer, and post-layer modules and analyze them
    sequential_blocks: List[nn.Module] = build_pipe_modules_for_analyzing(
        model, graph, modules_to_graphs, inputs=inputs
    )
    layer_infos: List[LayerInfo] = analyze_blocks(
        sequential_blocks,
        inputs=inputs,
        symbol_mapping=symbol_mapping,
        config=mist_config,
        grad_outputs="auto",
        has_pre_layer=True,
        has_post_layer=True,
    )

    return layer_infos


def get_common_providers_for_analysis_and_tuning(
    mist_config: MistConfig,
    num_hidden_layers: int,
    saved_path: str = None,
    force_rebuild: bool = True,
):
    model_config = mist_config.model
    training_config = mist_config.training
    strategy_config = mist_config.strategy
    hardware_config = mist_config.hardware
    tuning_config = mist_config.tuning
    model_config.tensor_parallel = True

    # Load the cached layer infos if possible
    saved_folder = "model_tracing_and_analysis_results"
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    if saved_path is None:
        saved_path = (
            f"model_{model_config.name}_ori_layers_{model_config.num_hidden_layers}_"
            f"analyzed_layers_{num_hidden_layers}_"
            f"f_{model_config.use_flash_attn}_"
            f"s_{model_config.max_position_embeddings}_v_{model_config.vocab_size}.pkl"
        )
        saved_path = os.path.join(saved_folder, saved_path)

    rebuild = force_rebuild or not os.path.exists(saved_path)

    if not rebuild:
        logger.info(f"Loading saved analyzed layer infos from {saved_path}")
        data = load_pickle(saved_path)
        assert isinstance(data, dict)
        symbol_manager = data["symbol_manager"]
        layer_infos = data["layer_infos"]
        symbol_mapping = data["symbol_mapping"]
        b = symbol_mapping["per_device_batch_size"]
        tp_size = symbol_mapping["tp_size"]

        # Load the symbol manager and symbol mapping
        gsm.reset()
        gsm.load_from_symbol_manager(symbol_manager)
        assert all(s in gsm.mapping for s in symbol_mapping.values()), (
            f"symbol_mapping contains symbols that are not in the global symbol manager: "
            f"{symbol_mapping}"
        )

    else:
        # Define the symbol mapping
        b = gsm.symbols("b", 2, integer=True, positive=True)
        tp_size = gsm.symbols("tp_size", 2, integer=True, positive=True)
        symbol_mapping = {"per_device_batch_size": b, "tp_size": tp_size}

    s = model_config.max_position_embeddings
    v = model_config.vocab_size

    model_provider = partial(
        base_model_provider,
        model_name=model_config.name,
        model_config=model_config,
        num_hidden_layers=num_hidden_layers,
        pre_process=True,
        post_process=True,
    )

    base_inputs_provider = get_inputs_provider(model_config.name)
    inputs_provider = partial(
        base_inputs_provider,
        batch_size=b,
        seq_len=s,
        device=get_device(torch.cuda.current_device()),
    )

    # # Create the model provider
    # model_provider, model_specific_config = get_model_provider(
    #     model_config, training_config, num_hidden_layers=num_hidden_layers
    # )

    # # Get the inputs provider
    # if model_config.name.startswith(("gpt", "llama")):

    #     def inputs_provider():
    #         input_ids = torch.randint(
    #             0, v, (b, s), dtype=torch.long, device=torch.cuda.current_device()
    #         )
    #         labels = torch.randint(
    #             0, v, (b, s), dtype=torch.long, device=torch.cuda.current_device()
    #         )
    #         inputs = {"input_ids": input_ids, "labels": labels}
    #         return inputs

    # elif model_config.name.startswith("bert"):

    #     def inputs_provider():
    #         input_ids = torch.randint(
    #             0, v, (b, s), dtype=torch.long, device=torch.cuda.current_device()
    #         )
    #         labels = torch.randint(
    #             0, v, (b, s), dtype=torch.long, device=torch.cuda.current_device()
    #         )
    #         next_sentence_label = torch.randint(
    #             0, 2, (b,), dtype=torch.long, device=torch.cuda.current_device()
    #         )
    #         inputs = {
    #             "input_ids": input_ids,
    #             "labels": labels,
    #             "next_sentence_label": next_sentence_label,
    #         }
    #         return inputs

    # else:
    #     raise ValueError(f"Unsupported model name: {model_config.name}")

    # Get the layer info provider
    layer_info_provider = partial(
        trace_and_analyze,
        model_provider=model_provider,
        inputs_provider=inputs_provider,
        symbol_mapping=symbol_mapping,
        mist_config=mist_config,
    )

    if rebuild:
        layer_infos = layer_info_provider()
        cuda_empty_cache()
        save_pickle(
            {
                "symbol_manager": gsm,
                "layer_infos": layer_infos,
                "symbol_mapping": symbol_mapping,
            },
            saved_path,
        )

    ret = {
        "symbol_manager": gsm,
        "layer_infos": layer_infos,
        "symbol_mapping": symbol_mapping,
        "model_provider": model_provider,
        "inputs_provider": inputs_provider,
        "layer_info_provider": layer_info_provider,
    }
    return ret
