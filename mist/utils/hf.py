import inspect
import random
from typing import Optional, List, Sequence, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from mist.logger import get_logger
from mist.utils.initialization import init_empty_weights

logger = get_logger(__name__)


def _generate_random_int(
    low: int = 10, high: int = 20, forbidden_values: Optional[List[int]] = None
):
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value


def _generate_dummy_input(model, input_name, shape, device=None):
    if input_name == "input_ids":
        output = torch.randint(0, 1000, shape, dtype=torch.long, device=device)
    elif input_name == "attention_mask":
        output = torch.randint(0, 2, shape, dtype=torch.long, device=device)
    elif input_name == "labels":
        output = torch.randint(0, 1000, shape, dtype=torch.long, device=device)
    else:
        raise NotImplementedError(
            f"Unsupported input name {input_name} for model {model.__class__.__name__}"
        )

    logger.debug(
        f"Generated dummy input for {input_name}: {output.shape}, {output.dtype}, {output.device}"
    )

    return output


def generate_dummy_inputs_for_hf(
    model: nn.Module,
    input_names: Optional[List[str]] = None,
    shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    return_dict: bool = False,
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
    device, optional
        the device to put the dummy inputs, by default None
    """

    def generate(model, name, shape, device=None):
        input = {name: _generate_dummy_input(model, name, shape, device=device)}

        return input

    if input_names is None:
        input_names = list(model.dummy_inputs.keys())
        logger.info(f"Using dummy input keys from model.dummy_inputs: {input_names}")

    if shape is None:
        batch_size = _generate_random_int()
        seq_len = _generate_random_int()
        shape = [batch_size, seq_len]

    dummy_inputs = {}
    for name in input_names:
        input = generate(model, name, shape, device=device)
        dummy_inputs.update(input)

    if "return_dict" in inspect.signature(model.forward).parameters:
        dummy_inputs["return_dict"] = return_dict

    return dummy_inputs


def create_meta_model(model_name_or_config: Union[str, AutoConfig]):
    """
    Create a meta model for the given model name/config in huggingface transformers.
    """
    if isinstance(model_name_or_config, str):
        model_config = AutoConfig.from_pretrained(
            model_name_or_config,
            trust_remote_code=True,
        )
    else:
        model_config = model_name_or_config

    if getattr(model_config, "use_cache", None) == True:
        logger.warning(
            f"The model config {model_name_or_config} has `use_cache=True`, which is "
            "not expected for training. Overriding it to `use_cache=False`."
        )
        model_config.use_cache = False

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=True,
        )
    return model, model_config


def create_meta_dummy_inputs(model, batch_size, seq_len, input_names=None):
    """
    Create dummy inputs for the meta model.
    """
    inputs_names = input_names or ["input_ids", "attention_mask", "labels"]
    inputs = generate_dummy_inputs_for_hf(
        model,
        input_names=input_names,
        shape=(batch_size, seq_len),
        device="meta",
        return_dict=False,
    )
    return inputs


def create_meta_model_and_inputs(
    model_name_or_config, batch_size, seq_len, input_names=None
):
    """
    Create a meta model and dummy inputs for the given model name in huggingface transformers.
    """
    model, model_config = create_meta_model(model_name_or_config)
    inputs = create_meta_dummy_inputs(
        model, batch_size, seq_len, input_names=input_names
    )

    # Check the runnability of the original model
    with torch.no_grad():
        model(**inputs)

    return model, model_config, inputs


# ====================================================================================================
# Model Specific Fixing


def fix_hf_model(model):
    """
    Fix the model for huggingface transformers.
    """
    # Reset the rotary embedding for falcon
    reset_rotary_embedding_for_falcon(model)

    return model


def reset_rotary_embedding_for_falcon(model):
    for name, module in model.named_modules():
        if module.__class__.__name__ == "RotaryEmbedding" and "maybe_rotary" in name:
            module.seq_len_cached = None
            logger.info(f"Resetting rotary embedding for {name}")
