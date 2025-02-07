from typing import Optional, Union, Sequence, Tuple, Dict, Any

import sympy as sp
import torch
from torch.distributed import ProcessGroup
from transformers import GPT2Config

from mist.config import ModelConfig, TrainingConfig
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.models.common import add_opt_config_to_mist_gpt_config
from mist.models.gpt import GPTLMHeadModel
from mist.models.gpt_config import MistGPTConfig
from mist.symbols import temporarily_set_sp_eq_ne


logger = get_logger()


def mist_model_config_to_mist_gpt_config_for_gpt(
    model_config: ModelConfig,
) -> MistGPTConfig:
    default_hf_config = GPT2Config()
    return MistGPTConfig(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.max_position_embeddings,
        n_embd=model_config.hidden_size,
        n_layer=model_config.num_hidden_layers,
        n_head=model_config.num_attention_heads,
        n_inner=model_config.intermediate_size,
        activation_function="gelu_new",
        # TODO(zhanda): here we still don't do any dropout even for GPT2 models
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        rms_norm=False,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        n_head_kv=getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        ),
    )


def get_gpt_model_provider_from_mist_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    num_hidden_layers: Optional[int] = None,
):
    name = model_config.name
    if not name.startswith("gpt"):
        raise ValueError(f"Model name {name} does not start with 'gpt'")

    if num_hidden_layers is not None:
        mist_gpt_config.n_layer = num_hidden_layers

    mist_gpt_config = mist_model_config_to_mist_gpt_config_for_gpt(model_config)
    mist_gpt_config = add_opt_config_to_mist_gpt_config(mist_gpt_config, model_config)
    logger.debug(f"Mist GPT config: {mist_gpt_config}")

    def model_provider(
        device: Union[torch.device, str] = "cuda",
        tp_process_groups: Optional[
            Sequence[Union[MistProcessGroup, ProcessGroup]]
        ] = None,
        pre_post_tp_process_group: Optional[
            Union[MistProcessGroup, ProcessGroup]
        ] = None,
    ):
        # This is a hack to make sure symbols can be supported when creating the model
        with temporarily_set_sp_eq_ne():
            model = GPTLMHeadModel(
                mist_gpt_config,
                process_groups=tp_process_groups,
                pre_post_process_group=pre_post_tp_process_group,
                device=device,
                dtype=training_config.params_dtype,
            )
        return model

    return model_provider, mist_gpt_config


def get_gpt_inputs_dummy_provider(
    batch_size: Union[sp.Basic, int],
    seq_len: Union[sp.Basic, int],
    vocab_size: Optional[Union[sp.basic, int]] = None,
) -> Dict[str, Any]:
    b = batch_size
    s = seq_len
    v = vocab_size if vocab_size is not None else 10000

    input_ids = torch.randint(
        0, v, (b, s), dtype=torch.long, device=torch.cuda.current_device()
    )
    labels = torch.randint(
        0, v, (b, s), dtype=torch.long, device=torch.cuda.current_device()
    )
    inputs = {"input_ids": input_ids, "labels": labels}
    return inputs
