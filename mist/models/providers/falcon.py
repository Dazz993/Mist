from typing import Optional, Union, Sequence, Tuple, Dict, Any

import sympy as sp
import torch
from torch.distributed import ProcessGroup
from transformers import FalconConfig

from mist.config import ModelConfig, TrainingConfig
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.models.common import add_opt_config_to_mist_gpt_config
from mist.models.gpt import GPTLMHeadModel
from mist.models.gpt_config import MistGPTConfig
from mist.symbols import temporarily_set_sp_eq_ne


logger = get_logger()


def falcon_config_to_gpt2_config(falcon_config: FalconConfig) -> MistGPTConfig:
    # The 40b config uses "n_head_kv" instead of "num_kv_heads"
    n_head_kv = getattr(
        falcon_config,
        "n_head_kv",
        1 if getattr(falcon_config, "multi_query", False) else falcon_config.n_head,
    )
    # HACK: the 40b config has 2 LN per layer instead of 1, but that's not reflected in the config.
    # So we have to infer it from the number of heads in the key/value block
    parallel_block_tied_norm = n_head_kv == 1
    return MistGPTConfig(
        vocab_size=falcon_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=falcon_config.hidden_size,
        n_layer=falcon_config.n_layer,
        n_head=falcon_config.n_head,
        n_inner=falcon_config.hidden_size * 4,
        activation_function="gelu",
        resid_pdrop=falcon_config.hidden_dropout,
        embd_pdrop=0.0,  # There doesn't seem to be any embedding dropout
        attn_pdrop=falcon_config.attention_dropout,
        layer_norm_epsilon=falcon_config.layer_norm_epsilon,
        initializer_range=falcon_config.initializer_range,
        bos_token_id=falcon_config.bos_token_id,
        eos_token_id=falcon_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        parallel_block=falcon_config.parallel_attn,
        n_head_kv=n_head_kv,
        parallel_block_tied_norm=parallel_block_tied_norm,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=False,
        tie_word_embeddings=True,
        qkv_proj_bias=falcon_config.bias,
        out_proj_bias=falcon_config.bias,
        mlp_fc1_bias=falcon_config.bias,
        mlp_fc2_bias=falcon_config.bias,
        lm_head_bias=False,
    )


def mist_model_config_to_mist_gpt_config_for_falcon(
    model_config: ModelConfig,
) -> MistGPTConfig:
    default_hf_config = FalconConfig()
    return MistGPTConfig(
        vocab_size=model_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=model_config.hidden_size,
        n_layer=model_config.num_hidden_layers,
        n_head=model_config.num_attention_heads,
        n_inner=model_config.intermediate_size,
        activation_function="gelu_new",
        # FIXME(zhanda): here we still don't do any dropout even for Falcon models
        # this is for the comparison fairness
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=getattr(
            model_config, "rms_norm_eps", default_hf_config.rms_norm_eps
        ),
        initializer_range=getattr(
            model_config, "initializer_range", default_hf_config.initializer_range
        ),
        bos_token_id=getattr(
            model_config, "bos_token_id", default_hf_config.bos_token_id
        ),
        eos_token_id=getattr(
            model_config, "eos_token_id", default_hf_config.eos_token_id
        ),
        # These are new arguments not in the original GPT2Config
        pad_token_id=getattr(
            model_config, "pad_token_id", default_hf_config.pad_token_id
        ),  # Idk if this does anything
        parallel_block=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        # FIXME(zhanda): here we still don't tie word embeddings for Falcon models
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        n_head_kv=getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        ),
    )


def get_falcon_model_provider_from_mist_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    num_hidden_layers: Optional[int] = None,
):
    name = model_config.name
    if not name.startswith("falcon"):
        raise ValueError(f"Model name {name} does not start with 'falcon'")

    if num_hidden_layers is not None:
        mist_gpt_config.n_layer = num_hidden_layers

    mist_gpt_config = mist_model_config_to_mist_gpt_config_for_falcon(model_config)
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


def get_falcon_inputs_dummy_provider(
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
