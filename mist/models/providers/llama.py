from typing import Optional, Union, Sequence, Tuple, Dict, Any

import sympy as sp
import torch
from torch.distributed import ProcessGroup
from transformers import LlamaConfig

from mist.config import ModelConfig, TrainingConfig
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.models.common import add_opt_config_to_mist_gpt_config
from mist.models.gpt import GPTLMHeadModel
from mist.models.gpt_config import MistGPTConfig
from mist.symbols import temporarily_set_sp_eq_ne


logger = get_logger()


def llama_config_to_mist_gpt_config(llama_config: LlamaConfig) -> MistGPTConfig:
    return MistGPTConfig(
        vocab_size=llama_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        rotary_emb_base=getattr(llama_config, "rotary_emb_base", 10000.0),
        n_head_kv=llama_config.num_key_value_heads,
    )


def mist_model_config_to_mist_gpt_config_for_llama(
    model_config: ModelConfig,
) -> MistGPTConfig:
    default_hf_config = LlamaConfig()
    return MistGPTConfig(
        vocab_size=model_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=model_config.hidden_size,
        n_layer=model_config.num_hidden_layers,
        n_head=model_config.num_attention_heads,
        n_inner=model_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
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
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        rotary_emb_base=getattr(model_config, "rotary_emb_base", 10000.0),
        n_head_kv=getattr(
            model_config, "num_key_value_heads", model_config.num_attention_heads
        ),
    )


def get_llama_model_provider_from_mist_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    num_hidden_layers: Optional[int] = None,
):
    name = model_config.name
    if not name.startswith("llama"):
        raise ValueError(f"Model name {name} does not start with 'llama'")

    if num_hidden_layers is not None:
        mist_gpt_config.n_layer = num_hidden_layers

    mist_gpt_config = llama_config_to_mist_gpt_config(model_config)
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


def get_llama_inputs_dummy_provider(
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
