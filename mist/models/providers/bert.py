from typing import Optional, Union, Sequence, Tuple, Dict, Any

import sympy as sp
import torch
from torch.distributed import ProcessGroup
from transformers import FalconConfig

from mist.config import ModelConfig, TrainingConfig
from mist.distributed.overrides import MistProcessGroup
from mist.logger import get_logger
from mist.models.common import add_opt_config_to_mist_gpt_config
from mist.models.bert import BertConfig, BertForPreTraining
from mist.symbols import temporarily_set_sp_eq_ne
from mist.utils.memory import materialize_module


def get_bert_model_provider(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    num_hidden_layers: Optional[int] = None,
):
    name = model_config.name
    if not name.startswith(("bert")):
        raise ValueError(f"Unsupported model name: {name}")

    num_hidden_layers = (
        num_hidden_layers
        if num_hidden_layers is not None
        else model_config.num_hidden_layers
    )

    bert_config = BertConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        hidden_act=model_config.activation_function,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=model_config.max_position_embeddings,
    )
    bert_config.only_return_loss = True
    bert_config.uses_fp32_loss = False
    bert_config.use_flash_attn = getattr(model_config, "use_flash_attn", False)
    bert_config.fused_dropout_add_ln = getattr(
        model_config, "fused_dropout_add_ln", False
    )

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
            model = BertForPreTraining(
                bert_config,
                process_groups=tp_process_groups,
                # pre_post_process_group=pre_post_tp_process_group,
                device=device,
                dtype=training_config.params_dtype,
            )
            model = model.to(training_config.params_dtype)
            device = torch.device(device)
            if device.type == "meta":
                model = model.to(device)
            else:
                model = materialize_module(model, device=device, inplace=True)
        return model

    return model_provider, bert_config
