from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ProcessGroup

from mist.config import ModelConfig
from mist.model.enums import AttnMaskType, PositionEmbeddingType, MLPType
import mist.distributed.op as dist_op
from mist.modules.losses import vocab_parallel_cross_entropy
from mist.model.language_model import TransformerLanguageModel
from mist.utils.device import get_device
from mist.utils.memory import assert_viewless_tensor
from mist.model.gpt import GPTModel, gpt_inputs_provider


class FalconModel(GPTModel):
    def __init__(
        self,
        model_config: ModelConfig,
        num_tokentypes: int = 0,
        pre_process: bool = True,
        post_process: bool = True,
        process_groups: Optional[Union[List[ProcessGroup], ProcessGroup]] = None,
        pre_post_process_group: Optional[ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            model_config=model_config,
            num_tokentypes=num_tokentypes,
            pre_process=pre_process,
            post_process=post_process,
            process_groups=process_groups,
            pre_post_process_group=pre_post_process_group,
            device=device,
            dtype=dtype,
        )

    def _verify_config(self, model_config: ModelConfig):
        assert model_config.add_encoder, "FalconModel requires add_decoder to be True"
        assert (
            not model_config.add_decoder
        ), "FalconModel requires add_decoder to be False"
        assert (
            not model_config.add_pooler
        ), "FalconModel requires add_pooler to be False"
        assert (
            model_config.parallel_block
        ), "FalconModel requires parallel_block to be True"
        assert model_config.position_embedding_type == PositionEmbeddingType.rotary
        assert model_config.encoder_attn_mask_type == AttnMaskType.causal
        assert model_config.mlp_type == MLPType.mlp
        assert not model_config.tie_word_embeddings
        assert not model_config.qkv_proj_bias
        assert not model_config.out_proj_bias
        assert not model_config.mlp_fc1_bias
        assert not model_config.mlp_fc2_bias


falcon_inputs_provider = gpt_inputs_provider
