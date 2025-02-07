"""Transformer based language model."""

from typing import Callable, Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import torch.distributed as dist

from mist.config import ModelConfig
from mist.model.enums import AttnMaskType, LayerType, MLPType
from mist.modules.embedding import ParallelEmbedding
from mist.model.transformer import ParallelTransformer
from mist.utils.device import get_device
import mist.distributed.op as dist_op
from mist.modules.fused_dense import ColumnParallelLinear


class Pooler(nn.Module):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hidden_size: int,
        init_method: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(Pooler, self).__init__()
        device = device or get_device()
        dtype = dtype or model_config.params_dtype
        self.dense = torch.nn.Linear(
            hidden_size, hidden_size, device=device, dtype=dtype
        )
        if init_method is not None:
            init_method(self.dense.weight)
        with torch.no_grad():
            self.dense.bias.zero_()
        self.sequence_parallel = model_config.sequence_parallel
        if self.sequence_parallel:
            raise NotImplementedError(
                "sequence parallelism is not supported for pooler layer"
            )

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        # if self.sequence_parallel:
        #     hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
        #         hidden_states, tensor_parallel_output_grad=False
        #     )

        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class TransformerLanguageModel(nn.Module):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        model_config: ModelConfig,
        pre_process: bool = True,
        post_process: bool = True,
        pre_post_process_group: Optional[dist.ProcessGroup] = None,
        process_groups: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(TransformerLanguageModel, self).__init__()
        self.model_config = model_config
        self.num_layers = model_config.num_hidden_layers
        self.hidden_size = model_config.hidden_size
        self.num_tokentypes = model_config.num_tokentypes
        self.encoder_attn_mask_type = model_config.encoder_attn_mask_type
        self.decoder_attn_mask_type = model_config.decoder_attn_mask_type
        self.add_encoder = model_config.add_encoder
        self.add_decoder = model_config.add_decoder
        self.add_pooler = model_config.add_pooler
        self.pre_process = pre_process
        self.post_process = post_process
        self.pre_post_process_group = pre_post_process_group
        if not isinstance(process_groups, list):
            process_groups = [process_groups] * self.num_layers
        assert len(process_groups) == self.num_layers
        self.process_groups = process_groups
        self.device = device or get_device()
        self.dtype = dtype or model_config.params_dtype
        self.tie_word_embeddings = model_config.tie_word_embeddings

        self.encoder_hidden_state = None

        s = model_config.max_position_embeddings
        l = model_config.num_hidden_layers
        v = model_config.vocab_size
        h = model_config.hidden_size
        mlp_mult_term = 64 if model_config.mlp_type == MLPType.gated_mlp else 16

        qkv_estimate = 6 * s * (h**2)
        attention_mat_estimate = 2 * (s**2) * h
        attention_vals_estimate = 2 * (s**2) * h
        linear_proj_estimate = 2 * s * (h**2)
        mlp_estimate = mlp_mult_term * s * h**2
        embedding_estimate = 6 * s * h * v

        per_layer_estimate = (
            qkv_estimate
            + attention_mat_estimate
            + attention_vals_estimate
            + linear_proj_estimate
            + mlp_estimate
        )
        self.flop_estimate = l * per_layer_estimate + embedding_estimate

        # Embeddings.
        if self.pre_process:
            self.embedding = ParallelEmbedding(
                model_config=model_config,
                process_group=self.pre_post_process_group,
                device=self.device,
                dtype=self.dtype,
            )
            self._embedding_key = "embedding"

        # Transformer.
        # Encoder (usually set to True, False if part of an encoder-decoder
        # architecture and in encoder-only stage).
        if self.add_encoder:
            self.encoder = ParallelTransformer(
                model_config=model_config,
                num_layers=self.num_layers,
                process_groups=self.process_groups,
                layer_type=LayerType.encoder,
                self_attn_mask_type=self.encoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                device=self.device,
                dtype=self.dtype,
            )
            self._encoder_key = "encoder"
        else:
            self.encoder = None

        # Decoder (usually set to False, True if part of an encoder-decoder
        # architecture and in decoder-only stage).
        if self.add_decoder:
            self.decoder = ParallelTransformer(
                model_config=model_config,
                num_layers=model_config.num_hidden_layers,
                process_groups=self.process_groups,
                layer_type=LayerType.decoder,
                self_attn_mask_type=model_config.decoder_attn_mask_type,
                pre_process=self.pre_process,
                post_process=self.post_process,
                device=self.device,
                dtype=self.dtype,
            )
            self._decoder_key = "decoder"
        else:
            self.decoder = None

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(
                    model_config=model_config,
                    hidden_size=model_config.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                self._pooler_key = "pooler"

            self.output_layer = ColumnParallelLinear(
                in_features=model_config.hidden_size,
                out_features=model_config.vocab_size,
                process_group=self.pre_post_process_group,
                bias=False,
                device=self.device,
                dtype=self.dtype,
            )  # Setting bias to False always to keep it consistent with embedding tying that also does not have a bias.
            self._output_layer_key = "output_layer"
            if self.tie_word_embeddings:
                assert (
                    self.embedding.word_embeddings.weight.data.size()
                    == self.output_layer.weight.data.size()
                ), "Tying requires the same dimensions for the word embeddings and the output layer"
                self.output_layer.weight.data = (
                    self.embedding.word_embeddings.weight.data
                )

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""

        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        if self.add_encoder and self.add_decoder:
            assert (
                len(input_tensor) == 1
            ), "input_tensor should only be length 1 for stage with both encoder and decoder"
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            assert (
                len(input_tensor) == 1
            ), "input_tensor should only be length 1 for stage with only encoder"
            self.encoder.set_input_tensor(input_tensor[0])
        elif self.add_decoder:
            if len(input_tensor) == 2:
                self.decoder.set_input_tensor(input_tensor[0])
                self.encoder_hidden_state = input_tensor[1]
            elif len(input_tensor) == 1:
                self.decoder.set_input_tensor(None)
                self.encoder_hidden_state = input_tensor[0]
            else:
                raise Exception("input_tensor must have either length 1 or 2")
        else:
            raise Exception("Stage must have at least either encoder or decoder")

    def forward(
        self,
        enc_input_ids,
        enc_position_ids=None,
        enc_attn_mask=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        enc_dec_attn_mask=None,
        tokentype_ids=None,
        inference_params=None,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden=False,
    ):
        # Encoder embedding.
        if self.pre_process:
            encoder_input = self.embedding(
                enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids
            )
        else:
            encoder_input = None

        # Run encoder.
        if enc_hidden_states is None:
            if self.encoder is not None:
                encoder_output = self.encoder(
                    encoder_input,
                    attention_mask=enc_attn_mask,
                    inference_params=inference_params,
                )
            else:
                encoder_output = self.encoder_hidden_state
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder embedding.
        if self.pre_process:
            decoder_input = self.embedding(dec_input_ids, dec_position_ids)
        else:
            decoder_input = None

        # Run decoder.
        decoder_output = self.decoder(
            decoder_input,
            dec_attn_mask,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
            inference_params=inference_params,
        )

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] = (
                self.embedding.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars
                )
            )
        if self.add_encoder:
            state_dict_[self._encoder_key] = (
                self.encoder.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars
                )
            )
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] = (
                    self.pooler.state_dict_for_save_checkpoint(
                        prefix=prefix, keep_vars=keep_vars
                    )
                )
            if not self.tie_embed_logits:
                state_dict_[self._lm_key] = self.lm_head.data
        if self.add_decoder:
            state_dict_[self._decoder_key] = (
                self.decoder.state_dict_for_save_checkpoint(
                    prefix=prefix, keep_vars=keep_vars
                )
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "_embeddings" in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Classifiaction head.
        if self.post_process and not self.tie_embed_logits:
            self.lm_head.data.copy_(state_dict[self._lm_key])

        # Encoder.
        if self.add_encoder:
            if self._encoder_key in state_dict:
                state_dict_ = state_dict[self._encoder_key]
            # For backward compatibility.
            elif "transformer" in state_dict:
                state_dict_ = state_dict["transformer"]
            else:
                # For backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if "transformer." in key:
                        state_dict_[key.split("transformer.")[1]] = state_dict[key]

            # For backward compatibility.
            state_dict_self_attention = {}
            for key in state_dict_.keys():
                if ".attention." in key:
                    state_dict_self_attention[
                        key.replace(".attention.", ".self_attention.")
                    ] = state_dict_[key]
                else:
                    state_dict_self_attention[key] = state_dict_[key]
            state_dict_ = state_dict_self_attention

            self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            if self.add_pooler:
                assert (
                    "pooler" in state_dict
                ), "could not find data for pooler in the checkpoint"
                self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)
        # Decoder.
        if self.add_decoder:
            assert (
                "decoder" in state_dict
            ), "could not find data for pooler in the checkpoint"
            self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
