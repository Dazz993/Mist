from typing import List, Tuple, Union, Optional, Dict, Any, Callable
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.ops import StochasticDepth

from mist.config import ModelConfig
from mist.model.enums import (
    AttnMaskType,
    PositionEmbeddingType,
    LayerType,
    MLPType,
    AttnType,
)
from mist.modules.utils import get_norm, get_activation
from mist.utils.device import get_device
from mist.modules.mha import ParallelMHA
from mist.modules.mlp import ParallelMLP, ParallelGatedMlp
from mist.utils.memory import make_viewless_tensor


def get_parallel_mha(
    model_config: ModelConfig,
    cross_attn: bool,
    causal: bool,
    process_group: Optional[dist.ProcessGroup] = None,
    layer_idx: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
):
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    softmax_scale = 1.0 if not model_config.scale_attn_weights else head_dim**-0.5
    if model_config.scale_attn_by_inverse_layer_idx:
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    rotary_emb_dim = int(model_config.rotary_emb_fraction * head_dim)
    return ParallelMHA(
        embed_dim=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        process_group=process_group,
        num_heads_kv=model_config.num_heads_kv,
        cross_attn=cross_attn,
        causal=causal,
        qkv_proj_bias=model_config.qkv_proj_bias,
        out_proj_bias=model_config.out_proj_bias,
        dropout=model_config.attention_dropout,
        softmax_scale=softmax_scale,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=model_config.rotary_emb_base,
        rotary_emb_scale_base=model_config.rotary_emb_scale_base,
        rotary_emb_interleaved=model_config.rotary_emb_interleaved,
        use_flash_attn=model_config.use_flash_attn,
        checkpointing=False,
        sequence_parallel=model_config.sequence_parallel,
        device=device or get_device(),
        dtype=dtype or model_config.params_dtype,
    )


def get_parallel_mlp(
    model_config: ModelConfig,
    layer_idx: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
):
    activation = get_activation(model_config.activation_function)
    kwargs = {}
    kwargs.update(
        in_features=model_config.hidden_size,
        hidden_features=model_config.intermediate_size,
        out_features=model_config.hidden_size,
        process_group=process_group,
        activation=activation,
        bias1=model_config.mlp_fc1_bias,
        bias2=model_config.mlp_fc2_bias,
        sequence_parallel=model_config.sequence_parallel,
        device=device or get_device(),
        dtype=dtype or model_config.params_dtype,
    )
    if model_config.mlp_type == MLPType.mlp:
        return ParallelMLP(**kwargs)
    elif model_config.mlp_type == MLPType.gated_mlp:
        return ParallelGatedMlp(**kwargs)
    else:
        raise ValueError(f"MLP type {model_config.mlp_type} not supported.")


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        layer_type=LayerType.encoder,
        self_attn_mask_type=AttnMaskType.padding,
        process_group: Optional[dist.ProcessGroup] = None,
        drop_path_rate=0.0,
        device=None,
        dtype=None,
    ):
        super(ParallelTransformerLayer, self).__init__()
        self.model_config = model_config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.self_attn_mask_type = self_attn_mask_type
        self.drop_path_rate = drop_path_rate
        self.device = device or get_device()
        self.dtype = dtype or model_config.params_dtype

        self.prenorm = model_config.prenorm
        self.parallel_block = model_config.parallel_block
        self.fp32_residual_connection = model_config.fp32_residual_connection

        # Normalize the input data.
        self.input_norm = get_norm(model_config, device=self.device, dtype=self.dtype)

        # Self attention.
        self.self_attention = get_parallel_mha(
            model_config=model_config,
            cross_attn=False,
            causal=self_attn_mask_type == AttnMaskType.causal,
            process_group=process_group,
            layer_idx=layer_idx,
            device=self.device,
            dtype=self.dtype,
        )
        self.hidden_dropout = model_config.hidden_dropout
        self.bias_dropout_fusion = model_config.bias_dropout_fusion
        self.drop_path = (
            StochasticDepth(drop_path_rate) if drop_path_rate > 0.0 else None
        )
        if self.parallel_block and drop_path_rate > 0.0:
            raise ValueError("Drop path is not supported with parallel block.")

        # Normalize the attention output
        # (if not using parallel block, then use the same norm of MLP).
        if not self.parallel_block:
            self.post_attention_norm = get_norm(
                model_config, device=self.device, dtype=self.dtype
            )

        # Cross attention.
        if self.layer_type in (LayerType.decoder,):
            self.inter_attention = get_parallel_mha(
                model_config=model_config,
                cross_attn=True,
                causal=False,
                process_group=process_group,
                layer_idx=layer_idx,
                device=self.device,
                dtype=self.dtype,
            )
            # Normalize the attention output.
            self.post_inter_attention_norm = get_norm(
                model_config, device=self.device, dtype=self.dtype
            )

        # MLP
        if model_config.num_experts is not None:
            raise NotImplementedError("Switch MLP not supported yet.")
            self.mlp = SwitchMLP(config)
        else:
            self.mlp = get_parallel_mlp(
                model_config=model_config,
                layer_idx=layer_idx,
                process_group=process_group,
                device=self.device,
                dtype=self.dtype,
            )

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = (
            nullcontext if use_nvfuser else torch.enable_grad
        )

    def decoder_cross_attention(
        self,
        encoder_output,
        enc_dec_attn_mask,
        norm_input,
        norm_output,
        bias_dropout_add_func,
    ):
        """Cross attention for a standard encoder-decoder model."""

        # Attention.
        attention_output = self.inter_attention(
            norm_output, x_kv=encoder_output, attention_mask=enc_dec_attn_mask
        )
        attention_bias = None

        # Residual connection.
        if self.prenorm:
            residual = norm_input
        else:
            residual = norm_output

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)

        # Bias-dropout-add.
        with self.bias_dropout_add_exec_handler():
            norm_input = bias_dropout_add_func(
                attention_output, attention_bias, residual, self.hidden_dropout
            )

        # Normalize.
        norm_output = self.post_inter_attention_norm(norm_input)

        return norm_input, norm_output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_output=None,
        enc_dec_attn_mask=None,
        inference_params=None,
    ):
        # jit scripting for a nn.module (with dropout) is not
        # trigerring the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        ###############
        # Transformer #
        ###############

        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        norm_output = self.input_norm(hidden_states)

        # Self attention.
        attention_output = self.self_attention(
            norm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
        )
        attention_bias = None

        # Residual connection.
        if self.prenorm:
            residual = hidden_states
        else:
            residual = norm_output

        # Consider the dropout and parallel-block
        if self.parallel_block:
            # used only if layer is decoder and prenorm is True
            # which seems a bit strange, but it's kept just in case for now
            norm_input = attention_output
        else:
            if self.drop_path is None:
                if attention_bias is not None:
                    attention_bias = attention_bias.expand_as(residual)
                with self.bias_dropout_add_exec_handler():
                    norm_input = bias_dropout_add_func(
                        attention_output, attention_bias, residual, self.hidden_dropout
                    )
            else:
                out = torch.nn.functional.dropout(
                    attention_output + attention_bias,
                    p=self.hidden_dropout,
                    training=self.training,
                )
                norm_input = residual + self.drop_path(out)
            # Layer norm post the self attention.
            norm_output = self.post_attention_norm(norm_input)

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            pass
        elif self.layer_type == LayerType.decoder:
            norm_input, norm_output = self.decoder_cross_attention(
                encoder_output,
                enc_dec_attn_mask,
                norm_input,
                norm_output,
                bias_dropout_add_func,
            )
        else:
            raise Exception("Unsupported layer type, '%s'." % self.layer_type.name)

        # MLP.
        mlp_output = self.mlp(norm_output)
        mlp_bias = None

        # Second residual connection and parallel block
        # (if using parallel block, keep the residual as the original one).
        if self.parallel_block:
            mlp_output = mlp_output + attention_output
        elif self.prenorm:
            residual = norm_input
        else:
            residual = norm_output

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output, mlp_bias, residual, self.hidden_dropout
                )

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = make_viewless_tensor(
                inp=output, requires_grad=output.requires_grad, keep_graph=True
            )

        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(
                mlp_output, p=self.hidden_dropout, training=self.training
            )
            output = residual + self.drop_path(out)

        return output


class ParallelTransformer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        num_layers: Optional[int] = None,
        process_groups: Optional[
            Union[List[dist.ProcessGroup], dist.ProcessGroup]
        ] = None,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_type: LayerType = LayerType.encoder,
        self_attn_mask_type: AttnMaskType = AttnMaskType.padding,
        pre_process: bool = True,
        post_process: bool = True,
        drop_path_rate: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(ParallelTransformer, self).__init__()
        self.model_config = model_config
        self.num_layers = num_layers or model_config.num_hidden_layers
        if not isinstance(process_groups, (list, tuple)):
            process_groups = [process_groups] * self.num_layers
        assert len(process_groups) == self.num_layers, (
            f"Number of process groups must match the number of layers, "
            f"got {len(process_groups)} process groups and {self.num_layers} layers."
        )
        self.process_groups = process_groups
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.layer_type = layer_type
        self.self_attn_mask_type = self_attn_mask_type
        self.pre_process = pre_process
        self.post_process = post_process
        self.drop_path_rate = drop_path_rate
        self.device = device
        self.dtype = dtype or model_config.params_dtype
        self.prenorm = model_config.prenorm

        # Other attributes
        self.input_tensor = None
        self.sequence_parallel = model_config.sequence_parallel

        # Number of layers.
        self.num_layers = num_layers
        self.drop_path_rates = [
            rate.item()
            for rate in torch.linspace(
                0, self.drop_path_rate, model_config.num_hidden_layers
            )
        ]

        def build_layer(layer_idx):
            return ParallelTransformerLayer(
                model_config,
                layer_idx=layer_idx,
                layer_type=layer_type,
                process_group=process_groups[layer_idx],
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_idx],
            )

        assert self.num_layers > 0, "Number of layers must be greater than 0."
        self.layers = torch.nn.ModuleList(
            [build_layer(i) for i in range(self.num_layers)]
        )

        if self.post_process and not self.prenorm:
            # Final layer norm before output.
            self.final_norm = get_norm(
                model_config, device=self.device, dtype=self.dtype
            )

    def _get_layer(self, layer_idx):
        return self.layers[layer_idx]

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_output=None,
        enc_dec_attn_mask=None,
        inference_params=None,
    ):
        # hidden_states: [b, s, h]

        # Checks.
        if inference_params:
            assert (
                self.recompute_granularity is None
            ), "inference does not work with activation checkpointing"

        if not self.pre_process:
            # See set_input_tensor()
            assert isinstance(self.input_tensor, torch.Tensor) or isinstance(
                hidden_states, torch.Tensor
            )
            if hidden_states is None:
                hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        # RNG context.
        if self.sequence_parallel:
            raise NotImplementedError("Sequence parallelism not supported yet.")
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # Forward layers.
        with rng_context:
            forward_kwargs = {
                "encoder_output": encoder_output,
                "enc_dec_attn_mask": enc_dec_attn_mask,
                "inference_params": inference_params,
            }
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, **forward_kwargs
                )

        # Final layer norm.
        if self.post_process and not self.prenorm:
            hidden_states = self.final_norm(hidden_states)

        return hidden_states

    def load_state_dict(self, state_dict, strict=True):
        """Customize load."""

        # Handle renaming layernorm -> norm in component names
        state_dict_ = {}
        for key in state_dict.keys():
            newkey = key.replace("layernorm", "norm")
            state_dict_[newkey] = state_dict[key]

        super().load_state_dict(state_dict_, strict)


# =============================================================================
# Helper Fused Func
# =============================================================================


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: Optional[torch.Tensor], residual: torch.Tensor, prob: float
) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)
