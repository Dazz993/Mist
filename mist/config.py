from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.distributed

from mist.model.enums import MLPType, PositionEmbeddingType, AttnMaskType
from mist.utils.common import str_to_torch_dtype

MB = 1 << 20
GB = 1 << 30


def get_model_config(cfg: DictConfig) -> ModelConfig:
    factory_kwargs = {
        "max_position_embeddings": cfg.training.max_sequence_length,
        "params_dtype": cfg.training.params_dtype,
        **cfg.model,
    }
    model_config = ModelConfig(**factory_kwargs)
    return model_config


def get_hardware_config(cfg: DictConfig) -> HardwareConfig:
    hardware_config = HardwareConfig(
        gpu_type=cfg.hardware.gpu_type,
        num_nodes=cfg.hardware.num_nodes,
        num_gpus_per_node=cfg.hardware.num_gpus_per_node,
        gpu_gpu_comm_params=cfg.hardware.gpu_gpu_comm_params,
        cpu_gpu_comm_params=cfg.hardware.cpu_gpu_comm_params,
        gpu_cpu_comm_params=cfg.hardware.gpu_cpu_comm_params,
        interference_model_params=cfg.hardware.interference_model_params,
        nvlink=cfg.hardware.nvlink,
        memory_capacity=cfg.hardware.memory_capacity,
    )
    return hardware_config


def get_training_config(cfg: DictConfig) -> TrainingConfig:
    training_config = TrainingConfig(
        global_batch_size=cfg.training.global_batch_size,
        max_sequence_length=cfg.training.max_sequence_length,
        params_dtype=cfg.training.params_dtype,
        exec_dtype=cfg.training.exec_dtype,
        optimizer_dtype=cfg.training.optimizer_dtype,
        autocast_enabled=cfg.training.autocast_enabled,
        optimizer_name=cfg.training.optimizer_name,
    )
    return training_config


def get_tuning_config(cfg: DictConfig) -> TuningConfig:
    if getattr(cfg, "tuning", None) is None or not getattr(
        cfg.tuning, "enabled", False
    ):
        return None
    tuning_config = TuningConfig(
        tuning_granularity=cfg.tuning.tuning_granularity,
        zero_2_and_3_enabled=cfg.tuning.zero_2_and_3_enabled,
        pre_post_strategy=cfg.tuning.pre_post_strategy,
        pre_post_strategy_array=cfg.tuning.pre_post_strategy_array,
        activation_checkpointing_tuning_enabled=cfg.tuning.activation_checkpointing_tuning_enabled,
        state_offloading_enabled=cfg.tuning.state_offloading_enabled,
        activation_offloading_enabled=cfg.tuning.activation_offloading_enabled,
        sample_size=cfg.tuning.sample_size,
    )
    return tuning_config


def get_strategy_config(cfg: DictConfig) -> StrategyConfig:
    if getattr(cfg, "strategy", None) is None or not getattr(
        cfg.strategy, "enabled", False
    ):
        return None
    strategy_config = StrategyConfig(
        layer_partitions=cfg.strategy.layer_partitions,
        device_assignment=cfg.strategy.device_assignment,
        gradient_checkpointing=cfg.strategy.gradient_checkpointing,
        gradient_accumulation_steps=cfg.strategy.gradient_accumulation_steps,
        stage_strategies=cfg.strategy.stage_strategies,
        pre_post_strategy=cfg.strategy.pre_post_strategy,
        pre_post_strategy_array=cfg.strategy.pre_post_strategy_array,
    )
    return strategy_config


class ModelConfig:
    def __init__(
        self,
        name: str,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_heads_kv: Optional[int] = None,
        # Dtypes related
        params_dtype: Optional[str] = "float16",
        fp32_residual_connection: bool = False,
        fp16_lm_cross_entropy: bool = True,
        # Embedding related
        max_position_embeddings: Optional[int] = None,
        vocab_size: Optional[int] = None,
        num_tokentypes: int = 0,
        position_embedding_type: Optional[str] = "absolute",
        padding_idx: Optional[int] = None,
        rotary_emb_base: Optional[float] = 10000.0,
        rotary_emb_scale_base: Optional[float] = None,
        rotary_emb_fraction: Optional[float] = 0.0,
        rotary_emb_interleaved: Optional[bool] = False,
        tie_word_embeddings: bool = False,
        # Model structure related
        add_encoder: Optional[bool] = None,
        add_decoder: Optional[bool] = None,
        encoder_attn_mask_type: str = "padding",
        decoder_attn_mask_type: str = "padding",
        add_pooler: bool = False,
        num_experts: Optional[int] = None,
        parallel_block: bool = False,
        parallel_block_tied_norm: bool = False,
        prenorm: bool = True,
        normalization: str = "LayerNorm",
        activation_function: str = "gelu_new",
        mlp_type: str = "mlp",
        multiple_of: int = 256,
        # Bias related
        qkv_proj_bias: bool = False,
        out_proj_bias: bool = False,
        mlp_fc1_bias: bool = False,
        mlp_fc2_bias: bool = False,
        # Dropout related
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        # Other attributes
        normalization_eps: float = 1e-6,
        scale_attn_weights: bool = False,
        scale_attn_by_inverse_layer_idx: bool = False,
        # Optimization knobs
        # Efficient kernels
        use_flash_attn: bool = False,
        bias_dropout_fusion: bool = True,
        # fused_mlp: Optional[bool] = None,
        # fused_dense_sqrelu_dense: Optional[bool] = None,
        # fused_bias_fc: Optional[bool] = None,
        # fused_dropout_add_ln: Optional[bool] = None,
        # Parallelism
        tensor_parallel: Optional[bool] = None,
        sequence_parallel: Optional[bool] = None,
    ):
        # Basic
        self.name = name.lower()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_heads_kv = num_heads_kv
        # Dtypes related
        self.params_dtype = str_to_torch_dtype(params_dtype)
        self.fp32_residual_connection = fp32_residual_connection
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        # Embedding related
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.num_tokentypes = num_tokentypes
        self.position_embedding_type = position_embedding_type
        self.padding_idx = padding_idx
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_fraction = rotary_emb_fraction
        self.rotary_emb_interleaved = rotary_emb_interleaved
        self.tie_word_embeddings = tie_word_embeddings
        # Model structure related
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.num_experts = num_experts
        self.parallel_block = parallel_block
        self.parallel_block_tied_norm = parallel_block_tied_norm
        self.prenorm = prenorm
        self.normalization = normalization
        self.activation_function = activation_function
        self.mlp_type = mlp_type
        self.multiple_of = multiple_of
        # Bias related
        self.qkv_proj_bias = qkv_proj_bias
        self.out_proj_bias = out_proj_bias
        self.mlp_fc1_bias = mlp_fc1_bias
        self.mlp_fc2_bias = mlp_fc2_bias
        # Dropout related
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        # Other attributes
        self.normalization_eps = normalization_eps
        self.scale_attn_weights = scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        # Optimization knobs
        # Efficient kernels
        self.use_flash_attn = use_flash_attn
        self.bias_dropout_fusion = bias_dropout_fusion
        # self.fused_mlp = fused_mlp
        # self.fused_dense_sqrelu_dense = fused_dense_sqrelu_dense
        # self.fused_bias_fc = fused_bias_fc
        # self.fused_dropout_add_ln = fused_dropout_add_ln
        # Parallelism
        self.tensor_parallel = tensor_parallel
        self.sequence_parallel = sequence_parallel

        self._verify_and_update()

    def _verify_and_update(self):
        # Basic
        assert self.name is not None
        assert self.hidden_size is not None
        assert self.mlp_type in MLPType.__members__, (
            f"mlp_type should be one of {MLPType.__members__}. " f"Got {self.mlp_type}."
        )
        self.mlp_type = MLPType[self.mlp_type]
        # update the intermediate size according to the hidden size
        if self.intermediate_size is None:
            if self.mlp_type == MLPType.mlp:
                self.intermediate_size = 4 * self.hidden_size
            else:
                feature_size = int(self.hidden_size * 8 / 3)
                feature_size = (
                    (feature_size + self.multiple_of - 1)
                    // self.multiple_of
                    * self.multiple_of
                )
                self.intermediate_size = int(feature_size)
        assert self.num_hidden_layers is not None
        assert self.num_attention_heads is not None
        self.num_heads_kv = self.num_heads_kv or self.num_attention_heads
        # Embedding related
        assert self.max_position_embeddings is not None
        assert self.vocab_size is not None
        assert self.position_embedding_type in PositionEmbeddingType.__members__, (
            f"position_embedding_type should be one of {PositionEmbeddingType.__members__}. "
            f"Got {self.position_embedding_type}."
        )
        self.position_embedding_type = PositionEmbeddingType[
            self.position_embedding_type
        ]
        if self.position_embedding_type == PositionEmbeddingType.rotary:
            assert self.rotary_emb_fraction is not None
        # Model structure related
        assert self.add_encoder is not None
        assert self.add_decoder is not None
        if self.add_encoder:
            assert self.encoder_attn_mask_type in AttnMaskType.__members__, (
                f"encoder_attn_mask_type should be one of {AttnMaskType.__members__}. "
                f"Got {self.encoder_attn_mask_type}."
            )
            self.encoder_attn_mask_type = AttnMaskType[self.encoder_attn_mask_type]
        if self.add_decoder:
            assert self.decoder_attn_mask_type in AttnMaskType.__members__, (
                f"decoder_attn_mask_type should be one of {AttnMaskType.__members__}. "
                f"Got {self.decoder_attn_mask_type}."
            )
            self.decoder_attn_mask_type = AttnMaskType[self.decoder_attn_mask_type]
        assert self.normalization.lower() in ["layernorm", "rmsnorm"]
        # Optimization
        if self.sequence_parallel:
            assert (
                self.tensor_parallel
            ), "tensor_parallel should be True if sequence_parallel is True."


class HardwareConfig:
    def __init__(
        self,
        gpu_type: Optional[str] = None,
        num_nodes: Optional[int] = None,
        num_gpus_per_node: Optional[int] = None,
        gpu_gpu_comm_params: Optional[List[int]] = None,
        cpu_gpu_comm_params: Optional[List[int]] = None,
        gpu_cpu_comm_params: Optional[List[int]] = None,
        interference_model_params: Optional[List[int]] = None,
        nvlink: Optional[bool] = None,
        memory_capacity: Optional[float] = None,
    ):
        self.gpu_type = gpu_type
        self.num_nodes = num_nodes
        self.num_gpus_per_node = num_gpus_per_node
        self.gpu_gpu_comm_params = tuple(gpu_gpu_comm_params)
        self.cpu_gpu_comm_params = tuple(cpu_gpu_comm_params)
        self.gpu_cpu_comm_params = tuple(gpu_cpu_comm_params)
        self.interference_model_params = tuple(interference_model_params)
        self.nvlink = nvlink
        self.memory_capacity = memory_capacity * 0.88

        self._verify()

    def _verify(self):
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            assert (
                self.num_nodes * self.num_gpus_per_node == world_size
            ), f"Number of devices should be {world_size}. Got {self.num_nodes * self.num_gpus_per_node}."
        if self.gpu_gpu_comm_params is not None:
            assert isinstance(self.gpu_gpu_comm_params, (list, tuple))
            assert len(self.gpu_gpu_comm_params) == 32
        if self.cpu_gpu_comm_params is not None:
            assert isinstance(self.cpu_gpu_comm_params, tuple)
            assert len(self.cpu_gpu_comm_params) == 2
        if self.gpu_cpu_comm_params is not None:
            assert isinstance(self.gpu_cpu_comm_params, tuple)
            assert len(self.gpu_cpu_comm_params) == 2
        if self.interference_model_params is not None:
            assert isinstance(self.interference_model_params, tuple)
            assert len(self.interference_model_params) == 28


class TrainingConfig:
    def __init__(
        self,
        global_batch_size: int,
        max_sequence_length: int,
        params_dtype: str,
        exec_dtype: str,
        optimizer_dtype: str,
        autocast_enabled: bool,
        optimizer_name: str,
        deallocate_pipeline_outputs: bool = False,
    ):
        self.global_batch_size = global_batch_size
        self.max_sequence_length = max_sequence_length
        self._params_dtype = params_dtype
        self._exec_dtype = exec_dtype
        self._optimizer_dtype = optimizer_dtype
        self.autocast_enabled = autocast_enabled
        self.optimizer_name = optimizer_name
        self.deallocate_pipeline_outputs = deallocate_pipeline_outputs

        self._verify()

    @property
    def params_dtype(self):
        return str_to_torch_dtype(self._params_dtype)

    @property
    def exec_dtype(self):
        return str_to_torch_dtype(self._exec_dtype)

    @property
    def optimizer_dtype(self):
        return str_to_torch_dtype(self._optimizer_dtype)

    def _verify(self):
        # TODO(zhanda): Implement this
        pass


class TuningConfig:
    def __init__(
        self,
        tuning_granularity: str,
        sample_size: int,
        pre_post_strategy: Optional[str] = None,
        pre_post_strategy_array: Optional[List[str]] = None,
        zero_2_and_3_enabled: Optional[bool] = None,
        activation_checkpointing_tuning_enabled: Optional[bool] = None,
        state_offloading_enabled: Optional[bool] = None,
        activation_offloading_enabled: Optional[bool] = None,
    ):
        self.tuning_granularity = tuning_granularity
        self.sample_size = int(sample_size)
        self.pre_post_strategy = pre_post_strategy
        self.pre_post_strategy_array = pre_post_strategy_array or []
        self.zero_2_and_3_enabled = zero_2_and_3_enabled
        self.activation_checkpointing_tuning_enabled = (
            activation_checkpointing_tuning_enabled
        )
        self.state_offloading_enabled = state_offloading_enabled
        self.activation_offloading_enabled = activation_offloading_enabled

        self._verify()

    def _verify(self):
        assert self.tuning_granularity in [
            "no-pp",  # No pipeline parallelism
            "uniform-pp",  # Even pipeline parallelism
            "uniform-device-pp",  # Even pipeline parallelism with uniform device assignment
            "uniform-device-pp-mip",  # Even pipeline parallelism with uniform device assignment and MIP
            "inter-stage",  # Inter-stage pipeline parallelism (all)
            "uniform-pp-simple-heuristic-mem-opt", # Simple heuristic memory optimization (a single number for CKPT and OFFLOAD)
        ]
        assert isinstance(self.sample_size, int)
        assert self.pre_post_strategy in [
            "preset",
            "dp",
            "intra-node-tp-with-ore",
            "intra-node-tp-without-ore",
        ]

        if self.pre_post_strategy == "preset":
            assert self.pre_post_strategy_array is not None, (
                "If pre_post_strategy is preset, pre_post_strategy_array should be provided. "
                f"Got {self.pre_post_strategy_array}."
            )
            assert len(self.pre_post_strategy_array) == 10, (
                "If pre_post_strategy is preset, pre_post_strategy_array should be a list of 9 elements. "
                "[0] batch_size, [1] dp_size, [2] tp_size, "
                "[3] wre_enabled, [4] gre_enabled, [5] ore_enabled, "
                "[6] wo_ratio, [7] go_ratio, [8] oo_ratio [9] ac_ratio. \n"
                f"Got {self.pre_post_strategy_array}."
            )
            pre_post_strategy_array = self.pre_post_strategy_array
            pre_post_strategy_array[0] = int(pre_post_strategy_array[0])
            pre_post_strategy_array[1] = int(pre_post_strategy_array[1])
            pre_post_strategy_array[2] = int(pre_post_strategy_array[2])
            pre_post_strategy_array[3] = bool(pre_post_strategy_array[3])
            pre_post_strategy_array[4] = bool(pre_post_strategy_array[4])
            pre_post_strategy_array[5] = bool(pre_post_strategy_array[5])
            pre_post_strategy_array[6] = float(pre_post_strategy_array[6])
            pre_post_strategy_array[7] = float(pre_post_strategy_array[7])
            pre_post_strategy_array[8] = float(pre_post_strategy_array[8])
            pre_post_strategy_array[9] = float(pre_post_strategy_array[9])


class StrategyConfig:
    def __init__(
        self,
        layer_partitions: List[int],
        device_assignment: List[Tuple[int, int]],
        gradient_checkpointing: List[int],
        gradient_accumulation_steps: int,
        stage_strategies: List[List[float]],
        pre_post_strategy: Union[str, List[float]] = "heuristic",
        pre_post_strategy_array: Optional[List[str]] = None,
    ):
        self.num_stages = len(layer_partitions)
        self.layer_partitions = layer_partitions
        self.device_assignment = device_assignment
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.stage_strategies = stage_strategies
        self.pre_post_strategy = pre_post_strategy
        self.pre_post_strategy_array = pre_post_strategy_array

        self._verify_and_update()

    def _verify_and_update(self):
        assert (
            len(self.layer_partitions)
            == len(self.device_assignment)
            == len(self.stage_strategies)
            == len(self.gradient_checkpointing)
        ), (
            "The length of layer_partitions, device_assignment, stage_strategies, and gradient_checkpointing should be the same. "
            f"Got {len(self.layer_partitions)}, {len(self.device_assignment)}, {len(self.stage_strategies)}, and {len(self.gradient_checkpointing)}."
        )

        for device_assignment in self.device_assignment:
            assert len(device_assignment) == 2, (
                "Device assignment should be a list of 2 elements. "
                "[0] num_nodes, [1] num_gpus_per_node. \n"
                f"Got {device_assignment}."
            )

        for stage_strategy in self.stage_strategies:
            error_msg = (
                "Stage strategy should be a list of 9 elements. "
                "[0] batch_size, [1] dp_size, [2] tp_size, "
                "[3] wre_enabled, [4] gre_enabled, [5] ore_enabled, "
                "[6] wo_ratio, [7] go_ratio, [8] oo_ratio, [9] ac_ratio. \n"
                f"Got {stage_strategy}."
            )
            assert len(stage_strategy) == 10, error_msg
            stage_strategy[0] = int(stage_strategy[0])
            stage_strategy[1] = int(stage_strategy[1])
            stage_strategy[2] = int(stage_strategy[2])
            stage_strategy[3] = bool(stage_strategy[3])
            stage_strategy[4] = bool(stage_strategy[4])
            stage_strategy[5] = bool(stage_strategy[5])
            stage_strategy[6] = float(stage_strategy[6])
            stage_strategy[7] = float(stage_strategy[7])
            stage_strategy[8] = float(stage_strategy[8])
            stage_strategy[9] = float(stage_strategy[9])

        self.device_assignment = [tuple(d) for d in self.device_assignment]
        self.stage_strategies = [tuple(s) for s in self.stage_strategies]

        if self.pre_post_strategy == "preset":
            assert self.pre_post_strategy_array is not None, (
                "If pre_post_strategy is preset, pre_post_strategy_array should be provided. "
                f"Got {self.pre_post_strategy_array}."
            )
            assert len(self.pre_post_strategy_array) == 10, (
                "If pre_post_strategy is preset, pre_post_strategy_array should be a list of 9 elements. "
                "[0] batch_size, [1] dp_size, [2] tp_size, "
                "[3] wre_enabled, [4] gre_enabled, [5] ore_enabled, "
                "[6] wo_ratio, [7] go_ratio, [8] oo_ratio, [9] ac_ratio \n"
                f"Got {self.pre_post_strategy_array}."
            )
            pre_post_strategy_array = self.pre_post_strategy_array
            pre_post_strategy_array[0] = int(pre_post_strategy_array[0])
            pre_post_strategy_array[1] = int(pre_post_strategy_array[1])
            pre_post_strategy_array[2] = int(pre_post_strategy_array[2])
            pre_post_strategy_array[3] = bool(pre_post_strategy_array[3])
            pre_post_strategy_array[4] = bool(pre_post_strategy_array[4])
            pre_post_strategy_array[5] = bool(pre_post_strategy_array[5])
            pre_post_strategy_array[6] = float(pre_post_strategy_array[6])
            pre_post_strategy_array[7] = float(pre_post_strategy_array[7])
            pre_post_strategy_array[8] = float(pre_post_strategy_array[8])
            pre_post_strategy_array[9] = float(pre_post_strategy_array[9])

    def verify_with_other_config(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        hardware_config: HardwareConfig,
    ):
        num_layers = model_config.num_hidden_layers
        global_batch_size = training_config.global_batch_size
        num_devices = hardware_config.num_nodes * hardware_config.num_gpus_per_node

        # Verify layer_partitions
        assert (
            sum(self.layer_partitions) == num_layers
        ), f"Sum of layer_partitions should be {num_layers}. Got {sum(self.layer_partitions)}."

        # Verify device_assignment
        assert (
            sum([n * m for n, m in self.device_assignment]) == num_devices
        ), f"Sum of device_assignment should be {num_devices}. Got {sum([n * m for n, m in self.device_assignment])}."

        # Verify gradient_checkpointing
        assert all(
            ckpt <= layers_in_stage
            for ckpt, layers_in_stage in zip(
                self.gradient_checkpointing, self.layer_partitions
            )
        ), f"Number of gradient checkpointing layers should be less than or equal to the number of layers in the stage. Got {self.gradient_checkpointing} and {self.layer_partitions}."

        # Verify stage strategy
        for stage_idx, stage_strategy in enumerate(self.stage_strategies):
            # Verify global batch size
            assert (
                stage_strategy[0] * stage_strategy[1] * self.gradient_accumulation_steps
                == global_batch_size
            ), (
                f"Global batch size should be {global_batch_size}. "
                f"Got {stage_strategy[0] * stage_strategy[1] * self.gradient_accumulation_steps}. "
                "Check your global batch size, gradient accumulation steps, and strategy (batch size, and dp size)."
            )

            # Verify devices
            assert stage_strategy[1] * stage_strategy[2] == np.prod(
                self.device_assignment[stage_idx]
            ), (
                f"Number of devices should be {np.prod(self.device_assignment[stage_idx])}. "
                f"Got {stage_strategy[1] * stage_strategy[2]}."
            )

        if self.pre_post_strategy == "preset":
            pre_post_strategy_array = self.pre_post_strategy_array
            # Verify pre_post_strategy_array
            assert (
                pre_post_strategy_array[0]
                * pre_post_strategy_array[1]
                * self.gradient_accumulation_steps
                == global_batch_size
            ), (
                f"Global batch size should be {global_batch_size}. "
                f"Got {pre_post_strategy_array[0] * pre_post_strategy_array[1]}. "
                "Check your global batch size, and strategy (batch size, and dp size)."
            )
            assert pre_post_strategy_array[1] * pre_post_strategy_array[2] == np.prod(
                self.device_assignment[0]
            ), (
                f"Number of devices should be {np.prod(self.device_assignment[0])}. "
                f"Got {pre_post_strategy_array[1] * pre_post_strategy_array[2]}."
            )
            assert pre_post_strategy_array[1] * pre_post_strategy_array[2] == np.prod(
                self.device_assignment[-1]
            ), (
                f"Number of devices should be {np.prod(self.device_assignment[-1])}. "
                f"Got {pre_post_strategy_array[1] * pre_post_strategy_array[2]}."
            )


class MistConfig:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        hardware_config: HardwareConfig,
        tuning_config: TuningConfig,
        strategy_config: StrategyConfig,
        original_config: Optional[DictConfig] = None,
    ):
        self.model = model_config
        self.training = training_config
        self.hardware = hardware_config
        self.tuning = tuning_config
        self.strategy = strategy_config
        self._original_config = original_config

        # NCCL timeout
        if (
            original_config is not None
            and getattr(original_config, "nccl_timeout", None) is not None
        ):
            self.nccl_timeout = original_config.nccl_timeout
        else:
            self.nccl_timeout = 120
        self.nccl_timeout = 60

    # Override getattr
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original_config, name)

    @classmethod
    def from_dict_config(cls, cfg: DictConfig):
        OmegaConf.resolve(cfg)
        model_config = get_model_config(cfg)
        hardware_config = get_hardware_config(cfg)
        training_config = get_training_config(cfg)
        tuning_config = get_tuning_config(cfg)
        strategy_config = get_strategy_config(cfg)
        # Verify strategy config with other configs
        if strategy_config is not None:
            strategy_config.verify_with_other_config(
                model_config, training_config, hardware_config
            )
        return cls(
            model_config=model_config,
            training_config=training_config,
            hardware_config=hardware_config,
            tuning_config=tuning_config,
            strategy_config=strategy_config,
            original_config=cfg,
        )


@dataclass(unsafe_hash=True)
class OldMistConfig:
    """Configuration for Mist."""

    # Model and datasets
    model_name: str = "gpt2"
    dataset_name: str = "wikitext-2"
    num_layers: int = 44
    sequence_length: int = 2048

    # Hardware
    gpu_type: str = None
    num_nodes: int = 16
    num_gpus_per_node: int = 8
    # Bandwidth in GB/s
    inter_node_gpu_gpu_bandwidth: float = 600.0
    intra_node_gpu_gpu_bandwidth: float = 600.0
    gpu_gpu_bandwidth: float = 150.0
    gpu_cpu_bandwidth: float = 32.0
    nvlink: bool = True
    memory_capacity: float = 24.0 * 0.9

    # Training
    global_batch_size: int = 2048
    params_dtype: str = "float16"
    exec_dtype: str = "float16"
    optimizer_dtype: str = "float32"
    autocast_enabled: bool = False
    optimizer_name: str = "adamw"

    # Optimization level
    # * Pipeline parallelism
    layer_partitioning_tuning_enabled: bool = True
    num_stages_candidates_if_tuning_disabled: Tuple[int] = (4,)
    device_assignment_tuning_enabled: bool = False
    grad_accumu_steps_tuning_enabled: bool = True
    grad_accumu_steps_tuning_candidates_if_auto_tuning_disabled: Tuple[int] = (8,)

    pre_post_strategy: str = "heuristic"
    strategy_granularity: str = (
        "stage"  # "model", "stage", "layer", "micro_batch", "phase"
    )
    share_strategy_for_fwd_bwd: bool = True
    ckpt_tuning_enabled: bool = True
    offloading_enabled: bool = True
    redundancy_sharding_enabled: bool = True
    # Optimization Auxiliary
    # share_strategy_for_pre_layer_and_post_layer: bool = True
    pre_layer_ckpt_tuning_enabled: bool = False
    pre_layer_offloading_enabled: bool = False
    pre_layer_redundancy_sharding_enabled: bool = False
    pre_layer_share_strategy_for_fwd_bwd: bool = True
    post_layer_ckpt_tuning_enabled: bool = False
    post_layer_offloading_enabled: bool = False
    post_layer_redundancy_sharding_enabled: bool = False
    post_layer_share_strategy_for_fwd_bwd: bool = True
    batch_size_candidates: Tuple[int] = (1, 2, 3, 4, 6, 8, 12, 16)
    num_offload_intervals: int = 10
    # Pipeline parallelism
    pp_enabled: bool = True
    pp_tuning_enabled: bool = False
    num_stages_candidates: Tuple[int] = (1, 2, 4, 8, 16)

    # Debugging flags
    use_dummy_inputs: bool = False

    @classmethod
    def from_yaml(cls, file_path):
        # TODO(zhanda): Implement this, now it is just a dummy
        return cls()
