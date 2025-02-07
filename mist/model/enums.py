import enum


class MLPType(enum.Enum):
    mlp = 1
    gated_mlp = 2


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2


class PositionEmbeddingType(enum.Enum):
    rotary = 1
    absolute = 2
