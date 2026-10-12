"""Configuration objects for the extracted MiniCPM4 backbone.

This dependency-minimal configuration preserves the fields required by the
archived SemBridge checkpoints.
"""

from __future__ import annotations

from pydantic import BaseModel


class RopeScalingConfig(BaseModel):
    type: str = "longrope"
    long_factor: list[float]
    short_factor: list[float]
    original_max_position_embeddings: int


class MiniCPM4Config(BaseModel):
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_scaling: RopeScalingConfig
    vocab_size: int
    use_mup: bool = True
    scale_emb: float = 1.0
    dim_model_base: int
    scale_depth: float = 1.0
    rope_theta: float = 10_000.0
    kv_channels: int | None = None
    speech_token_vocab_size: int | None = None
    use_speech_token: bool = False
    use_note_pitch: bool = False
