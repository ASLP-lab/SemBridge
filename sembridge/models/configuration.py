"""Configuration for the released SemBridge generator."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from .minicpm4 import MiniCPM4Config, RopeScalingConfig


class GeneratorConfig(BaseModel):
    """Resolved architecture for the continuous SemBridge generator."""

    architecture: str = "SemBridgeGenerator"
    text_vocab_size: int
    semantic_vocab_size: int = 16_384
    feat_dim: int = 64
    patch_size: int = 2
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    encoder_hidden_size: int = 1024
    encoder_intermediate_size: int = 4096
    encoder_num_layers: int = 4
    encoder_num_heads: int = 16
    encoder_num_key_value_heads: int = 16
    encoder_kv_channels: int | None = None
    dit_hidden_size: int = 1024
    dit_intermediate_size: int = 4096
    dit_num_layers: int = 4
    dit_num_heads: int = 16
    dit_num_key_value_heads: int = 16
    dit_kv_channels: int | None = None
    max_length: int = 4096
    max_position_embeddings: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    rope_scaling: RopeScalingConfig | None = None
    lm_kv_channels: int | None = None
    use_mup: bool = True
    scale_emb: float = 12.0
    dim_model_base: int = 256
    scale_depth: float = 1.4
    semantic_head_enabled: bool = True
    audio_start_token_id: int = 101
    audio_end_token_id: int = 102
    speech_task_token_id: int | None = None
    singing_task_token_id: int | None = None
    dit_mean_mode: bool = False

    @model_validator(mode="after")
    def validate_shapes(self) -> "GeneratorConfig":
        if self.feat_dim != 64 or self.patch_size != 2:
            raise ValueError("expected feat_dim=64 and patch_size=2")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("LM hidden size must be divisible by its attention heads")
        if self.encoder_hidden_size % self.encoder_num_heads:
            raise ValueError("encoder hidden size must be divisible by its attention heads")
        if self.dit_hidden_size % self.dit_num_heads:
            raise ValueError("DiT hidden size must be divisible by its attention heads")
        if self.dit_mean_mode:
            raise ValueError("mean-mode sampling is not supported")
        if self.text_vocab_size <= max(self.audio_start_token_id, self.audio_end_token_id):
            raise ValueError("text vocabulary does not contain the audio boundary token IDs")
        for name in ("speech_task_token_id", "singing_task_token_id"):
            value = getattr(self, name)
            if value is not None and not 0 <= value < self.text_vocab_size:
                raise ValueError(f"{name} is outside the configured text vocabulary")
        return self

    def task_token_id(self, task: str) -> int:
        if task == "tts":
            value = self.speech_task_token_id
        elif task == "svs":
            value = self.singing_task_token_id
        else:
            raise ValueError("task must be tts or svs")
        if value is None:
            raise ValueError(
                f"{task} task-token ID is missing from the model configuration"
            )
        return value

    def _rope(self, hidden_size: int, heads: int, kv_channels: int | None) -> RopeScalingConfig:
        head_dim = kv_channels or hidden_size // heads
        if self.rope_scaling is not None:
            expected = head_dim // 2
            if (
                len(self.rope_scaling.long_factor) != expected
                or len(self.rope_scaling.short_factor) != expected
            ):
                raise ValueError("RoPE factors do not match the attention head size")
            return self.rope_scaling.model_copy(deep=True)
        factors = [1.0] * (head_dim // 2)
        return RopeScalingConfig(
            type="longrope",
            long_factor=factors,
            short_factor=factors,
            original_max_position_embeddings=self.max_position_embeddings or self.max_length,
        )

    def lm_config(self) -> MiniCPM4Config:
        return MiniCPM4Config(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings or self.max_length,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            kv_channels=self.lm_kv_channels,
            rms_norm_eps=self.rms_norm_eps,
            rope_scaling=self._rope(
                self.hidden_size, self.num_attention_heads, self.lm_kv_channels
            ),
            vocab_size=self.text_vocab_size,
            use_mup=self.use_mup,
            scale_emb=self.scale_emb,
            dim_model_base=self.dim_model_base,
            scale_depth=self.scale_depth,
            rope_theta=self.rope_theta,
            speech_token_vocab_size=self.semantic_vocab_size,
            use_speech_token=self.semantic_head_enabled,
            use_note_pitch=True,
        )

    def encoder_config(self) -> MiniCPM4Config:
        return MiniCPM4Config(
            hidden_size=self.encoder_hidden_size,
            intermediate_size=self.encoder_intermediate_size,
            max_position_embeddings=self.max_position_embeddings or self.max_length,
            num_attention_heads=self.encoder_num_heads,
            num_hidden_layers=self.encoder_num_layers,
            num_key_value_heads=self.encoder_num_key_value_heads,
            kv_channels=self.encoder_kv_channels,
            rms_norm_eps=self.rms_norm_eps,
            rope_scaling=self._rope(
                self.encoder_hidden_size, self.encoder_num_heads, self.encoder_kv_channels
            ),
            vocab_size=0,
            use_mup=self.use_mup,
            scale_emb=self.scale_emb,
            dim_model_base=self.dim_model_base,
            scale_depth=self.scale_depth,
            rope_theta=self.rope_theta,
        )

    def dit_config(self) -> MiniCPM4Config:
        return MiniCPM4Config(
            hidden_size=self.dit_hidden_size,
            intermediate_size=self.dit_intermediate_size,
            max_position_embeddings=self.max_position_embeddings or self.max_length,
            num_attention_heads=self.dit_num_heads,
            num_hidden_layers=self.dit_num_layers,
            num_key_value_heads=self.dit_num_key_value_heads,
            kv_channels=self.dit_kv_channels,
            rms_norm_eps=self.rms_norm_eps,
            rope_scaling=self._rope(
                self.dit_hidden_size, self.dit_num_heads, self.dit_kv_channels
            ),
            vocab_size=0,
            use_mup=self.use_mup,
            scale_emb=self.scale_emb,
            dim_model_base=self.dim_model_base,
            scale_depth=self.scale_depth,
            rope_theta=self.rope_theta,
        )
