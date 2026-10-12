"""Typed model configuration and recipe/checkpoint adapters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, model_validator

from .cfm import CfmConfig
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
    semantic_anchor_layer: int | None = 32
    audio_start_token_id: int = 101
    audio_end_token_id: int = 102
    # Added-token IDs are tokenizer/checkpoint specific.  Do not silently use
    # the IDs from an older vocabulary for an archived checkpoint.
    speech_task_token_id: int | None = None
    singing_task_token_id: int | None = None
    cfm: CfmConfig = CfmConfig()
    dit_mean_mode: bool = False
    base_lm_layers: int | None = None
    merged_residual_layers: int = 0

    @model_validator(mode="after")
    def validate_shapes(self) -> "GeneratorConfig":
        if self.feat_dim != 64 or self.patch_size != 2:
            raise ValueError("paper path requires frame_dim=64 and patch_size=2")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("LM hidden size must be divisible by its attention heads")
        if self.encoder_hidden_size % self.encoder_num_heads:
            raise ValueError("encoder hidden size must be divisible by its attention heads")
        if self.dit_hidden_size % self.dit_num_heads:
            raise ValueError("DiT hidden size must be divisible by its attention heads")
        if self.semantic_head_enabled:
            if self.semantic_anchor_layer is None:
                raise ValueError("enabled semantic head requires an explicit anchor layer")
            if not 1 <= self.semantic_anchor_layer <= self.num_hidden_layers:
                raise ValueError("semantic anchor layer must be within the runtime LM depth")
        if self.dit_mean_mode:
            raise ValueError("dit_mean_mode is outside the released SemBridge paper path")
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
                f"{task} task-token ID is not encoded in this checkpoint; "
                "provide the reviewed tokenizer ID explicitly"
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
                raise ValueError("archived RoPE factors do not match the attention head size")
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


def generator_config_from_legacy(
    raw: dict[str, Any],
    *,
    anchor_layer: int | None = None,
    speech_task_token_id: int | None = None,
    singing_task_token_id: int | None = None,
    state_text_vocab_size: int | None = None,
    state_semantic_vocab_size: int | None = None,
    state_semantic_head_enabled: bool | None = None,
    state_lm_layers: int | None = None,
    state_lm_hidden_size: int | None = None,
    state_lm_intermediate_size: int | None = None,
    state_lm_num_heads: int | None = None,
    state_lm_num_kv_heads: int | None = None,
    state_lm_kv_channels: int | None = None,
    state_encoder_layers: int | None = None,
    state_encoder_hidden_size: int | None = None,
    state_encoder_intermediate_size: int | None = None,
    state_encoder_num_heads: int | None = None,
    state_encoder_num_kv_heads: int | None = None,
    state_encoder_kv_channels: int | None = None,
    state_dit_layers: int | None = None,
    state_dit_hidden_size: int | None = None,
    state_dit_intermediate_size: int | None = None,
    state_dit_num_heads: int | None = None,
    state_dit_num_kv_heads: int | None = None,
    state_dit_kv_channels: int | None = None,
) -> GeneratorConfig:
    """Resolve an archived generator ``config.json`` without mutating it."""

    source = deepcopy(raw)
    lm = source["lm_config"]
    encoder = source["encoder_config"]
    dit = source["dit_config"]
    residual = int(source.get("residual_lm_num_layers", 0))
    base_layers = int(lm["num_hidden_layers"])
    runtime_layers = state_lm_layers or (base_layers + residual)
    use_semantic = (
        bool(lm.get("use_speech_token", False))
        if state_semantic_head_enabled is None
        else state_semantic_head_enabled
    )
    resolved_anchor = anchor_layer if anchor_layer is not None else (runtime_layers if use_semantic else None)
    return GeneratorConfig(
        text_vocab_size=state_text_vocab_size or lm["vocab_size"],
        semantic_vocab_size=(
            state_semantic_vocab_size
            or lm.get("speech_token_vocab_size", 16_384)
        ),
        feat_dim=source.get("feat_dim", 64),
        patch_size=source.get("patch_size", 2),
        hidden_size=state_lm_hidden_size or lm["hidden_size"],
        intermediate_size=state_lm_intermediate_size or lm["intermediate_size"],
        num_hidden_layers=runtime_layers,
        num_attention_heads=state_lm_num_heads or lm["num_attention_heads"],
        num_key_value_heads=state_lm_num_kv_heads or lm["num_key_value_heads"],
        lm_kv_channels=state_lm_kv_channels or lm.get("kv_channels"),
        encoder_hidden_size=state_encoder_hidden_size or encoder["hidden_dim"],
        encoder_intermediate_size=(
            state_encoder_intermediate_size or encoder["ffn_dim"]
        ),
        encoder_num_layers=state_encoder_layers or encoder["num_layers"],
        encoder_num_heads=state_encoder_num_heads or encoder["num_heads"],
        encoder_num_key_value_heads=(
            state_encoder_num_kv_heads
            or encoder.get("num_key_value_heads", lm["num_key_value_heads"])
        ),
        encoder_kv_channels=state_encoder_kv_channels or encoder.get("kv_channels"),
        dit_hidden_size=state_dit_hidden_size or dit["hidden_dim"],
        dit_intermediate_size=state_dit_intermediate_size or dit["ffn_dim"],
        dit_num_layers=state_dit_layers or dit["num_layers"],
        dit_num_heads=state_dit_num_heads or dit["num_heads"],
        dit_num_key_value_heads=(
            state_dit_num_kv_heads
            or dit.get("num_key_value_heads", lm["num_key_value_heads"])
        ),
        dit_kv_channels=state_dit_kv_channels or dit.get("kv_channels"),
        max_length=source.get("max_length", lm["max_position_embeddings"]),
        max_position_embeddings=lm["max_position_embeddings"],
        rms_norm_eps=lm.get("rms_norm_eps", 1e-6),
        rope_theta=lm.get("rope_theta", 10_000.0),
        rope_scaling=lm.get("rope_scaling"),
        use_mup=lm.get("use_mup", True),
        scale_emb=lm.get("scale_emb", 1.0),
        dim_model_base=lm.get("dim_model_base", lm["hidden_size"]),
        scale_depth=lm.get("scale_depth", 1.0),
        semantic_head_enabled=use_semantic,
        semantic_anchor_layer=resolved_anchor,
        speech_task_token_id=speech_task_token_id,
        singing_task_token_id=singing_task_token_id,
        cfm=dit.get("cfm_config", {}),
        dit_mean_mode=source.get("dit_mean_mode", False),
        base_lm_layers=base_layers,
        merged_residual_layers=residual,
    )
