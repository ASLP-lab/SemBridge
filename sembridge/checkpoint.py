"""Strict generator checkpoint loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .models import (
    GeneratorConfig,
    SemBridgeGenerator,
    generator_config_from_legacy,
)

IGNORED_CHECKPOINT_PREFIXES = (
    "audio_vae.",
    "residual_lm.",
    "fsq_layer.",
    "nitp_projector.",
    "speech_token_feature_projector.",
)


@dataclass(frozen=True)
class LoadReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        return not self.missing_keys and not self.unexpected_keys


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_state_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load model weights") from exc
        state = load_file(str(path), device="cpu")
    else:
        payload = _torch_load(path)
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            state = payload["model"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state = payload["state_dict"]
        else:
            state = payload
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError(f"{path} does not contain a tensor state dict")
    return state


def _normalize_state_dict(
    state: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    normalized: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for original_key, value in state.items():
        key = original_key
        for prefix in ("module.", "generator."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith(IGNORED_CHECKPOINT_PREFIXES):
            ignored.append(original_key)
            continue
        normalized[key] = value
    return normalized, tuple(sorted(ignored))


def _state_architecture(
    state: dict[str, torch.Tensor], raw_config: dict[str, Any]
) -> dict[str, Any]:
    """Read shape-bearing overrides from weights when legacy config.json is stale."""

    normalized, _ = _normalize_state_dict(state)

    def layer_count(prefix: str) -> int | None:
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
        indices = {
            int(match.group(1))
            for key in normalized
            if (match := pattern.match(key)) is not None
        }
        if not indices:
            return None
        expected = set(range(max(indices) + 1))
        if indices != expected:
            raise ValueError(f"checkpoint has non-contiguous layers under {prefix}")
        return len(indices)

    embedding = normalized.get("base_lm.embed_tokens.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("checkpoint lacks base_lm.embed_tokens.weight")
    semantic = normalized.get("speech_token_head.weight")
    if semantic is not None and semantic.ndim != 2:
        raise ValueError("speech_token_head.weight must be a matrix")
    lm = raw_config["lm_config"]
    rope = lm.get("rope_scaling") or {}
    factors = rope.get("long_factor") or rope.get("short_factor") or []
    inferred_head_dim = len(factors) * 2 if factors else None

    def transformer_shapes(prefix: str, fallback: dict[str, Any]) -> dict[str, int]:
        q = normalized.get(prefix + "0.self_attn.q_proj.weight")
        k = normalized.get(prefix + "0.self_attn.k_proj.weight")
        gate = normalized.get(prefix + "0.mlp.gate_proj.weight")
        if q is None or k is None or gate is None:
            raise ValueError(f"checkpoint lacks complete transformer weights under {prefix}")
        hidden = int(q.size(1))
        head_dim = int(
            fallback.get("kv_channels")
            or inferred_head_dim
            or hidden // int(fallback["num_heads"])
        )
        if q.size(0) % head_dim or k.size(0) % head_dim:
            raise ValueError(f"checkpoint attention shapes under {prefix} do not match head_dim")
        return {
            "hidden_size": hidden,
            "intermediate_size": int(gate.size(0)),
            "num_heads": int(q.size(0) // head_dim),
            "num_kv_heads": int(k.size(0) // head_dim),
            "kv_channels": head_dim,
        }

    encoder = raw_config["encoder_config"]
    dit = raw_config["dit_config"]
    lm_shapes = transformer_shapes(
        "base_lm.layers.",
        {"num_heads": lm["num_attention_heads"], "kv_channels": lm.get("kv_channels")},
    )
    encoder_shapes = transformer_shapes(
        "feat_encoder.encoder.layers.",
        {"num_heads": encoder["num_heads"], "kv_channels": encoder.get("kv_channels")},
    )
    dit_shapes = transformer_shapes(
        "feat_decoder.estimator.decoder.layers.",
        {"num_heads": dit["num_heads"], "kv_channels": dit.get("kv_channels")},
    )
    return {
        "text_vocab_size": int(embedding.size(0)),
        "semantic_head_enabled": semantic is not None,
        "semantic_vocab_size": int(semantic.size(0)) if semantic is not None else None,
        "lm_layers": layer_count("base_lm.layers."),
        "lm_hidden_size": lm_shapes["hidden_size"],
        "lm_intermediate_size": lm_shapes["intermediate_size"],
        "lm_num_heads": lm_shapes["num_heads"],
        "lm_num_kv_heads": lm_shapes["num_kv_heads"],
        "lm_kv_channels": lm_shapes["kv_channels"],
        "encoder_layers": layer_count("feat_encoder.encoder.layers."),
        "encoder_hidden_size": encoder_shapes["hidden_size"],
        "encoder_intermediate_size": encoder_shapes["intermediate_size"],
        "encoder_num_heads": encoder_shapes["num_heads"],
        "encoder_num_kv_heads": encoder_shapes["num_kv_heads"],
        "encoder_kv_channels": encoder_shapes["kv_channels"],
        "dit_layers": layer_count("feat_decoder.estimator.decoder.layers."),
        "dit_hidden_size": dit_shapes["hidden_size"],
        "dit_intermediate_size": dit_shapes["intermediate_size"],
        "dit_num_heads": dit_shapes["num_heads"],
        "dit_num_kv_heads": dit_shapes["num_kv_heads"],
        "dit_kv_channels": dit_shapes["kv_channels"],
    }


def load_model_state(
    model: SemBridgeGenerator,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> LoadReport:
    normalized, ignored = _normalize_state_dict(state)
    embedding_key = "base_lm.embed_tokens.weight"
    if embedding_key in normalized:
        checkpoint_embedding = normalized[embedding_key]
        current_embedding = model.state_dict()[embedding_key]
        if checkpoint_embedding.shape != current_embedding.shape:
            if (
                checkpoint_embedding.ndim == 2
                and current_embedding.ndim == 2
                and checkpoint_embedding.size(1) == current_embedding.size(1)
                and checkpoint_embedding.size(0) < current_embedding.size(0)
            ):
                extended = current_embedding.clone()
                extended[: checkpoint_embedding.size(0)] = checkpoint_embedding
                normalized[embedding_key] = extended
            else:
                raise ValueError(
                    f"incompatible text embedding shape {tuple(checkpoint_embedding.shape)} "
                    f"for model {tuple(current_embedding.shape)}"
                )
    incompatible = model.load_state_dict(normalized, strict=False)
    missing = tuple(sorted(incompatible.missing_keys))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    report = LoadReport(missing, unexpected, ignored)
    if strict and not report.is_exact:
        raise ValueError(
            "checkpoint is not an exact SemBridge generator match: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return report


def _legacy_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "sembridge_metadata.json"
    if not metadata_path.is_file():
        return {}
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sembridge_metadata.json must contain an object")
    return value


def load_generator_checkpoint(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    anchor_layer: int | None = None,
    strict: bool = True,
) -> tuple[SemBridgeGenerator, LoadReport, dict[str, Any]]:
    """Load either a portable training checkpoint or an archived generator directory."""

    path = Path(checkpoint)
    metadata: dict[str, Any]
    if path.is_file():
        payload = _torch_load(path)
        if not isinstance(payload, dict) or "model_config" not in payload or "model" not in payload:
            raise ValueError("portable checkpoint lacks model_config/model fields")
        metadata = dict(payload.get("metadata", {}))
        config = GeneratorConfig.model_validate(payload["model_config"])
        if anchor_layer is not None and config.semantic_anchor_layer != anchor_layer:
            raise ValueError(
                f"portable checkpoint anchor is {config.semantic_anchor_layer}, "
                f"not requested layer {anchor_layer}"
            )
        state = payload["model"]
    elif path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = _legacy_metadata(path)
        candidates = (path / "model.safetensors", path / "pytorch_model.bin")
        weights = next((candidate for candidate in candidates if candidate.is_file()), None)
        if weights is None:
            raise FileNotFoundError("expected model.safetensors or pytorch_model.bin")
        state = _load_state_file(weights)
        if raw.get("architecture") == "SemBridgeGenerator":
            config = GeneratorConfig.model_validate(raw)
            if anchor_layer is not None and config.semantic_anchor_layer != anchor_layer:
                raise ValueError(
                    f"checkpoint anchor is {config.semantic_anchor_layer}, "
                    f"not requested layer {anchor_layer}"
                )
        else:
            state_architecture = _state_architecture(state, raw)
            resolved_anchor = anchor_layer or metadata.get("semantic_anchor_layer")
            if state_architecture["semantic_head_enabled"] and resolved_anchor is None:
                raise ValueError(
                    "semantic anchor depth is not encoded in archived weights; provide "
                    "--anchor-layer or sembridge_metadata.json"
                )
            config = generator_config_from_legacy(
                raw,
                anchor_layer=resolved_anchor,
                speech_task_token_id=(
                    int(metadata["speech_task_token_id"])
                    if metadata.get("speech_task_token_id") is not None
                    else None
                ),
                singing_task_token_id=(
                    int(metadata["singing_task_token_id"])
                    if metadata.get("singing_task_token_id") is not None
                    else None
                ),
                state_text_vocab_size=state_architecture["text_vocab_size"],
                state_semantic_vocab_size=state_architecture["semantic_vocab_size"],
                state_semantic_head_enabled=state_architecture["semantic_head_enabled"],
                state_lm_layers=state_architecture["lm_layers"],
                state_lm_hidden_size=state_architecture["lm_hidden_size"],
                state_lm_intermediate_size=state_architecture["lm_intermediate_size"],
                state_lm_num_heads=state_architecture["lm_num_heads"],
                state_lm_num_kv_heads=state_architecture["lm_num_kv_heads"],
                state_lm_kv_channels=state_architecture["lm_kv_channels"],
                state_encoder_layers=state_architecture["encoder_layers"],
                state_encoder_hidden_size=state_architecture["encoder_hidden_size"],
                state_encoder_intermediate_size=state_architecture[
                    "encoder_intermediate_size"
                ],
                state_encoder_num_heads=state_architecture["encoder_num_heads"],
                state_encoder_num_kv_heads=state_architecture["encoder_num_kv_heads"],
                state_encoder_kv_channels=state_architecture["encoder_kv_channels"],
                state_dit_layers=state_architecture["dit_layers"],
                state_dit_hidden_size=state_architecture["dit_hidden_size"],
                state_dit_intermediate_size=state_architecture["dit_intermediate_size"],
                state_dit_num_heads=state_architecture["dit_num_heads"],
                state_dit_num_kv_heads=state_architecture["dit_num_kv_heads"],
                state_dit_kv_channels=state_architecture["dit_kv_channels"],
            )
            metadata = {**metadata, "state_derived_architecture": state_architecture}
    else:
        raise FileNotFoundError(path)

    model = SemBridgeGenerator(config).to(device=device, dtype=dtype)
    report = load_model_state(model, state, strict=strict)
    model.eval()
    return model, report, metadata
