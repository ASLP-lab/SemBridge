"""High-level raw-text inference for the public SemBridge model directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_generator_checkpoint
from .hub import resolve_pretrained_path
from .infer import _patches_to_codec_latent, _write_pcm16
from .models import load_savae_pretrained
from .tokenizer import encode_segments, format_svs_notes, load_text_tokenizer


class SemBridge:
    """Unified TTS/SVS inference pipeline loaded from local files or the Hub."""

    def __init__(self, model_root: Path, *, device: str, dtype: torch.dtype) -> None:
        self.model_root = model_root
        self.device = torch.device(device)
        self.dtype = dtype
        self.model, report, self.metadata = load_generator_checkpoint(
            model_root, device=self.device, dtype=dtype, strict=True
        )
        if not report.is_exact:
            raise RuntimeError("public generator checkpoint did not load exactly")
        self.tokenizer = load_text_tokenizer(model_root)
        self.codec = load_savae_pretrained(
            model_root, device=self.device, strict=True
        )
        unified_config = model_root / "unified_savae_config.json"
        unified_weights = model_root / "unified_audiovae.safetensors"
        if unified_config.is_file() != unified_weights.is_file():
            raise FileNotFoundError(
                "unified codec requires both unified_savae_config.json and "
                "unified_audiovae.safetensors"
            )
        self.unified_codec = (
            load_savae_pretrained(
                model_root, device=self.device, strict=True, variant="unified"
            )
            if unified_config.is_file()
            else self.codec
        )
        self._validate_and_share_codec_encoder()
        if self.codec.latent_dim != self.model.config.feat_dim:
            raise ValueError("generator and codec latent dimensions differ")
        if len(self.tokenizer) != self.model.config.text_vocab_size or any(
            self.tokenizer.task_token_id(task) != self.model.config.task_token_id(task)
            for task in ("tts", "svs")
        ):
            raise ValueError("generator and tokenizer vocabulary/task IDs differ")
        generation_path = model_root / "generation_config.json"
        self.generation_config = (
            json.loads(generation_path.read_text(encoding="utf-8"))
            if generation_path.is_file()
            else {}
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype = "bfloat16",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "SemBridge":
        dtype_value = (
            {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype]
            if isinstance(dtype, str)
            else dtype
        )
        if str(device).startswith("cpu") and dtype_value != torch.float32:
            raise ValueError("CPU inference requires dtype='float32'")
        root = resolve_pretrained_path(
            path_or_repo_id, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(root, device=device, dtype=dtype_value)

    def _validate_and_share_codec_encoder(self) -> None:
        """Require one latent contract and one frozen encoder across both decoders."""

        if self.unified_codec is self.codec:
            return
        fields = ("sample_rate", "hop_length", "latent_dim", "use_psd")
        mismatched = {
            field: (getattr(self.codec, field), getattr(self.unified_codec, field))
            for field in fields
            if getattr(self.codec, field) != getattr(self.unified_codec, field)
        }
        if mismatched:
            raise ValueError(f"default/unified codec contract mismatch: {mismatched}")
        base_encoder = self.codec.encoder.state_dict()
        unified_encoder = self.unified_codec.encoder.state_dict()
        if base_encoder.keys() != unified_encoder.keys() or any(
            not torch.equal(base_encoder[key], unified_encoder[key])
            for key in base_encoder
        ):
            raise ValueError("default and unified codecs must contain the same encoder")
        self.unified_codec.encoder = self.codec.encoder
        if self.codec.fc_kappa is not None:
            base_kappa = self.codec.fc_kappa.state_dict()
            unified_kappa = self.unified_codec.fc_kappa.state_dict()
            if base_kappa.keys() != unified_kappa.keys() or any(
                not torch.equal(base_kappa[key], unified_kappa[key])
                for key in base_kappa
            ):
                raise ValueError("default and unified codecs must contain the same PSD head")
            self.unified_codec.fc_kappa = self.codec.fc_kappa

    def _codec_for_task(self, task: str, decoder: str = "auto"):
        normalized = task.lower()
        if normalized not in {"tts", "speech", "svs", "singing", "mixed", "unified"}:
            raise ValueError("task must be tts, svs, or unified")
        if decoder not in {"auto", "native", "bigvgan"}:
            raise ValueError("decoder must be auto, native, or bigvgan")
        if decoder != "auto":
            expected = "causal" if decoder == "native" else "bigvgan"
            for codec in (self.codec, self.unified_codec):
                if codec.config.decoder_type == expected:
                    return codec
            raise ValueError(f"checkpoint does not contain the requested {decoder} decoder")
        if normalized in {"tts", "speech"}:
            return self.codec
        return self.unified_codec

    def validation_summary(self, *, task: str = "tts", decoder: str = "auto") -> dict[str, Any]:
        """Report exact runtime loading, not author-release hashes or audio quality."""
        codec = self._codec_for_task(task, decoder)
        return {
            "checkpoint_exact": True,
            "validation": "runtime_load",
            "task": task,
            "decoder": decoder,
            "selected_decoder_type": codec.config.decoder_type,
            "selected_decoder_dim": codec.config.decoder_dim,
            "available_decoders": sorted({
                "native" if item.config.decoder_type == "causal" else "bigvgan"
                for item in (self.codec, self.unified_codec)
            }),
            "sample_rate": codec.sample_rate,
            "hop_length": codec.hop_length,
            "latent_dim": codec.latent_dim,
            "text_vocab_size": len(self.tokenizer),
            "lm_layers": self.model.config.num_hidden_layers,
        }

    def _load_prompt(
        self, path: str | Path | None, codec=None
    ) -> torch.Tensor | None:
        if path is None:
            return None
        codec = self.codec if codec is None else codec
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for prompt audio") from exc
        audio, sample_rate = torchaudio.load(str(path))
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != codec.sample_rate:
            audio = torchaudio.functional.resample(
                audio, sample_rate, codec.sample_rate
            )
        patch_samples = codec.hop_length * self.model.config.patch_size
        remainder = audio.size(-1) % patch_samples
        if remainder:
            audio = torch.nn.functional.pad(audio, (patch_samples - remainder, 0))
        with torch.inference_mode():
            latent = codec.encode(
                audio.to(self.device, dtype=torch.float32), codec.sample_rate
            )
        return codec.patch(latent).to(self.dtype)

    def _encode_target(
        self,
        *,
        task: str,
        text: str | None,
        notes: list[dict[str, Any]] | None,
        segments: list[dict[str, Any]] | None,
    ) -> tuple[list[int], list[int], str]:
        normalized = task.lower()
        if normalized in {"tts", "speech"}:
            if not text:
                raise ValueError("TTS inference requires text")
            ids = self.tokenizer.encode(text)
            task_ids = [self.tokenizer.task_token_id("speech")] * len(ids)
            return ids, task_ids, "tts"
        if normalized in {"svs", "singing"}:
            formatted = format_svs_notes(notes or [])
            ids = self.tokenizer.encode(formatted)
            task_ids = [self.tokenizer.task_token_id("singing")] * len(ids)
            return ids, task_ids, "svs"
        if normalized in {"mixed", "unified"}:
            if not segments:
                raise ValueError("unified inference requires speech/singing segments")
            ids, task_ids = encode_segments(self.tokenizer, segments)
            final_task = str(segments[-1].get("task", "")).lower()
            generation_task = "svs" if final_task in {"svs", "singing"} else "tts"
            return ids, task_ids, generation_task
        raise ValueError("task must be tts, svs, or unified")

    @torch.inference_mode()
    def generate(
        self,
        *,
        task: str,
        decoder: str = "auto",
        text: str | None = None,
        notes: list[dict[str, Any]] | None = None,
        segments: list[dict[str, Any]] | None = None,
        prompt_audio: str | Path | None = None,
        prompt_text: str | None = None,
        prompt_task: str | None = None,
        max_new_patches: int = 500,
        min_new_patches: int = 2,
        flow_steps: int | None = None,
        cfg_value: float | None = None,
        temperature: float | None = None,
        seed: int = 42,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(waveform, generated_patches)`` using auto/native/bigvgan decoding.

        Auto uses the default codec for TTS and the unified codec for SVS/mixed.
        Explicit selection never silently falls back to a different decoder.
        """

        if prompt_audio is None and prompt_text:
            raise ValueError("prompt_text requires prompt_audio")
        target_ids, target_task_ids, generation_task = self._encode_target(
            task=task, text=text, notes=notes, segments=segments
        )
        codec = self._codec_for_task(task, decoder)
        # Prompt encoding samples z. Seed it too so repeated requests and decoder
        # A/B runs use the same prompt context, not a fresh random latent.
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        prompt = self._load_prompt(prompt_audio, codec)
        prompt_ids = self.tokenizer.encode(prompt_text) if prompt_text else []
        if prompt is not None and not prompt_ids:
            raise ValueError("prompt_audio requires its matching prompt_text")
        if prompt_task is None:
            prompt_task = "singing" if task.lower() in {"svs", "singing"} else "speech"
        normalized_prompt_task = prompt_task.lower()
        if normalized_prompt_task in {"tts", "speech"}:
            prompt_task_name = "speech"
        elif normalized_prompt_task in {"svs", "singing"}:
            prompt_task_name = "singing"
        else:
            raise ValueError("prompt_task must be speech or singing")
        all_ids = prompt_ids + target_ids
        all_task_ids = (
            [self.tokenizer.task_token_id(prompt_task_name)] * len(prompt_ids)
            + target_task_ids
        )
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        patches = self.model.generate_patches(
            text_token_ids=torch.tensor(
                [all_ids], dtype=torch.long, device=self.device
            ),
            task=generation_task,
            prompt_patches=prompt,
            use_task_token=True,
            task_token_ids=torch.tensor(
                [all_task_ids], dtype=torch.long, device=self.device
            ),
            max_new_patches=max_new_patches,
            min_new_patches=min_new_patches,
            flow_steps=int(
                self.generation_config.get("flow_steps", 10)
                if flow_steps is None
                else flow_steps
            ),
            cfg_value=float(
                self.generation_config.get("cfg_value", 2.0)
                if cfg_value is None
                else cfg_value
            ),
            temperature=float(
                self.generation_config.get("temperature", 1.0)
                if temperature is None
                else temperature
            ),
            normalize_generated_patches=bool(codec.use_psd),
        )
        context_patches = min(2, prompt.size(1)) if prompt is not None else 0
        decode_patches = (
            torch.cat((prompt[:, -context_patches:], patches), dim=1)
            if context_patches
            else patches
        )
        waveform = codec.decode(
            _patches_to_codec_latent(decode_patches).float()
        )
        if context_patches:
            waveform = waveform[
                ..., context_patches * codec.hop_length * self.model.config.patch_size :
            ]
        return waveform.float().cpu(), patches.float().cpu()

    def generate_to_file(self, output: str | Path, **kwargs: Any) -> Path:
        destination = Path(output)
        if destination.suffix.lower() != ".wav":
            raise ValueError("raw SemBridge inference output must end in .wav")
        waveform, _ = self.generate(**kwargs)
        _write_pcm16(destination, waveform, self.codec.sample_rate)
        return destination
