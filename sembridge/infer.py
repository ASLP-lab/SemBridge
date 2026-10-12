"""Waveform output and acoustic-latent layout helpers."""

from pathlib import Path

import soundfile as sf
import torch


def _write_pcm16(path: Path, waveform: torch.Tensor, sample_rate: int = 44_100) -> None:
    samples = waveform.detach().float().cpu().clamp(-1, 1).flatten().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def _patches_to_codec_latent(patches: torch.Tensor) -> torch.Tensor:
    """Convert [B,T,2,64] patches to channel-first [B,64,2T] codec latents."""
    if patches.ndim != 4 or tuple(patches.shape[-2:]) != (2, 64):
        raise ValueError("generator patches must have shape [B,T,2,64]")
    return patches.reshape(patches.size(0), patches.size(1) * 2, 64).transpose(
        1, 2
    ).contiguous()
