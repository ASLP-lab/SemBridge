"""Semantic-acoustic VAE with native causal and BigVGAN waveform decoders."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from .bigvgan import BigVGANDecoder, BigVGANDecoderConfig


def _l2_norm(inputs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return inputs / inputs.norm(dim=-1, keepdim=True).clamp_min(eps)


class PowerSphericalDistribution:
    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-7) -> None:
        self.eps = eps
        self.mu = _l2_norm(mu, eps)
        self.kappa = kappa.clamp(1e-6, 1e4)
        self.dimension = self.mu.shape[-1]
        beta_value = 0.5 * (self.dimension - 1)
        self.alpha = self.kappa + beta_value
        self.beta = torch.as_tensor(
            beta_value, dtype=self.kappa.dtype, device=self.kappa.device
        ).expand_as(self.kappa)

    def rsample(self) -> torch.Tensor:
        scalar = (2.0 * Beta(self.alpha, self.beta).rsample() - 1.0).unsqueeze(-1)
        tangent = _l2_norm(
            torch.randn(
                *self.mu.shape[:-1],
                self.dimension - 1,
                device=self.mu.device,
                dtype=self.mu.dtype,
            ),
            self.eps,
        )
        sample = torch.cat(
            (scalar, torch.sqrt((1 - scalar.square()).clamp_min(0)) * tangent), dim=-1
        )
        first_axis = torch.zeros_like(self.mu)
        first_axis[..., 0] = 1.0
        reflection = _l2_norm(first_axis - self.mu, self.eps)
        result = sample - 2.0 * (sample * reflection).sum(-1, keepdim=True) * reflection
        parallel = (self.mu - first_axis).abs().sum(-1, keepdim=True) < 1e-6
        return torch.where(parallel, sample, result)


@torch.jit.script
def snake(inputs: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = inputs.shape
    flattened = inputs.reshape(shape[0], shape[1], -1)
    flattened = flattened + (alpha + 1e-9).reciprocal() * torch.sin(
        alpha * flattened
    ).square()
    return flattened.reshape(shape)


class Snake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return snake(inputs, self.alpha)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__padding = padding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(inputs, (self.__padding * 2, 0)))


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(
        self, *args, padding: int = 0, output_padding: int = 0, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = super().forward(inputs)
        trim = self.__padding * 2 - self.__output_padding
        return output[..., :-trim] if trim else output


def _causal_conv(*args, **kwargs) -> nn.Module:
    return weight_norm(CausalConv1d(*args, **kwargs))


def _causal_transpose_conv(*args, **kwargs) -> nn.Module:
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


class CausalResidualUnit(nn.Module):
    def __init__(
        self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1
    ) -> None:
        super().__init__()
        padding = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            _causal_conv(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=padding,
                groups=groups,
            ),
            Snake1d(dim),
            _causal_conv(dim, dim, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class CausalEncoderBlock(nn.Module):
    def __init__(
        self, output_dim: int = 16, input_dim: int | None = None, stride: int = 1, groups: int = 1
    ) -> None:
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            _causal_conv(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list[int] | None = None,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        strides = strides or [2, 4, 8, 8]
        blocks: list[nn.Module] = [_causal_conv(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            blocks.append(CausalEncoderBlock(d_model, stride=stride, groups=groups))
        self.fc_mu = _causal_conv(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = _causal_conv(d_model, latent_dim, kernel_size=3, padding=1)
        self.block = nn.Sequential(*blocks)
        self.enc_dim = d_model

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.block(inputs)
        return {"hidden_state": hidden, "mu": self.fc_mu(hidden), "logvar": self.fc_logvar(hidden)}


class NoiseBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = _causal_conv(dim, dim, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            inputs.size(0), 1, inputs.size(2), device=inputs.device, dtype=inputs.dtype
        )
        return inputs + noise * self.linear(inputs)


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        groups: int = 1,
        use_noise_block: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            Snake1d(input_dim),
            _causal_transpose_conv(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ) -> None:
        super().__init__()
        if depthwise:
            layers: list[nn.Module] = [
                _causal_conv(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                _causal_conv(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [_causal_conv(input_channel, channels, kernel_size=7, padding=3)]
        output_dim = channels
        for index, stride in enumerate(rates):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            groups = output_dim if depthwise else 1
            layers.append(
                CausalDecoderBlock(
                    input_dim, output_dim, stride, groups, use_noise_block
                )
            )
        layers.extend(
            [Snake1d(output_dim), _causal_conv(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class AudioVAEConfig(BaseModel):
    encoder_dim: int = 128
    encoder_rates: list[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_type: Literal["causal", "bigvgan"] = "causal"
    decoder_dim: int = 1536
    decoder_rates: list[int] = [8, 8, 5, 2]
    bigvgan: BigVGANDecoderConfig | None = None
    depthwise: bool = True
    sample_rate: int = 16_000
    use_noise_block: bool = False
    use_psd: bool = False
    is_finetuning: bool = False


class SemanticAcousticVAE(nn.Module):
    """Acoustic encoder, posterior, and decoder for continuous latent inference."""

    def __init__(self, config: AudioVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.sample_rate = config.sample_rate
        self.hop_length = math.prod(config.encoder_rates)
        self.chunk_size = self.hop_length
        self.use_psd = config.use_psd
        self.encoder = CausalEncoder(
            config.encoder_dim, config.latent_dim, config.encoder_rates, config.depthwise
        )
        if config.decoder_type == "bigvgan":
            if config.use_noise_block:
                raise ValueError("BigVGAN does not support the causal decoder noise block")
            self.decoder = BigVGANDecoder(
                config.latent_dim,
                config.decoder_dim,
                config.decoder_rates,
                config.bigvgan or BigVGANDecoderConfig(),
            )
        else:
            if config.bigvgan is not None:
                raise ValueError("AudioVAE.bigvgan requires decoder_type='bigvgan'")
            self.decoder = CausalDecoder(
                config.latent_dim,
                config.decoder_dim,
                config.decoder_rates,
                config.depthwise,
                use_noise_block=config.use_noise_block,
            )
        self.fc_kappa = (
            _causal_conv(self.encoder.enc_dim, 1, kernel_size=3, padding=1)
            if config.use_psd
            else None
        )
        if self.sample_rate != 44_100 or self.hop_length != 1764 or self.latent_dim != 64:
            raise ValueError("SemBridge SA-VAE must use 44.1 kHz, hop 1764, and 64-D latents")

    def encode(self, audio: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f"expected {self.sample_rate} Hz audio")
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        remainder = audio.size(-1) % self.hop_length
        if remainder:
            audio = F.pad(audio, (0, self.hop_length - remainder))
        encoded = self.encoder(audio)
        mean, logvar = encoded["mu"], encoded["logvar"]
        if self.use_psd:
            assert self.fc_kappa is not None
            # Preserve the archived adapter's normalization convention.
            direction = _l2_norm(mean)
            kappa = F.softplus(self.fc_kappa(encoded["hidden_state"])) + 1.0
            sample = PowerSphericalDistribution(
                direction.transpose(1, 2), kappa.squeeze(1)
            ).rsample()
            return sample.transpose(1, 2) * math.sqrt(self.latent_dim)
        if self.config.is_finetuning:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar.clamp(-12, 12))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError("latent must be [B,64,T] or [B,T,64]")
        if latent.size(1) != self.latent_dim and latent.size(2) == self.latent_dim:
            latent = latent.transpose(1, 2)
        return self.decoder(latent)

    def patch(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.size(1) == self.latent_dim:
            latent = latent.transpose(1, 2)
        usable = latent.size(1) - latent.size(1) % 2
        return latent[:, :usable].reshape(latent.size(0), usable // 2, 2, self.latent_dim)


_SAVAE_ARTIFACTS = {
    "default": ("savae_config.json", "audiovae.safetensors"),
    "unified": ("unified_savae_config.json", "unified_audiovae.safetensors"),
}


def _savae_artifact_paths(root: Path, variant: str) -> tuple[Path, Path]:
    try:
        config_name, weights_name = _SAVAE_ARTIFACTS[variant]
    except KeyError as exc:
        raise ValueError(f"unknown SA-VAE artifact variant: {variant!r}") from exc
    return root / config_name, root / weights_name


def save_savae_pretrained(
    model: SemanticAcousticVAE,
    destination: str | Path,
    *,
    variant: str = "default",
) -> Path:
    """Write the inference-only codec without pickle-based model payloads."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to export the SA-VAE") from exc
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    config_path, weights_path = _savae_artifact_paths(output, variant)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(weights_path))
    config_path.write_text(
        json.dumps(model.config.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_savae_pretrained(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    variant: str = "default",
) -> SemanticAcousticVAE:
    """Load a public inference-only SA-VAE directory."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load the SA-VAE") from exc
    root = Path(path)
    config_path, weights_path = _savae_artifact_paths(root, variant)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    config = AudioVAEConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    model = SemanticAcousticVAE(config).to(device)
    incompatible = model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise ValueError(
            "public SA-VAE checkpoint mismatch: "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model
