"""Pure-PyTorch BigVGAN decoder for SemBridge's fixed Code20 latents.

Adapted from NVIDIA BigVGAN and alias-free-torch. The public inference module
uses AMPBlock1 and excludes Hub downloads and custom CUDA activation kernels.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class BigVGANDecoderConfig(BaseModel):
    """BigVGAN architecture fields independent of the Code20 rate geometry."""

    model_config = ConfigDict(extra="forbid")

    resblock: Literal["1"] = "1"
    upsample_kernel_sizes: list[int] = Field(
        default_factory=lambda: [13, 13, 12, 5, 4]
    )
    resblock_kernel_sizes: list[int] = Field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = Field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    activation: Literal["snake", "snakebeta"] = "snakebeta"
    snake_logscale: bool = True
    use_bias_at_final: bool = False
    use_tanh_at_final: bool = False


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _init_conv(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        module.weight.data.normal_(mean, std)


class Snake(nn.Module):
    def __init__(self, channels: int, alpha_logscale: bool) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        initial = torch.zeros(channels) if alpha_logscale else torch.ones(channels)
        self.alpha = nn.Parameter(initial)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return inputs + torch.sin(alpha * inputs).square() / (alpha + 1e-9)


class SnakeBeta(nn.Module):
    def __init__(self, channels: int, alpha_logscale: bool) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        initial = torch.zeros(channels) if alpha_logscale else torch.ones(channels)
        self.alpha = nn.Parameter(initial.clone())
        self.beta = nn.Parameter(initial)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return inputs + torch.sin(alpha * inputs).square() / (beta + 1e-9)


def _kaiser_sinc_filter1d(
    cutoff: float,
    half_width: float,
    kernel_size: int,
) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (
            attenuation - 21
        )
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (
        torch.arange(-half_size, half_size) + 0.5
        if even
        else torch.arange(kernel_size) - half_size
    )
    values = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    values /= values.sum()
    return values.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer(
            "filter",
            _kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels = inputs.shape[1]
        inputs = F.pad(inputs, (self.pad, self.pad), mode="replicate")
        inputs = self.ratio * F.conv_transpose1d(
            inputs,
            self.filter.expand(channels, -1, -1),
            stride=self.ratio,
            groups=channels,
        )
        return inputs[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        even = kernel_size % 2 == 0
        self.ratio = ratio
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.register_buffer(
            "filter",
            _kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels = inputs.shape[1]
        inputs = F.pad(
            inputs, (self.pad_left, self.pad_right), mode="replicate"
        )
        return F.conv1d(
            inputs,
            self.filter.expand(channels, -1, -1),
            stride=self.ratio,
            groups=channels,
        )


class Activation1d(nn.Module):
    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.upsample = UpSample1d()
        self.activation = activation
        self.downsample = DownSample1d()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.activation(self.upsample(inputs)))


def _activation(name: str, channels: int, alpha_logscale: bool) -> Activation1d:
    if name == "snake":
        periodic = Snake(channels, alpha_logscale)
    elif name == "snakebeta":
        periodic = SnakeBeta(channels, alpha_logscale)
    else:
        raise ValueError(f"unsupported BigVGAN activation: {name!r}")
    return Activation1d(periodic)


class AMPBlock1(nn.Module):
    """BigVGAN anti-aliased multi-periodicity residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        activation: str,
        snake_logscale: bool,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=_padding(kernel_size, dilation),
                    )
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=_padding(kernel_size),
                    )
                )
                for _ in dilations
            ]
        )
        self.convs1.apply(_init_conv)
        self.convs2.apply(_init_conv)
        self.activations1 = nn.ModuleList(
            [_activation(activation, channels, snake_logscale) for _ in dilations]
        )
        self.activations2 = nn.ModuleList(
            [_activation(activation, channels, snake_logscale) for _ in dilations]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1,
            self.convs2,
            self.activations1,
            self.activations2,
        ):
            inputs = inputs + conv2(act2(conv1(act1(inputs))))
        return inputs


class BigVGANDecoder(nn.Module):
    """Non-causal decoder mapping ``[B, 64, T]`` to waveform samples."""

    def __init__(
        self,
        input_channels: int,
        initial_channels: int,
        upsample_rates: list[int],
        config: BigVGANDecoderConfig,
    ) -> None:
        super().__init__()
        kernels = list(config.upsample_kernel_sizes)
        if len(upsample_rates) != len(kernels):
            raise ValueError(
                "BigVGAN rates and kernels must have equal lengths: "
                f"{len(upsample_rates)} != {len(kernels)}"
            )
        if len(config.resblock_kernel_sizes) != len(
            config.resblock_dilation_sizes
        ):
            raise ValueError(
                "BigVGAN resblock kernels and dilation lists must have equal lengths"
            )
        if initial_channels % (2 ** len(upsample_rates)) != 0:
            raise ValueError(
                "BigVGAN initial channels must be divisible by 2 ** number of stages"
            )
        for rate, kernel in zip(upsample_rates, kernels):
            if kernel < rate or (kernel - rate) % 2:
                raise ValueError(
                    "BigVGAN kernel must preserve exact rate-scaled length: "
                    f"rate={rate}, kernel={kernel}"
                )

        self.upsample_rates = list(upsample_rates)
        self.config = config
        self.conv_pre = weight_norm(
            nn.Conv1d(input_channels, initial_channels, 7, padding=3)
        )
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for stage, (rate, kernel) in enumerate(zip(upsample_rates, kernels)):
            input_channels_for_stage = initial_channels // (2**stage)
            output_channels = initial_channels // (2 ** (stage + 1))
            upsample = weight_norm(
                nn.ConvTranspose1d(
                    input_channels_for_stage,
                    output_channels,
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                )
            )
            upsample.apply(_init_conv)
            self.ups.append(upsample)
            self.resblocks.append(
                nn.ModuleList(
                    [
                        AMPBlock1(
                            output_channels,
                            block_kernel,
                            list(dilations),
                            config.activation,
                            config.snake_logscale,
                        )
                        for block_kernel, dilations in zip(
                            config.resblock_kernel_sizes,
                            config.resblock_dilation_sizes,
                        )
                    ]
                )
            )

        final_channels = initial_channels // (2 ** len(upsample_rates))
        self.activation_post = _activation(
            config.activation,
            final_channels,
            config.snake_logscale,
        )
        self.conv_post = weight_norm(
            nn.Conv1d(
                final_channels,
                1,
                7,
                padding=3,
                bias=config.use_bias_at_final,
            )
        )
        self.conv_post.apply(_init_conv)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv_pre(inputs)
        for upsample, blocks in zip(self.ups, self.resblocks):
            inputs = upsample(inputs)
            combined = None
            for block in blocks:
                block_output = block(inputs)
                combined = (
                    block_output if combined is None else combined + block_output
                )
            inputs = combined / len(blocks)
        inputs = self.conv_post(self.activation_post(inputs))
        if self.config.use_tanh_at_final:
            return torch.tanh(inputs)
        return torch.clamp(inputs, min=-1.0, max=1.0)
