"""Patch-local diffusion Transformer used by SemBridge flow matching."""

from __future__ import annotations

import math

import torch
from torch import nn

from .minicpm4 import MiniCPM4Config, MiniCPMModel


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("SinusoidalPosEmb requires an even dimension of at least four")
        self.dim = dim

    def forward(self, values: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        if values.ndim < 1:
            values = values.unsqueeze(0)
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, dtype=values.dtype, device=values.device)
            * (-math.log(10_000.0) / (half - 1))
        )
        embedding = scale * values.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class SemBridgeLocalDiT(nn.Module):
    def __init__(self, config: MiniCPM4Config, in_channels: int = 64) -> None:
        super().__init__()
        if config.vocab_size != 0:
            raise ValueError("local DiT MiniCPM4 config must use vocab_size=0")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config
        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, in_channels, bias=True)
        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        self.delta_time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        self.decoder = MiniCPMModel(config)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.in_proj(x.transpose(1, 2).contiguous())
        condition = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix_length = condition.size(1)
        time = self.time_mlp(self.time_embeddings(t).to(hidden.dtype))
        delta = self.delta_time_mlp(self.time_embeddings(dt).to(hidden.dtype))
        hidden = torch.cat(((mu + time + delta).unsqueeze(1), condition, hidden), dim=1)
        hidden, _ = self.decoder(hidden, is_causal=False)
        hidden = self.out_proj(hidden[:, prefix_length + 1 :, :])
        return hidden.transpose(1, 2).contiguous()
