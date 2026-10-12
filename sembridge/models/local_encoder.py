"""SemBridge local encoder for 2x64-frame acoustic patches."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .minicpm4 import MiniCPM4Config, MiniCPMModel


class SemBridgeLocalEncoder(nn.Module):
    """Aggregate each acoustic patch without flattening its two latent frames."""

    def __init__(self, config: MiniCPM4Config, input_dim: int = 64) -> None:
        super().__init__()
        if config.vocab_size != 0:
            raise ValueError("local encoder MiniCPM4 config must use vocab_size=0")
        self.config = config
        self.special_token = nn.Parameter(torch.randn(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.encoder = MiniCPMModel(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("local encoder expects [batch, patches, patch_size, frame_dim]")
        batch, patches, _, _ = inputs.shape
        hidden = self.in_proj(inputs)
        special = self.special_token.expand(batch, patches, 1, -1)
        hidden = torch.cat((special, hidden), dim=2)
        hidden = rearrange(hidden, "b t p c -> (b t) p c")
        encoded, _ = self.encoder(hidden, is_causal=False)
        return rearrange(encoded[:, 0, :], "(b t) c -> b t c", b=batch)
