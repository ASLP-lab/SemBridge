"""Classifier-free guidance and Euler sampling for acoustic patches."""

from __future__ import annotations

import math

import torch

from .local_dit import SemBridgeLocalDiT


class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        estimator: SemBridgeLocalDiT,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        if n_timesteps < 1:
            raise ValueError("n_timesteps must be positive")
        noise = torch.randn(
            (mu.size(0), self.in_channels, patch_size),
            device=mu.device,
            dtype=mu.dtype,
        ) * temperature
        span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        span = span + sway_sampling_coef * (torch.cos(math.pi / 2 * span) - 1 + span)
        return self.solve_euler(noise, span, mu, cond, cfg_value, use_cfg_zero_star)

    @staticmethod
    def optimized_scale(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return (positive * negative).sum(1, keepdim=True) / (
            negative.square().sum(1, keepdim=True) + 1e-8
        )

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        current = t_span[0]
        delta = t_span[0] - t_span[1]
        zero_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_steps:
                velocity = torch.zeros_like(x)
            else:
                batch = x.size(0)
                x_in = torch.cat((x, x), dim=0)
                mu_in = torch.cat((mu, torch.zeros_like(mu)), dim=0)
                time_in = current.expand(2 * batch)
                dt_in = torch.zeros(
                    2 * batch, device=x.device, dtype=x.dtype
                )
                cond_in = torch.cat((cond, cond), dim=0)
                positive, negative = self.estimator(
                    x_in, mu_in, time_in, cond_in, dt_in
                ).chunk(2, dim=0)
                if use_cfg_zero_star:
                    scale = self.optimized_scale(
                        positive.reshape(batch, -1), negative.reshape(batch, -1)
                    ).view(batch, *([1] * (positive.ndim - 1)))
                else:
                    scale = 1.0
                velocity = negative * scale + cfg_value * (positive - negative * scale)
            x = x - delta * velocity
            current = current - delta
            if step < len(t_span) - 1:
                delta = current - t_span[step + 1]
        return x
