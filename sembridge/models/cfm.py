"""Conditional flow matching objective and Euler sampler used by SemBridge."""

from __future__ import annotations

import math

import torch
from pydantic import BaseModel
from torch.nn import functional as F

from .local_dit import SemBridgeLocalDiT


class CfmConfig(BaseModel):
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0


class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfm_params: CfmConfig,
        estimator: SemBridgeLocalDiT,
        mean_mode: bool = False,
    ) -> None:
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.reg_loss_type = cfm_params.reg_loss_type
        self.ratio_r_neq_t_range = cfm_params.ratio_r_neq_t_range
        self.noise_cond_prob_range = cfm_params.noise_cond_prob_range
        self.noise_cond_scale = cfm_params.noise_cond_scale
        self.in_channels = in_channels
        self.mean_mode = mean_mode
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
                dt_in = delta.expand(2 * batch) if self.mean_mode else torch.zeros(
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

    def sample_r_t(
        self,
        x: torch.Tensor,
        mu: float = -0.4,
        sigma: float = 1.0,
        ratio_r_neq_t: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        if self.t_scheduler == "log-norm":
            r = torch.sigmoid(torch.randn(batch, device=x.device, dtype=x.dtype) * sigma + mu)
            t = torch.sigmoid(torch.randn(batch, device=x.device, dtype=x.dtype) * sigma + mu)
        elif self.t_scheduler == "uniform":
            r = torch.rand(batch, device=x.device, dtype=x.dtype)
            t = torch.rand(batch, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"unsupported t_scheduler: {self.t_scheduler}")
        independent = torch.rand(batch, device=x.device) < ratio_r_neq_t
        lower, upper = torch.minimum(r, t), torch.maximum(r, t)
        return torch.where(independent, lower, t), torch.where(independent, upper, t)

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        progress: float = 0.0,
    ) -> torch.Tensor:
        if self.mean_mode:
            raise NotImplementedError(
                "mean-mode JVP training is outside the SemBridge paper path"
            )
        batch = x1.size(0)
        if self.training_cfg_rate > 0:
            keep = (torch.rand(batch, device=x1.device) > self.training_cfg_rate).to(mu.dtype)
            mu = mu * keep.unsqueeze(1)
        if cond is None:
            cond = torch.zeros_like(x1)
        noise_probability = self.noise_cond_prob_range[0] + progress * (
            self.noise_cond_prob_range[1] - self.noise_cond_prob_range[0]
        )
        noisy = (torch.rand(batch, device=x1.device) < noise_probability).to(x1.dtype)
        cond = cond + noisy.view(-1, 1, 1) * torch.randn_like(cond) * self.noise_cond_scale
        _, time = self.sample_r_t(x1)
        noise = torch.randn_like(x1)
        mixed = (1 - time.view(-1, 1, 1)) * x1 + time.view(-1, 1, 1) * noise
        target = noise - x1
        predicted = self.estimator(mixed, mu, time, cond, torch.zeros_like(time))
        losses = F.mse_loss(predicted, target.detach(), reduction="none").mean(dim=1)
        if tgt_mask is None:
            return losses.mean()
        mask = tgt_mask.squeeze(1).to(losses.dtype)
        return (losses * mask).sum() / mask.sum().clamp_min(1.0)
