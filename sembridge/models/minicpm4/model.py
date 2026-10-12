"""MiniCPM4 decoder backbone used by SemBridge.

Parameter names are retained for SemBridge checkpoint compatibility.
Grouped-query attention is expanded explicitly for compatibility with
PyTorch releases without ``enable_gqa`` support.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import StaticKVCache
from .config import MiniCPM4Config


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    original_dtype = hidden.dtype
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = (hidden * torch.rsqrt(variance + eps)).to(original_dtype)
    return normalized * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = query.dtype
    query_float = query.float()
    key_float = key.float()
    query = query_float * cos + rotate_half(query_float) * sin
    key = key_float * cos + rotate_half(key_float) * sin
    return query.to(dtype), key.to(dtype)


def _repeat_kv(hidden: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden
    return hidden.repeat_interleave(groups, dim=1)


class MiniCPMLongRoPE(nn.Module):
    def __init__(self, config: MiniCPM4Config) -> None:
        super().__init__()
        self.dim = config.kv_channels or config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.short_factor = config.rope_scaling.short_factor
        self.long_factor = config.rope_scaling.long_factor
        self.original_max_position_embeddings = config.rope_scaling.original_max_position_embeddings
        expected = self.dim // 2
        if self.dim % 2 or len(self.short_factor) != expected or len(self.long_factor) != expected:
            raise ValueError(
                "RoPE factors must each contain head_dim / 2 values and head_dim must be even"
            )
        ratio = self.max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(
            1 + math.log(ratio) / math.log(self.original_max_position_embeddings)
        )
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(self.max_position_embeddings, inv_freq.device, torch.float32)

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        factors = self.long_factor if seq_len > self.original_max_position_embeddings else self.short_factor
        extension = torch.tensor(factors, device=device, dtype=torch.float32)
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        frequencies = torch.outer(positions, 1.0 / extension) * self.inv_freq.to(device=device)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.cos_cached = embedding.cos().to(dtype) * self.scaling_factor
        self.sin_cached = embedding.sin().to(dtype) * self.scaling_factor

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = int(position_ids.max().item()) + 1 if position_ids.numel() else 0
        if maximum > self.cos_cached.size(0):
            self._set_cos_sin_cache(maximum, position_ids.device, torch.float32)
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.kv_channels or config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        if self.num_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _project(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, query_length, _ = hidden_states.shape
        query, key, value = self._project(hidden_states)
        cos, sin = position_emb
        if cos.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        cached_key, cached_value = key, value
        key = _repeat_kv(key, self.num_key_value_groups).contiguous()
        value = _repeat_kv(value, self.num_key_value_groups).contiguous()
        output = F.scaled_dot_product_attention(
            query.contiguous(),
            key,
            value,
            attn_mask=attention_mask,
            is_causal=is_causal and attention_mask is None,
        )
        output = output.transpose(1, 2).reshape(
            batch, query_length, self.num_heads * self.head_dim
        )
        return self.o_proj(output), (cached_key, cached_value)

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor],
        position_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch = hidden_states.size(0)
        query, key, value = self._project(hidden_states.unsqueeze(1))
        cos, sin = position_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id : position_id + 1, :] = key
        value_cache[:, :, position_id : position_id + 1, :] = value
        expanded_key = _repeat_kv(key_cache, self.num_key_value_groups).contiguous()
        expanded_value = _repeat_kv(value_cache, self.num_key_value_groups).contiguous()
        valid = (
            torch.arange(key_cache.size(2), device=key_cache.device) <= position_id
        ).view(1, 1, 1, -1)
        output = F.scaled_dot_product_attention(
            query.contiguous(), expanded_key, expanded_value, attn_mask=valid
        )
        output = output.transpose(1, 2).reshape(batch, self.num_heads * self.head_dim)
        return self.o_proj(output)


class MiniCPMMLP(nn.Module):
    def __init__(self, config: MiniCPM4Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(inputs)) * self.up_proj(inputs))


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM4Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = MiniCPMAttention(config, layer_idx)
        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers
        self.use_mup = config.use_mup

    def _residual(self, residual: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if self.use_mup:
            hidden = hidden * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        return residual + hidden

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states, cache = self.self_attn(
            self.input_layernorm(hidden_states), position_emb, is_causal, attention_mask
        )
        hidden_states = self._residual(residual, hidden_states)
        residual = hidden_states
        hidden_states = self._residual(
            residual, self.mlp(self.post_attention_layernorm(hidden_states))
        )
        return hidden_states, cache

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: tuple[torch.Tensor, torch.Tensor],
        position_id: int,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn.forward_step(
            self.input_layernorm(hidden_states), position_emb, position_id, kv_cache
        )
        hidden_states = self._residual(residual, hidden_states)
        residual = hidden_states
        return self._residual(residual, self.mlp(self.post_attention_layernorm(hidden_states)))


class MiniCPMModel(nn.Module):
    def __init__(self, config: MiniCPM4Config) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens: nn.Module
        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Identity()
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = MiniCPMRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope_emb = MiniCPMLongRoPE(config)
        self.kv_cache: StaticKVCache | None = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        is_causal: bool = True,
        return_hidden_layer: int | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.size(1), device=inputs_embeds.device, dtype=torch.long
            )
        else:
            position_ids = position_ids.to(inputs_embeds.device, dtype=torch.long)
        if return_hidden_layer is not None and not 1 <= return_hidden_layer <= len(self.layers):
            raise ValueError(
                f"return_hidden_layer must be in [1, {len(self.layers)}], got {return_hidden_layer}"
            )
        position_emb = self.rope_emb(position_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device, dtype=torch.bool)
            length = inputs_embeds.size(1)
            if attention_mask.ndim == 2:
                if attention_mask.shape != (inputs_embeds.size(0), length):
                    raise ValueError("2-D attention_mask must have shape [batch, length]")
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.ndim != 4:
                raise ValueError("attention_mask must be [B,K], [B,Q,K], or [B,1,Q,K]")
            if attention_mask.size(-1) != length:
                raise ValueError("attention_mask key length differs from the input sequence")
            if attention_mask.size(-2) not in {1, length}:
                raise ValueError("attention_mask query length must be 1 or the input length")
            if is_causal:
                causal = torch.ones(
                    length, length, dtype=torch.bool, device=inputs_embeds.device
                ).tril()
                attention_mask = attention_mask & causal.view(1, 1, length, length)
        hidden_states = inputs_embeds
        caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        selected = None
        for layer_index, layer in enumerate(self.layers, start=1):
            hidden_states, cache = layer(
                hidden_states, position_emb, is_causal, attention_mask
            )
            caches.append(cache)
            if layer_index == return_hidden_layer:
                selected = hidden_states
        final = self.norm(hidden_states)
        if return_hidden_layer is not None:
            return final, caches, selected
        return final, caches

    def forward_step(self, inputs_embeds: torch.Tensor, position_id: int) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("KV cache is not set up")
        position_tensor = torch.tensor([position_id], device=inputs_embeds.device, dtype=torch.long)
        position_emb = self.rope_emb(position_tensor)
        hidden_states = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden_states = layer.forward_step(
                hidden_states,
                position_emb,
                position_id,
                self.kv_cache.get_layer_cache(index),
            )
        return self.norm(hidden_states)

    def setup_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.kv_cache = StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            dim_kv_head=(
                self.config.kv_channels
                or self.config.hidden_size // self.config.num_attention_heads
            ),
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
