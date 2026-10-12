"""Static key/value cache used by autoregressive MiniCPM4 inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class StaticKVCache:
    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        dim_kv_head: int,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
        max_length: int,
    ) -> None:
        self.max_length = max_length
        self.num_layers = num_layers
        self.kv_cache = torch.zeros(
            2,
            num_layers,
            batch_size,
            num_kv_heads,
            max_length,
            dim_kv_head,
            device=device,
            dtype=dtype,
        )
        self.current_length = 0

    def get_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")
        position = self.current_length
        self.current_length += 1
        return position

    def fill_caches(self, caches: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if len(caches) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer caches, received {len(caches)}")
        length = caches[0][0].size(2)
        if length > self.max_length:
            raise ValueError(f"prefix length {length} exceeds KV cache capacity {self.max_length}")
        self.current_length = length
        self.kv_cache.zero_()
        for index, (key, value) in enumerate(caches):
            self.kv_cache[0, index, :, :, :length, :] = key
            self.kv_cache[1, index, :, :, :length, :] = value
