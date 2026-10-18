"""Autoregressive inference for continuous acoustic patches."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .cfm import UnifiedCFM
from .configuration import GeneratorConfig
from .local_dit import SemBridgeLocalDiT
from .local_encoder import SemBridgeLocalEncoder
from .minicpm4 import MiniCPMModel


class SemBridgeGenerator(nn.Module):
    """Generate acoustic patches with the released SemBridge weights."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size

        self.base_lm = MiniCPMModel(config.lm_config())
        self.feat_encoder = SemBridgeLocalEncoder(
            config.encoder_config(), input_dim=config.feat_dim
        )
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            estimator=SemBridgeLocalDiT(
                config.dit_config(), in_channels=config.feat_dim
            ),
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_hidden_size, config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.hidden_size, config.dit_hidden_size)
        self.stop_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.hidden_size, 2, bias=False)
        # Kept for checkpoint loading; generation does not use the semantic head.
        if config.semantic_head_enabled:
            self.speech_token_head: nn.Linear | None = nn.Linear(
                config.hidden_size, config.semantic_vocab_size
            )
        else:
            self.speech_token_head = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _embed_inputs(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_feats: torch.Tensor,
        audio_mask: torch.Tensor,
        task_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        feat_embed = self.enc_to_lm_proj(self.feat_encoder(audio_feats))
        scale = self.config.scale_emb if self.config.use_mup else 1.0
        text_embed = self.base_lm.embed_tokens(text_tokens) * scale
        if task_tokens is not None:
            if task_tokens.shape != text_tokens.shape:
                raise ValueError("task_tokens must align with every packed sequence position")
            text_embed = text_embed + self.base_lm.embed_tokens(task_tokens)
        return (
            text_mask.unsqueeze(-1) * text_embed
            + audio_mask.unsqueeze(-1) * feat_embed
        )

    def _build_inference_prefix(
        self,
        text_token_ids: torch.Tensor,
        prompt_patches: torch.Tensor | None,
        task: str,
        use_task_token: bool,
        task_token_id: int | None,
        task_token_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if text_token_ids.ndim != 2:
            raise ValueError("text_token_ids must be [batch, length]")
        batch, text_length = text_token_ids.shape
        if batch != 1:
            raise ValueError(
                "generation currently accepts one request at a time so per-sample stop "
                "lengths cannot be confused with padded patches"
            )
        if text_length == 0:
            raise ValueError("text_token_ids cannot be empty")
        if text_token_ids.min() < 0 or text_token_ids.max() >= self.config.text_vocab_size:
            raise ValueError("text token ID is outside the configured text vocabulary")
        boundary = torch.full(
            (batch, 1),
            self.config.audio_start_token_id,
            dtype=torch.long,
            device=text_token_ids.device,
        )
        packed_text = torch.cat((text_token_ids, boundary), dim=1)
        prefix_text_length = packed_text.size(1)
        if prompt_patches is None:
            prompt_patches = torch.zeros(
                batch,
                0,
                self.patch_size,
                self.feat_dim,
                device=text_token_ids.device,
                dtype=self.dtype,
            )
        if prompt_patches.ndim != 4 or tuple(prompt_patches.shape[-2:]) != (
            self.patch_size,
            self.feat_dim,
        ):
            raise ValueError(
                f"prompt_patches must be [B,T,{self.patch_size},{self.feat_dim}]"
            )
        prompt_patches = prompt_patches.to(self.device, dtype=self.dtype)
        prompt_length = prompt_patches.size(1)
        packed_text = torch.cat(
            (
                packed_text,
                torch.zeros(batch, prompt_length, dtype=torch.long, device=self.device),
            ),
            dim=1,
        )
        text_mask = torch.cat(
            (
                torch.ones(batch, prefix_text_length, device=self.device, dtype=self.dtype),
                torch.zeros(batch, prompt_length, device=self.device, dtype=self.dtype),
            ),
            dim=1,
        )
        audio_mask = 1.0 - text_mask
        audio_feats = torch.cat(
            (
                torch.zeros(
                    batch,
                    prefix_text_length,
                    self.patch_size,
                    self.feat_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
                prompt_patches,
            ),
            dim=1,
        )
        task_tokens = None
        if use_task_token:
            task_tokens = torch.zeros_like(packed_text)
            if task_token_id is not None and task_token_ids is not None:
                raise ValueError("provide scalar task_token_id or aligned task_token_ids, not both")
            if task_token_ids is not None:
                task_token_ids = task_token_ids.to(self.device, dtype=torch.long)
                if task_token_ids.shape != (batch, text_length):
                    raise ValueError("task_token_ids must align with all text positions")
                if task_token_ids.min() < 0 or task_token_ids.max() >= self.config.text_vocab_size:
                    raise ValueError("task token ID is outside the configured text vocabulary")
                task_tokens[:, :text_length] = task_token_ids
            else:
                resolved_task_id = (
                    self.config.task_token_id(task)
                    if task_token_id is None
                    else int(task_token_id)
                )
                if not 0 <= resolved_task_id < self.config.text_vocab_size:
                    raise ValueError("task token ID is outside the configured text vocabulary")
                task_tokens[:, :text_length] = resolved_task_id
        elif task_token_id is not None or task_token_ids is not None:
            raise ValueError("task token IDs require use_task_token=true")
        return packed_text, text_mask, audio_feats, audio_mask, task_tokens

    @torch.inference_mode()
    def generate_patches(
        self,
        *,
        text_token_ids: torch.Tensor,
        task: str,
        prompt_patches: torch.Tensor | None = None,
        use_task_token: bool | None = None,
        task_token_id: int | None = None,
        task_token_ids: torch.Tensor | None = None,
        max_new_patches: int = 50,
        min_new_patches: int = 2,
        flow_steps: int = 10,
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        stop_threshold: float = 0.5,
        use_cfg_zero_star: bool = True,
        normalize_generated_patches: bool = False,
    ) -> torch.Tensor:
        """Generate only continuous 2x64 patches; never call the semantic head."""

        if task not in {"tts", "svs"}:
            raise ValueError("task must be tts or svs")
        if max_new_patches < 1 or flow_steps < 2:
            raise ValueError("max_new_patches must be positive and flow_steps must be at least 2")
        if use_task_token is None:
            use_task_token = task == "svs" or task_token_id is not None or task_token_ids is not None
        text_token_ids = text_token_ids.to(self.device, dtype=torch.long)
        packed_text, text_mask, audio_feats, audio_mask, task_tokens = self._build_inference_prefix(
            text_token_ids,
            prompt_patches,
            task,
            use_task_token,
            task_token_id,
            task_token_ids,
        )
        prefix_length = packed_text.size(1)
        if prefix_length + max_new_patches > self.config.max_length:
            raise ValueError("prefix plus requested generation exceeds model max_length")
        combined = self._embed_inputs(
            packed_text, text_mask, audio_feats, audio_mask, task_tokens
        )
        self.base_lm.setup_cache(
            batch_size=packed_text.size(0),
            max_length=self.config.max_length,
            device=self.device,
            dtype=self.dtype,
        )
        encoded, caches = self.base_lm(inputs_embeds=combined, is_causal=True)
        assert self.base_lm.kv_cache is not None
        self.base_lm.kv_cache.fill_caches(caches)
        lm_hidden = encoded[:, -1, :].to(self.dtype)
        if audio_mask.sum() > 0:
            previous_patch = audio_feats[:, -1, :, :]
        else:
            previous_patch = torch.zeros(
                packed_text.size(0),
                self.patch_size,
                self.feat_dim,
                device=self.device,
                dtype=self.dtype,
            )
        generated: list[torch.Tensor] = []
        for index in range(max_new_patches):
            dit_hidden = self.lm_to_dit_proj(lm_hidden)
            patch = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=previous_patch.transpose(1, 2).contiguous(),
                n_timesteps=flow_steps,
                cfg_value=cfg_value,
                temperature=temperature,
                use_cfg_zero_star=use_cfg_zero_star,
            ).transpose(1, 2).contiguous()
            if normalize_generated_patches:
                patch = F.normalize(patch.float(), dim=-1).to(self.dtype) * self.feat_dim**0.5
            generated.append(patch)
            stop_probability = self.stop_head(
                self.stop_actn(self.stop_proj(lm_hidden))
            ).softmax(dim=-1)[:, 1]
            # Preserve the archived ``i > min_len`` stopping convention.
            if index > min_new_patches and bool(
                torch.all(stop_probability >= stop_threshold)
            ):
                break
            patch_embed = self.enc_to_lm_proj(self.feat_encoder(patch.unsqueeze(1)))[:, 0]
            position = self.base_lm.kv_cache.step()
            lm_hidden = self.base_lm.forward_step(patch_embed, position).to(self.dtype)
            previous_patch = patch
        return torch.stack(generated, dim=1)
