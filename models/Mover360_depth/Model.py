"""
BaseModel: wraps Flux2Transformer2DModel using native Flux2 image conditioning.

Conditioning approach:
  - Every conditioning image is first encoded by the Flux2 VAE into native 32-channel latents
  - RGB/reference inputs use the standard 32 -> 128 patchify+BN projection
  - Mask guidance keeps the legacy 32 -> 128 path for compatibility
  - Depth-aware mask guidance concatenates mask and depth latents (32 + 32 = 64)
    before a learned 64 -> 128 patchify projection
  - Multiple conditioning images are concatenated to the noise latent stream as usual
  - After transformer, only noise tokens are extracted
"""

import warnings
warnings.filterwarnings("ignore", message="Ccross_attention_kwargs")

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput


class BaseModel(nn.Module):
    def __init__(
        self,
        flux_transformer,
        latent_bn_mean: torch.Tensor,
        latent_bn_std: torch.Tensor,
    ):
        super().__init__()
        self.flux_transformer = flux_transformer

        in_channels = flux_transformer.config.in_channels  # 128 for Flux2
        if in_channels % 4 != 0:
            raise ValueError(f"Flux2 in_channels must be divisible by 4, got {in_channels}")
        self.vae_latent_channels = in_channels // 4
        # Reproduce Flux2's native patchify+BN path on image latents exactly.
        self.input_image_proj = nn.Conv2d(
            self.vae_latent_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.mask_guidance_proj = nn.Conv2d(
            self.vae_latent_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        self.mask_depth_guidance_proj = nn.Conv2d(
            self.vae_latent_channels * 2,
            in_channels,
            kernel_size=2,
            stride=2,
            bias=True,
        )
        with torch.no_grad():
            self.input_image_proj.weight.zero_()
            self.input_image_proj.bias.zero_()
            self.mask_guidance_proj.weight.zero_()
            self.mask_guidance_proj.bias.zero_()
            self.mask_depth_guidance_proj.weight.zero_()
            self.mask_depth_guidance_proj.bias.zero_()

            latent_bn_mean = latent_bn_mean.detach().view(-1)
            latent_bn_std = latent_bn_std.detach().view(-1)
            patch_offsets = ((0, 0), (0, 1), (1, 0), (1, 1))

            expected_patchified_channels = self.vae_latent_channels * len(patch_offsets)
            if latent_bn_mean.numel() != expected_patchified_channels or latent_bn_std.numel() != expected_patchified_channels:
                raise ValueError(
                    "Unexpected Flux2 latent BN stats shape: "
                    f"mean={latent_bn_mean.numel()}, std={latent_bn_std.numel()}, "
                    f"expected={expected_patchified_channels}"
                )

            for latent_idx in range(self.vae_latent_channels):
                for patch_idx, (kh, kw) in enumerate(patch_offsets):
                    out_idx = latent_idx * 4 + patch_idx
                    scale = 1.0 / latent_bn_std[out_idx]
                    shift = -latent_bn_mean[out_idx] / latent_bn_std[out_idx]
                    self.input_image_proj.weight[out_idx, latent_idx, kh, kw] = scale
                    self.input_image_proj.bias[out_idx] = shift
                    self.mask_guidance_proj.weight[out_idx, latent_idx, kh, kw] = scale
                    self.mask_guidance_proj.bias[out_idx] = shift
                    # Keep the legacy mask path at initialization and learn the depth half from scratch.
                    self.mask_depth_guidance_proj.weight[out_idx, latent_idx, kh, kw] = scale
                    self.mask_depth_guidance_proj.bias[out_idx] = shift

        self.trainable_parameters = [
            (list(self.input_image_proj.parameters()), 1.0),
            (list(self.mask_guidance_proj.parameters()), 1.0),
            (list(self.mask_depth_guidance_proj.parameters()), 1.0),
        ]

    # ---- Flux2 latent packing/unpacking utilities ----

    @staticmethod
    def _pack_latents(latents):
        """Pack [B, C, H, W] → [B, H*W, C] for Flux2 transformer input."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _prepare_latent_ids(latents):
        """Generate 4D position IDs (T, H, W, L) for noise latent patches. Returns [B, H*W, 4]."""
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _prepare_image_ids(image_latents, scale=10):
        """
        Generate 4D position IDs (T, H, W, L) for conditioning image latents.
        Each image gets a unique T-coordinate: T = scale, 2*scale, 3*scale, ...
        This distinguishes conditioning images from noise patches (T=0) and from each other.
        """
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            if x.dim() == 4:
                x = x.squeeze(0)
            _, height, width = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)
        return image_latent_ids

    @staticmethod
    def _group_image_latents_by_sample(image_latents: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """
        Regroup legacy slot-major image latents into per-sample lists.

        Older code paths built the list as:
          [erp_0, erp_1, ..., erp_{N-1}, ref_0, ref_1, ..., ref_{N-1}, ...]
        This helper converts it back to:
          [[erp_0, ref_0, ...], [erp_1, ref_1, ...], ...]
        """
        num_samples = sum(1 for latent in image_latents if latent.shape[-1] != latent.shape[-2])
        if num_samples == 0:
            num_samples = len(image_latents)

        if num_samples == 0:
            return []

        if len(image_latents) % num_samples != 0:
            raise ValueError(
                f"Cannot evenly regroup {len(image_latents)} image latents into per-sample conditions; "
                f"detected {num_samples} ERP samples."
            )

        grouped_latents = [[] for _ in range(num_samples)]
        for latent_idx, latent in enumerate(image_latents):
            grouped_latents[latent_idx % num_samples].append(latent)

        num_images_per_sample = {len(latents) for latents in grouped_latents}
        if len(num_images_per_sample) != 1:
            raise ValueError(
                f"Inconsistent conditioning image count per sample: {sorted(num_images_per_sample)}"
            )

        return grouped_latents

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.FloatTensor],
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        input_img_latents: Optional[List[torch.Tensor]] = None,
        guidance: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        """
        Args:
            hidden_states: [B, C, H, W] patchified+BN-normalized noise latent (C=128)
            timestep: diffusion timestep (in [0, 1] range, will be scaled to [0, 1000] internally)
            encoder_hidden_states: [B, text_seq_len, joint_attention_dim] from Qwen3
            txt_ids: [B, text_seq_len, 4] text position IDs
            input_img_latents: per-sample conditioning image latents. Each sample is a list of
                `(kind, latent)` pairs ordered the same way as the prompt refers to them,
                where `kind` is one of `input`, `ref`, `mask_ref`, or `mask_depth_ref`.
            guidance: [B] guidance scale (None for distilled klein models)
            return_dict: whether to return dict or tuple
        """
        if hidden_states.dim() == 5:
            hidden_states = hidden_states.squeeze(1)
        batch_size, num_channels, h, w = hidden_states.shape
        num_noise_tokens = h * w

        expected_dtype = next(self.flux_transformer.parameters()).dtype
        hidden_states = hidden_states.to(dtype=expected_dtype)

        # ==== 1. Prepare noise latent IDs and pack ====
        img_ids = self._prepare_latent_ids(hidden_states).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        hidden_states = self._pack_latents(hidden_states)  # [B, H*W, 128]

        # ==== 2. Process and concatenate input image latents (native Flux2 approach) ====
        if input_img_latents and len(input_img_latents) > 0:
            if isinstance(input_img_latents[0], list):
                grouped_latents = input_img_latents
            else:
                grouped_latents = self._group_image_latents_by_sample(input_img_latents)

            grouped_projected = [
                [
                    self._project_condition_latent(kind, latent, expected_dtype)
                    for kind, latent in map(self._split_condition_latent, sample_latents)
                ]
                for sample_latents in grouped_latents
            ]

            num_condition_samples = len(grouped_projected)
            if num_condition_samples == 0:
                raise ValueError("Received image conditioning latents but failed to recover sample grouping.")
            if batch_size % num_condition_samples != 0:
                raise ValueError(
                    f"Batch size ({batch_size}) is not divisible by conditioning sample count "
                    f"({num_condition_samples})."
                )

            per_sample_image_tokens = []
            per_sample_image_ids = []
            for sample_latents in grouped_projected:
                packed_images = [self._pack_latents(latent).squeeze(0) for latent in sample_latents]
                sample_tokens = torch.cat(packed_images, dim=0)
                sample_ids = self._prepare_image_ids(sample_latents).squeeze(0)
                per_sample_image_tokens.append(sample_tokens)
                per_sample_image_ids.append(sample_ids)

            image_token_lengths = {tokens.shape[0] for tokens in per_sample_image_tokens}
            if len(image_token_lengths) != 1:
                raise ValueError(
                    f"All samples must have the same number of conditioning tokens, got "
                    f"{sorted(image_token_lengths)}."
                )

            all_image_tokens = torch.stack(per_sample_image_tokens, dim=0).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            image_ids = torch.stack(per_sample_image_ids, dim=0).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )

            cfg_copies = batch_size // num_condition_samples
            if cfg_copies > 1:
                all_image_tokens = all_image_tokens.repeat(cfg_copies, 1, 1)
                image_ids = image_ids.repeat(cfg_copies, 1, 1)

            # Concatenate to noise stream (native Flux2 image conditioning)
            hidden_states = torch.cat([hidden_states, all_image_tokens], dim=1)
            img_ids = torch.cat([img_ids, image_ids], dim=1)

        # ==== 3. Timestep + guidance embedding and modulation ====
        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.flux_transformer.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.flux_transformer.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.flux_transformer.double_stream_modulation_txt(temb)
        single_stream_mod = self.flux_transformer.single_stream_modulation(temb)

        # ==== 4. Input projections ====
        hidden_states = self.flux_transformer.x_embedder(hidden_states)  # [B, num_patches, inner_dim]
        encoder_hidden_states = self.flux_transformer.context_embedder(
            encoder_hidden_states.to(hidden_states.dtype)
        )  # [B, text_seq_len, inner_dim]

        num_txt_tokens = encoder_hidden_states.shape[1]

        # ==== 5. Position embeddings (RoPE) ====
        if txt_ids.ndim == 3:
            txt_ids_2d = txt_ids[0]
        else:
            txt_ids_2d = txt_ids
        if img_ids.ndim == 3:
            img_ids_2d = img_ids[0]
        else:
            img_ids_2d = img_ids

        image_rotary_emb = self.flux_transformer.pos_embed(img_ids_2d)
        text_rotary_emb = self.flux_transformer.pos_embed(txt_ids_2d)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # ==== 6. Joint attention transformer blocks (dual stream) ====
        for block in self.flux_transformer.transformer_blocks:
            if torch.is_grad_enabled() and self.flux_transformer.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self.flux_transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                )

        # ==== 7. Merge streams ====
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # ==== 8. Single-stream transformer blocks ====
        for block in self.flux_transformer.single_transformer_blocks:
            if torch.is_grad_enabled() and self.flux_transformer.gradient_checkpointing:
                hidden_states = self.flux_transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                )

        # ==== 9. Extract ONLY noise tokens (discard text + reference image tokens) ====
        # Token layout: [text_tokens | noise_tokens | reference_image_tokens]
        hidden_states = hidden_states[:, num_txt_tokens:num_txt_tokens + num_noise_tokens, ...]
        hidden_states = self.flux_transformer.norm_out(hidden_states, temb)
        output = self.flux_transformer.proj_out(hidden_states)
        # output: [B, num_noise_tokens, 128]

        # ==== 10. Reshape to image format ====
        output = output.permute(0, 2, 1).reshape(batch_size, -1, h, w)
        # output: [B, 128, H, W]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @staticmethod
    def _split_condition_latent(
        condition_latent: Union[torch.Tensor, Tuple[str, torch.Tensor]]
    ) -> Tuple[str, torch.Tensor]:
        if isinstance(condition_latent, tuple):
            if len(condition_latent) != 2:
                raise ValueError(
                    "Condition latent tuples must be `(kind, latent)`, "
                    f"got {len(condition_latent)} elements."
                )
            kind, latent = condition_latent
            return kind, latent

        # Backward compatibility with legacy callers that passed bare tensors only.
        return "input", condition_latent

    def _project_condition_latent(
        self,
        kind: str,
        latent: torch.Tensor,
        expected_dtype: torch.dtype,
    ) -> torch.Tensor:
        projected_dtype = expected_dtype
        if kind == "mask_depth_ref":
            projection = self.mask_depth_guidance_proj
            latent = latent.to(dtype=projection.weight.dtype)
            return projection(latent).to(dtype=projected_dtype)
        if kind == "mask_ref":
            projection = self.mask_guidance_proj
            latent = latent.to(dtype=projection.weight.dtype)
            return projection(latent).to(dtype=projected_dtype)
        if kind in {"input", "ref"}:
            projection = self.input_image_proj
            latent = latent.to(dtype=projection.weight.dtype)
            return projection(latent).to(dtype=projected_dtype)
        raise ValueError(f"Unsupported condition latent kind: {kind!r}")
