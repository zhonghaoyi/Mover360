from .PanoGenerator import PanoGenerator
import torch
import os
from PIL import Image, ImageDraw
from lightning.pytorch.utilities import rank_zero_only
from ..modules.utils import tensor_to_image
from .Model import BaseModel
from .EvalPanoGen import (
    ensure_mover360_test_metrics,
    update_mover360_test_metrics,
    finalize_mover360_test_metrics,
)
from einops import rearrange
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import inspect


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute mu for Flux2 dynamic shifting (from official Flux2KleinPipeline)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def point_map_peak_pixel(point_map_2d):
    """Return the strongest pixel from a precomputed point map."""
    point_map = torch.as_tensor(point_map_2d)
    if point_map.numel() == 0:
        return None

    max_value = point_map.max()
    if float(max_value) <= 0.0:
        return None

    _, width = point_map.shape
    peak_index = int(torch.argmax(point_map).item())
    y = peak_index // max(width, 1)
    x = peak_index % max(width, 1)
    return x, y


def draw_point_visualization(base_image, point_xy, core_radius_px=16, marker_radius=6):
    """Render the point location together with its fixed-size core region."""
    vis = np.asarray(base_image)
    if vis.ndim == 2:
        vis = np.repeat(vis[..., None], 3, axis=2)
    elif vis.ndim == 3 and vis.shape[2] == 1:
        vis = np.repeat(vis, 3, axis=2)

    vis = vis.astype(np.uint8)
    if point_xy is None:
        return Image.fromarray(vis).convert("RGB")

    x, y = point_xy
    height, width = vis.shape[:2]
    core_radius_px = max(int(core_radius_px), 0)

    if core_radius_px > 0:
        y1 = max(0, y - core_radius_px)
        y2 = min(height - 1, y + core_radius_px)
        x1 = x - core_radius_px
        x2 = x + core_radius_px
        segments = []
        if x1 < 0:
            segments.append((0, y1, x2, y2))
            segments.append((width + x1, y1, width - 1, y2))
        elif x2 >= width:
            segments.append((x1, y1, width - 1, y2))
            segments.append((0, y1, x2 - width, y2))
        else:
            segments.append((x1, y1, x2, y2))

        for sx1, sy1, sx2, sy2 in segments:
            sx1 = max(0, min(width - 1, sx1))
            sx2 = max(0, min(width - 1, sx2))
            if sx2 < sx1 or sy2 < sy1:
                continue
            region = vis[sy1:sy2 + 1, sx1:sx2 + 1].astype(np.float32)
            tint = np.array([255.0, 180.0, 40.0], dtype=np.float32)
            vis[sy1:sy2 + 1, sx1:sx2 + 1] = np.clip(region * 0.55 + tint * 0.45, 0, 255).astype(np.uint8)

    vis_image = Image.fromarray(vis).convert("RGB")
    marker_radius = max(int(marker_radius), 1)
    draw = ImageDraw.Draw(vis_image)

    if core_radius_px > 0:
        y1 = max(0, y - core_radius_px)
        y2 = min(height - 1, y + core_radius_px)
        x1 = x - core_radius_px
        x2 = x + core_radius_px
        segments = []
        if x1 < 0:
            segments.append((0, y1, x2, y2))
            segments.append((width + x1, y1, width - 1, y2))
        elif x2 >= width:
            segments.append((x1, y1, width - 1, y2))
            segments.append((0, y1, x2 - width, y2))
        else:
            segments.append((x1, y1, x2, y2))

        for sx1, sy1, sx2, sy2 in segments:
            sx1 = max(0, min(width - 1, sx1))
            sx2 = max(0, min(width - 1, sx2))
            if sx2 < sx1 or sy2 < sy1:
                continue
            draw.rectangle((sx1, sy1, sx2, sy2), outline=(255, 255, 255), width=2)

    draw.line((x - marker_radius - 2, y, x + marker_radius + 2, y), fill=(255, 255, 255), width=1)
    draw.line((x, y - marker_radius - 2, x, y + marker_radius + 2), fill=(255, 255, 255), width=1)
    draw.ellipse(
        (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
        fill=(255, 64, 64),
        outline=(255, 255, 255),
    )
    return vis_image


def _to_uint8_hwc_image(image):
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)

    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image.astype(np.uint8)


def compose_labeled_image_grid(
    panels,
    label_height=28,
    gap=8,
    background=(255, 255, 255),
    panels_per_row=3,
):
    if not panels:
        return None

    panel_images = []
    for title, image in panels:
        panel_image = Image.fromarray(_to_uint8_hwc_image(image)).convert("RGB")
        labeled_panel = Image.new(
            "RGB",
            (panel_image.width, panel_image.height + label_height),
            background,
        )
        labeled_panel.paste(panel_image, (0, label_height))
        draw = ImageDraw.Draw(labeled_panel)
        draw.text((8, 6), title, fill=(0, 0, 0))
        panel_images.append(labeled_panel)

    if panels_per_row is None or panels_per_row <= 0:
        panels_per_row = len(panel_images)

    rows = [
        panel_images[row_start:row_start + panels_per_row]
        for row_start in range(0, len(panel_images), panels_per_row)
    ]
    row_widths = [
        sum(panel_image.width for panel_image in row) + gap * max(len(row) - 1, 0)
        for row in rows
    ]
    row_heights = [
        max(panel_image.height for panel_image in row)
        for row in rows
    ]
    total_width = max(row_widths)
    total_height = sum(row_heights) + gap * max(len(rows) - 1, 0)
    canvas = Image.new("RGB", (total_width, total_height), background)

    offset_y = 0
    for row, row_height in zip(rows, row_heights):
        offset_x = 0
        for panel_image in row:
            canvas.paste(panel_image, (offset_x, offset_y))
            offset_x += panel_image.width + gap
        offset_y += row_height + gap

    return np.array(canvas)


class Mover360_depth(PanoGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._val_log_entries = []
        self._val_logged_samples = 0

        self.instantiate_model()

        if self.hparams.ckpt_path is not None:
            print(f"Loading checkpoint from {self.hparams.ckpt_path}")
            state_dict = self._load_checkpoint_state_dict(self.hparams.ckpt_path)
            self.convert_state_dict(state_dict)
            if self.hparams.load_trainable_ckpt_only:
                state_dict, trainable_keys = self._filter_checkpoint_trainable_weights(state_dict)
                self._load_checkpoint_weights(state_dict, expected_keys=trainable_keys)
            else:
                self._load_checkpoint_weights(state_dict, strict=True)
            print(f"Successfully loaded checkpoint weights")
        else:
            print("No checkpoint path provided, using random initialization")

    def instantiate_model(self):
        pano_flux = self.load_pano()
        latent_bn_mean = self.vae.bn.running_mean.detach().clone()
        latent_bn_std = torch.sqrt(
            self.vae.bn.running_var.detach().clone() + self.vae.config.batch_norm_eps
        )
        self.mv_base_model = BaseModel(pano_flux, latent_bn_mean, latent_bn_std)
        self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    @staticmethod
    def _load_checkpoint_state_dict(ckpt_path):
        load_kwargs = {
            "map_location": "cpu",
            "weights_only": True,
        }
        try:
            checkpoint = torch.load(ckpt_path, mmap=True, **load_kwargs)
        except TypeError:
            checkpoint = torch.load(ckpt_path, **load_kwargs)
        return checkpoint["state_dict"]

    def _filter_checkpoint_trainable_weights(self, state_dict):
        trainable_keys = {
            name
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in trainable_keys
        }
        dropped_count = len(state_dict) - len(filtered_state_dict)
        print(
            "Loading trainable checkpoint weights only: "
            f"{len(filtered_state_dict)} kept, {dropped_count} frozen/non-parameter keys skipped"
        )
        return filtered_state_dict, trainable_keys

    def _load_checkpoint_weights(self, state_dict, expected_keys=None, strict=False):
        strict_error = None
        if strict:
            try:
                self.load_state_dict(state_dict, strict=True)
                return
            except RuntimeError as exc:
                strict_error = exc

        current_state_dict = self.state_dict()
        filtered_state_dict = {}
        dropped_keys = []

        drop_input_proj = False
        proj_weight_key = "mv_base_model.input_image_proj.weight"
        if proj_weight_key in state_dict and proj_weight_key in current_state_dict:
            drop_input_proj = state_dict[proj_weight_key].shape != current_state_dict[proj_weight_key].shape

        for key, value in state_dict.items():
            if drop_input_proj and key.startswith("mv_base_model.input_image_proj."):
                dropped_keys.append(
                    f"{key}: {tuple(value.shape)} -> {tuple(current_state_dict[key].shape)}"
                )
                continue

            if key not in current_state_dict:
                continue

            if current_state_dict[key].shape != value.shape:
                dropped_keys.append(
                    f"{key}: {tuple(value.shape)} -> {tuple(current_state_dict[key].shape)}"
                )
                continue

            filtered_state_dict[key] = value

        incompatible = self.load_state_dict(filtered_state_dict, strict=False)
        if strict_error is not None:
            print(f"Strict checkpoint load failed, falling back to compatible keys only: {strict_error}")
        print(f"Loaded {len(filtered_state_dict)} compatible checkpoint tensors")
        if dropped_keys:
            preview = ", ".join(dropped_keys[:6])
            print(f"Dropped {len(dropped_keys)} incompatible checkpoint keys: {preview}")
        if expected_keys is not None:
            missing_keys = sorted(expected_keys - set(filtered_state_dict))
        else:
            missing_keys = incompatible.missing_keys
        if missing_keys:
            preview = ", ".join(missing_keys[:6])
            print(f"Missing trainable keys after partial load ({len(missing_keys)}): {preview}")
        if incompatible.unexpected_keys:
            preview = ", ".join(incompatible.unexpected_keys[:6])
            print(f"Unexpected keys after partial load ({len(incompatible.unexpected_keys)}): {preview}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # Flux2 latent shape in patchified space: [B, C*4, H/(vae_scale*2), W/(vae_scale*2)]
        # C=32 (Flux2 VAE latent channels), C*4=128 after patchification
        # Spatial: H/16, W/16 (vae_scale=8, patchify halves again)
        h = 2 * (int(height) // (self.vae_scale_factor * 2))
        w = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (
            batch_size,
            num_channels_latents * 4,
            h // 2,
            w // 2,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def sample_x0(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
            x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [torch.randn_like(img_start) for img_start in x1]
        else:
            x0 = torch.randn_like(x1)
        return x0

    def sample_timestep(self, x1):
        u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
        t = 1 / (1 + torch.exp(-u))
        t = t.to(x1[0])
        return t

    def _build_condition_image_groups(
        self,
        input_images,
        ref_images=None,
        mask_guidance_images=None,
        depth_guidance_images=None,
        use_ref=False,
        use_mask_guidance=False,
        sample_has_ref=None,
    ):
        condition_image_groups = []
        if ref_images is not None and ref_images.shape[0] != input_images.shape[0]:
            raise ValueError(
                f"reference image batch ({ref_images.shape[0]}) does not match input image batch "
                f"({input_images.shape[0]})"
            )
        if mask_guidance_images is not None and mask_guidance_images.shape[0] != input_images.shape[0]:
            raise ValueError(
                f"mask guidance batch ({mask_guidance_images.shape[0]}) does not match input image batch "
                f"({input_images.shape[0]})"
            )
        if depth_guidance_images is not None and depth_guidance_images.shape[0] != input_images.shape[0]:
            raise ValueError(
                f"depth guidance batch ({depth_guidance_images.shape[0]}) does not match input image batch "
                f"({input_images.shape[0]})"
            )
        normalized_sample_has_ref = None
        if sample_has_ref is not None and len(sample_has_ref) != input_images.shape[0]:
            raise ValueError(
                f"`sample_has_ref` length ({len(sample_has_ref)}) does not match input image batch "
                f"({input_images.shape[0]})"
            )
        elif sample_has_ref is not None:
            normalized_sample_has_ref = []
            for has_ref in sample_has_ref:
                if isinstance(has_ref, torch.Tensor):
                    normalized_sample_has_ref.append(bool(has_ref.reshape(-1)[0].item()))
                else:
                    normalized_sample_has_ref.append(bool(has_ref))

        ref_slot_is_active = use_ref and ref_images is not None
        if ref_slot_is_active and normalized_sample_has_ref is not None and not any(normalized_sample_has_ref):
            raise ValueError(
                "Reference conditioning was enabled for the batch, but `sample_has_ref` reports no valid "
                "reference samples."
            )

        for i in range(input_images.shape[0]):
            sample_images = [("input", input_images[i:i+1])]
            sample_uses_ref = ref_slot_is_active
            if sample_uses_ref:
                # Training/inference may mix samples with and without a real reference image
                # in the same batch. Upstream code already fills missing refs with zero-valued
                # placeholders, and we still append a ref slot for every sample so the
                # prompt image numbering stays aligned with the conditioning token layout.
                sample_images.append(("ref", ref_images[i:i+1]))
            if use_mask_guidance and mask_guidance_images is not None:
                if depth_guidance_images is not None:
                    sample_images.append(
                        ("mask_depth_ref", (mask_guidance_images[i:i+1], depth_guidance_images[i:i+1]))
                    )
                else:
                    sample_images.append(("mask_ref", mask_guidance_images[i:i+1]))
            condition_image_groups.append(sample_images)
        return condition_image_groups

    @staticmethod
    def _parse_inference_target_guidance_modes(mode_value):
        # `all` is a multi-pass save mode, not a single fused guidance tensor.
        if isinstance(mode_value, (list, tuple)):
            raw_modes = [str(mode).strip().lower() for mode in mode_value]
        else:
            normalized = str(mode_value).strip().lower()
            if normalized == "all":
                raw_modes = ["mask", "bbox", "point"]
            else:
                for separator in ("+", ",", "|", "/"):
                    normalized = normalized.replace(separator, " ")
                raw_modes = [token for token in normalized.split() if token]

        expanded_modes = []
        for mode in raw_modes:
            if mode == "all":
                expanded_modes.extend(["mask", "bbox", "point"])
            else:
                expanded_modes.append(mode)

        valid_modes = {"both", "bbox", "point", "mask"}
        parsed_modes = []
        for mode in expanded_modes:
            if mode not in valid_modes:
                raise ValueError(
                    "inference_target_guidance_mode must be one of {'all', 'both', 'bbox', 'point', 'mask'} "
                    f"or a separator-joined combination of single modes, got {mode_value!r}"
                )
            if mode not in parsed_modes:
                parsed_modes.append(mode)

        if not parsed_modes:
            raise ValueError("inference_target_guidance_mode must contain at least one valid mode")
        return parsed_modes

    @staticmethod
    def _select_primary_inference_guidance_mode(parsed_modes):
        if not parsed_modes:
            raise ValueError("parsed_modes must contain at least one guidance mode")
        return parsed_modes[0]

    @staticmethod
    def _append_path_suffix(path, suffix):
        if not suffix:
            return path
        stem, ext = os.path.splitext(path)
        return f"{stem}_{suffix}{ext}"

    @staticmethod
    def _normalize_inference_source_mask_type(batch):
        source_mask_type = batch.get("inference_source_mask_type", "bbox")
        if isinstance(source_mask_type, (list, tuple)):
            source_mask_type = source_mask_type[0] if source_mask_type else "bbox"
        normalized = str(source_mask_type or "bbox").strip().lower()
        if normalized in {"bbox", "box"}:
            return "bbox"
        if normalized in {"full", "mask", "fine"}:
            return "full"
        raise ValueError(
            "inference_source_mask_type must be one of {'bbox', 'full'}, "
            f"got {source_mask_type!r}"
        )

    @staticmethod
    def _replace_source_mask_channel(base_mask, source_mask):
        if base_mask is None or source_mask is None:
            return base_mask
        if base_mask.ndim == 5 and source_mask.ndim == 5 and base_mask.shape[2] > 1 and source_mask.shape[2] > 1:
            base_mask = base_mask.clone()
            base_mask[:, :, 1:2] = source_mask[:, :, 1:2]
            return base_mask
        if base_mask.ndim == 4 and source_mask.ndim == 4 and base_mask.shape[1] > 1 and source_mask.shape[1] > 1:
            base_mask = base_mask.clone()
            base_mask[:, 1:2] = source_mask[:, 1:2]
            return base_mask
        return base_mask

    @staticmethod
    def _select_inference_guidance_mask_batch(batch, guidance_mode):
        if guidance_mode == "mask":
            selected_mask = batch.get("pano_full_mask", batch.get("pano_mask"))
            if Mover360_depth._normalize_inference_source_mask_type(batch) == "bbox":
                selected_mask = Mover360_depth._replace_source_mask_channel(selected_mask, batch.get("pano_mask"))
            return selected_mask
        return batch.get("pano_mask")

    def _resolve_inference_target_guidance_probs(self, mode=None):
        if mode is None:
            parsed_modes = self._parse_inference_target_guidance_modes(
                self.hparams.inference_target_guidance_mode
            )
            mode = self._select_primary_inference_guidance_mode(parsed_modes)

        if mode == "both":
            return 1.0, 1.0
        if mode in {"bbox", "mask"}:
            return 1.0, 0.0
        if mode == "point":
            return 0.0, 1.0
        raise ValueError(
            "inference_target_guidance_mode must be one of {'both', 'bbox', 'point', 'mask'}, "
            f"got {mode!r}"
        )

    def _resolve_training_target_guidance_probs(self):
        bbox_prob = float(self.hparams.target_bbox_guidance_use_prob)
        point_prob = float(self.hparams.point_guidance_use_prob)
        if bbox_prob < 0.0 or point_prob < 0.0:
            raise ValueError(
                "target_bbox_guidance_use_prob and point_guidance_use_prob must be non-negative, "
                f"got {bbox_prob} and {point_prob}"
            )

        total = bbox_prob + point_prob
        if total <= 0.0:
            raise ValueError(
                "target_bbox_guidance_use_prob + point_guidance_use_prob must be > 0 "
                "for exclusive target guidance sampling"
            )

        return bbox_prob / total, point_prob / total

    @staticmethod
    def _expand_sample_functions(functions, repeat_count):
        expanded = []
        for function_name in functions:
            expanded.extend([str(function_name).lower()] * repeat_count)
        return expanded

    @staticmethod
    def _expand_sample_flags(flags, repeat_count):
        expanded = []
        for flag in flags:
            expanded.extend([bool(flag)] * repeat_count)
        return expanded

    @staticmethod
    def _sample_has_valid_ref(batch, index):
        has_ref = batch.get('has_ref')
        if has_ref is None:
            return True

        sample_flag = has_ref[index]
        if isinstance(sample_flag, torch.Tensor):
            return bool(sample_flag.reshape(-1)[0].item())
        return bool(sample_flag)

    @staticmethod
    def _prompt_text_at(prompts, index, fallback=None):
        if isinstance(prompts, str):
            return prompts
        if prompts is not None and index < len(prompts):
            return prompts[index]
        return fallback

    def _resolve_inference_prompt_info(self, batch, guidance_mode=None, guidance_mask_batch=None):
        if guidance_mask_batch is None:
            guidance_mask_batch = self._select_inference_guidance_mask_batch(batch, guidance_mode)

        refs_in_batch = 'refs' in batch and len(batch['refs']) > 0
        sample_has_ref = []
        for i in range(len(batch['function'])):
            has_ref = (
                batch['function'][i] == 'add'
                and refs_in_batch
                and self._sample_has_valid_ref(batch, i)
            )
            sample_has_ref.append(has_ref)

        use_img_cfg = True
        use_ref = any(sample_has_ref) and self.hparams.use_ref_in_inference
        use_mask_guidance = guidance_mask_batch is not None and self.hparams.use_mask_in_inference

        prompt_inputs = self.get_flux2_pano_prompts(
            batch,
            use_img_cfg=use_img_cfg,
            use_ref=use_ref,
            use_mask_guidance=use_mask_guidance,
            sample_has_ref=sample_has_ref,
        )
        processed_data = self.multimodal_processor(
            prompt_inputs,
            use_img_cfg=use_img_cfg,
            mode="val",
        )
        return {
            "prompt_inputs": prompt_inputs,
            "prompts": processed_data["prompts"],
            "sample_has_ref": sample_has_ref,
            "use_img_cfg": use_img_cfg,
            "use_ref": use_ref,
            "use_mask_guidance": use_mask_guidance,
            "guidance_mode": guidance_mode,
        }

    def _build_mask_guidance_images(
        self,
        pano_mask,
        functions=None,
        bbox_use_prob=1.0,
        point_use_prob=1.0,
        mutually_exclusive_target_guidance=False,
    ):
        if pano_mask is None:
            return None

        if pano_mask.ndim != 4 or pano_mask.shape[1] < 3:
            raise ValueError(
                f"`pano_mask` must be [B, >=3, H, W] with a precomputed target point channel, "
                f"got {tuple(pano_mask.shape)}"
            )
        target_mask = pano_mask[:, 0:1]
        source_mask = pano_mask[:, 1:2]
        target_point_map = pano_mask[:, 2:3].to(dtype=target_mask.dtype).clamp(0, 1)

        bbox_use_prob = float(bbox_use_prob)
        bbox_use_prob = min(max(bbox_use_prob, 0.0), 1.0)
        point_use_prob = float(point_use_prob)
        point_use_prob = min(max(point_use_prob, 0.0), 1.0)

        batch_size = target_mask.shape[0]
        selector_shape = (batch_size, 1, 1, 1)
        bbox_available = target_mask.amax(dim=(1, 2, 3), keepdim=True) > 0
        point_available = target_point_map.amax(dim=(1, 2, 3), keepdim=True) > 0

        target_available = bbox_available | point_available

        if mutually_exclusive_target_guidance:
            random_choice = torch.rand(selector_shape, device=target_mask.device)
            bbox_selector = (random_choice < bbox_use_prob) & bbox_available
            point_selector = (random_choice >= bbox_use_prob) & point_available

            missing_target = (~bbox_selector) & (~point_selector) & target_available
            bbox_selector = bbox_selector | (missing_target & bbox_available)
            point_selector = point_selector | (missing_target & (~bbox_available) & point_available)
        else:
            if bbox_use_prob >= 1.0:
                bbox_selector = torch.ones(selector_shape, device=target_mask.device, dtype=torch.bool)
            elif bbox_use_prob <= 0.0:
                bbox_selector = torch.zeros(selector_shape, device=target_mask.device, dtype=torch.bool)
            else:
                bbox_selector = torch.rand(selector_shape, device=target_mask.device) < bbox_use_prob

            if point_use_prob >= 1.0:
                point_selector = torch.ones(selector_shape, device=target_mask.device, dtype=torch.bool)
            elif point_use_prob <= 0.0:
                point_selector = torch.zeros(selector_shape, device=target_mask.device, dtype=torch.bool)
            else:
                point_selector = torch.rand(selector_shape, device=target_mask.device) < point_use_prob

            bbox_selector = bbox_selector & bbox_available
            point_selector = point_selector & point_available

            missing_target = (~bbox_selector) & (~point_selector) & target_available
            bbox_fallback = missing_target & bbox_available
            point_fallback = missing_target & (~bbox_available) & point_available
            bbox_selector = bbox_selector | bbox_fallback
            point_selector = point_selector | point_fallback

        target_bbox_channel = torch.where(bbox_selector, target_mask, torch.zeros_like(target_mask))
        target_point_channel = torch.where(point_selector, target_point_map, torch.zeros_like(target_point_map))

        if functions is not None:
            if len(functions) != batch_size:
                raise ValueError(
                    f"`functions` must have length {batch_size} when building mask guidance, got {len(functions)}"
                )
            normalized_functions = [str(function_name).lower() for function_name in functions]
            add_selector = torch.tensor(
                [function_name == 'add' for function_name in normalized_functions],
                device=target_mask.device,
                dtype=torch.bool,
            ).view(batch_size, 1, 1, 1)
            remove_selector = torch.tensor(
                [function_name == 'remove' for function_name in normalized_functions],
                device=target_mask.device,
                dtype=torch.bool,
            ).view(batch_size, 1, 1, 1)
            source_mask = torch.where(add_selector, torch.zeros_like(source_mask), source_mask)
            target_bbox_channel = torch.where(
                remove_selector, torch.zeros_like(target_bbox_channel), target_bbox_channel
            )
            target_point_channel = torch.where(
                remove_selector, torch.zeros_like(target_point_channel), target_point_channel
            )

        mask_guidance = torch.cat([source_mask, target_bbox_channel, target_point_channel], dim=1)
        mask_guidance = mask_guidance.clamp(0, 1)
        return mask_guidance * 2 - 1

    def _encode_condition_images(self, condition_image_groups, device):
        """
        Encode condition images in Flux2's multi-reference order.

        `condition_image_groups` is a per-sample list ordered exactly like the prompt
        refers to them, e.g. image 1 = input panorama, image 2 = reference image.
        Panorama-shaped inputs receive circular padding before VAE encoding and are
        unpadded afterwards. Native reference images keep their original square path.
        The condition kind is preserved so BaseModel can choose the correct projection
        path for natural images vs. mask guidance. Depth-aware mask guidance encodes
        instruction-map and depth panoramas separately, then concatenates the two
        32-channel VAE latents into a single 64-channel conditioning latent.
        """
        flat_entries = []
        for sample_images in condition_image_groups:
            for kind, image in sample_images:
                if kind == "mask_depth_ref":
                    mask_image, depth_image = image
                    flat_entries.append(("mask_depth_ref", "mask", self.pad_pano(mask_image)))
                    flat_entries.append(("mask_depth_ref", "depth", self.pad_pano(depth_image)))
                else:
                    flat_entries.append(
                        (kind, None, self.pad_pano(image) if kind in {"input", "mask_ref"} else image)
                    )

        input_img_latents = self.encode_image_latents(
            [entry[2] for entry in flat_entries],
            device=device,
        )

        grouped_input_img_latents = []
        flat_idx = 0
        for sample_images in condition_image_groups:
            sample_latents = []
            for kind, _ in sample_images:
                if kind == "mask_depth_ref":
                    mask_latent = self.unpad_pano(input_img_latents[flat_idx], latent=True)
                    depth_latent = self.unpad_pano(input_img_latents[flat_idx + 1], latent=True)
                    sample_latents.append((kind, torch.cat([mask_latent, depth_latent], dim=1)))
                    flat_idx += 2
                else:
                    latent = input_img_latents[flat_idx]
                    if kind in {"input", "mask_ref"}:
                        latent = self.unpad_pano(latent, latent=True)
                    sample_latents.append((kind, latent))
                    flat_idx += 1

            grouped_input_img_latents.append(sample_latents)

        return grouped_input_img_latents

    def training_step(self, batch, batch_idx):
        device = batch['pano'].device
        dtype = batch['pano'].dtype
        b = batch['pano'].shape[0]

        height = batch['height'][0]
        width = batch['width'][0]

        input_images_list = []
        target_images_list = []
        ref_images_list = []
        pano_mask_list = []
        refs_in_batch = 'refs' in batch and len(batch['refs']) > 0

        for i in range(b):
            sample_ref = batch['refs'][i] if refs_in_batch and self._sample_has_valid_ref(batch, i) else None
            if batch['function'][i] == 'add':
                input_images_list.append(batch['remove_pano'][i])
                target_images_list.append(batch['pano'][i])
                ref_images_list.append(sample_ref)
                pano_mask_list.append(batch['pano_mask'][i])
            elif batch['function'][i] == 'remove':
                input_images_list.append(batch['pano'][i])
                target_images_list.append(batch['remove_pano'][i])
                ref_images_list.append(None)
                pano_mask_list.append(batch['pano_mask'][i])
            elif batch['function'][i] == 'move':
                input_images_list.append(batch['remove_pano'][i])
                target_images_list.append(batch['pano'][i])
                ref_images_list.append(None)
                pano_mask_list.append(batch['pano_mask'][i])

        input_images = torch.stack(input_images_list, dim=0)
        target_images = torch.stack(target_images_list, dim=0)
        pano_mask = torch.stack(pano_mask_list, dim=0)

        sample_has_ref = [ref_img is not None for ref_img in ref_images_list]
        has_ref_images = any(ref_img is not None for ref_img in ref_images_list)
        if has_ref_images:
            ref_images_filled = []
            for ref_img in ref_images_list:
                if ref_img is not None:
                    ref_images_filled.append(ref_img)
                else:
                    first_ref = next(ref for ref in ref_images_list if ref is not None)
                    ref_images_filled.append(torch.zeros_like(first_ref))
            ref_images = torch.stack(ref_images_filled, dim=0)

        input_images = rearrange(input_images, 'b m c h w -> (b m) c h w')
        if has_ref_images:
            ref_images = rearrange(ref_images, 'b m c h w -> (b m) c h w')
        target_images = rearrange(target_images, 'b m c h w -> (b m) c h w')
        pano_mask = rearrange(pano_mask, 'b m c h w -> (b m) c h w')
        input_depth_images = None
        if batch.get('input_depth') is not None:
            input_depth_images = rearrange(batch['input_depth'], 'b m c h w -> (b m) c h w')
        functions = self._expand_sample_functions(batch['function'], batch['pano'].shape[1])
        expanded_sample_has_ref = self._expand_sample_flags(sample_has_ref, batch['pano'].shape[1])

        random_p_img = torch.rand(1, device=device).item()
        random_p_mask = torch.rand(1, device=device).item()
        random_p_ref = torch.rand(1, device=device).item()
        random_p_depth = torch.rand(1, device=device).item()
        use_mask = random_p_mask < self.hparams.edit_mask_use_prob
        use_img_cfg = random_p_img < self.hparams.image_use_prob
        use_mask_guidance = use_mask and pano_mask is not None
        use_ref = random_p_ref < self.hparams.ref_use_prob and has_ref_images
        use_depth_guidance = (
            use_mask_guidance
            and input_depth_images is not None
            and random_p_depth < self.hparams.depth_use_prob
        )
        bbox_use_prob_train, point_use_prob_train = self._resolve_training_target_guidance_probs()
        mask_guidance_images = (
            self._build_mask_guidance_images(
                pano_mask,
                functions=functions,
                bbox_use_prob=bbox_use_prob_train,
                point_use_prob=point_use_prob_train,
                mutually_exclusive_target_guidance=True,
            )
            if use_mask_guidance
            else None
        )

        # ---- Text processing via FluxMultiModalProcessor ----
        prompt_inputs = self.get_flux2_pano_prompts(
            batch,
            use_img_cfg=use_img_cfg,
            use_ref=use_ref,
            use_mask_guidance=use_mask_guidance,
            sample_has_ref=sample_has_ref,
        )
        processed_data = self.multimodal_processor(
            prompt_inputs,
            use_img_cfg=use_img_cfg,
            mode="train",
        )
        prompts = processed_data["prompts"]

        # ---- Encode text with Qwen3 ----
        encoder_hidden_states, text_ids = self.encode_prompt(prompts, device)

        # ---- Encode input images via VAE ----
        if use_img_cfg:
            condition_image_groups = self._build_condition_image_groups(
                input_images,
                ref_images=ref_images if has_ref_images else None,
                mask_guidance_images=mask_guidance_images,
                depth_guidance_images=input_depth_images if use_depth_guidance else None,
                use_ref=use_ref,
                use_mask_guidance=use_mask_guidance,
                sample_has_ref=expanded_sample_has_ref,
            )
            input_img_latents = self._encode_condition_images(
                condition_image_groups,
                device,
            )
        else:
            input_img_latents = []

        # ---- Encode target images (patchified + BN normalized) ----
        padded_target_images = self.pad_pano(target_images)
        target_img_latents = self.encode_image(padded_target_images, device=device)
        unpadded_target_img_latents = []
        for latent in target_img_latents:
            unpadded_target_img_latents.append(self.unpad_pano(latent, patchified=True))
        target_img_latents = torch.stack(unpadded_target_img_latents, dim=0).squeeze(1)

        # ---- Flow matching: sample noise and timestep ----
        x0 = self.sample_x0(target_img_latents)
        t = self.sample_timestep(target_img_latents)
        if isinstance(target_img_latents, (list, tuple)):
            noise_z = [(1 - t[i]) * target_img_latents[i] + t[i] * x0[i] for i in range(b)]
            ut = [(x0[i] - target_img_latents[i]) for i in range(b)]
            noise_z = torch.stack(noise_z, dim=0)
            ut = torch.stack(ut, dim=0)
        else:
            dims = [1] * (len(target_img_latents.size()) - 1)
            t_ = t.view(t.size(0), *dims)
            noise_z = (1 - t_) * target_img_latents + t_ * x0
            ut = x0 - target_img_latents

        # ---- Forward pass through BaseModel ----
        denoise = self.mv_base_model(
            hidden_states=noise_z,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            input_img_latents=input_img_latents,
            return_dict=False,
        )[0]

        # ---- Loss computation (same as OmniGen version) ----
        # if use_img_cfg:
        #     # Keep the loss weighting path in the original Flux2 patchified latent space.
        #     padded_input_images = self.pad_pano(input_images)
        #     input_img_latents_for_loss = self.encode_image(padded_input_images, device=device)
        #     input_img_latents_for_loss = [
        #         self.unpad_pano(latent, patchified=True) for latent in input_img_latents_for_loss
        #     ]
        #     input_img_latents_for_loss = torch.stack(input_img_latents_for_loss, dim=0).squeeze(1)
        #     assert len(input_img_latents_for_loss) == len(target_img_latents), \
        #         f"input count ({len(input_img_latents_for_loss)}) != target count ({len(target_img_latents)})"

        #     patch_weight = []
        #     for i in range(len(target_img_latents)):
        #         temp_x = target_img_latents[i]
        #         w = torch.ones_like(temp_x).detach()
        #         input_x = input_img_latents_for_loss[i]

        #         target_ch = temp_x.shape[0]
        #         if input_x.shape[0] > target_ch:
        #             input_x = input_x[:target_ch]

        #         if input_x.shape != temp_x.shape:
        #             print(f"Warning: the {i}th image shape does not match - input: {input_x.shape}, target: {temp_x.shape}")
        #             w = w * 0
        #             patch_weight.append(w)
        #             continue

        #         diff = torch.abs(temp_x - input_x).detach()
        #         diff_mean = torch.mean(diff)
        #         if diff_mean < 0.001:
        #             w = w * 0
        #         elif diff_mean <= 0.8:
        #             weight = self._compute_denoise_style_raw_weight(diff_mean).item()
        #             w[diff > 0.3] = weight
        #         patch_weight.append(w)

        #     squared_error = torch.nn.functional.mse_loss(ut, denoise, reduction='none')
        #     stacked_patch_weights = torch.stack(patch_weight, dim=0)
        #     weighted_squared_error = squared_error * stacked_patch_weights
        #     denoise_loss = torch.mean(weighted_squared_error)
        # else:
        loss = torch.nn.functional.mse_loss(ut, denoise)

        used_input_image_proj = use_img_cfg
        used_mask_guidance_proj = use_img_cfg and use_mask_guidance and not use_depth_guidance
        used_mask_depth_guidance_proj = use_img_cfg and use_depth_guidance

        # DDP dummy loss to keep all trainable params in gradient graph.
        if not used_input_image_proj:
            dummy_input = torch.zeros(
                1,
                self.mv_base_model.input_image_proj.in_channels,
                2,
                2,
                device=device,
                dtype=self.mv_base_model.input_image_proj.weight.dtype,
            )
            loss = loss + 0.0 * self.mv_base_model.input_image_proj(dummy_input).sum()

        if not used_mask_guidance_proj:
            dummy_mask_input = torch.zeros(
                1,
                self.mv_base_model.mask_guidance_proj.in_channels,
                2,
                2,
                device=device,
                dtype=self.mv_base_model.mask_guidance_proj.weight.dtype,
            )
            loss = loss + 0.0 * self.mv_base_model.mask_guidance_proj(dummy_mask_input).sum()

        if not used_mask_depth_guidance_proj:
            dummy_mask_depth_input = torch.zeros(
                1,
                self.mv_base_model.mask_depth_guidance_proj.in_channels,
                2,
                2,
                device=device,
                dtype=self.mv_base_model.mask_depth_guidance_proj.weight.dtype,
            )
            loss = loss + 0.0 * self.mv_base_model.mask_depth_guidance_proj(dummy_mask_depth_input).sum()

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss_denoise', loss)
        # self.log('train/loss_pano', loss)
        return loss

    @torch.no_grad()
    def inference(self, batch, return_aux=False, guidance_mode=None, guidance_mask_batch=None, generator=None):
        device = batch['pano'].device
        dtype = batch['pano'].dtype
        b = batch['pano'].shape[0]
        height = batch['height'][0]
        width = batch['width'][0]

        if guidance_mode is None:
            guidance_modes = self._parse_inference_target_guidance_modes(
                self.hparams.inference_target_guidance_mode
            )
            guidance_mode = self._select_primary_inference_guidance_mode(guidance_modes)

        input_images_list = []
        ref_images_list = []
        refs_in_batch = 'refs' in batch and len(batch['refs']) > 0

        for i in range(b):
            sample_ref = batch['refs'][i] if refs_in_batch and self._sample_has_valid_ref(batch, i) else None
            if batch['function'][i] == 'add':
                input_images_list.append(batch['remove_pano'][i])
                ref_images_list.append(sample_ref)
            elif batch['function'][i] == 'remove':
                input_images_list.append(batch['pano'][i])
                ref_images_list.append(None)
            elif batch['function'][i] == 'move':
                input_images_list.append(batch['remove_pano'][i])
                ref_images_list.append(None)

        input_images = torch.stack(input_images_list, dim=0)

        sample_has_ref = [ref_img is not None for ref_img in ref_images_list]
        has_ref_images = any(ref_img is not None for ref_img in ref_images_list)
        if has_ref_images:
            ref_images_filled = []
            for ref_img in ref_images_list:
                if ref_img is not None:
                    ref_images_filled.append(ref_img)
                else:
                    first_ref = next(ref for ref in ref_images_list if ref is not None)
                    ref_images_filled.append(torch.zeros_like(first_ref))
            ref_images = torch.stack(ref_images_filled, dim=0)

        input_images = rearrange(input_images, 'b m c h w -> (b m) c h w')
        if has_ref_images:
            ref_images = rearrange(ref_images, 'b m c h w -> (b m) c h w')
        input_depth_images = None
        if batch.get('input_depth') is not None:
            input_depth_images = rearrange(batch['input_depth'], 'b m c h w -> (b m) c h w')

        if guidance_mask_batch is None:
            guidance_mask_batch = self._select_inference_guidance_mask_batch(batch, guidance_mode)

        has_pano_mask = guidance_mask_batch is not None
        use_mask_in_inf = has_pano_mask and self.hparams.use_mask_in_inference
        if has_pano_mask:
            pano_mask = rearrange(guidance_mask_batch, 'b m c h w -> (b m) c h w')
        else:
            pano_mask = None
        functions = self._expand_sample_functions(batch['function'], batch['pano'].shape[1])
        expanded_sample_has_ref = self._expand_sample_flags(sample_has_ref, batch['pano'].shape[1])
        use_ref = has_ref_images and self.hparams.use_ref_in_inference
        bbox_use_prob_in_inf, point_use_prob_in_inf = self._resolve_inference_target_guidance_probs(guidance_mode)
        mask_guidance_images = (
            self._build_mask_guidance_images(
                pano_mask,
                functions=functions,
                bbox_use_prob=bbox_use_prob_in_inf,
                point_use_prob=point_use_prob_in_inf,
            )
            if use_mask_in_inf
            else None
        )
        use_img_cfg = True
        do_classifier_free_guidance = self.hparams.guidance_scale > 1.0 and not self.is_distilled

        # ---- Text processing ----
        prompt_info = self._resolve_inference_prompt_info(
            batch,
            guidance_mode=guidance_mode,
            guidance_mask_batch=guidance_mask_batch,
        )
        prompts = prompt_info["prompts"]
        encoder_hidden_states, text_ids = self.encode_prompt(prompts, device)
        negative_encoder_hidden_states = None
        negative_text_ids = None
        if do_classifier_free_guidance:
            negative_prompts = [""] * len(prompts)
            negative_encoder_hidden_states, negative_text_ids = self.encode_prompt(negative_prompts, device)

        if getattr(self, "offload_text_encoder_after_encode", False):
            self.offload_text_encoder()

        # ---- Encode input images ----
        if use_img_cfg:
            condition_image_groups = self._build_condition_image_groups(
                input_images,
                ref_images=ref_images if has_ref_images else None,
                mask_guidance_images=mask_guidance_images,
                depth_guidance_images=input_depth_images if use_mask_in_inf else None,
                use_ref=use_ref,
                use_mask_guidance=use_mask_in_inf,
                sample_has_ref=expanded_sample_has_ref,
            )
            input_img_latents = self._encode_condition_images(
                condition_image_groups,
                device,
            )
        else:
            input_img_latents = []

        # ---- Prepare noise latents ----
        # Compute mu for Flux2 dynamic shifting (official empirical formula)
        h_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_latent = 2 * (int(width) // (self.vae_scale_factor * 2))
        image_seq_len = (h_latent // 2) * (w_latent // 2)  # patchified spatial tokens
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.hparams.inference_timesteps)

        sigmas = np.linspace(1, 0, self.hparams.inference_timesteps + 1)[:self.hparams.inference_timesteps]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, self.hparams.inference_timesteps, device, None, sigmas=sigmas, mu=mu
        )
        self._num_timesteps = len(timesteps)

        # Flux2: in_channels=128 is patchified, actual VAE latent channels = in_channels // 4 = 32
        latent_channels = self.transformer_channel // 4
        latents = self.prepare_latents(
            b * 1,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator=generator,
            latents=None,
        )

        # Match the official Flux2 Klein pipeline and start denoising from index 0.
        self.scheduler.set_begin_index(0)

        # ---- Denoising loop ----
        for i, t in enumerate(timesteps):
            # Scheduler returns timesteps in [0, 1000], but Model.forward multiplies by 1000 internally.
            # Must divide by 1000 here to match training convention where t ∈ [0, 1].
            # (Same as original Flux2KleinPipeline: `timestep=timestep / 1000`)
            timestep = (t / 1000).expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.mv_base_model(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=text_ids,
                input_img_latents=input_img_latents,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                neg_noise_pred = self.mv_base_model(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=negative_encoder_hidden_states,
                    txt_ids=negative_text_ids,
                    input_img_latents=input_img_latents,
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + self.hparams.guidance_scale * (noise_pred - neg_noise_pred)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # ---- Decode latents (BN denormalize → unpatchify → VAE decode) ----
        latents = latents.to(self.vae.dtype)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean

        latents = self._unpatchify_latents(latents)
        latents = self.pad_pano(latents, latent=True)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.unpad_pano(image)

        if return_aux:
            return image, prompt_info
        return image

    @torch.no_grad()
    def on_validation_epoch_start(self):
        self._val_log_entries = []
        self._val_logged_samples = 0

    @staticmethod
    def _to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def _build_combined_visual_summary(self, entry):
        panels = [
            ("Edit Input", entry["pano_before"]),
            ("Edit Pred", entry["pano_pred"]),
            ("Edit GT", entry["pano_after"]),
        ]

        if entry.get("ref_image") is not None:
            panels.append((entry.get("ref_caption") or "Reference", entry["ref_image"]))
        if entry.get("pano_mask_src") is not None:
            panels.append(("Source Mask", entry["pano_mask_src"]))
        if entry.get("pano_mask_dst") is not None:
            panels.append(("Target Mask", entry["pano_mask_dst"]))
        if entry.get("pano_mask_point") is not None:
            panels.append(("Target Point Mask", entry["pano_mask_point"]))
        if entry.get("pano_mask_point_vis") is not None:
            panels.append(("Target Point Vis", entry["pano_mask_point_vis"]))
        if entry.get("pano_layout_cond") is not None:
            panels.append(("Layout Cond", entry["pano_layout_cond"]))

        return compose_labeled_image_grid(panels, panels_per_row=3)

    def _build_val_log_entry(
        self,
        pano_pred,
        pano_before,
        pano_after,
        pano_mask,
        pano_prompt,
        view_id,
        function_name,
        batch_idx,
        sample_offset,
        pano_layout_cond=None,
        ref_image=None,
        ref_caption=None,
    ):
        prompt = pano_prompt[0] if pano_prompt else None
        entry = {
            "view_id": view_id[0] if view_id else None,
            "function": function_name,
            "batch_idx": int(batch_idx),
            "sample_offset": int(sample_offset),
            "pano_prompt": prompt,
            "pano_pred": self._to_cpu(pano_pred[0]),
            "pano_before": self._to_cpu(pano_before[0, 0]),
            "pano_after": self._to_cpu(pano_after[0, 0]),
            "ref_image": None,
            "ref_caption": ref_caption,
            "pano_layout_cond": None,
            "pano_mask_src": None,
            "pano_mask_dst": None,
            "pano_mask_point": None,
            "pano_mask_point_vis": None,
        }

        if ref_image is not None:
            entry["ref_image"] = self._to_cpu(ref_image[0, 0])

        if pano_mask is not None:
            target_mask = pano_mask[0, 0, 0]
            if pano_mask.shape[2] < 3:
                raise ValueError(
                    f"`pano_mask` must include a precomputed target point channel, got {tuple(pano_mask.shape)}"
                )
            point_mask = pano_mask[0, 0, 2:3].clamp(0, 1)
            target_point_pixel = point_map_peak_pixel(point_mask[0])
            point_vis = draw_point_visualization(
                (target_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                target_point_pixel,
            )
            entry["pano_mask_src"] = self._to_cpu(pano_mask[0, 0, 1:2])
            entry["pano_mask_dst"] = self._to_cpu(pano_mask[0, 0, 0:1])
            entry["pano_mask_point"] = self._to_cpu(point_mask)
            entry["pano_mask_point_vis"] = torch.from_numpy(np.array(point_vis)).permute(2, 0, 1).float() / 127.5 - 1.0

        if pano_layout_cond is not None:
            entry["pano_layout_cond"] = self._to_cpu(pano_layout_cond[0, 0])

        return entry

    def on_validation_epoch_end(self):
        self._val_log_entries = []
        self._val_logged_samples = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        max_log_samples = int(getattr(self.hparams, "val_log_max_samples", 4) or 0)
        should_log_images = (
            self.trainer.is_global_zero
            and max_log_samples > 0
            and self._val_logged_samples < max_log_samples
        )

        step_entries = []
        guidance_modes = self._parse_inference_target_guidance_modes(
            self.hparams.inference_target_guidance_mode
        )
        multiple_modes = len(guidance_modes) > 1

        for guidance_mode in guidance_modes:
            guidance_mask_batch = self._select_inference_guidance_mask_batch(batch, guidance_mode)
            inference_result = self.inference(
                batch,
                return_aux=should_log_images,
                guidance_mode=guidance_mode,
                guidance_mask_batch=guidance_mask_batch,
            )
            if not should_log_images:
                continue

            pano_pred, inference_aux = inference_result
            actual_prompts = inference_aux.get("prompts")

            for i in range(len(batch['function'])):
                if guidance_mask_batch is not None:
                    mask_i = guidance_mask_batch[i:i+1]
                else:
                    mask_i = None
                sample_id = batch['pano_id'][i] if 'pano_id' in batch else batch['object_id'][i]
                prompt_text = self._prompt_text_at(
                    actual_prompts,
                    i,
                    fallback=batch['pano_prompt'][i],
                )
                if multiple_modes:
                    sample_id = f"{sample_id}_{guidance_mode}"
                    prompt_text = f"[{guidance_mode}] {prompt_text}"

                ref_i = None
                ref_caption = None
                if batch['function'][i] == 'add' and 'refs' in batch and len(batch['refs']) > 0 and self._sample_has_valid_ref(batch, i):
                    ref_i = batch['refs'][i:i+1]
                    ref_source_ids = batch.get('ref_source_id')
                    if ref_source_ids is not None:
                        ref_caption = f"Reference: {ref_source_ids[i]}"
                    else:
                        ref_img_paths = batch.get('ref_img_path')
                        if ref_img_paths is not None:
                            ref_caption = f"Reference: {os.path.basename(ref_img_paths[i])}"
                        else:
                            ref_caption = "Reference"

                if batch['function'][i] == 'add':
                    step_entries.append(self._build_val_log_entry(
                        pano_pred[i:i+1],
                        pano_before=batch['remove_pano'][i:i+1],
                        pano_after=batch['pano'][i:i+1],
                        pano_mask=mask_i,
                        ref_image=ref_i,
                        ref_caption=ref_caption,
                        pano_prompt=[prompt_text],
                        view_id=[sample_id],
                        function_name=batch['function'][i],
                        batch_idx=batch_idx,
                        sample_offset=i,
                        pano_layout_cond=batch.get('pano_layout_cond')[i:i+1] if batch.get('pano_layout_cond') is not None else None,
                    ))
                elif batch['function'][i] == 'remove':
                    step_entries.append(self._build_val_log_entry(
                        pano_pred[i:i+1],
                        pano_before=batch['pano'][i:i+1],
                        pano_after=batch['remove_pano'][i:i+1],
                        pano_mask=mask_i,
                        ref_image=ref_i,
                        ref_caption=ref_caption,
                        pano_prompt=[prompt_text],
                        view_id=[sample_id],
                        function_name=batch['function'][i],
                        batch_idx=batch_idx,
                        sample_offset=i,
                        pano_layout_cond=batch.get('pano_layout_cond')[i:i+1] if batch.get('pano_layout_cond') is not None else None,
                    ))
                elif batch['function'][i] == 'move':
                    step_entries.append(self._build_val_log_entry(
                        pano_pred[i:i+1],
                        pano_before=batch['remove_pano'][i:i+1],
                        pano_after=batch['pano'][i:i+1],
                        pano_mask=mask_i,
                        ref_image=ref_i,
                        ref_caption=ref_caption,
                        pano_prompt=[prompt_text],
                        view_id=[sample_id],
                        function_name=batch['function'][i],
                        batch_idx=batch_idx,
                        sample_offset=i,
                        pano_layout_cond=batch.get('pano_layout_cond')[i:i+1] if batch.get('pano_layout_cond') is not None else None,
                    ))

        if should_log_images:
            remaining = max_log_samples - self._val_logged_samples
            entries_to_log = step_entries[:remaining]
            for entry in entries_to_log:
                self.log_val_entry(entry)
            self._val_logged_samples += len(entries_to_log)

    def inference_and_save(self, batch, output_dir, ext='png', precomputed_predictions=None):
        if len(batch.get('function', [])) != 1:
            raise ValueError(
                "inference_and_save expects a single-sample batch. Slice batched inputs before saving."
            )
        if 'pano_id' in batch:
            identifier = batch['pano_id'][0]
        elif 'view_id' in batch:
            identifier = batch['view_id'][0]
        else:
            identifier = 'unknown'

        instruction_text = None
        if 'instruction_text' in batch:
            instruction_text = batch['instruction_text'][0]
        elif 'instruction' in batch and batch['instruction'][0] is not None:
            instruction_text = batch['instruction'][0].split('.')[0]

        ref_img_path = batch.get('ref_img_path', [])
        has_ref_img_path = (
            isinstance(ref_img_path, list)
            and len(ref_img_path) > 0
            and bool(ref_img_path[0])
        )
        if has_ref_img_path:
            ref_name = ref_img_path[0].split('/')[-1].split('.')[0]
            ref_prompt = batch.get('ref_prompt', [None])[0]
            if ref_prompt is not None:
                instruction = ref_prompt.split('.')[0]
                prompt_path = os.path.join(output_dir, f"{identifier}_{ref_name}_{instruction}.txt")
            else:
                prompt_path = os.path.join(output_dir, f"{identifier}_{ref_name}.txt")
        else:
            if instruction_text is not None:
                safe_instruction = instruction_text.replace(' ', '_').replace('/', '_')[:50]
                prompt_path = os.path.join(output_dir, f"{identifier}_{safe_instruction}.txt")
            else:
                prompt_path = os.path.join(output_dir, f"{identifier}.txt")

        os.makedirs(output_dir, exist_ok=True)

        if has_ref_img_path:
            ref_name = ref_img_path[0].split('/')[-1].split('.')[0]
            ref_prompt = batch.get('ref_prompt', [None])[0]
            if ref_prompt is not None:
                instruction = ref_prompt.split('.')[0]
                path = os.path.join(output_dir, f"{identifier}_{ref_name}_{instruction}.{ext}")
            else:
                path = os.path.join(output_dir, f"{identifier}_{ref_name}.{ext}")
        else:
            if instruction_text is not None:
                safe_instruction = instruction_text.replace(' ', '_').replace('/', '_')[:50]
                path = os.path.join(output_dir, f"{identifier}_{safe_instruction}.{ext}")
            else:
                path = os.path.join(output_dir, f"{identifier}.{ext}")

        guidance_modes = self._parse_inference_target_guidance_modes(
            self.hparams.inference_target_guidance_mode
        )
        multiple_modes = len(guidance_modes) > 1
        prompt_lines = []

        for guidance_mode in guidance_modes:
            path_suffix = guidance_mode if multiple_modes else None
            output_path = self._append_path_suffix(path, path_suffix)
            guidance_mask_batch = self._select_inference_guidance_mask_batch(batch, guidance_mode)
            prompt_info = self._resolve_inference_prompt_info(
                batch,
                guidance_mode=guidance_mode,
                guidance_mask_batch=guidance_mask_batch,
            )
            prompt_text = self._prompt_text_at(
                prompt_info.get("prompts"),
                0,
                fallback=batch['pano_prompt'][0],
            )
            if multiple_modes:
                prompt_text = f"[{guidance_mode}] {prompt_text}"
            prompt_lines.append(prompt_text)

            if os.path.exists(output_path):
                continue

            pano_pred = None
            if precomputed_predictions is not None:
                pano_pred = precomputed_predictions.get(guidance_mode)
            if pano_pred is None:
                pano_pred = self.inference(
                    batch,
                    guidance_mode=guidance_mode,
                    guidance_mask_batch=guidance_mask_batch,
                )

            image_to_save = tensor_to_image(pano_pred[0]) if isinstance(pano_pred, torch.Tensor) else pano_pred[0]
            Image.fromarray(np.asarray(image_to_save).astype(np.uint8)).save(output_path)
            print(f"Saved prediction to: {output_path}")

        with open(prompt_path, 'w') as f:
            f.write('\n'.join(prompt_lines) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        pano_pred = batch.get('pano_pred')
        used_offline_predictions = pano_pred is not None

        if not used_offline_predictions:
            parsed_guidance_modes = self._parse_inference_target_guidance_modes(
                self.hparams.inference_target_guidance_mode
            )
            primary_guidance_mode = self._select_primary_inference_guidance_mode(parsed_guidance_modes)
            guidance_mask_batch = self._select_inference_guidance_mask_batch(batch, primary_guidance_mode)
            pano_pred = self.inference(
                batch,
                guidance_mode=primary_guidance_mode,
                guidance_mask_batch=guidance_mask_batch,
            )

        pano_height = int(batch['height'][0]) if 'height' in batch else int(pano_pred.shape[-2])
        ensure_mover360_test_metrics(self, pano_height=pano_height)
        update_mover360_test_metrics(self, batch, pano_pred, batch_idx)

        if used_offline_predictions:
            return

        for sample_idx in range(len(batch['function'])):
            sample_batch = self._slice_batch_for_saving(batch, sample_idx)
            sample_pred = pano_pred[sample_idx:sample_idx + 1]
            output_dir = os.path.join(self.logger.save_dir, 'test', sample_batch['pano_id'][0])
            self.inference_and_save(
                sample_batch,
                output_dir,
                precomputed_predictions={primary_guidance_mode: sample_pred},
            )

    @torch.no_grad()
    def on_test_end(self):
        finalize_mover360_test_metrics(self)

    @torch.no_grad()
    @rank_zero_only
    def log_val_entry(self, entry):
        view_id = entry.get("view_id")
        pano_prompt = entry.get("pano_prompt")
        combined_vis = self._build_combined_visual_summary(entry)
        if combined_vis is None:
            return

        log_dict = {
            f'val/sample_{view_id}': self.temp_wandb_image(
                combined_vis,
                pano_prompt if pano_prompt else None,
            ),
        }

        self.logger.experiment.log(log_dict)
