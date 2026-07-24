import os
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from torch import nn
import wandb
import numpy as np

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError

from ..faed.FAED import FrechetAutoEncoderDistance
from .PanoGenerator import PanoBase


DEFAULT_MIN_FOV_DEGREES = 35.0
DEFAULT_MAX_FOV_DEGREES = 150.0
DEFAULT_CONTEXT_SCALE = 1.2
DEFAULT_PERSPECTIVE_SIZE = 512
DEFAULT_MASK_THRESHOLD = 0.0
DEFAULT_CLIP_SCORE_MODEL_NAME = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEFAULT_DINOV3_MODEL_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DEFAULT_HUGGINGFACE_CACHE = None
DEFAULT_DINOV3_CACHE_DIR = None


def _flatten_pano_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 5:
        return rearrange(image, 'b l c h w -> (b l) c h w')
    if image.ndim == 4:
        return image
    raise ValueError(f"Expected panorama batch with 4 or 5 dims, got shape {tuple(image.shape)}")


def _flatten_mask_batch(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 5:
        return rearrange(mask, 'b l c h w -> (b l) c h w')
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Expected mask batch with 4 or 5 dims, got shape {tuple(mask.shape)}")


def _select_function_aware_gt_batch(batch) -> torch.Tensor:
    if "function" not in batch or "remove_pano" not in batch:
        return batch["pano"]

    pano_gt = []
    for sample_idx, function_name in enumerate(batch["function"]):
        if str(function_name).lower() == "remove":
            pano_gt.append(batch["remove_pano"][sample_idx])
        else:
            pano_gt.append(batch["pano"][sample_idx])
    return torch.stack(pano_gt, dim=0)


def _module_hparam(module, name, default):
    hparams = getattr(module, "hparams", None)
    if hparams is None:
        return default
    if hasattr(hparams, name):
        value = getattr(hparams, name)
        return default if value is None else value
    get = getattr(hparams, "get", None)
    if get is not None:
        value = get(name, default)
        return default if value is None else value
    return default


def _resolve_dinov3_cache_dir(module):
    dinov3_cache_dir = _module_hparam(module, "dinov3_cache_dir", None)
    if dinov3_cache_dir is not None:
        return dinov3_cache_dir
    return _module_hparam(module, "huggingface_cache", DEFAULT_HUGGINGFACE_CACHE)


def _resolve_clip_cache_dir(module):
    return _module_hparam(module, "huggingface_cache", DEFAULT_HUGGINGFACE_CACHE)


def _normalize_vector_np(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def _make_camera_basis_np(theta_degrees: float, phi_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.deg2rad(theta_degrees)
    phi = np.deg2rad(phi_degrees)

    forward = np.array(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
            np.cos(phi) * np.cos(theta),
        ],
        dtype=np.float64,
    )
    forward = _normalize_vector_np(forward)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([np.cos(theta), 0.0, -np.sin(theta)], dtype=np.float64)
    right = _normalize_vector_np(right)

    up = _normalize_vector_np(np.cross(forward, right))
    return forward, right, up


def _extract_circular_box_xyxy(mask_2d: np.ndarray) -> np.ndarray | None:
    mask = np.asarray(mask_2d, dtype=bool)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    height, width = mask.shape
    y1 = float(ys.min()) / float(max(height, 1))
    y2 = (float(ys.max()) + 1.0) / float(max(height, 1))

    x_unique = np.unique(xs)
    if x_unique.size >= width:
        return np.array([0.0, y1, 1.0, y2], dtype=np.float32)

    if x_unique.size == 1:
        start = int(x_unique[0])
        end = int(x_unique[0])
    else:
        diffs = x_unique[1:] - x_unique[:-1]
        wrap_gap = np.array([x_unique[0] + width - x_unique[-1]], dtype=diffs.dtype)
        gaps = np.concatenate([diffs, wrap_gap], axis=0)
        largest_gap_idx = int(np.argmax(gaps))
        start = int(x_unique[(largest_gap_idx + 1) % x_unique.size])
        end = int(x_unique[largest_gap_idx])

    span = float(((end - start) % width) + 1)
    if span >= width:
        return np.array([0.0, y1, 1.0, y2], dtype=np.float32)

    x1 = float(start) / float(max(width, 1))
    x2 = ((float(start) + span) % float(width)) / float(max(width, 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _box_center_x(box_xyxy: np.ndarray | None) -> float | None:
    if box_xyxy is None:
        return None

    x1, _, x2, _ = [float(value) for value in box_xyxy]
    if x2 >= x1:
        return 0.5 * (x1 + x2)

    span = (1.0 - x1) + x2
    return (x1 + 0.5 * span) % 1.0


def _normalized_erp_to_directions_np(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    theta = xs * (2.0 * np.pi) - np.pi
    phi = (0.5 - ys) * np.pi
    return np.stack(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
            np.cos(phi) * np.cos(theta),
        ],
        axis=-1,
    )


def _sample_bbox_points_np(box_xyxy: np.ndarray, sample_count: int = 7) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x_span = x2 - x1 if x2 >= x1 else (1.0 - x1) + x2
    y_span = max(y2 - y1, 0.0)

    xs = (x1 + np.linspace(0.0, x_span, sample_count, dtype=np.float64)) % 1.0
    ys = y1 + np.linspace(0.0, y_span, sample_count, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.reshape(-1), grid_y.reshape(-1)


def _clamp_fov(fov_degrees: float, min_fov_degrees: float, max_fov_degrees: float) -> float:
    min_fov_degrees = min(max(float(min_fov_degrees), 1.0), 178.0)
    max_fov_degrees = min(max(float(max_fov_degrees), min_fov_degrees), 178.0)
    return min(max(float(fov_degrees), min_fov_degrees), max_fov_degrees)


def _fov_degrees_from_half_plane_extent(half_plane_extent: float) -> float:
    return float(np.rad2deg(2.0 * np.arctan(max(float(half_plane_extent), 1e-6))))


def _compute_adaptive_fov_degrees(
    box_xyxy: np.ndarray,
    theta_degrees: float,
    phi_degrees: float,
    context_scale: float,
    min_fov_degrees: float,
    max_fov_degrees: float,
) -> tuple[float, float]:
    forward, right, up = _make_camera_basis_np(theta_degrees=theta_degrees, phi_degrees=phi_degrees)
    sample_xs, sample_ys = _sample_bbox_points_np(box_xyxy)
    directions = _normalized_erp_to_directions_np(sample_xs, sample_ys)

    depth = directions @ forward
    valid = depth > 1e-5
    if not np.any(valid):
        max_fov = _clamp_fov(max_fov_degrees, min_fov_degrees, max_fov_degrees)
        return max_fov, max_fov

    directions = directions[valid]
    depth = depth[valid]
    plane_x = (directions @ right) / depth
    plane_y = (directions @ up) / depth

    object_half_width = max(
        abs(float(plane_x.min())),
        abs(float(plane_x.max())),
        0.5 * float(plane_x.max() - plane_x.min()),
    )
    object_half_height = max(
        abs(float(plane_y.min())),
        abs(float(plane_y.max())),
        0.5 * float(plane_y.max() - plane_y.min()),
    )

    context_scale = max(float(context_scale), 1.0)
    required_horizontal_fov = _fov_degrees_from_half_plane_extent(object_half_width * context_scale)
    required_vertical_fov = _fov_degrees_from_half_plane_extent(object_half_height * context_scale)
    required_fov = max(required_horizontal_fov, required_vertical_fov)
    fov = _clamp_fov(required_fov, min_fov_degrees, max_fov_degrees)
    return fov, required_fov


def _find_mask_view(
    mask_2d: torch.Tensor,
    threshold: float,
    context_scale: float,
    min_fov_degrees: float,
    max_fov_degrees: float,
) -> tuple[float, float, float] | None:
    binary_mask = mask_2d.detach().to(torch.float32).cpu().numpy() > float(threshold)
    box_xyxy = _extract_circular_box_xyxy(binary_mask)
    if box_xyxy is None:
        return None

    center_x = _box_center_x(box_xyxy)
    if center_x is None:
        return None

    _, y1, _, y2 = [float(value) for value in box_xyxy]
    center_y = min(max(0.5 * (y1 + y2), 0.0), 1.0)
    center_x = min(max(center_x, 0.0), 1.0)

    theta_degrees = center_x * 360.0 - 180.0
    phi_degrees = 90.0 - center_y * 180.0
    fov_degrees, _ = _compute_adaptive_fov_degrees(
        box_xyxy=box_xyxy,
        theta_degrees=theta_degrees,
        phi_degrees=phi_degrees,
        context_scale=context_scale,
        min_fov_degrees=min_fov_degrees,
        max_fov_degrees=max_fov_degrees,
    )
    return theta_degrees, phi_degrees, fov_degrees


def _make_camera_basis_torch(theta_degrees: float, phi_degrees: float, device, dtype):
    theta = torch.tensor(np.deg2rad(theta_degrees), device=device, dtype=dtype)
    phi = torch.tensor(np.deg2rad(phi_degrees), device=device, dtype=dtype)

    forward = torch.stack(
        [
            torch.cos(phi) * torch.sin(theta),
            torch.sin(phi),
            torch.cos(phi) * torch.cos(theta),
        ]
    )
    forward = forward / forward.norm().clamp_min(1e-8)

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    right = torch.linalg.cross(world_up, forward)
    if float(right.norm().item()) < 1e-8:
        right = torch.stack([torch.cos(theta), torch.zeros_like(theta), -torch.sin(theta)])
    right = right / right.norm().clamp_min(1e-8)

    up = torch.linalg.cross(forward, right)
    up = up / up.norm().clamp_min(1e-8)
    return forward, right, up


def _sample_bilinear_wrap_x(image: torch.Tensor, sample_x: torch.Tensor, sample_y: torch.Tensor) -> torch.Tensor:
    channels, height, width = image.shape
    image_float = image.to(torch.float32)

    sample_x = torch.remainder(sample_x, float(width))
    sample_y = sample_y.clamp(0.0, float(height - 1))

    x0 = torch.floor(sample_x).to(torch.long)
    y0 = torch.floor(sample_y).to(torch.long)
    x1 = (x0 + 1) % width
    y1 = (y0 + 1).clamp(0, height - 1)

    wx = sample_x - x0.to(sample_x.dtype)
    wy = sample_y - y0.to(sample_y.dtype)

    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy

    sampled = (
        image_float[:, y0, x0] * wa.unsqueeze(0)
        + image_float[:, y0, x1] * wb.unsqueeze(0)
        + image_float[:, y1, x0] * wc.unsqueeze(0)
        + image_float[:, y1, x1] * wd.unsqueeze(0)
    )
    return sampled


def _erp_to_perspective_tensor(
    image: torch.Tensor,
    theta_degrees: float,
    phi_degrees: float,
    fov_degrees: float,
    output_width: int,
    output_height: int,
) -> torch.Tensor:
    _, pano_height, pano_width = image.shape
    device = image.device
    dtype = torch.float32

    x_axis = ((torch.arange(output_width, device=device, dtype=dtype) + 0.5) / float(output_width)) * 2.0 - 1.0
    y_axis = 1.0 - ((torch.arange(output_height, device=device, dtype=dtype) + 0.5) / float(output_height)) * 2.0
    plane_y, plane_x = torch.meshgrid(y_axis, x_axis, indexing="ij")

    fov_scale = torch.tan(torch.tensor(np.deg2rad(fov_degrees) * 0.5, device=device, dtype=dtype))
    forward, right, up = _make_camera_basis_torch(theta_degrees, phi_degrees, device=device, dtype=dtype)

    directions = (
        forward.view(1, 1, 3)
        + plane_x.unsqueeze(-1) * fov_scale * right.view(1, 1, 3)
        + plane_y.unsqueeze(-1) * fov_scale * up.view(1, 1, 3)
    )
    directions = directions / directions.norm(dim=2, keepdim=True).clamp_min(1e-8)

    yaw = torch.atan2(directions[..., 0], directions[..., 2])
    pitch = torch.asin(directions[..., 1].clamp(-1.0, 1.0))

    sample_x = (yaw / (2.0 * torch.pi) + 0.5) * float(pano_width)
    sample_y = (0.5 - pitch / torch.pi) * float(pano_height)
    return _sample_bilinear_wrap_x(image, sample_x, sample_y)


def _select_projection_mask_batch(batch, pano_pred: torch.Tensor) -> torch.Tensor | None:
    mask = batch.get("pano_full_mask")
    if mask is None:
        mask = batch.get("pano_mask")
    if mask is None:
        return None

    mask = _flatten_mask_batch(mask).to(device=pano_pred.device, dtype=torch.float32)
    if mask.shape[1] == 1:
        return mask[:, 0:1]

    target_mask = mask[:, 0:1]
    source_mask = mask[:, 1:2]
    target_available = target_mask.amax(dim=(1, 2, 3), keepdim=True) > 0
    return torch.where(target_available, target_mask, source_mask)


def _build_adaptive_perspective_metric_tensors(
    pano_gt_uint8: torch.Tensor,
    pano_pred_uint8: torch.Tensor,
    projection_mask: torch.Tensor | None,
    output_size: int,
    context_scale: float,
    min_fov_degrees: float,
    max_fov_degrees: float,
    mask_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    output_size = int(output_size)
    output_size = max(output_size, 1)

    pers_gt = []
    pers_pred = []
    pers_masks = []
    if projection_mask is None:
        projection_mask = torch.ones(
            pano_gt_uint8.shape[0],
            1,
            pano_gt_uint8.shape[-2],
            pano_gt_uint8.shape[-1],
            device=pano_gt_uint8.device,
            dtype=torch.float32,
        )

    for sample_idx in range(pano_gt_uint8.shape[0]):
        view = _find_mask_view(
            projection_mask[sample_idx, 0],
            threshold=mask_threshold,
            context_scale=context_scale,
            min_fov_degrees=min_fov_degrees,
            max_fov_degrees=max_fov_degrees,
        )
        if view is None:
            view = (0.0, 0.0, _clamp_fov(max_fov_degrees, min_fov_degrees, max_fov_degrees))
        theta_degrees, phi_degrees, fov_degrees = view

        pers_gt.append(
            _erp_to_perspective_tensor(
                pano_gt_uint8[sample_idx],
                theta_degrees=theta_degrees,
                phi_degrees=phi_degrees,
                fov_degrees=fov_degrees,
                output_width=output_size,
                output_height=output_size,
            )
        )
        pers_pred.append(
            _erp_to_perspective_tensor(
                pano_pred_uint8[sample_idx],
                theta_degrees=theta_degrees,
                phi_degrees=phi_degrees,
                fov_degrees=fov_degrees,
                output_width=output_size,
                output_height=output_size,
            )
        )
        pers_masks.append(
            _erp_to_perspective_tensor(
                projection_mask[sample_idx],
                theta_degrees=theta_degrees,
                phi_degrees=phi_degrees,
                fov_degrees=fov_degrees,
                output_width=output_size,
                output_height=output_size,
            ).clamp(0.0, 1.0)
        )

    pers_gt = torch.stack(pers_gt, dim=0).round().clamp(0, 255).to(torch.uint8)
    pers_pred = torch.stack(pers_pred, dim=0).round().clamp(0, 255).to(torch.uint8)
    pers_masks = torch.stack(pers_masks, dim=0)
    return pers_gt, pers_pred, pers_masks


def _to_uint8_metric_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.dtype == torch.uint8:
        return image

    image = image.detach().to(torch.float32)
    if image.numel() == 0:
        return image.to(torch.uint8)

    min_val = float(image.min().item())
    max_val = float(image.max().item())

    if min_val >= -1.5 and max_val <= 1.5:
        image = (image / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0
    elif min_val >= 0.0 and max_val <= 1.5:
        image = image.clamp(0.0, 1.0) * 255.0
    else:
        image = image.clamp(0.0, 255.0)

    return image.round().to(torch.uint8)


def _to_lpips_input(image_uint8: torch.Tensor) -> torch.Tensor:
    return image_uint8.to(torch.float32) / 255.0


def _metric_value_to_python(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
    return value


def _should_enable_shadow_mse(module) -> bool:
    guidance_mode = getattr(getattr(module, "hparams", None), "inference_target_guidance_mode", None)
    if guidance_mode is None:
        return False
    return str(guidance_mode).strip().lower() == "mask"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _strip_metric_prompt_prefix(prompt_text: str) -> str:
    original_text = str(prompt_text or "").strip()
    prompt_text = original_text
    image_prefixes = (
        "Image 1 is the panorama to edit.",
        "Image 2 is the object reference.",
        "Image 3 marks the target region.",
        "Image 2 marks the target region.",
        "Image 2 marks the object to remove.",
        "Image 2 marks the source and target regions.",
    )

    changed = True
    while changed:
        changed = False
        for prefix in image_prefixes:
            if prompt_text.startswith(prefix):
                prompt_text = prompt_text[len(prefix):].strip()
                changed = True
    return prompt_text or original_text


def _as_text_list(value, batch_size: int) -> list[str]:
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()

    try:
        items = list(value)
    except TypeError:
        items = [value] * batch_size

    texts = []
    for item in items[:batch_size]:
        if isinstance(item, bytes):
            texts.append(item.decode("utf-8", errors="ignore"))
        elif isinstance(item, torch.Tensor):
            item = item.detach().cpu()
            texts.append(str(item.item() if item.numel() == 1 else item.tolist()))
        else:
            texts.append(str(item))

    if len(texts) < batch_size:
        texts.extend([""] * (batch_size - len(texts)))
    return texts


def _resolve_clip_prompts(batch, batch_size: int) -> list[str]:
    for key in ("instruction_text", "pano_prompt_without_img", "pano_prompt"):
        if key not in batch:
            continue
        prompts = [_strip_metric_prompt_prefix(text) for text in _as_text_list(batch[key], batch_size)]
        if any(prompt.strip() for prompt in prompts):
            return prompts
    return [""] * batch_size


def _tensor_batch_to_pil_images(images_uint8: torch.Tensor) -> list[Image.Image]:
    images = images_uint8.detach().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW image tensor, got shape {tuple(images.shape)}")

    pil_images = []
    for image in images:
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] < 3:
            image = image.repeat(3, 1, 1)[:3]
        elif image.shape[0] > 3:
            image = image[:3]
        image = image.to(torch.uint8)
        array = image.permute(1, 2, 0).contiguous().numpy()
        pil_images.append(Image.fromarray(array, mode="RGB"))
    return pil_images


class DinoV3Similarity(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_MODEL_NAME,
        cache_dir: str | None = DEFAULT_DINOV3_CACHE_DIR,
        device: str | None = None,
        local_files_only: bool = False,
        hf_token: str | None = None,
        batch_size: int = 2,
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device_name = device
        self.local_files_only = bool(local_files_only)
        self.hf_token = hf_token
        self.batch_size = max(int(batch_size), 1)
        self._processor = None
        self._model = None
        self._update_called = False
        self.register_buffer("similarity_sum", torch.tensor(0.0, dtype=torch.float64), persistent=False)
        self.register_buffer("similarity_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def _resolved_device(self) -> torch.device:
        if self.device_name:
            return torch.device(self.device_name)
        if self.similarity_sum.is_cuda:
            return self.similarity_sum.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv3 similarity needs transformers with DINOv3 support. "
                "Install the project environment before running test metrics."
            ) from exc

        cache_dir_arg = None
        if self.cache_dir:
            cache_dir = Path(self.cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_dir_arg = str(cache_dir)
            os.environ.setdefault("HF_HUB_CACHE", cache_dir_arg)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir_arg)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir_arg)
            if cache_dir.name == "hub":
                os.environ.setdefault("HF_HOME", str(cache_dir.parent))

        hf_token = self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)

        def load_model_pair(local_files_only: bool):
            processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=cache_dir_arg,
                local_files_only=local_files_only,
                token=hf_token,
            )
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=cache_dir_arg,
                local_files_only=local_files_only,
                token=hf_token,
            )
            return processor, model

        first_local_only = self.local_files_only or (cache_dir_arg is not None and not hf_token)
        try:
            processor, model = load_model_pair(local_files_only=first_local_only)
        except Exception as first_exc:
            if self.local_files_only or hf_token:
                raise RuntimeError(
                    f"Could not load DINOv3 model '{self.model_name}' "
                    f"(cache_dir={cache_dir_arg}, local_files_only={first_local_only}, "
                    f"token_provided={bool(hf_token)})."
                ) from first_exc
            try:
                processor, model = load_model_pair(local_files_only=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load DINOv3 model '{self.model_name}' "
                    f"(cache_dir={cache_dir_arg}, local_files_only=False, "
                    f"token_provided={bool(hf_token)}). Local cache load also failed: {first_exc}"
                ) from exc

        model.to(self._resolved_device())
        model.requires_grad_(False)
        model.eval()
        self._processor = processor
        self._model = model

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        self._load_model()
        features = []
        device = next(self._model.parameters()).device
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                image_batch = images[start:start + self.batch_size]
                inputs = self._processor(images=image_batch, return_tensors="pt")
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                    if hasattr(value, "to")
                }
                outputs = self._model(**inputs)
                if getattr(outputs, "pooler_output", None) is not None:
                    feats = outputs.pooler_output
                elif getattr(outputs, "last_hidden_state", None) is not None:
                    feats = outputs.last_hidden_state[:, 0]
                else:
                    first_output = outputs[0]
                    feats = first_output[:, 0] if first_output.ndim == 3 else first_output
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                features.append(feats.detach())
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def update(self, pred_images_uint8: torch.Tensor, gt_images_uint8: torch.Tensor) -> None:
        pred_images = _tensor_batch_to_pil_images(pred_images_uint8)
        gt_images = _tensor_batch_to_pil_images(gt_images_uint8)
        sample_count = min(len(pred_images), len(gt_images))
        if sample_count <= 0:
            return

        pred_features = self._encode(pred_images[:sample_count])
        gt_features = self._encode(gt_images[:sample_count])
        similarities = (pred_features * gt_features).sum(dim=-1)
        self.similarity_sum += similarities.sum().to(
            device=self.similarity_sum.device,
            dtype=self.similarity_sum.dtype,
        )
        self.similarity_count += torch.tensor(
            similarities.numel(),
            device=self.similarity_count.device,
            dtype=self.similarity_count.dtype,
        )
        self._update_called = True

    def compute(self) -> torch.Tensor:
        count = self.similarity_count.clamp_min(1).to(torch.float64)
        return (self.similarity_sum / count).to(torch.float32)

    def reset(self) -> None:
        self.similarity_sum.zero_()
        self.similarity_count.zero_()
        self._update_called = False


class CompatibleCLIPScore(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_SCORE_MODEL_NAME,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._image_processor = None
        self._tokenizer = None
        self._model = None
        self._update_called = False
        self.register_buffer("score_sum", torch.tensor(0.0, dtype=torch.float64), persistent=False)
        self.register_buffer("score_count", torch.tensor(0, dtype=torch.long), persistent=False)

    def _load_model(self) -> None:
        if self._model is not None and self._image_processor is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel
        except ImportError as exc:
            raise RuntimeError("CLIPScore needs transformers installed.") from exc

        cache_dir_arg = None
        if self.cache_dir:
            cache_dir = Path(self.cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_dir_arg = str(cache_dir)
            os.environ.setdefault("HF_HUB_CACHE", cache_dir_arg)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir_arg)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir_arg)
            if cache_dir.name == "hub":
                os.environ.setdefault("HF_HOME", str(cache_dir.parent))

        self._image_processor = AutoImageProcessor.from_pretrained(self.model_name, cache_dir=cache_dir_arg)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir_arg)
        self._model = CLIPModel.from_pretrained(self.model_name, cache_dir=cache_dir_arg)
        self._model.to(self.score_sum.device)
        self._model.requires_grad_(False)
        self._model.eval()

    @staticmethod
    def _feature_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        pooler_output = getattr(output, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        if isinstance(output, (tuple, list)) and output:
            first_output = output[0]
            if isinstance(first_output, torch.Tensor):
                return first_output
        raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")

    @torch.no_grad()
    def update(self, images: torch.Tensor | list[torch.Tensor], text: str | list[str]) -> None:
        self._load_model()
        if not isinstance(images, list):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = list(images)
        if not all(image.ndim == 3 for image in images):
            raise ValueError("Expected CLIPScore images with shape [C, H, W] or [N, C, H, W].")

        if not isinstance(text, list):
            text = [text]
        if len(text) != len(images):
            raise ValueError(f"Expected {len(images)} CLIP prompts, got {len(text)}.")

        device = self.score_sum.device
        processed_images = self._image_processor(
            images=[image.detach().cpu() for image in images],
            return_tensors="pt",
        )
        processed_text = self._tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        pixel_values = processed_images["pixel_values"].to(device)
        input_ids = processed_text["input_ids"].to(device)
        attention_mask = processed_text.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        image_features = self._feature_tensor(self._model.get_image_features(pixel_values=pixel_values))
        text_features = self._feature_tensor(
            self._model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        scores = 100.0 * (image_features * text_features).sum(dim=-1)
        self.score_sum += scores.sum().to(device=device, dtype=self.score_sum.dtype)
        self.score_count += torch.tensor(scores.numel(), device=device, dtype=self.score_count.dtype)
        self._update_called = True

    def compute(self) -> torch.Tensor:
        count = self.score_count.clamp_min(1).to(torch.float64)
        score = self.score_sum / count
        return torch.max(score.to(torch.float32), torch.zeros((), device=score.device, dtype=torch.float32))

    def reset(self) -> None:
        self.score_sum.zero_()
        self.score_count.zero_()
        self._update_called = False


def ensure_mover360_test_metrics(module, pano_height: int, log_test_samples: int = 0):
    if getattr(module, "_mover360_test_metrics_ready", False):
        return

    dinov3_model_name = _module_hparam(module, "dinov3_model_name", DEFAULT_DINOV3_MODEL_NAME)
    dinov3_cache_dir = _resolve_dinov3_cache_dir(module)
    dinov3_device = _module_hparam(module, "dinov3_device", None)
    dinov3_local_files_only = _as_bool(_module_hparam(module, "dinov3_local_files_only", False))
    dinov3_hf_token = _module_hparam(module, "dinov3_hf_token", None)
    dinov3_batch_size = int(_module_hparam(module, "dinov3_batch_size", 2))

    eval_metrics = {
        "FID": FrechetInceptionDistance(feature=2048),
        "FEAD": FrechetAutoEncoderDistance(pano_height=int(pano_height)),
        "PSNR": PeakSignalNoiseRatio(data_range=255.0),
        "SSIM": StructuralSimilarityIndexMeasure(data_range=255.0),
        "LPIPS": LearnedPerceptualImagePatchSimilarity(normalize=True),
        "DINOv3_similarity": DinoV3Similarity(
            model_name=dinov3_model_name,
            cache_dir=dinov3_cache_dir,
            device=dinov3_device,
            local_files_only=dinov3_local_files_only,
            hf_token=dinov3_hf_token,
            batch_size=dinov3_batch_size,
        ),
    }
    if _should_enable_shadow_mse(module):
        eval_metrics["shadow_mse"] = MeanSquaredError()

    module.eval_metrics = nn.ModuleDict(eval_metrics)
    module.eval_metrics.requires_grad_(False)
    module._mover360_test_metrics_ready = True
    module._mover360_log_test_samples = int(log_test_samples)


@torch.no_grad()
def update_mover360_test_metrics(module, batch, pano_pred: torch.Tensor, batch_idx: int):
    if not getattr(module, "_mover360_test_metrics_ready", False):
        pano_height = int(batch["height"][0]) if "height" in batch else int(pano_pred.shape[-2])
        ensure_mover360_test_metrics(module, pano_height=pano_height)

    module.eval_metrics.to(device=pano_pred.device)

    pano_gt_uint8 = _to_uint8_metric_tensor(_flatten_pano_batch(_select_function_aware_gt_batch(batch)))
    pano_pred_uint8 = _to_uint8_metric_tensor(_flatten_pano_batch(pano_pred))

    module.eval_metrics["FEAD"].update(pano_gt_uint8, real=True)
    module.eval_metrics["FEAD"].update(pano_pred_uint8, real=False)
    module.eval_metrics["PSNR"].update(
        pano_pred_uint8.to(torch.float32),
        pano_gt_uint8.to(torch.float32),
    )

    perspective_size = int(_module_hparam(module, "test_perspective_size", DEFAULT_PERSPECTIVE_SIZE))
    context_scale = float(_module_hparam(module, "test_perspective_context_scale", DEFAULT_CONTEXT_SCALE))
    min_fov_degrees = float(_module_hparam(module, "test_perspective_min_fov", DEFAULT_MIN_FOV_DEGREES))
    max_fov_degrees = float(_module_hparam(module, "test_perspective_max_fov", DEFAULT_MAX_FOV_DEGREES))
    mask_threshold = float(_module_hparam(module, "test_perspective_mask_threshold", DEFAULT_MASK_THRESHOLD))
    projection_mask = _select_projection_mask_batch(batch, pano_pred)
    pers_gt_uint8, pers_pred_uint8, pers_mask = _build_adaptive_perspective_metric_tensors(
        pano_gt_uint8=pano_gt_uint8,
        pano_pred_uint8=pano_pred_uint8,
        projection_mask=projection_mask,
        output_size=perspective_size,
        context_scale=context_scale,
        min_fov_degrees=min_fov_degrees,
        max_fov_degrees=max_fov_degrees,
        mask_threshold=mask_threshold,
    )

    module.eval_metrics["FID"].update(pers_gt_uint8, real=True)
    module.eval_metrics["FID"].update(pers_pred_uint8, real=False)

    module.eval_metrics["SSIM"].update(
        pers_pred_uint8.to(torch.float32),
        pers_gt_uint8.to(torch.float32),
    )
    module.eval_metrics["LPIPS"].update(
        _to_lpips_input(pers_pred_uint8),
        _to_lpips_input(pers_gt_uint8),
    )
    module.eval_metrics["DINOv3_similarity"].update(pers_pred_uint8, pers_gt_uint8)

    if "shadow_mse" in module.eval_metrics and pers_mask is not None:
        edit_region_mask = pers_mask > 0.5
        non_edit_region_mask = ~edit_region_mask
        if torch.any(non_edit_region_mask):
            pers_gt_float = pers_gt_uint8.to(torch.float32) / 255.0
            pers_pred_float = pers_pred_uint8.to(torch.float32) / 255.0
            expanded_mask = non_edit_region_mask.expand_as(pers_gt_float)
            module.eval_metrics["shadow_mse"].update(
                pers_pred_float[expanded_mask],
                pers_gt_float[expanded_mask],
            )


@torch.no_grad()
def finalize_mover360_test_metrics(module):
    if not getattr(module, "_mover360_test_metrics_ready", False):
        return

    test_metrics = {}
    for key, metric in module.eval_metrics.items():
        if not getattr(metric, "_update_called", False):
            continue
        test_metrics[key] = _metric_value_to_python(metric.compute())

    if not test_metrics:
        return

    trainer = getattr(module, "trainer", None)
    is_global_zero = trainer is None or trainer.is_global_zero
    if not is_global_zero:
        return

    if wandb.run is not None:
        wandb.summary.update(test_metrics)
    logger = getattr(module, "logger", None)
    experiment = getattr(logger, "experiment", None) if logger is not None else None
    if experiment is not None and hasattr(experiment, "log"):
        metrics_table = wandb.Table(columns=list(test_metrics.keys()), data=[list(test_metrics.values())])
        experiment.log({"test/metrics": metrics_table})


class EvalPanoGen(PanoBase):
    def __init__(
            self,
            log_test_samples: int = 0,
            pano_height: int = 512,
            data: str = None,
            test_perspective_size: int = DEFAULT_PERSPECTIVE_SIZE,
            test_perspective_context_scale: float = DEFAULT_CONTEXT_SCALE,
            test_perspective_min_fov: float = DEFAULT_MIN_FOV_DEGREES,
            test_perspective_max_fov: float = DEFAULT_MAX_FOV_DEGREES,
            test_perspective_mask_threshold: float = DEFAULT_MASK_THRESHOLD,
            clip_score_model_name: str = DEFAULT_CLIP_SCORE_MODEL_NAME,
            dinov3_model_name: str = DEFAULT_DINOV3_MODEL_NAME,
            dinov3_cache_dir: str | None = DEFAULT_DINOV3_CACHE_DIR,
            dinov3_local_files_only: bool = False,
            dinov3_hf_token: str | None = None,
            dinov3_batch_size: int = 2,
            dinov3_device: str | None = None,
            huggingface_cache: str | None = DEFAULT_HUGGINGFACE_CACHE,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        ensure_mover360_test_metrics(self, pano_height=pano_height, log_test_samples=log_test_samples)

    def load_from_checkpoint(self, *args, **kwargs):
        if "strict" not in kwargs:
            kwargs["strict"] = False
        return super().load_from_checkpoint(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        if "strict" not in kwargs:
            kwargs["strict"] = False
        return super().load_state_dict(*args, **kwargs)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch = self.trainer.strategy.batch_to_device(batch)
        update_mover360_test_metrics(self, batch, batch["pano_pred"], batch_idx)

    @torch.no_grad()
    def on_test_end(self):
        finalize_mover360_test_metrics(self)
