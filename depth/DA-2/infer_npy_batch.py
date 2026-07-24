import argparse
import logging
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from runtime_bootstrap import ensure_runtime_env, preload_bundled_cuda_libs

ensure_runtime_env()
preload_bundled_cuda_libs()

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from da2.model.spherevit import SphereViT
from da2.utils.base import load_config
from da2.utils.io import read_cv2_image, torch_transform
from da2.utils.vis import colorize_distance
from tqdm.auto import tqdm

# Precompute the MovieRenders_PredictDepth condition maps for a data folder:
#   python depth/DA-2/infer_npy_batch.py --input_root data/UE5_data/Test_all_new/Saved
# Multi-GPU:
#   torchrun --standalone --nproc_per_node=4 depth/DA-2/infer_npy_batch.py \
#       --input_root data/UE5_data/Test_all_new/Saved --batch_size 2


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DA-2 on all images under an input directory with multi-GPU sharding "
            "and batched inference."
        )
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "Root directory containing input images. If it contains a "
            "MovieRenders_Normal child, that child is used automatically. "
            "Images are discovered recursively."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root directory for predicted depth .npy outputs. Defaults to a "
            "MovieRenders_PredictDepth directory next to the resolved "
            "MovieRenders_Normal input root."
        ),
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default=None,
        help=(
            "Optional root directory for masks. Relative paths should mirror input_root. "
            "Missing masks fall back to all-valid."
        ),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/infer.json",
        help="Path to the DA-2 config file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help=(
            "Per-process batch size. Images with different resolutions are "
            "automatically split into separate batches."
        ),
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Also save a colorized depth visualization next to each .npy file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs. By default, completed samples are skipped.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of sorted images to process. Useful for debugging.",
    )
    return parser.parse_args()


def resolve_user_path(path_str):
    return Path(path_str).expanduser().resolve()


def resolve_config_path(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def read_mask(mask_path, image_shape):
    if mask_path is None:
        return np.ones(image_shape[1:], dtype=bool)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {mask_path}")
    return mask > 0


def list_image_paths(input_root):
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    image_paths = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {input_root}")
    return image_paths


def resolve_image_root(input_root):
    normal_root = input_root / "MovieRenders_Normal"
    if normal_root.is_dir():
        return normal_root
    return input_root


def resolve_output_paths(image_path, input_root, output_root, save_vis):
    relative_path = image_path.relative_to(input_root)
    npy_path = output_root / relative_path.with_suffix(".npy")
    if save_vis:
        vis_path = npy_path.with_name(f"{npy_path.stem}_vis.png")
    else:
        vis_path = None
    return npy_path, vis_path


def prepare_runtime(config_path, project_dir):
    logging.basicConfig(
        format="%(asctime)s --> %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    config = load_config(str(config_path))
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=config["accelerator"]["timeout"]))
    accelerator = Accelerator(
        gradient_accumulation_steps=config["accelerator"]["accumulation_nsteps"],
        mixed_precision=config["accelerator"]["mixed_precision"],
        log_with=config["accelerator"]["report_to"],
        project_config=ProjectConfiguration(project_dir=str(project_dir)),
        kwargs_handlers=[kwargs],
    )
    logger = get_logger(__name__, log_level="INFO")
    config["env"]["logger"] = logger
    set_seed(config["env"]["seed"])
    if config["env"]["verbose"]:
        logger.info(f"Config: {config_path}")
        logger.info(f"Running on device: {accelerator.device}")
    return config, accelerator


def configure_local_cuda_device():
    if not torch.cuda.is_available():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            "torchrun requested more local processes than visible GPUs. "
            f"LOCAL_RANK={local_rank}, visible_gpu_count={device_count}, "
            f"CUDA_VISIBLE_DEVICES={visible_devices}"
        )
    torch.cuda.set_device(local_rank)


def load_model_for_inference(config, accelerator):
    logger = config["env"]["logger"]
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            logger.info("Loading DA-2 weights on main process")
            model = SphereViT.from_pretrained("haodongli/DA-2", config=config)
        else:
            model = None
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            model = SphereViT.from_pretrained("haodongli/DA-2", config=config)
    else:
        model = SphereViT.from_pretrained("haodongli/DA-2", config=config)

    model = model.to(accelerator.device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if config["env"]["verbose"]:
        logger.info(f"Model's dtype: {next(model.parameters()).dtype}.")
    config["spherevit"]["dtype"] = next(model.parameters()).dtype
    return model


def infer_distance_batch(model, images, device_type):
    if device_type in {"cpu", "mps"}:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device_type)
    with autocast_ctx, torch.no_grad():
        distances = model(images).detach().float().cpu().numpy()
    if distances.ndim == 2:
        distances = distances[None, ...]
    return distances


def build_output_record(image_path, input_root, output_root, mask_root, save_vis):
    npy_path, vis_path = resolve_output_paths(
        image_path=image_path,
        input_root=input_root,
        output_root=output_root,
        save_vis=save_vis,
    )
    mask_path = None
    if mask_root is not None:
        mask_path = mask_root / image_path.relative_to(input_root)

    return {
        "image_path": image_path,
        "mask_path": mask_path,
        "npy_path": npy_path,
        "vis_path": vis_path,
    }


def load_image_record(record, save_vis):
    try:
        cv2_image = read_cv2_image(str(record["image_path"]))
    except Exception as exc:
        raise RuntimeError(f"Failed to read image: {record['image_path']}") from exc

    image = torch_transform(cv2_image).astype(np.float32, copy=False)
    record["image"] = image
    if save_vis:
        record["mask"] = read_mask(record["mask_path"], image.shape)
    return record


def outputs_exist(record, save_vis):
    if not record["npy_path"].exists():
        return False
    if save_vis and record["vis_path"] is not None and not record["vis_path"].exists():
        return False
    return True


def save_predictions(records, distances, save_vis):
    for record, distance in zip(records, distances, strict=True):
        record["npy_path"].parent.mkdir(parents=True, exist_ok=True)
        np.save(record["npy_path"], distance)
        if save_vis and record["vis_path"] is not None:
            colorize_distance(distance.copy(), record["mask"]).save(record["vis_path"])


def process_batch(records, model, model_dtype, device, device_type, save_vis):
    batch = np.stack([record["image"] for record in records], axis=0)
    batch = torch.from_numpy(batch).to(device=device, dtype=model_dtype)
    distances = infer_distance_batch(model, batch, device_type)
    save_predictions(records, distances, save_vis)
    return len(records)


def summarize_counts(accelerator, assigned_count, written_count, skipped_count):
    local_counts = torch.tensor(
        [assigned_count, written_count, skipped_count],
        device=accelerator.device,
        dtype=torch.long,
    )
    reduced_counts = accelerator.reduce(local_counts, reduction="sum")
    return [int(value) for value in reduced_counts.cpu().tolist()]


def create_progress_bar(accelerator, total_count):
    if total_count == 0:
        return None

    if accelerator.num_processes > 1:
        desc = f"Rank {accelerator.process_index}"
        position = accelerator.process_index
        leave = accelerator.process_index == 0
    else:
        desc = "Predicting depth"
        position = 0
        leave = True

    return tqdm(
        total=total_count,
        desc=desc,
        position=position,
        dynamic_ncols=True,
        leave=leave,
    )


def refresh_progress_bar(progress_bar, written_count, skipped_count):
    if progress_bar is None:
        return
    progress_bar.set_postfix(written=written_count, skipped=skipped_count)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    configure_local_cuda_device()
    input_root = resolve_user_path(args.input_root)
    image_root = resolve_image_root(input_root)
    if args.output_root is not None:
        output_root = resolve_user_path(args.output_root)
    elif image_root.name == "MovieRenders_Normal":
        output_root = image_root.parent / "MovieRenders_PredictDepth"
    else:
        raise ValueError(
            "--output_root is required when the input root does not resolve to "
            "a MovieRenders_Normal directory."
        )
    mask_root = resolve_user_path(args.mask_root) if args.mask_root else None
    config_path = resolve_config_path(args.config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_paths(image_root)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print("No images selected after applying --limit.")
        return

    config, accelerator = prepare_runtime(
        config_path=config_path,
        project_dir=output_root / "_accelerate_runtime",
    )
    model = load_model_for_inference(config, accelerator).eval()
    model_dtype = config["spherevit"]["dtype"]

    if accelerator.is_main_process:
        if image_root != input_root:
            config["env"]["logger"].info(f"Input root: {input_root}")
            config["env"]["logger"].info(f"Using image root: {image_root}")
        config["env"]["logger"].info(f"Found {len(image_paths)} images under {image_root}")
        config["env"]["logger"].info(f"Predictions will be written to {output_root}")
        config["env"]["logger"].info(
            "Visible GPUs: "
            f"{torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')})"
        )

    assigned_paths = image_paths[accelerator.process_index :: accelerator.num_processes]
    pending_records = []
    pending_shape = None
    written_count = 0
    skipped_count = 0
    progress_bar = create_progress_bar(accelerator, len(assigned_paths))
    refresh_progress_bar(progress_bar, written_count, skipped_count)

    for image_path in assigned_paths:
        record = build_output_record(
            image_path=image_path,
            input_root=image_root,
            output_root=output_root,
            mask_root=mask_root,
            save_vis=args.save_vis,
        )

        if not args.overwrite and outputs_exist(record, args.save_vis):
            skipped_count += 1
            if progress_bar is not None:
                progress_bar.update(1)
            refresh_progress_bar(progress_bar, written_count, skipped_count)
            continue

        record = load_image_record(record, save_vis=args.save_vis)

        image_shape = tuple(record["image"].shape)
        if pending_records and (
            len(pending_records) >= args.batch_size or image_shape != pending_shape
        ):
            processed_count = len(pending_records)
            written_count += process_batch(
                pending_records,
                model=model,
                model_dtype=model_dtype,
                device=accelerator.device,
                device_type=accelerator.device.type,
                save_vis=args.save_vis,
            )
            if progress_bar is not None:
                progress_bar.update(processed_count)
            refresh_progress_bar(progress_bar, written_count, skipped_count)
            pending_records = []
            pending_shape = None

        if not pending_records:
            pending_shape = image_shape
        pending_records.append(record)

    if pending_records:
        processed_count = len(pending_records)
        written_count += process_batch(
            pending_records,
            model=model,
            model_dtype=model_dtype,
            device=accelerator.device,
            device_type=accelerator.device.type,
            save_vis=args.save_vis,
        )
        if progress_bar is not None:
            progress_bar.update(processed_count)
        refresh_progress_bar(progress_bar, written_count, skipped_count)

    if progress_bar is not None:
        progress_bar.close()

    accelerator.wait_for_everyone()
    total_assigned, total_written, total_skipped = summarize_counts(
        accelerator=accelerator,
        assigned_count=len(assigned_paths),
        written_count=written_count,
        skipped_count=skipped_count,
    )

    if accelerator.is_main_process:
        config["env"]["logger"].info(
            "Finished batched inference. "
            f"Assigned={total_assigned}, written={total_written}, skipped={total_skipped}."
        )


if __name__ == "__main__":
    main()
