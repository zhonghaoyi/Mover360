import argparse
import logging
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import cv2
import numpy as np
from runtime_bootstrap import ensure_runtime_env, preload_bundled_cuda_libs

ensure_runtime_env()
preload_bundled_cuda_libs()

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from da2.utils.base import load_config
from da2.utils.io import read_cv2_image, tensorize, torch_transform
from da2.model.spherevit import SphereViT
from da2.utils.vis import colorize_distance


MODEL_DTYPE = torch.float32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DA-2 on a single image and save the predicted depth map as a .npy file."
    )
    parser.add_argument("--image_path", type=str, default='/vol/graphics-solar/zhonghaoy/360mover/depth/DA-2/assets/demos/CineCameraActor1_1_0004.png', help="Path to the input panorama image.")
    parser.add_argument("--mask_path", type=str, default=None, help="Optional path to a binary mask image.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/infer.json",
        help="Path to the DA-2 config file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='/vol/graphics-solar/zhonghaoy/360mover/depth/DA-2/output',
        help="Output .npy path, or an output directory where <image_name>.npy will be written.",
    )
    parser.add_argument(
        "--save_vis",
        default=True,
        action="store_true",
        help="Also save a colorized depth visualization next to the .npy output.",
    )
    parser.add_argument(
        "--vis_path",
        type=str,
        default='/vol/graphics-solar/zhonghaoy/360mover/depth/DA-2/output/vis.png',
        help="Optional path for the colorized depth visualization.",
    )
    return parser.parse_args()


def read_mask(mask_path, image_shape):
    if mask_path is None:
        return np.ones(image_shape[1:], dtype=bool)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {mask_path}")
    return mask > 0


def resolve_output_paths(image_path, output_path, vis_path, save_vis):
    image_stem = Path(image_path).stem
    if output_path is None:
        npy_path = Path("output/depth_npy") / f"{image_stem}.npy"
    else:
        output_target = Path(output_path)
        if output_target.suffix.lower() == ".npy":
            npy_path = output_target
        else:
            npy_path = output_target / f"{image_stem}.npy"

    npy_path.parent.mkdir(parents=True, exist_ok=True)

    if vis_path is not None:
        vis_output_path = Path(vis_path)
    elif save_vis:
        vis_output_path = npy_path.with_name(f"{npy_path.stem}_vis.png")
    else:
        vis_output_path = None

    if vis_output_path is not None:
        vis_output_path.parent.mkdir(parents=True, exist_ok=True)

    return npy_path, vis_output_path


def prepare_runtime(config_path, project_dir):
    logging.basicConfig(
        format="%(asctime)s --> %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    config = load_config(config_path)
    config["accelerator"]["mixed_precision"] = "no"
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


def load_model(config, accelerator):
    model = SphereViT.from_pretrained("haodongli/DA-2", config=config)
    model = model.to(accelerator.device, MODEL_DTYPE)
    torch.cuda.empty_cache()
    model = accelerator.prepare(model)
    if accelerator.num_processes > 1:
        model = model.module
    config["spherevit"]["dtype"] = MODEL_DTYPE
    if config["env"]["verbose"]:
        config["env"]["logger"].info(f"Loaded model in {next(model.parameters()).dtype}.")
    return model.eval()


def load_single_image(image_path, mask_path, model_dtype, device):
    cv2_image = read_cv2_image(image_path)
    image = torch_transform(cv2_image)
    mask = read_mask(mask_path, image.shape)
    image = tensorize(image, model_dtype, device)
    return image, mask


def infer_distance(model, image, device_type):
    if MODEL_DTYPE == torch.float32 or device_type in {"cpu", "mps"}:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type=device_type, dtype=MODEL_DTYPE)
    with autocast_ctx, torch.no_grad():
        distance = model(image).detach().float().cpu().numpy().squeeze()
    return distance


def main():
    args = parse_args()
    npy_path, vis_output_path = resolve_output_paths(
        image_path=args.image_path,
        output_path=args.output_path,
        vis_path=args.vis_path,
        save_vis=args.save_vis,
    )
    config, accelerator = prepare_runtime(args.config_path, project_dir=npy_path.parent)
    model = load_model(config, accelerator)
    image, mask = load_single_image(
        image_path=args.image_path,
        mask_path=args.mask_path,
        model_dtype=config["spherevit"]["dtype"],
        device=accelerator.device,
    )
    distance = infer_distance(model, image, accelerator.device.type)
    np.save(npy_path, distance)
    config["env"]["logger"].info(f"Saved depth npy to {npy_path}")

    if vis_output_path is not None:
        colorize_distance(distance.copy(), mask).save(vis_output_path)
        config["env"]["logger"].info(f"Saved depth visualization to {vis_output_path}")


if __name__ == "__main__":
    main()
