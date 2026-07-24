"""
FluxMultiModalProcessor: adapted for Flux2 Klein's single Qwen3 text encoder.

Flux2 Klein vs Flux1 processor differences:
  Flux1: CLIP tokenizer (pooled) + T5 tokenizer (sequence), dual text encoders
  Flux2: single Qwen3 tokenizer + Qwen3ForCausalLM encoder, multi-layer hidden state concatenation

The processor handles:
  1. Prompt construction for CFG (conditional / unconditional / image-only)
  2. Image preprocessing for VAE encoding
"""

import re
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def crop_image(pil_image, max_image_size):
    """
    Crop the image so that its height and width does not exceed `max_image_size`, while ensuring both the height and
    width are multiples of 16.
    """
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % 16) // 2
    crop_y2 = arr.shape[0] % 16 - crop_y1
    crop_x1 = (arr.shape[1] % 16) // 2
    crop_x2 = arr.shape[1] % 16 - crop_x1
    arr = arr[crop_y1 : arr.shape[0] - crop_y2, crop_x1 : arr.shape[1] - crop_x2]
    return Image.fromarray(arr)


class FluxMultiModalProcessor:
    """
    Multimodal processor for Flux2 Klein.

    Flux2 Klein uses a single Qwen3 text encoder. This processor constructs
    the prompt lists for CFG and handles image preprocessing. The actual text
    encoding (Qwen3 tokenization + forward) is done in PanoGenerator.encode_prompt.
    """

    def __init__(self, tokenizer, max_image_size: int = 1024):
        self.tokenizer = tokenizer  # Qwen2TokenizerFast
        self.max_image_size = max_image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: crop_image(pil_image, max_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        return self.image_transform(image)

    def _strip_image_tags(self, text):
        """Remove <img><|image_N|></img> tags from text for clean tokenization."""
        cleaned = re.sub(r'<img><\|image_\d+\|></img>\s*', '', text).strip()
        return cleaned if cleaned else text

    def __call__(
        self,
        instructions: List[str],
        input_images: Optional[List[List]] = None,
        height: int = 1024,
        width: int = 1024,
        negative_prompt: str = "",
        use_img_cfg: bool = True,
        num_images_per_prompt: int = 1,
        mode: str = "train",
    ) -> Dict:
        """
        Clean prompts for Flux2 Klein and preprocess images.

        Returns a dict with:
          - "prompts": list of prompt strings (ready for Qwen3 encoding)
          - "input_pixel_values": list of preprocessed image tensors
        """
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]

        batch_size = len(instructions)

        clean_prompts = [self._strip_image_tags(p) for p in instructions]

        pixel_values = []
        if input_images is not None:
            for img_list in input_images:
                if img_list is not None:
                    processed = [self.process_image(x) for x in img_list if x is not None]
                    if processed:
                        pixel_values.extend([x.unsqueeze(0) for x in processed])

        return {
            # The official Flux2 Klein pipeline encodes unconditional prompts separately
            # during CFG, so the processor always returns only cleaned conditional prompts.
            "prompts": clean_prompts,
            "input_pixel_values": pixel_values,
        }
