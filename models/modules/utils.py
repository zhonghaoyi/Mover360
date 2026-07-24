import torch
from einops import rearrange
import lightning as L
import tempfile
from PIL import Image
import wandb


def tensor_to_image(image):
    """
    Convert a PyTorch Tensor (shape: [..., c, h, w], dtype could be bfloat16/float16/float32/uint8)
    to a NumPy HWC uint8 image.
    """
    # 1. If bfloat16, convert to float32 first
    if image.dtype == torch.bfloat16 or image.dtype == torch.float16:
        image = image.to(torch.float32)

    # 2. Handle different data types and ranges
    if image.dtype != torch.uint8:
        # Check data range to decide how to process
        image_max = image.max()
        # If data is in [-1,1] range, first normalize to [0,1] then scale to [0,255]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round()
        
    # 3. Convert to uint8 and rearrange dimensions
    image = image.cpu().numpy().astype('uint8')
    image = rearrange(image, '... c h w -> ... h w c')
    return image



class WandbLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.TemporaryDirectory()

    def temp_wandb_image(self, image, prompt=None):
        if isinstance(image, torch.Tensor):
            image = tensor_to_image(image)
        
        # Ensure image has correct dimensions
        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)  # Remove batch dimension
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)  # Remove single channel dimension
        
        # Ensure image is 3D (H, W, C) or 2D (H, W)
        if image.ndim == 3 and image.shape[-1] not in [1, 3]:
            # If the last dimension is not a channel dimension, readjust
            if image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
        
        img_path = tempfile.NamedTemporaryFile(
            dir=self.temp_dir.name, suffix=".jpg", delete=False).name
        Image.fromarray(image.squeeze()).save(img_path)
        return wandb.Image(img_path, caption=prompt if prompt else None)

    def __del__(self):
        self.temp_dir.cleanup()
