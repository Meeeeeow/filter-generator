import numpy as np
from PIL import Image
import os

def add_dirt_overlay(img_path, dirt_texture_path=None, dirt_strength=0.6, threshold=40):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Use provided dirt texture or default
    if dirt_texture_path is None:
        dirt_texture_path = os.path.join(os.path.dirname(__file__), '../Images/img_1.png')
    dirt = Image.open(dirt_texture_path).convert('L')

    # Remove writing/watermark at the bottom (crop bottom 12%)
    orig_w, orig_h = dirt.size
    crop_h = int(orig_h * 0.88)  # Keep top 88%
    dirt = dirt.crop((0, 0, orig_w, crop_h))

    # Resize cropped dirt to match input image
    dirt = dirt.resize((w, h), Image.BICUBIC)

    # Normalize dirt to [0, 1]
    dirt_np = np.array(dirt, dtype=np.float32) / 255.0
    # Invert so black=dirty, white=clean
    dirt_mask_np = 1.0 - dirt_np
    # Strengthen the dirt effect
    dirt_mask_np = dirt_mask_np * dirt_strength

    # Apply dirt: darken the image where dirt is present
    img_np = np.array(img, dtype=np.float32) / 255.0
    degraded_np = img_np * (1.0 - dirt_mask_np[..., None])
    degraded_img = Image.fromarray(np.clip(degraded_np * 255, 0, 255).astype(np.uint8))

    # Create a binary mask for dirt
    mask = (dirt_mask_np > (threshold / 255.0)).astype(np.uint8) * 255
    dirt_mask_img = Image.fromarray(mask.astype(np.uint8), mode='L')

    return degraded_img, dirt_mask_img
