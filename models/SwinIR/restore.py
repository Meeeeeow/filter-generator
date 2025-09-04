import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from .network_swinir import SwinIR
import torch
import numpy as np
from PIL import Image

def load_swinir_model(model_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def infer_patches(model, img_tensor, patch_size=128, overlap=16):
    # img_tensor: (1, 3, H, W)
    _, _, H, W = img_tensor.shape
    stride = patch_size - overlap
    output = torch.zeros(1, 3, H * 2, W * 2)  # upscale=2
    weight = torch.zeros(1, 3, H * 2, W * 2)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            patch = img_tensor[:, :, y:y_end, x:x_end]
            pad_h = patch_size - (y_end - y)
            pad_w = patch_size - (x_end - x)
            if pad_h > 0 or pad_w > 0:
                # Use 'replicate' if padding is too large for 'reflect'
                pad_mode = 'reflect'
                if (pad_h >= patch.shape[2]) or (pad_w >= patch.shape[3]):
                    pad_mode = 'replicate'
                patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h), mode=pad_mode)
            with torch.no_grad():
                out_patch = model(patch)
            out_patch = out_patch[:, :, : (y_end - y) * 2, : (x_end - x) * 2]
            output[:, :, y * 2 : y_end * 2, x * 2 : x_end * 2] += out_patch
            weight[:, :, y * 2 : y_end * 2, x * 2 : x_end * 2] += 1
    output /= weight
    return output

def restore_images_with_swinir(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            if max(img_tensor.shape[2:]) > 256:
                restored = infer_patches(model, img_tensor, patch_size=128, overlap=16)
            else:
                restored = model(img_tensor)
        restored_img = restored.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
        restored_img = (restored_img * 255).astype(np.uint8)
        Image.fromarray(restored_img).save(output_path)

def restore_with_swinir():
    model_path = 'models/SwinIR/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
    model = load_swinir_model(model_path)
    restore_images_with_swinir('Output_image/raindrops', 'Restore_image/raindrops', model)
    restore_images_with_swinir('Output_image/fingerprints', 'Restore_image/fingerprints', model)
