import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

def add_scratch_overlay(img_path, num_scratches=8, min_len=0.3, max_len=0.8, min_width=1, max_width=3):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Create transparent overlay and mask
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask)

    for _ in range(num_scratches):
        # Random start and end points
        length = random.uniform(min_len, max_len) * min(w, h)
        angle = random.uniform(0, 2 * np.pi)
        x0 = random.randint(0, w-1)
        y0 = random.randint(0, h-1)
        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))
        width = random.randint(min_width, max_width)
        # Random brightness (white to light gray)
        brightness = random.randint(180, 255)
        color = (brightness, brightness, brightness, random.randint(120, 200))
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
        mask_draw.line([(x0, y0), (x1, y1)], fill=255, width=width)

    # Optionally blur overlay and mask for realism
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    mask = mask.point(lambda p: 255 if p > 32 else 0)

    # Composite overlay onto image
    img_rgba = img.convert('RGBA')
    degraded_img = Image.alpha_composite(img_rgba, overlay).convert('RGB')

    return degraded_img, mask

