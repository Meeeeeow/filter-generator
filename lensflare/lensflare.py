import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
import math

def add_lensflare_overlay(img_path, num_ghosts=5, num_streaks=2):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Create transparent overlay and mask
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask)

    # 1. Pick a light source position near the edge/corner
    edge = random.choice(['top', 'bottom', 'left', 'right', 'topleft', 'topright', 'bottomleft', 'bottomright'])
    margin = int(0.08 * min(w, h))
    if edge == 'top':
        sx, sy = random.randint(margin, w-margin), margin
    elif edge == 'bottom':
        sx, sy = random.randint(margin, w-margin), h-margin
    elif edge == 'left':
        sx, sy = margin, random.randint(margin, h-margin)
    elif edge == 'right':
        sx, sy = w-margin, random.randint(margin, h-margin)
    elif edge == 'topleft':
        sx, sy = margin, margin
    elif edge == 'topright':
        sx, sy = w-margin, margin
    elif edge == 'bottomleft':
        sx, sy = margin, h-margin
    else:
        sx, sy = w-margin, h-margin

    # 2. Compute line from source to center
    cx, cy = w//2, h//2
    dx, dy = cx-sx, cy-sy

    # 3. Add a soft radial glow at the light source
    glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_radius = int(min(w, h) * random.uniform(0.18, 0.28))
    for i in range(6, 0, -1):
        alpha = int(60 * (i/6)**2)
        color = (255, 255, 220, alpha)
        r = int(glow_radius * i / 6)
        bbox = [sx-r, sy-r, sx+r, sy+r]
        glow_draw.ellipse(bbox, fill=color)
    overlay = Image.alpha_composite(overlay, glow)

    # 4. Add faint rings/arcs
    for i in range(random.randint(1, 3)):
        ring_radius = int(min(w, h) * random.uniform(0.18, 0.45))
        ring_width = int(ring_radius * random.uniform(0.04, 0.09))
        ring_alpha = random.randint(18, 38)
        ring_color = (255, 255, 200, ring_alpha)
        ring = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ring_draw = ImageDraw.Draw(ring)
        bbox = [sx-ring_radius, sy-ring_radius, sx+ring_radius, sy+ring_radius]
        ring_draw.ellipse(bbox, outline=ring_color, width=ring_width)
        overlay = Image.alpha_composite(overlay, ring)

    # 5. Place flare elements (ghosts) along the line
    for i in range(num_ghosts):
        t = (i+1)/(num_ghosts+1)
        fx = int(sx + dx * t + random.uniform(-0.04, 0.04)*w)
        fy = int(sy + dy * t + random.uniform(-0.04, 0.04)*h)
        max_radius = int(min(w, h) * random.uniform(0.04, 0.12) * (1-t+0.2))
        radius = int(max_radius * random.uniform(0.7, 1.1))
        # More color variation
        color_choices = [
            (255, 255, 200), (200, 220, 255), (180, 255, 220), (255, 200, 255), (255, 220, 180), (255, 255, 255)
        ]
        color = random.choice(color_choices) + (random.randint(30, 80),)
        bbox = [fx-radius, fy-radius, fx+radius, fy+radius]
        draw.ellipse(bbox, fill=color)
        mask_draw.ellipse(bbox, fill=255)

    # 6. Add a bright core at the light source
    core_radius = int(min(w, h) * random.uniform(0.04, 0.09))
    core_color = (255, 255, 230, random.randint(120, 200))
    bbox = [sx-core_radius, sy-core_radius, sx+core_radius, sy+core_radius]
    draw.ellipse(bbox, fill=core_color)
    mask_draw.ellipse(bbox, fill=255)

    # 7. Add soft, colored streaks
    angle = math.atan2(dy, dx)
    for _ in range(num_streaks):
        streak_angle = angle + random.uniform(-0.18, 0.18)
        length = int(min(w, h) * random.uniform(0.5, 1.0))
        width = random.randint(8, 18)
        x0 = int(cx - length/2 * math.cos(streak_angle))
        y0 = int(cy - length/2 * math.sin(streak_angle))
        x1 = int(cx + length/2 * math.cos(streak_angle))
        y1 = int(cy + length/2 * math.sin(streak_angle))
        # Multi-layered streaks
        for s in range(3, 0, -1):
            streak_alpha = int(40 * s / 3)
            streak_color = (255, 255, 220, streak_alpha)
            draw.line([(x0, y0), (x1, y1)], fill=streak_color, width=width + 2*s)
            mask_draw.line([(x0, y0), (x1, y1)], fill=255, width=width + 2*s)

    # 8. Blur overlay and mask for realism
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(8, 18)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(6, 12)))
    mask = mask.point(lambda p: 255 if p > 32 else 0)

    # 9. Composite overlay onto image
    img_rgba = img.convert('RGBA')
    degraded_img = Image.alpha_composite(img_rgba, overlay).convert('RGB')

    return degraded_img, mask
