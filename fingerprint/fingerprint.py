import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import random
import math


def create_fingerprint_ridges(width, height):
    # Create a base image for ridges
    ridges = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(ridges)

    center_x, center_y = width // 2, height // 2

    # Create swirl patterns (like fingerprints) - increased number of swirls
    for i in range(random.randint(3, 5)):  # Increased number of swirls: 3-5
        start_angle = random.uniform(0, 2 * math.pi)
        swirl_center_x = center_x + random.randint(-width // 4, width // 4)
        swirl_center_y = center_y + random.randint(-height // 4, height // 4)

        # Draw curved lines around the center with proper density
        for r in range(6, min(width, height) // 2, 6):  # Better spacing between ridges
            points = []
            for a in range(0, 360, 8):  # Increased density: 8 degree steps
                angle = math.radians(a)
                # Add some randomness to the radius
                var_r = r + random.randint(-2, 2)
                x = swirl_center_x + var_r * math.cos(angle + start_angle)
                y = swirl_center_y + var_r * math.sin(angle + start_angle)

                # Ensure points are within bounds
                x = max(0, min(width - 1, int(x)))
                y = max(0, min(height - 1, int(y)))
                points.append((x, y))

            # Draw the curve with higher frequency
            if len(points) > 1 and random.random() < 0.8:  # 80% chance to draw this curve
                draw.line(points, fill=255, width=1)

    return ridges


def add_fingerprint_smudge(img_path, num_prints=None):
    if num_prints is None:
        num_prints = random.randint(2, 3)  # Quantity: 2-3 smudges

    img = Image.open(img_path).convert("RGBA")
    w_img, h_img = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    label_mask = Image.new("L", img.size, 0)

    for _ in range(num_prints):
        # Vary sizes: small, medium, and large
        size_type = random.choice(["small", "medium", "medium", "large"])  # More medium for balance

        if size_type == "small":
            w = random.randint(100, 140)
            h = random.randint(120, 160)  # More oval shape for fingertips
        elif size_type == "medium":
            w = random.randint(140, 180)
            h = random.randint(160, 200)  # More oval shape for fingertips
        else:  # large
            w = random.randint(180, 220)
            h = random.randint(200, 240)  # More oval shape for fingertips

        # Create fingerprint pattern
        ridges = create_fingerprint_ridges(w, h)

        # Create a base for the smudge with more realistic fingertip shape
        fp = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(fp)

        # Create a more realistic fingertip shape (not perfect oval)
        # Draw an oval with some irregularity to mimic actual fingertip
        padding_x = random.randint(8, 15)
        padding_y = random.randint(5, 12)

        # Add some randomness to make it less perfect
        oval_bbox = (
            padding_x + random.randint(-3, 3),
            padding_y + random.randint(-3, 3),
            w - padding_x + random.randint(-3, 3),
            h - padding_y + random.randint(-3, 3)
        )
        draw.ellipse(oval_bbox, fill=255)

        # Combine ridges with the base shape
        fp = ImageChops.multiply(fp, ridges)

        # Apply blur to create a smudge effect but preserve ridge details
        fp = fp.filter(ImageFilter.GaussianBlur(random.uniform(1.8, 2.8)))  # Slightly more blur

        # Normalize contrast with reduced intensity
        arr = np.array(fp, dtype=np.float32)
        if np.ptp(arr) > 0:  # Avoid division by zero
            arr = (arr - arr.min()) / np.ptp(arr) * 200  # Reduced intensity (200)
        fp = Image.fromarray(arr.astype(np.uint8))

        # Convert to RGBA with lighter gray tint for less intensity
        tint_choices = [
            (90, 90, 90),  # Lighter gray
            (100, 100, 100),  # Slightly lighter gray
            (80, 80, 80),  # Medium gray
            (110, 110, 110),  # Light gray
        ]
        tint = random.choice(tint_choices)

        fp_r = fp.point(lambda p: int(tint[0] * (p / 255.0)))
        fp_g = fp.point(lambda p: int(tint[1] * (p / 255.0)))
        fp_b = fp.point(lambda p: int(tint[2] * (p / 255.0)))

        # Reduce alpha slightly to make smudges less intense
        fp_a = fp.point(lambda p: int(min(255, p * random.uniform(1.2, 1.6))))  # Reduced alpha

        fp_rgba = Image.merge("RGBA", (fp_r, fp_g, fp_b, fp_a))

        # Distort smudge with scaling + rotation
        angle = random.uniform(-30, 30)
        scale_x = random.uniform(0.9, 1.2)  # Less distortion
        scale_y = random.uniform(0.9, 1.2)  # Less distortion

        fp_rgba = fp_rgba.rotate(angle, expand=True, resample=Image.BICUBIC)
        new_w = max(1, int(fp_rgba.width * scale_x))
        new_h = max(1, int(fp_rgba.height * scale_y))
        fp_rgba = fp_rgba.resize((new_w, new_h), resample=Image.BICUBIC)

        # Light blur for smeared effect (preserve ridge details)
        fp_rgba = fp_rgba.filter(ImageFilter.GaussianBlur(random.uniform(1.0, 1.8)))

        # Paste on overlay
        paste_x = random.randint(0, max(1, w_img - fp_rgba.width))
        paste_y = random.randint(0, max(1, h_img - fp_rgba.height))

        # Create a temporary overlay for this smudge
        temp_overlay = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        temp_overlay.paste(fp_rgba, (paste_x, paste_y), fp_rgba)
        overlay = Image.alpha_composite(overlay, temp_overlay)

        # Update label (binary mask of smudge area)
        # Extract alpha channel and threshold it
        alpha_chan = fp_rgba.split()[-1]
        binary_mask = alpha_chan.point(lambda p: 255 if p > 30 else 0)

        # Create a temporary label mask for this smudge
        temp_label = Image.new("L", label_mask.size, 0)
        temp_label.paste(binary_mask, (paste_x, paste_y), binary_mask)

        # Combine with existing label mask using ImageChops.lighter (keeps the maximum value)
        label_mask = ImageChops.lighter(label_mask, temp_label)

    # Final composite
    combined = Image.alpha_composite(img, overlay).convert("RGB")

    # Ensure label mask is binary (0 or 255)
    label_mask = label_mask.point(lambda p: 255 if p > 0 else 0)

    return combined, label_mask