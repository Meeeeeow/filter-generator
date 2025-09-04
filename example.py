from __future__ import absolute_import

import os
from raindrop.dropgenerator import generateDrops
from raindrop.config import cfg
from fingerprint.fingerprint import add_fingerprint_smudge   # ✅ import fingerprints
from dirt.dirt import add_dirt_overlay
from scratch.scratch import add_scratch_overlay
from lensflare.lensflare import add_lensflare_overlay

from PIL import Image
import numpy as np


# def restore():
#     from models.SwinIR.restore import restore_with_swinir
#     restore_with_swinir()

def process_raindrops(image_folder_path, outputimg_folder_path, outputlabel_folder_path):
    """Process images with raindrop generation"""
    for file_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, file_name)

        output_image, output_label = generateDrops(image_path, cfg)

        # Save image
        save_path = os.path.join(outputimg_folder_path, file_name)
        output_image.save(save_path)

        # Save label
        save_path = os.path.join(outputlabel_folder_path, file_name)
        output_label = np.array(output_label)
        if not isinstance(output_label, Image.Image):
            output_label_img = Image.fromarray((output_label * 255).astype(np.uint8))
        else:
            output_label_img = Image.fromarray((np.array(output_label) * 255).astype(np.uint8))

        output_label_img.save(save_path)


def process_fingerprints(image_folder_path, outputimg_folder_path, outputlabel_folder_path):
    """Process images with fingerprint generation"""
    for file_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, file_name)

        output_image, output_label = add_fingerprint_smudge(image_path)

        # Save image
        save_path = os.path.join(outputimg_folder_path, file_name)
        output_image.save(save_path)

        # Save label (mask) directly as returned (no conversion)
        save_path = os.path.join(outputlabel_folder_path, file_name)
        output_label.save(save_path)


def process_dirt(image_folder_path, outputimg_folder_path, outputlabel_folder_path):
    """Process images with dirt overlay generation"""
    for file_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, file_name)

        output_image, output_label = add_dirt_overlay(image_path)

        # Save image
        save_path = os.path.join(outputimg_folder_path, file_name)
        output_image.save(save_path)

        # Save label (mask) directly as returned
        save_path = os.path.join(outputlabel_folder_path, file_name)
        output_label.save(save_path)


def process_scratch(image_folder_path, outputimg_folder_path, outputlabel_folder_path):
    """Process images with scratch overlay generation"""
    for file_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, file_name)

        output_image, output_label = add_scratch_overlay(image_path)

        # Save image
        save_path = os.path.join(outputimg_folder_path, file_name)
        output_image.save(save_path)

        # Save label (mask) directly as returned
        save_path = os.path.join(outputlabel_folder_path, file_name)
        output_label.save(save_path)


def process_lensflare(image_folder_path, outputimg_folder_path, outputlabel_folder_path):
    """Process images with lens flare overlay generation"""
    for file_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, file_name)

        output_image, output_label = add_lensflare_overlay(image_path)

        # Save image
        save_path = os.path.join(outputimg_folder_path, file_name)
        output_image.save(save_path)

        # Save label (mask) directly as returned
        save_path = os.path.join(outputlabel_folder_path, file_name)
        output_label.save(save_path)


def main():
    # Input images
    image_folder_path = "Images"

    # --- Raindrops ---
    raindrop_img_folder = "Output_image/raindrops"
    raindrop_label_folder = "Output_label/raindrops"
    os.makedirs(raindrop_img_folder, exist_ok=True)
    os.makedirs(raindrop_label_folder, exist_ok=True)
    process_raindrops(image_folder_path, raindrop_img_folder, raindrop_label_folder)

    # --- Fingerprints ---
    fingerprint_img_folder = "Output_image/fingerprints"
    fingerprint_label_folder = "Output_label/fingerprints"
    os.makedirs(fingerprint_img_folder, exist_ok=True)
    os.makedirs(fingerprint_label_folder, exist_ok=True)
    process_fingerprints(image_folder_path, fingerprint_img_folder, fingerprint_label_folder)

    # --- Dirt Overlay ---
    dirt_img_folder = "Output_image/dirt"
    dirt_label_folder = "Output_label/dirt"
    os.makedirs(dirt_img_folder, exist_ok=True)
    os.makedirs(dirt_label_folder, exist_ok=True)
    process_dirt(image_folder_path, dirt_img_folder, dirt_label_folder)

    # --- Scratch Overlay ---
    scratch_img_folder = "Output_image/scratch"
    scratch_label_folder = "Output_label/scratch"
    os.makedirs(scratch_img_folder, exist_ok=True)
    os.makedirs(scratch_label_folder, exist_ok=True)
    process_scratch(image_folder_path, scratch_img_folder, scratch_label_folder)

    # --- Lens Flare Overlay ---
    lensflare_img_folder = "Output_image/lensflare"
    lensflare_label_folder = "Output_label/lensflare"
    os.makedirs(lensflare_img_folder, exist_ok=True)
    os.makedirs(lensflare_label_folder, exist_ok=True)
    process_lensflare(image_folder_path, lensflare_img_folder, lensflare_label_folder)
    # Restore images using SwinIR
    # restore()


if __name__ == "__main__":
    main()

# Example usage (uncomment to use):
# show_fingerprint_and_mask('Restore_image/fingerprints', 'Output_label/fingerprints')
