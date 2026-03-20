#!/usr/bin/env python3
"""
Extract granite texture from UGRAN product photos.

Each photo contains a 60x30 cm granite slab (vertical rectangle, 2:1 aspect ratio)
on a light gray background with "UGRAN" logo at the bottom.

This script:
1. Detects the slab boundaries (dark stone on light background)
2. Crops only the top half of the slab to avoid the UGRAN logo
3. Result: square texture (30x30 cm proportions)
4. Saves to textures/ directory
"""

import os
import sys
import numpy as np
from PIL import Image


def find_slab_bounds(img):
    """Find the bounding box of the granite slab in the image.

    The slab is a dark rectangle on a light gray background.
    We detect it by finding where the image transitions from light to dark.
    """
    arr = np.array(img.convert('L'))  # grayscale
    h, w = arr.shape

    # The background is light gray (~200+), the slab is darker
    # Use adaptive threshold based on the image
    # Sample corners to determine background color
    corners = [
        arr[0:50, 0:50].mean(),
        arr[0:50, w-50:w].mean(),
        arr[h-50:h, 0:50].mean(),
        arr[h-50:h, w-50:w].mean(),
    ]
    bg_level = np.mean(corners)

    # Threshold: pixels significantly darker than background are part of the slab
    threshold = bg_level - 40

    # Create binary mask
    mask = arr < threshold

    # Find rows and columns that have significant dark content
    row_sums = mask.sum(axis=1)
    col_sums = mask.sum(axis=0)

    # Threshold for considering a row/col as part of the slab
    row_thresh = w * 0.05  # at least 5% of width must be dark
    col_thresh = h * 0.05  # at least 5% of height must be dark

    dark_rows = np.where(row_sums > row_thresh)[0]
    dark_cols = np.where(col_sums > col_thresh)[0]

    if len(dark_rows) == 0 or len(dark_cols) == 0:
        print("  WARNING: Could not detect slab boundaries, using center crop")
        # Fallback: assume slab is centered, ~40% width, ~65% height
        cx, cy = w // 2, h // 2
        sw, sh = int(w * 0.20), int(h * 0.32)
        return (cx - sw, cy - sh, cx + sw, cy + sh)

    top = dark_rows[0]
    bottom = dark_rows[-1]
    left = dark_cols[0]
    right = dark_cols[-1]

    # Add small padding inward to avoid edge artifacts (shadow, light bleed)
    pad_x = int((right - left) * 0.02)
    pad_y = int((bottom - top) * 0.02)

    left += pad_x
    right -= pad_x
    top += pad_y
    bottom -= pad_y

    return (left, top, right, bottom)


def extract_texture(input_path, output_path):
    """Extract square texture from a slab photo."""
    img = Image.open(input_path)
    print(f"Processing: {input_path}")
    print(f"  Image size: {img.size[0]}x{img.size[1]}")

    # Find the slab
    left, top, right, bottom = find_slab_bounds(img)
    slab_w = right - left
    slab_h = bottom - top
    print(f"  Detected slab: ({left}, {top}) to ({right}, {bottom})")
    print(f"  Slab size: {slab_w}x{slab_h} px")
    print(f"  Slab aspect ratio: {slab_h/slab_w:.2f} (expected ~2.0 for 60x30)")

    # The slab is 60x30 cm (height x width), aspect ratio ~2:1
    # We take the top half to get 30x30 cm (square), avoiding the UGRAN logo
    # The logo is at the bottom of the image, overlapping the lower part of the slab

    # Calculate the top half of the slab
    half_height = slab_h // 2

    # Crop the top square portion of the slab
    # Use the slab width as the side length, take from top
    square_side = min(slab_w, half_height)

    # Center the square crop horizontally within the slab
    crop_left = left + (slab_w - square_side) // 2
    crop_right = crop_left + square_side
    crop_top = top
    crop_bottom = top + square_side

    print(f"  Cropping square: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom})")
    print(f"  Output size: {square_side}x{square_side} px (30x30 cm proportions)")

    texture = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Save as high-quality image
    texture.save(output_path, quality=95)
    print(f"  Saved: {output_path}")
    print()

    return texture


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, "source_photos")
    output_dir = os.path.join(script_dir, "textures")

    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(extensions)
    ]

    if not images:
        print(f"No images found in {source_dir}/")
        print(f"Please add your UGRAN granite photos to the source_photos/ directory.")
        sys.exit(1)

    images.sort()
    print(f"Found {len(images)} image(s) to process\n")

    for filename in images:
        input_path = os.path.join(source_dir, filename)

        # Output filename: texture_<original_name>.jpg
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"texture_{name}.jpg")

        extract_texture(input_path, output_path)

    print(f"Done! {len(images)} texture(s) extracted to {output_dir}/")


if __name__ == "__main__":
    main()
