"""
SMART DEPTH VISION — NYU Depth V2 Label Fixer
---------------------------------------------
NYU label images look completely BLACK because they are stored as
16-bit PNGs where pixel values represent CLASS IDs (0 to ~894).
Normal image viewers show these as black since they expect 0-255.

This script normalizes and colorizes them so they are actually visible.
Run: python fix_nyu_labels.py
"""

import cv2
import numpy as np
from pathlib import Path
import os

# ── UPDATE THESE PATHS ─────────────────────────────────────
NYU_LABELS_PATH = r"C:\Users\Ankush\Desktop\Btech\Capstone\Capstone_project\extracted\labels"
NYU_OUTPUT_PATH = r"C:\Users\Ankush\Desktop\Btech\Capstone\Capstone_project\extracted\labels_fixed"
# ──────────────────────────────────────────────────────────

# Nice colormap for class visualization
COLORMAPS = {
    "jet":   cv2.COLORMAP_JET,
    "hsv":   cv2.COLORMAP_HSV,
    "turbo": cv2.COLORMAP_TURBO,
}


def fix_label_image(label_path, output_dir):
    """
    Fix a single NYU label image.
    Saves 3 versions: normalized grayscale, jet colormap, turbo colormap.
    """
    # Read as 16-bit (this is crucial — cv2.IMREAD_ANYDEPTH)
    label = cv2.imread(str(label_path), cv2.IMREAD_ANYDEPTH)

    if label is None:
        print(f"  [Skip] Cannot read: {label_path.name}")
        return

    # Print unique class count for first image
    unique_classes = np.unique(label)

    # Normalize to 0–255 for visualization
    label_norm = cv2.normalize(label, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    stem = label_path.stem

    # Save normalized grayscale
    cv2.imwrite(str(output_dir / f"{stem}_gray.png"), label_norm)

    # Save with jet colormap (good for segmentation visualization)
    jet = cv2.applyColorMap(label_norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{stem}_jet.png"), jet)

    # Save with turbo colormap (modern, better contrast)
    turbo = cv2.applyColorMap(label_norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_dir / f"{stem}_turbo.png"), turbo)

    return len(unique_classes)


def main():
    print("=" * 50)
    print("   NYU Depth V2 — Label Visualizer Fix")
    print("=" * 50)

    labels_dir = Path(NYU_LABELS_PATH)
    output_dir = Path(NYU_OUTPUT_PATH)

    if not labels_dir.exists():
        print(f"\n[ERROR] Labels path not found:\n  {NYU_LABELS_PATH}")
        print("  Please update NYU_LABELS_PATH in fix_nyu_labels.py")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(labels_dir.glob("*.png")) + list(labels_dir.glob("*.jpg"))

    if not label_files:
        print(f"\n[ERROR] No image files found in:\n  {NYU_LABELS_PATH}")
        return

    print(f"\n[Found] {len(label_files)} label images")
    print(f"[Output] Saving fixed images to:\n  {NYU_OUTPUT_PATH}\n")

    for i, lf in enumerate(label_files[:50]):  # Process first 50 as sample
        n_classes = fix_label_image(lf, output_dir)
        if i == 0 and n_classes:
            print(f"  First image: {n_classes} unique class labels found")
        if i % 10 == 0:
            print(f"  Processed {i+1}/{min(50, len(label_files))}...")

    print(f"\n[Done] Fixed label images saved to:\n  {NYU_OUTPUT_PATH}")
    print("  Each label now has 3 versions: _gray, _jet, _turbo")
    print("\n  WHY they were black:")
    print("  NYU labels are 16-bit PNGs with class IDs (0-894)")
    print("  Normal viewers show them black because they expect 0-255 range")
    print("  This script normalizes and colourizes them properly.")


if __name__ == "__main__":
    main()
