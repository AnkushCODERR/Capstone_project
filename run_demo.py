"""
SMART DEPTH VISION — Demo Runner
Processes mentor's dataset (rgbd-scenes-v2) and saves visual results.
Run this file: python run_demo.py
"""

import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pipeline import load_models, process_image

# ─────────────────────────────────────────────────────────
#  CONFIGURATION — Update paths here if needed
# ─────────────────────────────────────────────────────────
DATASET_PATH  = r"C:\Users\Ankush\Desktop\Btech\Capstone\rgbd-scenes-v2\imgs"
OUTPUT_PATH   = r"C:\Users\Ankush\Desktop\Btech\Capstone\results"
SCENES        = None   # None = all scenes, or set like ["scene_01","scene_02"]
IMAGES_PER_SCENE = 10  # How many images to process per scene (keep low for demo speed)
SHOW_LIVE     = False  # Set True to show each result live while processing


def get_scene_pairs(scene_path):
    """
    Find all (color, depth) image pairs in a scene folder.
    Pattern: XXXXX-color.png paired with XXXXX-depth.png
    """
    scene_path = Path(scene_path)
    color_files = sorted(scene_path.glob("*-color.png"))
    pairs = []
    for color_file in color_files:
        depth_file = scene_path / color_file.name.replace("-color.png", "-depth.png")
        if depth_file.exists():
            pairs.append((color_file, depth_file))
    return pairs


def draw_summary_card(scene_name, stats_all, w=640, h=200):
    """Create a summary statistics card image for a scene."""
    card = np.zeros((h, w, 3), dtype=np.uint8)
    card[:] = (30, 30, 30)

    total = sum(len(s) for s in stats_all)
    total_3d = sum(1 for s in stats_all for obj in s if obj["dimension"] == "3D")
    total_2d = total - total_3d

    cv2.putText(card, f"Scene: {scene_name}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(card, f"Images processed : {len(stats_all)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(card, f"Total detections : {total}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(card, f"3D objects       : {total_3d}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
    cv2.putText(card, f"2D objects       : {total_2d}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 1)

    # Bar chart
    bar_x = 380
    if total > 0:
        bar_w = 220
        frac_3d = total_3d / total
        cv2.rectangle(card, (bar_x, 110), (bar_x + bar_w, 140), (60, 60, 60), -1)
        cv2.rectangle(card, (bar_x, 110),
                      (bar_x + int(bar_w * frac_3d), 140), (0, 200, 0), -1)
        cv2.rectangle(card, (bar_x + int(bar_w * frac_3d), 110),
                      (bar_x + bar_w, 140), (0, 80, 200), -1)
        cv2.putText(card, f"3D: {frac_3d*100:.0f}%  2D: {(1-frac_3d)*100:.0f}%",
                    (bar_x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return card


def main():
    print("=" * 55)
    print("       SMART DEPTH VISION — Demo Runner")
    print("=" * 55)

    # ── Validate dataset path ──────────────────────────────
    dataset_root = Path(DATASET_PATH)
    if not dataset_root.exists():
        print(f"\n[ERROR] Dataset path not found:\n  {DATASET_PATH}")
        print("  Please update DATASET_PATH in run_demo.py")
        return

    # ── Create output folder ───────────────────────────────
    output_root = Path(OUTPUT_PATH)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[Output] Results will be saved to:\n  {OUTPUT_PATH}\n")

    # ── Load models once ───────────────────────────────────
    print("[Models] Loading YOLOv8 + MiDaS...")
    yolo, midas, transform, device = load_models()

    # ── Find scenes ────────────────────────────────────────
    all_scenes = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    if SCENES is not None:
        all_scenes = [d for d in all_scenes if d.name in SCENES]

    print(f"[Dataset] Found {len(all_scenes)} scenes to process")
    print(f"[Config]  Processing {IMAGES_PER_SCENE} images per scene\n")

    global_stats = {}
    total_time_start = time.time()

    # ── Process each scene ─────────────────────────────────
    for scene_dir in all_scenes:
        scene_name = scene_dir.name
        print(f"─── Processing: {scene_name} ───────────────────────")

        # Output folder for this scene
        scene_output = output_root / scene_name
        scene_output.mkdir(exist_ok=True)

        # Get image pairs
        pairs = get_scene_pairs(scene_dir)
        if not pairs:
            print(f"  [Skip] No color-depth pairs found in {scene_name}")
            continue

        # Pick evenly spaced samples
        step = max(1, len(pairs) // IMAGES_PER_SCENE)
        selected_pairs = pairs[::step][:IMAGES_PER_SCENE]

        print(f"  Total pairs: {len(pairs)} | Sampling: {len(selected_pairs)}")

        scene_stats = []
        scene_start = time.time()

        for idx, (color_path, depth_path) in enumerate(
                tqdm(selected_pairs, desc=f"  {scene_name}", ncols=70)):

            t0 = time.time()
            result_img, stats = process_image(
                rgb_path=color_path,
                yolo=yolo,
                midas=midas,
                transform=transform,
                device=device,
                depth_path=depth_path   # pass GT depth for 3-panel view
            )
            t1 = time.time()

            if result_img is None:
                continue

            scene_stats.append(stats)

            # Save result image
            out_filename = f"{color_path.stem}_result.jpg"
            out_path = scene_output / out_filename
            cv2.imwrite(str(out_path), result_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            if SHOW_LIVE:
                display = cv2.resize(result_img, (1280, 400))
                cv2.imshow("Smart Depth Vision", display)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break

        # Scene summary card
        summary_card = draw_summary_card(scene_name, scene_stats)
        cv2.imwrite(str(scene_output / "00_summary.jpg"), summary_card)

        # Scene stats to JSON
        scene_time = time.time() - scene_start
        total_det = sum(len(s) for s in scene_stats)
        total_3d  = sum(1 for s in scene_stats for o in s if o["dimension"] == "3D")
        total_2d  = total_det - total_3d

        global_stats[scene_name] = {
            "images_processed": len(scene_stats),
            "total_detections": total_det,
            "3D_objects": total_3d,
            "2D_objects": total_2d,
            "processing_time_sec": round(scene_time, 2)
        }

        print(f"  Done! Detections: {total_det}  |  3D: {total_3d}  2D: {total_2d}  "
              f"|  Time: {scene_time:.1f}s")

    if SHOW_LIVE:
        cv2.destroyAllWindows()

    # ── Save global summary JSON ───────────────────────────
    summary_path = output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(global_stats, f, indent=2)

    total_time = time.time() - total_time_start
    total_images = sum(v["images_processed"] for v in global_stats.values())
    total_det_all = sum(v["total_detections"] for v in global_stats.values())

    print("\n" + "=" * 55)
    print("          PROCESSING COMPLETE!")
    print("=" * 55)
    print(f"  Scenes processed   : {len(global_stats)}")
    print(f"  Images processed   : {total_images}")
    print(f"  Total detections   : {total_det_all}")
    print(f"  Total time         : {total_time:.1f}s")
    print(f"  Results saved to   : {OUTPUT_PATH}")
    print(f"  Summary JSON       : {summary_path}")
    print("=" * 55)
    print("\n  Open the results folder to see all output images!")


if __name__ == "__main__":
    main()
