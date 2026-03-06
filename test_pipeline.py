"""
SMART DEPTH VISION — Quick Test
--------------------------------
Run this FIRST before run_demo.py to verify everything is working.
Tests on just 2 images from scene_01.
Run: python test_pipeline.py
"""

import cv2
from pathlib import Path
from pipeline import load_models, process_image

DATASET_PATH = r"C:\Users\Ankush\Desktop\Btech\Capstone\rgbd-scenes-v2\imgs"
TEST_SCENE   = "scene_01"
OUTPUT_PATH  = r"C:\Users\Ankush\Desktop\Btech\Capstone\results\test"

def main():
    print("=" * 50)
    print("   Smart Depth Vision — Quick Test")
    print("=" * 50)

    # Find 2 test images
    scene_path = Path(DATASET_PATH) / TEST_SCENE
    color_files = sorted(scene_path.glob("*-color.png"))[:2]

    if not color_files:
        print(f"\n[ERROR] No color images found in {scene_path}")
        return

    print(f"\n[Test] Using {len(color_files)} images from {TEST_SCENE}")

    # Load models
    print("\n[Loading] Models...")
    yolo, midas, transform, device = load_models()

    # Output
    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process
    for i, color_path in enumerate(color_files):
        depth_path = scene_path / color_path.name.replace("-color.png", "-depth.png")
        print(f"\n[Processing] {color_path.name}")

        result_img, stats = process_image(
            rgb_path=color_path,
            yolo=yolo, midas=midas,
            transform=transform, device=device,
            depth_path=depth_path if depth_path.exists() else None
        )

        if result_img is not None:
            out_path = out_dir / f"test_{i+1}_result.jpg"
            cv2.imwrite(str(out_path), result_img)
            print(f"  Saved: {out_path}")
            print(f"  Detections: {len(stats)}")
            for obj in stats:
                print(f"    - {obj['class']:15s} [{obj['dimension']}] "
                      f"| depth_std={obj['depth_std']:.1f} "
                      f"| dim_conf={obj['dim_confidence']:.0f}%")

    print("\n[SUCCESS] Test complete! Check results in:")
    print(f"  {OUTPUT_PATH}")
    print("\nIf results look good, run: python run_demo.py")

if __name__ == "__main__":
    main()
