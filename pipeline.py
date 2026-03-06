"""
SMART DEPTH VISION
Real-Time Monocular Spatial Awareness Using YOLOv8 and Deep Depth Estimation
Core Pipeline: Object Detection + Depth Estimation + 2D/3D Classification
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
DEPTH_STD_THRESHOLD = 8.0   # depth std-dev above this → 3D object
CONFIDENCE_THRESHOLD = 0.35  # YOLO detection confidence


# ─────────────────────────────────────────
#  LOAD MODELS (called once at startup)
# ─────────────────────────────────────────
def load_models():
    """Load YOLOv8 and MiDaS models onto GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Device] Using: {device}")

    # YOLOv8 - object detection
    print("  [YOLOv8] Loading model...")
    yolo = YOLO("yolov8n.pt")  # downloads automatically on first run

    # MiDaS - depth estimation
    print("  [MiDaS] Loading model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device)
    midas.eval()

    # MiDaS transform
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    print("  [Models] Both models loaded successfully!\n")
    return yolo, midas, transform, device


# ─────────────────────────────────────────
#  DEPTH ESTIMATION
# ─────────────────────────────────────────
def estimate_depth(rgb_image, midas, transform, device):
    """
    Given an RGB image (numpy array BGR), return a normalized depth map.
    Returns depth map as numpy float32 array (same HxW as input).
    """
    # MiDaS expects RGB
    img_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to 0-255 for visualization
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_normalized = np.zeros_like(depth)

    return depth.astype(np.float32), depth_normalized.astype(np.uint8)


# ─────────────────────────────────────────
#  2D vs 3D CLASSIFICATION LOGIC
# ─────────────────────────────────────────
def classify_2d_or_3d(depth_map, box, threshold=DEPTH_STD_THRESHOLD):
    """
    Core classification function.
    
    Logic: A real 3D object in the world will have VARYING depth values 
    within its bounding box (it has volume, surfaces at different distances).
    A 2D object (poster, screen, flat image) will have UNIFORM depth values
    (everything is at the same flat distance).
    
    Args:
        depth_map: full image depth map (float32)
        box: bounding box [x1, y1, x2, y2]
        threshold: std deviation threshold
    
    Returns:
        label: "3D" or "2D"
        std_val: the actual std deviation value
        confidence: how confident we are (0-100%)
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Clamp to image bounds
    h, w = depth_map.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Extract depth patch within bounding box
    depth_patch = depth_map[y1:y2, x1:x2]

    if depth_patch.size == 0:
        return "2D", 0.0, 0.0

    std_val = float(np.std(depth_patch))

    # Classification
    if std_val >= threshold:
        label = "3D"
        # Confidence: how far above threshold (capped at 100%)
        confidence = min(100.0, ((std_val - threshold) / threshold) * 100 + 60)
    else:
        label = "2D"
        # Confidence: how far below threshold
        confidence = min(100.0, ((threshold - std_val) / threshold) * 100 + 60)

    return label, std_val, confidence


# ─────────────────────────────────────────
#  MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────
def process_image(rgb_path, yolo, midas, transform, device,
                  depth_path=None, threshold=DEPTH_STD_THRESHOLD):
    """
    Full pipeline for one image:
    1. Load RGB image
    2. Run YOLOv8 detection
    3. Run MiDaS depth estimation
    4. Classify each detected object as 2D or 3D
    5. Return annotated result image + stats

    Args:
        rgb_path: path to color image
        yolo, midas, transform, device: loaded models
        depth_path: optional path to ground truth depth image
        threshold: 2D/3D classification threshold

    Returns:
        result_image: annotated image (numpy BGR)
        stats: list of dicts with detection info
    """
    # Load image
    img = cv2.imread(str(rgb_path))
    if img is None:
        print(f"  [Warning] Could not load image: {rgb_path}")
        return None, []

    h, w = img.shape[:2]

    # ── Step 1: YOLO Detection ──────────────────
    yolo_results = yolo(img, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    boxes = yolo_results.boxes

    # ── Step 2: MiDaS Depth Estimation ──────────
    depth_raw, depth_vis = estimate_depth(img, midas, transform, device)

    # Create colourmap depth for display
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # ── Step 3: Build output image ───────────────
    # Left: RGB with detections | Right: Depth colormap
    output_img = img.copy()
    stats = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = yolo.names[cls_id]

            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            # ── Step 4: 2D/3D Classification ────
            label_3d, std_val, confidence = classify_2d_or_3d(
                depth_raw, coords, threshold
            )

            # Choose colour: GREEN for 3D, RED for 2D
            color = (0, 255, 0) if label_3d == "3D" else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

            # Label text
            text_top = f"{class_name} [{label_3d}]"
            text_bot = f"conf:{conf:.2f} std:{std_val:.1f}"

            # Measure text size for clean background
            (tw1, th1), _ = cv2.getTextSize(text_top, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            (tw2, th2), _ = cv2.getTextSize(text_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_w = max(tw1, tw2) + 10

            # Place label BELOW top-left corner if too close to top edge
            label_y = y1 - 44 if y1 > 50 else y2 + 44

            # Background for text
            cv2.rectangle(output_img,
                          (x1, label_y),
                          (x1 + box_w, label_y + 44),
                          color, -1)
            cv2.putText(output_img, text_top, (x1 + 4, label_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(output_img, text_bot, (x1 + 4, label_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Also draw box on depth colormap
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, 2)
            cv2.putText(depth_colormap, f"{label_3d}", (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stats.append({
                "class": class_name,
                "dimension": label_3d,
                "yolo_conf": round(conf, 3),
                "depth_std": round(std_val, 2),
                "dim_confidence": round(confidence, 1),
                "bbox": [x1, y1, x2, y2]
            })

    # ── Step 5: Side-by-side layout ─────────────
    # Add headers
    header_h = 30
    rgb_header = np.zeros((header_h, w, 3), dtype=np.uint8)
    dep_header = np.zeros((header_h, w, 3), dtype=np.uint8)

    cv2.putText(rgb_header, "RGB + YOLOv8 Detection + 2D/3D Label",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(dep_header, "MiDaS Depth Estimation Map",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    left_panel = np.vstack([rgb_header, output_img])
    right_panel = np.vstack([dep_header, depth_colormap])

    # If ground truth depth provided, add third panel
    if depth_path is not None:
        gt_depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if gt_depth is not None:
            gt_norm = cv2.normalize(gt_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gt_color = cv2.applyColorMap(gt_norm, cv2.COLORMAP_JET)
            gt_color_resized = cv2.resize(gt_color, (w, h))
            gt_header = np.zeros((header_h, w, 3), dtype=np.uint8)
            cv2.putText(gt_header, "Ground Truth Depth",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            gt_panel = np.vstack([gt_header, gt_color_resized])
            result_image = np.hstack([left_panel, right_panel, gt_panel])
        else:
            result_image = np.hstack([left_panel, right_panel])
    else:
        result_image = np.hstack([left_panel, right_panel])

    return result_image, stats
