"""V2 demo: extract watch from reference photo, overlay on wrist at correct scale."""
import math
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.models import WatchSpec
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm

load_dotenv()

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260907_124808.jpg"))
WATCH_REF = Path("cache/gshock_upright.jpg")


def extract_watch_from_ref(ref_path):
    """Extract watch from a product shot with light/white background.

    Uses brightness thresholding — no SAM needed for clean product photos.
    Returns RGBA image of the watch.
    """
    ref = cv2.imread(str(ref_path))
    if ref is None:
        raise FileNotFoundError(f"Cannot load reference: {ref_path}")
    h, w = ref.shape[:2]
    print(f"  Reference image: {w}x{h}")

    # Convert to grayscale and threshold: watch is dark, background is light
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # Otsu's method finds the optimal threshold between dark watch and light bg
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up: remove small noise, fill holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the watch)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No watch found in reference image")
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    # Feather edges for natural blending
    clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 1.5)

    # Crop to bounding box with padding
    ys, xs = np.where(clean_mask > 0)
    pad = 5
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad)

    cropped = ref[y1:y2, x1:x2]
    cropped_mask = clean_mask[y1:y2, x1:x2]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask

    print(f"  Extracted watch: {rgba.shape[1]}x{rgba.shape[0]}")
    return rgba


def composite_2d(background, watch_face, center, rotation_deg, target_width_px):
    """Composite watch onto background using Poisson blending for natural color/light match."""
    bh, bw = background.shape[:2]
    fh, fw = watch_face.shape[:2]

    # Scale watch to physical size
    scale = target_width_px / fw
    new_w = max(1, round(fw * scale))
    new_h = max(1, round(fh * scale))
    scaled = cv2.resize(watch_face, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # Place and rotate onto canvas
    canvas = np.zeros((bh, bw, 4), dtype=np.uint8)
    fg_center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(fg_center, -rotation_deg, 1.0)
    rot_mat[0, 2] += center[0] - new_w / 2
    rot_mat[1, 2] += center[1] - new_h / 2
    cv2.warpAffine(scaled, rot_mat, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)

    alpha = canvas[:, :, 3]

    # Build mask for seamlessClone (needs 255 where watch is, 0 elsewhere)
    mask = (alpha > 128).astype(np.uint8) * 255

    # Erode mask slightly to avoid edge artifacts in Poisson blending
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Build the source image (BGR, no alpha) on the same canvas size
    src = canvas[:, :, :3].copy()

    # Poisson blending center point
    cx, cy = int(center[0]), int(center[1])
    # Clamp center to be valid for seamlessClone
    cx = max(new_w // 2, min(bw - new_w // 2, cx))
    cy = max(new_h // 2, min(bh - new_h // 2, cy))

    # Check mask has content
    if mask.sum() == 0:
        print("  WARNING: empty mask, falling back to alpha blend")
        return background.copy()

    try:
        # MIXED_CLONE preserves the source texture while matching bg lighting
        result = cv2.seamlessClone(src, background, mask, (cx, cy), cv2.MIXED_CLONE)
    except cv2.error as e:
        print(f"  WARNING: seamlessClone failed ({e}), falling back to alpha blend")
        result = background.copy()
        alpha_f = alpha.astype(np.float32) / 255.0
        alpha_f = cv2.GaussianBlur(alpha_f, (3, 3), 0.5)
        for c in range(3):
            fg = canvas[:, :, c].astype(np.float32)
            bg = result[:, :, c].astype(np.float32)
            result[:, :, c] = np.clip(fg * alpha_f + bg * (1.0 - alpha_f), 0, 255).astype(np.uint8)

    return result


def main():
    spec = WatchSpec(
        reference="casio-gshock-dw5600",
        name="Casio G-Shock DW-5600",
        case_diameter_mm=42.8,
        lug_to_lug_mm=48.9,
        thickness_mm=13.4,
        band_width_mm=25.0,
        case_shape="square",
        photo_paths=[],
    )

    wrist_image = cv2.imread(str(WRIST_IMAGE))
    h, w = wrist_image.shape[:2]
    print(f"Wrist image: {w}x{h}")

    # Load SAM 2
    print("Loading SAM 2 on CPU...")
    sam_model = build_sam2("configs/sam2/sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device="cpu")
    predictor = SAM2ImagePredictor(sam_model)

    # Detect wrist
    print("Detecting wrist...")
    wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
    landmarks = detect_wrist(wrist_rgb)
    print(f"  Wrist: ({landmarks.wrist_point[0]:.0f}, {landmarks.wrist_point[1]:.0f})")
    print(f"  Forearm angle: {np.degrees(landmarks.forearm_angle_rad):.1f} deg")

    # Segment wrist for width
    print("Segmenting wrist...")
    segment = segment_wrist(wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, predictor)
    print(f"  Wrist width: {segment.width_px:.0f}px")

    # Scale calibration
    circumference_mm = 175.0
    px_per_mm = compute_px_per_mm(segment.width_px, circumference_mm, landmarks.pose_angle_rad)
    target_width_px = spec.case_diameter_mm * px_per_mm
    print(f"  px_per_mm: {px_per_mm:.2f}")
    print(f"  Watch will be {target_width_px:.0f}px wide")

    # Watch placement — shifted up the forearm from the wrist landmark
    shift_mm = 35.0
    shift_px = shift_mm * px_per_mm
    fore_dir = np.array([math.cos(landmarks.forearm_angle_rad), math.sin(landmarks.forearm_angle_rad)])
    watch_cx = landmarks.wrist_point[0] + fore_dir[0] * shift_px
    watch_cy = landmarks.wrist_point[1] + fore_dir[1] * shift_px
    print(f"  Watch center: ({watch_cx:.0f}, {watch_cy:.0f})")

    # Extract watch from product shot (simple bg removal, no SAM needed)
    print("Extracting watch from product photo...")
    watch_face = extract_watch_from_ref(WATCH_REF)
    cv2.imwrite(str(Path("outputs") / "extracted_watch.png"), watch_face)

    # forearm_angle points hand→elbow. +180 so 12 o'clock faces fingers.
    rotation_deg = np.degrees(landmarks.forearm_angle_rad) + 180
    print(f"  Rotation: {rotation_deg:.1f} deg")

    print("Compositing...")
    result = composite_2d(wrist_image, watch_face, (watch_cx, watch_cy), rotation_deg, target_width_px)

    # Draw debug overlay: wrist point, watch center, forearm direction
    debug = result.copy()
    wp = (int(landmarks.wrist_point[0]), int(landmarks.wrist_point[1]))
    wc = (int(watch_cx), int(watch_cy))
    cv2.circle(debug, wp, 10, (0, 255, 0), 2)
    cv2.circle(debug, wc, 10, (0, 0, 255), 2)
    cv2.line(debug, wp, wc, (255, 255, 0), 2)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "demo_v2.png"), result)
    cv2.imwrite(str(out_dir / "demo_v2_debug.png"), debug)
    cv2.imwrite(str(out_dir / "watch_face.png"), watch_face)
    print(f"\nSaved to {out_dir / 'demo_v2.png'}")
    print(f"Debug overlay: {out_dir / 'demo_v2_debug.png'}")


if __name__ == "__main__":
    main()
