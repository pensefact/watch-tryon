"""V7 demo: GPT-Image-2.5 Sunburst instruction-based editing.

Same deterministic composite as v5/v6, but instead of mask-based inpainting
(which regenerates the watch region), we use instruction-based editing
that preserves structure and only harmonizes what needs harmonizing.
"""
import base64
import gc
import json
import math
import os
import tempfile
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

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260908_092226.jpg"))
WATCH_REF = Path(os.path.expanduser(
    "~/Downloads/G-Shock-x-Charles-Darwin-Foundation-GAB2100DF-1A-Watch-Black-front-1024x1024.jpg"
))


# ---------------------------------------------------------------------------
# 1. Watch segmentation (same as v5/v6)
# ---------------------------------------------------------------------------

def segment_watch(product_path: Path) -> np.ndarray:
    img = cv2.imread(str(product_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load: {product_path}")
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No watch found in product photo")
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 2.0)

    ys, xs = np.where(clean_mask > 0)
    pad = 10
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad)

    cropped = img[y1:y2, x1:x2]
    cropped_mask = clean_mask[y1:y2, x1:x2]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask
    return rgba


# ---------------------------------------------------------------------------
# 2. Composite (same as v5/v6)
# ---------------------------------------------------------------------------

def composite_watch(
    background: np.ndarray,
    watch_rgba: np.ndarray,
    center: tuple[float, float],
    rotation_deg: float,
    target_width_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    bh, bw = background.shape[:2]
    rh, rw = watch_rgba.shape[:2]

    alpha = watch_rgba[:, :, 3]
    cols = np.any(alpha > 10, axis=0)
    if not cols.any():
        return background.copy(), np.zeros((bh, bw), dtype=np.uint8)
    x_min, x_max = np.where(cols)[0][[0, -1]]
    watch_render_width = x_max - x_min

    scale = target_width_px / max(watch_render_width, 1)
    new_w = max(1, round(rw * scale))
    new_h = max(1, round(rh * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(watch_rgba, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((bh, bw, 4), dtype=np.uint8)
    fg_center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(fg_center, -rotation_deg, 1.0)
    rot_mat[0, 2] += center[0] - new_w / 2
    rot_mat[1, 2] += center[1] - new_h / 2
    cv2.warpAffine(scaled, rot_mat, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)

    watch_mask = canvas[:, :, 3].copy()

    shadow_offset = max(4, int(target_width_px * 0.02))
    shadow_sigma = target_width_px * 0.04
    shadow_alpha = watch_mask.astype(np.float32) / 255.0
    M_shadow = np.float32([[1, 0, shadow_offset], [0, 1, shadow_offset]])
    shadow_alpha = cv2.warpAffine(shadow_alpha, M_shadow, (bw, bh))
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (0, 0), shadow_sigma)
    shadow_alpha = np.clip(shadow_alpha * 0.5, 0, 1)

    result = background.copy()
    for c in range(3):
        bg = result[:, :, c].astype(np.float32)
        result[:, :, c] = np.clip(bg * (1.0 - shadow_alpha * 0.4), 0, 255).astype(np.uint8)

    feather_r = max(3, int(target_width_px * 0.012))
    feather_k = feather_r * 2 + 1
    alpha_f = watch_mask.astype(np.float32) / 255.0
    alpha_u8 = (alpha_f * 255).astype(np.uint8)
    erode_k = max(1, feather_r // 2)
    alpha_u8 = cv2.erode(alpha_u8, np.ones((erode_k, erode_k), np.uint8), iterations=1)
    alpha_f = cv2.GaussianBlur(
        alpha_u8.astype(np.float32) / 255.0, (feather_k, feather_k), feather_r * 0.5
    )

    for c in range(3):
        fg = canvas[:, :, c].astype(np.float32)
        bg = result[:, :, c].astype(np.float32)
        result[:, :, c] = np.clip(fg * alpha_f + bg * (1.0 - alpha_f), 0, 255).astype(np.uint8)

    return result, watch_mask


# ---------------------------------------------------------------------------
# 3. GPT-Image-2.5 Sunburst instruction-based harmonization
# ---------------------------------------------------------------------------

def _upload_image(img_bgr: np.ndarray, max_dim: int = 1536) -> str:
    import fal_client
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(path, img_bgr)
    url = fal_client.upload_file(path)
    os.unlink(path)
    return url


def harmonize_gpt(
    composite: np.ndarray,
    product_photo_path: Path,
    watch_name: str,
    quality: str = "high",
) -> np.ndarray:
    """Use GPT-Image-2.5 Sunburst via fal.ai to harmonize the composite."""
    import fal_client

    composite_url = _upload_image(composite, max_dim=2048)

    prompt = (
        f"This photo shows a {watch_name} watch composited onto someone's wrist. "
        "The watch was digitally placed from a product photo and looks pasted — "
        "the lighting doesn't match the scene, the strap edges are harsh, "
        "and the strap sits flat on top of the wrist instead of wrapping around it. "
        "Harmonize the watch into the scene: "
        "match the warm indoor lighting, soften the strap-to-skin edges, "
        "add subtle shadows under the case and strap, "
        "make the strap look like it wraps naturally around the wrist. "
        "CRITICAL: preserve every detail on the watch face exactly as-is — "
        "the dial text, indices, hands, G-SHOCK branding, PROTECTION text, "
        "and digital sub-display must remain pixel-perfect and legible. "
        "Do not change the watch model, colors, or any text on the dial or bezel. "
        "Only fix the integration into the scene."
    )

    print(f"    Sending to GPT-Image-2.5 Sunburst via fal.ai (quality={quality})...")
    print(f"    Prompt: {prompt[:120]}...")

    result = fal_client.subscribe("openai/gpt-image-2.5/sunburst/edit", arguments={
        "prompt": prompt,
        "image_urls": [composite_url],
        "quality": quality,
        "num_images": 1,
        "output_format": "png",
    })

    output_url = result["images"][0]["url"]
    print(f"    Downloading result...")

    import httpx
    resp = httpx.get(output_url)
    arr = np.frombuffer(resp.content, np.uint8)
    harmonized = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if harmonized is None:
        raise RuntimeError("Failed to decode harmonized image")

    if harmonized.shape[:2] != composite.shape[:2]:
        harmonized = cv2.resize(harmonized, (composite.shape[1], composite.shape[0]))

    return harmonized


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    fal_key = os.getenv("FAL_KEY", "")
    if not fal_key:
        print("ERROR: FAL_KEY not set in .env")
        return

    spec = WatchSpec(
        reference="casio-gshock-gab2100df",
        name="Casio G-Shock GA-B2100DF-1A",
        case_diameter_mm=45.4,
        lug_to_lug_mm=48.5,
        thickness_mm=11.8,
        band_width_mm=22.0,
        case_shape="round",
        photo_paths=[str(WATCH_REF)],
    )

    wrist_image = cv2.imread(str(WRIST_IMAGE))
    if wrist_image is None:
        print(f"ERROR: Cannot load wrist image: {WRIST_IMAGE}")
        return
    h, w = wrist_image.shape[:2]
    print(f"Wrist image: {w}x{h}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # --- Step 1: Segment watch ---
    print("\n=== Watch Segmentation ===")
    watch_rgba = segment_watch(WATCH_REF)
    print(f"  Segmented: {watch_rgba.shape[1]}x{watch_rgba.shape[0]}")

    # --- Step 2: Wrist detection + calibration ---
    print("\n=== Wrist Detection ===")
    print("Loading SAM 2 on CPU...")
    sam_model = build_sam2(
        "configs/sam2/sam2_hiera_s.yaml",
        "checkpoints/sam2_hiera_small.pt",
        device="cpu",
    )
    predictor = SAM2ImagePredictor(sam_model)

    wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
    landmarks = detect_wrist(wrist_rgb)
    forearm_deg = np.degrees(landmarks.forearm_angle_rad)
    print(f"  Wrist: ({landmarks.wrist_point[0]:.0f}, {landmarks.wrist_point[1]:.0f})")
    print(f"  Forearm angle: {forearm_deg:.1f} deg")

    segment = segment_wrist(
        wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, predictor
    )
    print(f"  Wrist width: {segment.width_px:.0f}px")

    circumference_mm = 175.0
    px_per_mm = compute_px_per_mm(segment.width_px, circumference_mm, landmarks.pose_angle_rad)
    target_width_px = spec.case_diameter_mm * px_per_mm
    print(f"  px_per_mm: {px_per_mm:.2f}")
    print(f"  Watch target width: {target_width_px:.0f}px")

    shift_mm = 25.0
    shift_px = shift_mm * px_per_mm
    fore_dir = np.array([
        math.cos(landmarks.forearm_angle_rad),
        math.sin(landmarks.forearm_angle_rad),
    ])
    watch_cx = landmarks.wrist_point[0] + fore_dir[0] * shift_px
    watch_cy = landmarks.wrist_point[1] + fore_dir[1] * shift_px
    print(f"  Watch center: ({watch_cx:.0f}, {watch_cy:.0f})")

    del sam_model, predictor, segment, wrist_rgb
    gc.collect()
    print("  Freed detection models")

    # --- Step 3: Composite ---
    print("\n=== Compositing ===")
    rotation_deg = forearm_deg + 180
    composite, watch_mask = composite_watch(
        wrist_image, watch_rgba, (watch_cx, watch_cy), rotation_deg, target_width_px
    )
    cv2.imwrite(str(output_dir / "v7_composite_raw.png"), composite)
    print(f"  Raw composite saved")

    # --- Step 4: GPT-Image harmonization ---
    print("\n=== Harmonization (GPT-Image-2.5 Sunburst) ===")
    try:
        harmonized = harmonize_gpt(composite, WATCH_REF, spec.name)
        cv2.imwrite(str(output_dir / "v7_harmonized.png"), harmonized)
        final = harmonized
        print(f"  Harmonized output saved")
    except Exception as e:
        print(f"  Harmonization failed: {e}")
        print(f"  Using raw composite as fallback")
        final = composite

    out_path = output_dir / "demo_v7.png"
    cv2.imwrite(str(out_path), final)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
