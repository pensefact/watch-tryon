"""V5 demo: Product photo segmentation → perspective warp → composite → diffusion harmonization.

No 3D reconstruction or Blender. The product photo IS the watch asset.
Size accuracy comes from physical calibration (wrist width → px/mm → exact pixel dimensions).
Visual realism comes from diffusion-based harmonization (IC-Light V2 via Replicate).
"""
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.models import WatchSpec
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm

load_dotenv()

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260908_092226.jpg"))
GROUND_TRUTH = Path(os.path.expanduser("~/Downloads/20260908_092120.jpg"))
WATCH_REF = Path(os.path.expanduser(
    "~/Downloads/G-Shock-x-Charles-Darwin-Foundation-GAB2100DF-1A-Watch-Black-front-1024x1024.jpg"
))


# ---------------------------------------------------------------------------
# 1. Watch segmentation from product photo
# ---------------------------------------------------------------------------

def segment_watch(product_path: Path) -> np.ndarray:
    """Extract watch RGBA from product photo (dark watch on light background)."""
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

    # Crop to bounding box
    ys, xs = np.where(clean_mask > 0)
    pad = 10
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad)

    cropped = img[y1:y2, x1:x2]
    cropped_mask = clean_mask[y1:y2, x1:x2]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask
    return rgba


def crop_to_case(watch_rgba: np.ndarray, stub_fraction: float = 0.15) -> np.ndarray:
    """Crop to watch case + short strap stubs, feather strap edges."""
    alpha = watch_rgba[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    if not rows.any():
        return watch_rgba

    y_min, y_max = np.where(rows)[0][[0, -1]]

    row_widths = np.zeros(watch_rgba.shape[0])
    for r in range(y_min, y_max + 1):
        cols = np.where(alpha[r] > 10)[0]
        if len(cols) > 0:
            row_widths[r] = cols[-1] - cols[0]

    max_w = row_widths.max()
    case_rows = np.where(row_widths > max_w * 0.7)[0]
    case_top, case_bot = case_rows[0], case_rows[-1]
    case_h = case_bot - case_top

    stub_px = int(case_h * stub_fraction)
    crop_top = max(0, case_top - stub_px)
    crop_bot = min(watch_rgba.shape[0], case_bot + stub_px)

    cropped = watch_rgba[crop_top:crop_bot].copy()

    # Feather strap stubs
    fade_top = case_top - crop_top
    fade_bot = crop_bot - case_bot
    if fade_top > 2:
        for i in range(fade_top):
            t = i / fade_top
            cropped[i, :, 3] = (cropped[i, :, 3].astype(np.float32) * t).astype(np.uint8)
    if fade_bot > 2:
        for i in range(fade_bot):
            row = cropped.shape[0] - 1 - i
            t = i / fade_bot
            cropped[row, :, 3] = (cropped[row, :, 3].astype(np.float32) * t).astype(np.uint8)

    return cropped


# ---------------------------------------------------------------------------
# 2. Composite: place watch on wrist at exact physical dimensions
# ---------------------------------------------------------------------------

def composite_watch(
    background: np.ndarray,
    watch_rgba: np.ndarray,
    center: tuple[float, float],
    rotation_deg: float,
    target_width_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite watch onto background. Returns (result_bgr, watch_mask_u8)."""
    bh, bw = background.shape[:2]
    rh, rw = watch_rgba.shape[:2]

    # Scale to exact physical size
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

    # Place and rotate onto canvas
    canvas = np.zeros((bh, bw, 4), dtype=np.uint8)
    fg_center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(fg_center, -rotation_deg, 1.0)
    rot_mat[0, 2] += center[0] - new_w / 2
    rot_mat[1, 2] += center[1] - new_h / 2
    cv2.warpAffine(scaled, rot_mat, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)

    # Extract mask before blending
    watch_mask = canvas[:, :, 3].copy()

    # Drop shadow
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

    # Alpha blend with edge feathering
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
# 3. Diffusion harmonization via fal.ai Flux inpainting + IP-Adapter
# ---------------------------------------------------------------------------

def _upload_image(img_bgr: np.ndarray, max_dim: int = 1536) -> str:
    """Resize if needed, write to temp file, upload to fal.ai, return URL."""
    import fal_client

    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    path = tempfile.mktemp(suffix=".png")
    cv2.imwrite(path, img_bgr)
    url = fal_client.upload_file(path)
    os.unlink(path)
    return url


def harmonize(
    composite: np.ndarray,
    mask: np.ndarray,
    product_photo_path: Path,
    prompt: str = "a black Casio G-Shock watch worn on a wrist, photorealistic, indoor warm lighting",
    strength: float = 0.35,
    ip_adapter_scale: float = 0.7,
) -> np.ndarray:
    """Harmonize composite using fal.ai Flux inpainting with IP-Adapter.

    The IP-Adapter anchors the watch appearance to the product photo,
    allowing higher denoising for better lighting/shadow integration
    without hallucinating watch details.
    """
    import fal_client

    # Downscale for API (max 1536px on longest side)
    max_dim = 1536
    h, w = composite.shape[:2]
    if max(h, w) > max_dim:
        api_scale = max_dim / max(h, w)
        api_w, api_h = int(w * api_scale), int(h * api_scale)
    else:
        api_scale = 1.0
        api_w, api_h = w, h

    # Dilate mask so the model can work on edges, strap-to-skin transitions, shadows
    dilate_px = max(20, int(mask.shape[0] * 0.03))
    expanded_mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=2)
    expanded_mask = cv2.GaussianBlur(expanded_mask, (0, 0), dilate_px * 0.5)

    print(f"  Uploading images to fal.ai ({api_w}x{api_h})...")
    composite_url = _upload_image(composite, max_dim)
    mask_rgb = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
    mask_url = _upload_image(mask_rgb, max_dim)

    product = cv2.imread(str(product_photo_path))
    product_url = _upload_image(product, max_dim)

    print(f"  Running Flux inpainting (strength={strength}, ip_scale={ip_adapter_scale})...")
    try:
        result = fal_client.subscribe("fal-ai/flux-general/inpainting", arguments={
            "prompt": prompt,
            "image_url": composite_url,
            "mask_url": mask_url,
            "strength": strength,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "image_size": {
                "width": api_w,
                "height": api_h,
            },
            "ip_adapters": [{
                "path": "XLabs-AI/flux-ip-adapter",
                "image_url": product_url,
                "scale": ip_adapter_scale,
                "weight_name": "ip_adapter.safetensors",
                "image_encoder_path": "openai/clip-vit-large-patch14",
            }],
        })

        output_url = result["images"][0]["url"]
        print(f"  Downloading result...")

        import httpx
        resp = httpx.get(output_url)
        arr = np.frombuffer(resp.content, np.uint8)
        harmonized = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if harmonized is None:
            raise RuntimeError("Failed to decode harmonized image")

        if harmonized.shape[:2] != composite.shape[:2]:
            harmonized = cv2.resize(harmonized, (composite.shape[1], composite.shape[0]))

        return harmonized

    except Exception as e:
        print(f"  Harmonization failed: {e}")
        print("  Falling back to unharmonized composite")
        return composite


# ---------------------------------------------------------------------------
# 4. Ground truth scoring
# ---------------------------------------------------------------------------

def score_against_ground_truth(
    composite: np.ndarray,
    ground_truth_path: Path,
    wrist_center: tuple[float, float],
    target_width_px: float,
) -> dict:
    """Compare composite against ground truth photo wearing the watch."""
    gt = cv2.imread(str(ground_truth_path))
    if gt is None:
        raise FileNotFoundError(f"Cannot load ground truth: {ground_truth_path}")

    if composite.shape[:2] != gt.shape[:2]:
        gt = cv2.resize(gt, (composite.shape[1], composite.shape[0]))

    margin = int(target_width_px * 0.75)
    cx, cy = int(wrist_center[0]), int(wrist_center[1])
    h, w = composite.shape[:2]
    y1, y2 = max(0, cy - margin), min(h, cy + margin)
    x1, x2 = max(0, cx - margin), min(w, cx + margin)

    comp_crop = composite[y1:y2, x1:x2]
    gt_crop = gt[y1:y2, x1:x2]

    comp_gray = cv2.cvtColor(comp_crop, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(gt_gray, comp_gray, full=True)

    hist_scores = []
    for c in range(3):
        h_comp = cv2.calcHist([comp_crop], [c], None, [64], [0, 256])
        h_gt = cv2.calcHist([gt_crop], [c], None, [64], [0, 256])
        cv2.normalize(h_comp, h_comp)
        cv2.normalize(h_gt, h_gt)
        hist_scores.append(cv2.compareHist(h_comp, h_gt, cv2.HISTCMP_CORREL))
    hist_score = float(np.mean(hist_scores))

    edges_comp = cv2.Canny(comp_gray, 50, 150)
    edges_gt = cv2.Canny(gt_gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_comp_d = cv2.dilate(edges_comp, kernel, iterations=1)
    edges_gt_d = cv2.dilate(edges_gt, kernel, iterations=1)
    overlap = np.logical_and(edges_comp > 0, edges_gt_d > 0).sum()
    total = max(1, (edges_comp > 0).sum())
    edge_precision = overlap / total
    overlap_rev = np.logical_and(edges_gt > 0, edges_comp_d > 0).sum()
    total_rev = max(1, (edges_gt > 0).sum())
    edge_recall = overlap_rev / total_rev
    edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-6)

    combined = ssim_score * 40 + hist_score * 30 + edge_f1 * 30
    combined = max(0, min(100, combined))

    output_dir = Path("outputs")
    cv2.imwrite(str(output_dir / "eval_comp_crop.png"), comp_crop)
    cv2.imwrite(str(output_dir / "eval_gt_crop.png"), gt_crop)

    return {
        "ssim": round(ssim_score, 4),
        "hist_corr": round(hist_score, 4),
        "edge_f1": round(edge_f1, 4),
        "combined": round(combined, 2),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    fal_key = os.getenv("FAL_KEY", "")

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

    # --- Step 1: Segment watch from product photo ---
    print("\n=== Watch Segmentation ===")
    watch_rgba = segment_watch(WATCH_REF)
    print(f"  Segmented: {watch_rgba.shape[1]}x{watch_rgba.shape[0]}")
    cv2.imwrite(str(output_dir / "v5_watch_segmented.png"), watch_rgba)

    cv2.imwrite(str(output_dir / "v5_watch_segmented_full.png"), watch_rgba)

    # --- Step 2: Wrist detection + calibration ---
    print("\n=== Wrist Detection ===")
    print("Loading SAM 2 on CPU...")
    sam_model = build_sam2(
        "configs/sam2/sam2_hiera_s.yaml",
        "checkpoints/sam2_hiera_small.pt",
        device="cpu",
    )
    predictor = SAM2ImagePredictor(sam_model)

    print("Detecting wrist...")
    wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
    landmarks = detect_wrist(wrist_rgb)
    forearm_deg = np.degrees(landmarks.forearm_angle_rad)
    print(f"  Wrist: ({landmarks.wrist_point[0]:.0f}, {landmarks.wrist_point[1]:.0f})")
    print(f"  Forearm angle: {forearm_deg:.1f} deg")

    print("Segmenting wrist...")
    segment = segment_wrist(
        wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, predictor
    )
    print(f"  Wrist width: {segment.width_px:.0f}px")

    circumference_mm = 175.0
    px_per_mm = compute_px_per_mm(segment.width_px, circumference_mm, landmarks.pose_angle_rad)
    target_width_px = spec.case_diameter_mm * px_per_mm
    print(f"  px_per_mm: {px_per_mm:.2f}")
    print(f"  Watch target width: {target_width_px:.0f}px (={spec.case_diameter_mm}mm)")

    # Watch placement
    shift_mm = 25.0
    shift_px = shift_mm * px_per_mm
    fore_dir = np.array([
        math.cos(landmarks.forearm_angle_rad),
        math.sin(landmarks.forearm_angle_rad),
    ])
    watch_cx = landmarks.wrist_point[0] + fore_dir[0] * shift_px
    watch_cy = landmarks.wrist_point[1] + fore_dir[1] * shift_px
    print(f"  Watch center: ({watch_cx:.0f}, {watch_cy:.0f})")

    # --- Step 3: Composite at exact physical dimensions ---
    print("\n=== Compositing ===")
    rotation_deg = forearm_deg + 180
    print(f"  Rotation: {rotation_deg:.1f} deg")

    composite, watch_mask = composite_watch(
        wrist_image, watch_rgba, (watch_cx, watch_cy), rotation_deg, target_width_px
    )
    cv2.imwrite(str(output_dir / "v5_composite_raw.png"), composite)
    cv2.imwrite(str(output_dir / "v5_watch_mask.png"), watch_mask)
    print(f"  Raw composite saved")

    # --- Step 4: Harmonization ---
    if fal_key:
        print("\n=== Harmonization (fal.ai Flux + IP-Adapter) ===")
        harmonized = harmonize(
            composite, watch_mask, WATCH_REF,
            prompt=f"a black Casio G-Shock watch worn on a wrist, photorealistic, indoor warm lighting",
            strength=0.75,
            ip_adapter_scale=0.9,
        )
        cv2.imwrite(str(output_dir / "v5_harmonized.png"), harmonized)
        final = harmonized
        print(f"  Harmonized output saved")
    else:
        print("\n=== Skipping harmonization (no FAL_KEY in .env) ===")
        final = composite

    out_path = output_dir / "demo_v5.png"
    cv2.imwrite(str(out_path), final)
    print(f"\nResult saved to {out_path}")

    # --- Ground truth evaluation ---
    if GROUND_TRUTH.exists():
        print("\n=== Ground Truth Evaluation ===")
        scores = score_against_ground_truth(
            composite=final,
            ground_truth_path=GROUND_TRUTH,
            wrist_center=(watch_cx, watch_cy),
            target_width_px=target_width_px,
        )
        print(f"  SSIM:           {scores['ssim']:.4f}")
        print(f"  Histogram corr: {scores['hist_corr']:.4f}")
        print(f"  Edge F1:        {scores['edge_f1']:.4f}")
        print(f"  Combined score: {scores['combined']:.1f} / 100")


if __name__ == "__main__":
    main()
