"""V6 demo: Agentic watch try-on pipeline.

Deterministic compose → Generate (fal.ai Flux + IP-Adapter) → Evaluate (Claude vision)
→ Adjust parameters → Loop until brand-accurate + naturally integrated.
"""
import base64
import gc
import json
import math
import os
import tempfile
from pathlib import Path

import anthropic
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

MAX_ITERATIONS = 5


# ---------------------------------------------------------------------------
# 1. Watch segmentation (unchanged from v5)
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
# 2. Composite (unchanged from v5)
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
# 3. Harmonization via fal.ai Flux inpainting + IP-Adapter
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


def harmonize(
    composite: np.ndarray,
    mask: np.ndarray,
    product_photo_path: Path,
    prompt: str,
    strength: float,
    ip_adapter_scale: float,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
) -> np.ndarray:
    import fal_client

    max_dim = 1536
    h, w = composite.shape[:2]
    if max(h, w) > max_dim:
        api_scale = max_dim / max(h, w)
        api_w, api_h = int(w * api_scale), int(h * api_scale)
    else:
        api_w, api_h = w, h

    dilate_px = max(20, int(mask.shape[0] * 0.03))
    expanded_mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=2)
    expanded_mask = cv2.GaussianBlur(expanded_mask, (0, 0), dilate_px * 0.5)

    print(f"    Uploading images ({api_w}x{api_h})...")
    composite_url = _upload_image(composite, max_dim)
    mask_rgb = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
    mask_url = _upload_image(mask_rgb, max_dim)

    product = cv2.imread(str(product_photo_path))
    product_url = _upload_image(product, max_dim)

    print(f"    Flux inpainting: strength={strength:.2f}, ip_scale={ip_adapter_scale:.2f}, "
          f"steps={num_steps}, guidance={guidance_scale:.1f}")

    result = fal_client.subscribe("fal-ai/flux-general/inpainting", arguments={
        "prompt": prompt,
        "image_url": composite_url,
        "mask_url": mask_url,
        "strength": strength,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "image_size": {"width": api_w, "height": api_h},
        "ip_adapters": [{
            "path": "XLabs-AI/flux-ip-adapter",
            "image_url": product_url,
            "scale": ip_adapter_scale,
            "weight_name": "ip_adapter.safetensors",
            "image_encoder_path": "openai/clip-vit-large-patch14",
        }],
    })

    output_url = result["images"][0]["url"]

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
# 4. Claude vision evaluator
# ---------------------------------------------------------------------------

def _img_to_b64(img_bgr: np.ndarray, max_dim: int = 1024) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def evaluate_result(
    harmonized: np.ndarray,
    product_photo_path: Path,
    iteration: int,
    params: dict,
) -> dict:
    """Use Claude vision to evaluate how well the harmonized result matches the product photo.

    Returns dict with:
      - pass: bool (accept this result)
      - brand_accuracy: 1-10
      - natural_integration: 1-10
      - issues: list of specific problems
      - parameter_adjustments: dict of suggested param changes
    """
    client = anthropic.Anthropic()

    harmonized_b64 = _img_to_b64(harmonized)

    product = cv2.imread(str(product_photo_path))
    product_b64 = _img_to_b64(product)

    eval_prompt = f"""You are evaluating iteration {iteration} of a watch try-on pipeline.

IMAGE 1 (left/first): The product reference photo of a Casio G-Shock GA-B2100DF-1A watch.
IMAGE 2 (right/second): The generated result — this watch composited onto someone's wrist, then processed by a diffusion model to harmonize lighting and edges.

Current generation parameters:
- strength (denoising): {params['strength']:.2f} (0=preserve input exactly, 1=generate freely)
- ip_adapter_scale: {params['ip_adapter_scale']:.2f} (how strongly to anchor to product photo appearance)
- guidance_scale: {params['guidance_scale']:.1f}
- num_steps: {params['num_steps']}

Evaluate the result on two axes:

1. **Brand accuracy** (1-10): Does the watch in the result match the product photo?
   - Is it the same watch model? (round analog face, not digital)
   - Are the dial markings/indices correct? (no garbled text, correct "G-SHOCK" branding)
   - Is the bezel shape right?
   - Are colors accurate? (black case, dark green/olive accents)

2. **Natural integration** (1-10): Does the watch look naturally worn?
   - Lighting consistency with the scene
   - No visible paste/cutout edges
   - Strap wraps believably around wrist
   - Appropriate shadows
   - No hallucinated artifacts (extra fingers, distorted skin, impossible geometry)

Respond with ONLY valid JSON (no markdown fences):
{{
  "brand_accuracy": <1-10>,
  "natural_integration": <1-10>,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "parameter_adjustments": {{
    "strength": <suggested new value or null if fine>,
    "ip_adapter_scale": <suggested new value or null>,
    "guidance_scale": <suggested new value or null>,
    "num_steps": <suggested new value or null>,
    "prompt_additions": "<extra prompt text to add, or null>"
  }},
  "reasoning": "<1-2 sentences explaining your evaluation>"
}}

Guidelines for parameter adjustments:
- If watch details are WRONG (hallucinated dial, wrong shape): DECREASE strength, INCREASE ip_adapter_scale
- If watch looks PASTED (harsh edges, wrong lighting): INCREASE strength slightly, keep ip_adapter_scale high
- If there are HALLUCINATED ARTIFACTS (extra fingers, distorted anatomy): DECREASE strength significantly
- Sweet spot is usually strength 0.4-0.65 with ip_adapter_scale 0.85-0.95
- Don't suggest changes larger than ±0.1 for strength or ±0.1 for ip_adapter_scale per iteration"""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": product_b64}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": harmonized_b64}},
                {"type": "text", "text": eval_prompt},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    evaluation = json.loads(raw)

    evaluation["pass"] = (
        evaluation["brand_accuracy"] >= 7
        and evaluation["natural_integration"] >= 7
        and len(evaluation.get("issues", [])) <= 1
    )

    return evaluation


# ---------------------------------------------------------------------------
# 5. Agentic loop
# ---------------------------------------------------------------------------

def agentic_harmonize(
    composite: np.ndarray,
    mask: np.ndarray,
    product_photo_path: Path,
    watch_name: str,
) -> tuple[np.ndarray, list[dict]]:
    """Run the generate → evaluate → adjust loop.

    Returns (best_result, history) where history contains all iterations.
    """
    params = {
        "strength": 0.55,
        "ip_adapter_scale": 0.90,
        "guidance_scale": 3.5,
        "num_steps": 28,
        "prompt": f"a {watch_name} watch worn on a wrist, photorealistic, natural indoor lighting, "
                  "sharp watch details, correct dial markings",
    }

    history = []
    best_result = None
    best_score = -1

    output_dir = Path("outputs")

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"\n  --- Iteration {i}/{MAX_ITERATIONS} ---")
        print(f"    Params: strength={params['strength']:.2f}, "
              f"ip_scale={params['ip_adapter_scale']:.2f}, "
              f"guidance={params['guidance_scale']:.1f}")

        # Generate
        try:
            result = harmonize(
                composite, mask, product_photo_path,
                prompt=params["prompt"],
                strength=params["strength"],
                ip_adapter_scale=params["ip_adapter_scale"],
                num_steps=params["num_steps"],
                guidance_scale=params["guidance_scale"],
            )
        except Exception as e:
            print(f"    Generation failed: {e}")
            history.append({"iteration": i, "error": str(e)})
            continue

        cv2.imwrite(str(output_dir / f"v6_iter{i}.png"), result)
        print(f"    Saved iteration {i} result")

        # Evaluate
        print(f"    Evaluating with Claude vision...")
        try:
            evaluation = evaluate_result(result, product_photo_path, i, params)
        except Exception as e:
            print(f"    Evaluation failed: {e}")
            history.append({"iteration": i, "error": f"eval: {e}"})
            continue

        score = (evaluation["brand_accuracy"] + evaluation["natural_integration"]) / 2
        print(f"    Brand accuracy: {evaluation['brand_accuracy']}/10")
        print(f"    Natural integration: {evaluation['natural_integration']}/10")
        print(f"    Combined: {score:.1f}/10")
        if evaluation.get("issues"):
            for issue in evaluation["issues"]:
                print(f"    Issue: {issue}")
        print(f"    Reasoning: {evaluation.get('reasoning', '')}")

        record = {
            "iteration": i,
            "params": dict(params),
            "evaluation": evaluation,
            "score": score,
        }
        history.append(record)

        if score > best_score:
            best_score = score
            best_result = result.copy()

        if evaluation["pass"]:
            print(f"    >>> PASSED — accepting result from iteration {i}")
            break

        # Adjust parameters for next iteration
        adj = evaluation.get("parameter_adjustments", {})
        if adj.get("strength") is not None:
            params["strength"] = np.clip(adj["strength"], 0.2, 0.85)
        if adj.get("ip_adapter_scale") is not None:
            params["ip_adapter_scale"] = np.clip(adj["ip_adapter_scale"], 0.5, 1.0)
        if adj.get("guidance_scale") is not None:
            params["guidance_scale"] = np.clip(adj["guidance_scale"], 1.5, 7.0)
        if adj.get("num_steps") is not None:
            params["num_steps"] = int(np.clip(adj["num_steps"], 20, 50))
        if adj.get("prompt_additions"):
            base = f"a {watch_name} watch worn on a wrist, photorealistic, natural indoor lighting"
            params["prompt"] = f"{base}, {adj['prompt_additions']}"

        print(f"    Adjusting for next iteration...")
    else:
        print(f"\n  Reached max iterations ({MAX_ITERATIONS}). Using best result (score {best_score:.1f}).")

    if best_result is None:
        print("  All iterations failed. Returning raw composite.")
        best_result = composite

    return best_result, history


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    fal_key = os.getenv("FAL_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not fal_key:
        print("ERROR: FAL_KEY not set in .env")
        return
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        print("  Needed for Claude vision evaluation in the agentic loop.")
        print("  Get one at https://console.anthropic.com/")
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
    cv2.imwrite(str(output_dir / "v6_watch_segmented.png"), watch_rgba)

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

    # Free SAM2 + MediaPipe before the memory-heavy loop
    del sam_model, predictor, segment, wrist_rgb
    gc.collect()
    print("  Freed detection models")

    # --- Step 3: Composite ---
    print("\n=== Compositing ===")
    rotation_deg = forearm_deg + 180
    composite, watch_mask = composite_watch(
        wrist_image, watch_rgba, (watch_cx, watch_cy), rotation_deg, target_width_px
    )
    cv2.imwrite(str(output_dir / "v6_composite_raw.png"), composite)
    cv2.imwrite(str(output_dir / "v6_watch_mask.png"), watch_mask)
    print(f"  Raw composite saved")

    # --- Step 4: Agentic harmonization loop ---
    print("\n=== Agentic Harmonization ===")
    final, history = agentic_harmonize(
        composite, watch_mask, WATCH_REF, spec.name
    )

    out_path = output_dir / "demo_v6.png"
    cv2.imwrite(str(out_path), final)
    print(f"\nFinal result saved to {out_path}")

    # Save run history
    history_path = output_dir / "v6_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Run history saved to {history_path}")

    # Summary
    if history:
        scores = [h["score"] for h in history if "score" in h]
        if scores:
            print(f"\nScore progression: {' → '.join(f'{s:.1f}' for s in scores)}")
            passed = any(h.get("evaluation", {}).get("pass") for h in history)
            if passed:
                print("Result: ACCEPTED by evaluator")
            else:
                print(f"Result: Best of {len(scores)} attempts (score {max(scores):.1f}/10)")


if __name__ == "__main__":
    main()
