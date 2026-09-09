"""V11 demo: No composite — let the model place the watch from scratch.

Sends bare wrist photo + product photo + physical specs to GPT-Image-2.5
Sunburst and lets the model handle all placement, sizing, and perspective.
No SAM2, no MediaPipe, no calibration, no compositing.
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260908_092226.jpg"))
WATCH_REF = Path(os.path.expanduser(
    "~/Downloads/G-Shock-x-Charles-Darwin-Foundation-GAB2100DF-1A-Watch-Black-front-1024x1024.jpg"
))


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


def main():
    fal_key = os.getenv("FAL_KEY", "")
    if not fal_key:
        print("ERROR: FAL_KEY not set in .env")
        return

    watch_name = "Casio G-Shock GA-B2100DF-1A"
    case_diameter_mm = 45.4
    band_width_mm = 22.0
    circumference_mm = 175.0

    wrist_image = cv2.imread(str(WRIST_IMAGE))
    if wrist_image is None:
        print(f"ERROR: Cannot load wrist image: {WRIST_IMAGE}")
        return

    product_image = cv2.imread(str(WATCH_REF))
    if product_image is None:
        print(f"ERROR: Cannot load product image: {WATCH_REF}")
        return

    print(f"Wrist image: {wrist_image.shape[1]}x{wrist_image.shape[0]}")
    print(f"Product image: {product_image.shape[1]}x{product_image.shape[0]}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    import fal_client

    print("\n=== Uploading images ===")
    product_url = _upload_image(product_image, max_dim=1024)
    wrist_url = _upload_image(wrist_image, max_dim=2048)
    print("  Done")

    prompt = (
        "You are given two images:\n"
        "1. A product photo of a watch on a white background\n"
        "2. A bare wrist photo — no watch is being worn\n\n"
        f"The watch is a {watch_name}: {case_diameter_mm}mm case diameter, "
        f"{band_width_mm}mm band width. The person's wrist circumference is {circumference_mm}mm.\n\n"
        "Place the watch from the product photo onto the bare wrist in the second image, "
        "as if the person is naturally wearing it. Specifically:\n"
        "- Position it at the wrist crease (where hand meets forearm)\n"
        f"- Size it correctly: a {case_diameter_mm}mm watch on a {circumference_mm}mm wrist\n"
        "- Apply proper perspective foreshortening for the camera angle\n"
        "- The strap should wrap naturally around the wrist with proper curvature\n"
        "- Match the warm indoor lighting of the scene\n"
        "- Add natural shadows under the case and strap\n"
        "- Add subtle glass reflection/glare on the crystal\n\n"
        "CRITICAL: preserve every detail on the watch face exactly as shown in the product photo — "
        "the dial text, indices, hands, G-SHOCK branding, PROTECTION text, "
        "and digital sub-display must remain pixel-perfect and legible. "
        "Do not change any text on the dial or bezel.\n"
        "IMPORTANT: The strap color is dark grey/olive green (NOT pure black) — "
        "preserve this color exactly as shown in the product photo."
    )

    print("\n=== Generating (GPT-Image-2.5 Sunburst) ===")
    print(f"    Prompt: {prompt[:120]}...")

    result = fal_client.subscribe("openai/gpt-image-2.5/sunburst/edit", arguments={
        "prompt": prompt,
        "image_urls": [product_url, wrist_url],
        "quality": "high",
        "num_images": 1,
        "output_format": "png",
    })

    output_url = result["images"][0]["url"]
    print("    Downloading result...")

    import httpx
    resp = httpx.get(output_url)
    arr = np.frombuffer(resp.content, np.uint8)
    final = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if final is None:
        print("ERROR: Failed to decode result")
        return

    out_path = output_dir / "demo_v11.png"
    cv2.imwrite(str(out_path), final)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
