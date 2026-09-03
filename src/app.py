import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv

from src.models import WatchSpec
from src.pipeline import run_still_pipeline

load_dotenv()

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("outputs")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def try_on(
    wrist_image: np.ndarray,
    circumference_cm: float,
    watch_reference: str,
    watch_name: str,
    case_diameter_mm: float,
    lug_to_lug_mm: float,
    thickness_mm: float,
    band_width_mm: float,
    case_shape: str,
    watch_photo: np.ndarray,
) -> np.ndarray:
    api_key = os.environ.get("PIAPI_API_KEY", "")
    if not api_key:
        raise gr.Error("PIAPI_API_KEY not set in .env")

    # Save watch photo to a temp file for the reconstruction API
    watch_photo_path = CACHE_DIR / f"{watch_reference}_photo.png"
    cv2.imwrite(str(watch_photo_path), cv2.cvtColor(watch_photo, cv2.COLOR_RGB2BGR))

    spec = WatchSpec(
        reference=watch_reference,
        name=watch_name,
        case_diameter_mm=case_diameter_mm,
        lug_to_lug_mm=lug_to_lug_mm,
        thickness_mm=thickness_mm,
        band_width_mm=band_width_mm,
        case_shape=case_shape,
        photo_paths=[watch_photo_path],
    )

    wrist_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)

    # SAM 2 predictor — lazy-loaded
    predictor = _load_sam_predictor()

    result_bgr = run_still_pipeline(
        watch_spec=spec,
        wrist_image=wrist_bgr,
        circumference_mm=circumference_cm * 10,
        api_key=api_key,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        sam_predictor=predictor,
    )

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def try_on_video(
    wrist_video: str,  # Gradio returns video as file path
    circumference_cm: float,
    watch_reference: str,
    watch_name: str,
    case_diameter_mm: float,
    lug_to_lug_mm: float,
    thickness_mm: float,
    band_width_mm: float,
    case_shape: str,
    watch_photo: np.ndarray,
) -> str:
    api_key = os.environ.get("PIAPI_API_KEY", "")
    if not api_key:
        raise gr.Error("PIAPI_API_KEY not set in .env")

    watch_photo_path = CACHE_DIR / f"{watch_reference}_photo.png"
    cv2.imwrite(str(watch_photo_path), cv2.cvtColor(watch_photo, cv2.COLOR_RGB2BGR))

    spec = WatchSpec(
        reference=watch_reference,
        name=watch_name,
        case_diameter_mm=case_diameter_mm,
        lug_to_lug_mm=lug_to_lug_mm,
        thickness_mm=thickness_mm,
        band_width_mm=band_width_mm,
        case_shape=case_shape,
        photo_paths=[watch_photo_path],
    )

    from src.video.pipeline import run_video_pipeline

    predictor = _load_sam_predictor()

    # SAM 2 video predictor
    from sam2.build_sam import build_sam2_video_predictor
    video_predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2_hiera_small.pt")

    output_path = OUTPUT_DIR / f"{watch_reference}_tryon.mp4"
    run_video_pipeline(
        watch_spec=spec,
        video_path=Path(wrist_video),
        circumference_mm=circumference_cm * 10,
        api_key=api_key,
        cache_dir=CACHE_DIR,
        output_path=output_path,
        sam_predictor=predictor,
        sam_video_predictor=video_predictor,
    )

    return str(output_path)


_sam_predictor_cache = None


def _load_sam_predictor():
    global _sam_predictor_cache
    if _sam_predictor_cache is not None:
        return _sam_predictor_cache

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2_hiera_small.pt")
        _sam_predictor_cache = SAM2ImagePredictor(sam_model)
        return _sam_predictor_cache
    except ImportError:
        raise gr.Error(
            "SAM 2 not installed. Run: pip install sam2 "
            "and download sam2_hiera_small.pt"
        )


def _build_shared_inputs() -> list:
    """Build a fresh set of watch/wrist spec inputs (Gradio components can't be reused across interfaces)."""
    return [
        gr.Number(label="Wrist Circumference (cm)", value=17.5, minimum=10, maximum=25),
        gr.Textbox(label="Watch Reference", placeholder="e.g. 126710BLNR"),
        gr.Textbox(label="Watch Name", placeholder="e.g. Rolex GMT-Master II"),
        gr.Number(label="Case Diameter (mm)", value=42.0, minimum=20, maximum=60),
        gr.Number(label="Lug-to-Lug (mm)", value=48.0, minimum=25, maximum=65),
        gr.Number(label="Thickness (mm)", value=13.0, minimum=5, maximum=25),
        gr.Number(label="Band Width (mm)", value=20.0, minimum=10, maximum=30),
        gr.Dropdown(label="Case Shape", choices=["round", "square", "tonneau"], value="round"),
        gr.Image(label="Watch Photo (front-facing)", type="numpy"),
    ]


still_interface = gr.Interface(
    fn=try_on,
    inputs=[gr.Image(label="Wrist Photo", type="numpy"), *_build_shared_inputs()],
    outputs=gr.Image(label="Try-On Result", type="numpy"),
    title="Watch Virtual Try-On",
    description="Upload a wrist photo and watch details to see how the watch looks on your wrist at accurate scale.",
)

video_interface = gr.Interface(
    fn=try_on_video,
    inputs=[gr.Video(label="Wrist Video"), *_build_shared_inputs()],
    outputs=gr.Video(label="Try-On Result"),
    title="Watch Virtual Try-On (Video)",
    description="Upload a wrist video and watch details to see the watch tracked on your wrist across frames.",
)

demo = gr.TabbedInterface(
    [still_interface, video_interface],
    ["Still Image", "Video"],
    title="Watch Virtual Try-On",
)

if __name__ == "__main__":
    demo.launch()
