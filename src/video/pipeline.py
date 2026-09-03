import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.models import WatchSpec, PlacementSpec
from src.watch_pipeline.reconstruction import reconstruct_watch
from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import measure_wrist_width
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction
from src.composition.renderer import render_watch
from src.composition.compositor import composite_watch
from src.video.tracker import OneEuroFilter, track_wrist_video


def run_video_pipeline(
    watch_spec: WatchSpec,
    video_path: Path,
    circumference_mm: float,
    api_key: str,
    cache_dir: Path,
    output_path: Path,
    sam_predictor,
    sam_video_predictor,
) -> Path:
    """Run the full video try-on pipeline."""
    import asyncio

    # --- Watch pipeline (same as stills, cached) ---
    photo_path = watch_spec.photo_paths[0]
    mesh_path = asyncio.run(
        reconstruct_watch(photo_path, api_key, cache_dir, watch_spec.reference)
    )
    mesh = load_and_scale_mesh(mesh_path, watch_spec)
    scaled_mesh_path = cache_dir / f"{watch_spec.reference}_scaled.glb"
    mesh.export(str(scaled_mesh_path))

    # --- Read video metadata ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames in video")

    # --- Wrist detection on first frame ---
    landmarks = detect_wrist(frames[0])
    light_dir = estimate_light_direction(frames[0])

    # --- SAM 2 video tracking ---
    masks = track_wrist_video(video_path, landmarks.wrist_point, sam_video_predictor)

    # --- Smoothing filters ---
    cx_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    cy_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.3)
    scale_filter = OneEuroFilter(min_cutoff=0.5, beta=0.2)

    # --- Per-frame render + composite ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_dir = Path(tmp_dir) / "frames"
        frame_dir.mkdir()

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            t = i / fps

            # Extract placement from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # No wrist detected this frame — use previous placement
                cv2.imwrite(str(frame_dir / f"{i:06d}.png"), frame)
                continue

            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                cv2.imwrite(str(frame_dir / f"{i:06d}.png"), frame)
                continue

            raw_cx = M["m10"] / M["m00"]
            raw_cy = M["m01"] / M["m00"]

            # Forearm angle from mask orientation
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                raw_angle = ellipse[2]
            else:
                raw_angle = 0.0

            raw_width = measure_wrist_width(mask, math.radians(raw_angle))
            raw_scale = compute_px_per_mm(raw_width, circumference_mm, landmarks.pose_angle_rad)

            # Smooth
            cx = cx_filter(t, raw_cx)
            cy = cy_filter(t, raw_cy)
            angle = angle_filter(t, raw_angle)
            px_per_mm = scale_filter(t, raw_scale)

            placement = PlacementSpec(
                center_x=cx,
                center_y=cy,
                rotation_deg=angle,
                px_per_mm=px_per_mm,
                light_direction=light_dir,
                wrist_mask=mask,
            )

            render_dir = Path(tmp_dir) / f"render_{i:06d}"
            passes = render_watch(scaled_mesh_path, placement, render_dir, image_size=(w, h))
            composited = composite_watch(frame, passes, placement, watch_spec.case_diameter_mm)
            cv2.imwrite(str(frame_dir / f"{i:06d}.png"), composited)

        # --- Encode to video ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frame_dir / "%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    return output_path
