import asyncio
from pathlib import Path

import numpy as np

from src.models import WatchSpec, PlacementSpec
from src.watch_pipeline.reconstruction import reconstruct_watch
from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction
from src.composition.renderer import render_watch
from src.composition.compositor import composite_watch


def run_still_pipeline(
    watch_spec: WatchSpec,
    wrist_image: np.ndarray,
    circumference_mm: float,
    api_key: str,
    cache_dir: Path,
    output_dir: Path,
    sam_predictor=None,
) -> np.ndarray:
    """Run the full still-image try-on pipeline."""
    # --- Watch pipeline ---
    photo_path = watch_spec.photo_paths[0]
    mesh_path = asyncio.run(
        reconstruct_watch(photo_path, api_key, cache_dir, watch_spec.reference)
    )
    mesh = load_and_scale_mesh(mesh_path, watch_spec)

    # Save scaled mesh for Blender
    scaled_mesh_path = cache_dir / f"{watch_spec.reference}_scaled.glb"
    mesh.export(str(scaled_mesh_path))

    # --- Wrist pipeline ---
    landmarks = detect_wrist(wrist_image)
    segment = segment_wrist(
        wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, sam_predictor
    )
    px_per_mm = compute_px_per_mm(
        segment.width_px, circumference_mm, landmarks.pose_angle_rad
    )
    light_dir = estimate_light_direction(wrist_image)

    placement = PlacementSpec(
        center_x=landmarks.wrist_point[0],
        center_y=landmarks.wrist_point[1],
        rotation_deg=np.degrees(landmarks.forearm_angle_rad),
        px_per_mm=px_per_mm,
        light_direction=light_dir,
        wrist_mask=segment.mask,
    )

    # --- Composition ---
    h, w = wrist_image.shape[:2]
    passes = render_watch(scaled_mesh_path, placement, output_dir / "renders", image_size=(w, h))
    result = composite_watch(wrist_image, passes, placement)

    return result
