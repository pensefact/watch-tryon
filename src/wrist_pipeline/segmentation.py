import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WristSegment:
    mask: np.ndarray
    contour: np.ndarray
    width_px: float


def measure_wrist_width(
    mask: np.ndarray,
    forearm_angle_rad: float,
    wrist_point: tuple[float, float] | None = None,
) -> float:
    """Measure wrist width perpendicular to forearm at the wrist point."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask")

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2).astype(np.float64)

    perp_angle = forearm_angle_rad + math.pi / 2
    perp_dir = np.array([math.cos(perp_angle), math.sin(perp_angle)])
    fore_dir = np.array([math.cos(forearm_angle_rad), math.sin(forearm_angle_rad)])

    if wrist_point is not None:
        wp = np.array(wrist_point)
        fore_proj = (points - wp) @ fore_dir
        # Keep only contour points within a narrow band around the wrist
        band = np.percentile(np.abs(fore_proj), 15)
        band = max(band, 10.0)
        near = np.abs(fore_proj) < band
        if near.sum() >= 2:
            points = points[near]

    projections = points @ perp_dir
    width = projections.max() - projections.min()
    return float(width)


def segment_wrist(
    image: np.ndarray,
    point_prompt: tuple[float, float],
    forearm_angle_rad: float,
    predictor,
) -> WristSegment:
    """Segment the wrist using SAM 2 with a point + bounding box prompt."""
    predictor.set_image(image)

    h, w = image.shape[:2]
    px, py = point_prompt

    # Bounding box around the wrist area — roughly 20% of image size
    box_half = min(w, h) * 0.10
    box = np.array([
        max(0, px - box_half),
        max(0, py - box_half),
        min(w, px + box_half),
        min(h, py + box_half),
    ])

    input_point = np.array([[px, py]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=True,
    )

    best_idx = np.argmax(scores)
    mask = (masks[best_idx] * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else np.array([])

    width = measure_wrist_width(mask, forearm_angle_rad, wrist_point=point_prompt)

    return WristSegment(
        mask=mask,
        contour=largest_contour.reshape(-1, 2) if len(largest_contour) > 0 else np.array([]),
        width_px=width,
    )
