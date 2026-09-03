import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WristSegment:
    mask: np.ndarray
    contour: np.ndarray
    width_px: float


def measure_wrist_width(mask: np.ndarray, forearm_angle_rad: float) -> float:
    """Measure wrist width perpendicular to the forearm direction."""
    # Find contour points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask")

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2).astype(np.float64)

    # Project contour points onto the axis perpendicular to the forearm
    perp_angle = forearm_angle_rad + math.pi / 2
    perp_dir = np.array([math.cos(perp_angle), math.sin(perp_angle)])

    projections = points @ perp_dir
    width = projections.max() - projections.min()
    return float(width)


def segment_wrist(
    image: np.ndarray,
    point_prompt: tuple[float, float],
    forearm_angle_rad: float,
    predictor,
) -> WristSegment:
    """Segment the wrist using SAM 2 with a point prompt."""
    predictor.set_image(image)

    input_point = np.array([[point_prompt[0], point_prompt[1]]])
    input_label = np.array([1])  # foreground

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Take highest-scoring mask
    best_idx = np.argmax(scores)
    mask = (masks[best_idx] * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else np.array([])

    width = measure_wrist_width(mask, forearm_angle_rad)

    return WristSegment(
        mask=mask,
        contour=largest_contour.reshape(-1, 2) if len(largest_contour) > 0 else np.array([]),
        width_px=width,
    )
