import cv2
import numpy as np


def estimate_light_direction(image: np.ndarray) -> tuple[float, float, float]:
    """Estimate dominant light direction from image brightness distribution.

    Returns a normalized (x, y, z) direction vector.
    x: positive = right, y: positive = down, z: positive = toward camera.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float64)

    h, w = gray.shape
    total_brightness = gray.sum()

    if total_brightness < 1.0:
        return (0.0, 0.0, 1.0)

    # Compute brightness-weighted centroid
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    cx = (gray * x_coords).sum() / total_brightness
    cy = (gray * y_coords).sum() / total_brightness

    # Map centroid to [-1, 1] relative to image center
    dx = (cx - w / 2) / (w / 2)
    dy = (cy - h / 2) / (h / 2)

    # How concentrated is the brightness? If uniform, default to overhead.
    brightness_std = gray.std() / max(gray.mean(), 1.0)
    concentration = min(brightness_std / 0.5, 1.0)

    # Scale lateral components by concentration — uniform light stays overhead
    dx *= concentration
    dy *= concentration
    dz = max(0.3, 1.0 - abs(dx) - abs(dy))

    # Normalize
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    if length < 1e-6:
        return (0.0, 0.0, 1.0)

    return (float(dx / length), float(dy / length), float(dz / length))
