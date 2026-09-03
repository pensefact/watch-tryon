import math


# Wrist cross-section modeled as an ellipse.
# Major axis (front-facing) = diameter = circumference / pi.
# Minor axis (depth) estimated at ~0.7× major axis (typical wrist is flatter than it is wide).
_WRIST_ECCENTRICITY = 0.7


def estimate_visible_width(circumference_mm: float, pose_angle_rad: float) -> float:
    """Estimate the visible wrist width in mm given circumference and viewing angle.

    angle=0 means camera faces the top of the wrist (dial side).
    angle=pi/2 means camera sees the wrist edge-on (from the thumb/pinky side).
    """
    major_axis = circumference_mm / math.pi
    minor_axis = major_axis * _WRIST_ECCENTRICITY

    # Visible width is the ellipse projection at the given angle
    a = major_axis / 2
    b = minor_axis / 2
    visible_half_width = math.sqrt(
        (a * math.cos(pose_angle_rad)) ** 2 + (b * math.sin(pose_angle_rad)) ** 2
    )
    return 2 * visible_half_width


def compute_px_per_mm(
    wrist_width_px: float, circumference_mm: float, pose_angle_rad: float
) -> float:
    """Compute pixels-per-millimeter at the wrist surface."""
    if wrist_width_px <= 0:
        raise ValueError(f"wrist_width_px must be positive, got {wrist_width_px}")
    if circumference_mm <= 0:
        raise ValueError(f"circumference_mm must be positive, got {circumference_mm}")

    visible_width_mm = estimate_visible_width(circumference_mm, pose_angle_rad)
    return wrist_width_px / visible_width_mm
