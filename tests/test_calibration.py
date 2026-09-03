import math
import pytest
from src.wrist_pipeline.calibration import compute_px_per_mm, estimate_visible_width


def test_visible_width_front_facing():
    """When wrist faces camera straight on (angle=0), visible width = diameter = circumference/pi."""
    circumference_mm = 175.0
    width = estimate_visible_width(circumference_mm, pose_angle_rad=0.0)
    expected_diameter = circumference_mm / math.pi
    assert abs(width - expected_diameter) < 0.01


def test_visible_width_angled():
    """At 45 degrees, visible width should be less than the full diameter."""
    circumference_mm = 175.0
    front_width = estimate_visible_width(circumference_mm, pose_angle_rad=0.0)
    angled_width = estimate_visible_width(circumference_mm, pose_angle_rad=math.pi / 4)
    assert angled_width < front_width
    assert angled_width > 0


def test_visible_width_edge_on():
    """At 90 degrees (edge-on), visible width approaches the minor axis of the ellipse."""
    circumference_mm = 175.0
    width = estimate_visible_width(circumference_mm, pose_angle_rad=math.pi / 2)
    assert width > 0
    front_width = estimate_visible_width(circumference_mm, pose_angle_rad=0.0)
    assert width < front_width


def test_px_per_mm_basic():
    """If wrist is 200px wide in image and 55.7mm actual width, scale = 200/55.7."""
    circumference_mm = 175.0
    wrist_width_px = 200.0
    px_per_mm = compute_px_per_mm(wrist_width_px, circumference_mm, pose_angle_rad=0.0)
    expected_diameter = circumference_mm / math.pi
    expected_scale = wrist_width_px / expected_diameter
    assert abs(px_per_mm - expected_scale) < 0.01


def test_px_per_mm_larger_wrist():
    """Larger wrist at same pixel width = smaller px/mm (further from camera or thinner wrist)."""
    small_wrist = compute_px_per_mm(200.0, 160.0, 0.0)
    large_wrist = compute_px_per_mm(200.0, 200.0, 0.0)
    assert small_wrist > large_wrist


def test_px_per_mm_rejects_invalid():
    with pytest.raises(ValueError):
        compute_px_per_mm(0.0, 175.0, 0.0)
    with pytest.raises(ValueError):
        compute_px_per_mm(200.0, 0.0, 0.0)
