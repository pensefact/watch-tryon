import numpy as np
from src.wrist_pipeline.lighting import estimate_light_direction


def test_light_from_top_left():
    """Bright region in top-left → light direction should have negative x, negative y."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[0:50, 0:50] = 255  # bright top-left
    dx, dy, dz = estimate_light_direction(image)
    assert dx < 0, f"Expected negative x for top-left light, got {dx}"
    assert dy < 0, f"Expected negative y for top-left light, got {dy}"


def test_light_from_right():
    """Bright region on right → light direction should have positive x."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[50:150, 150:200] = 255  # bright right side
    dx, dy, dz = estimate_light_direction(image)
    assert dx > 0, f"Expected positive x for right light, got {dx}"


def test_returns_normalized():
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    image[0:100, 100:200] = 255
    dx, dy, dz = estimate_light_direction(image)
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    assert abs(length - 1.0) < 0.01, f"Direction should be normalized, length={length}"


def test_uniform_image_returns_overhead():
    """Uniform brightness → default to overhead lighting (0, 0, 1)."""
    image = np.full((200, 200, 3), 128, dtype=np.uint8)
    dx, dy, dz = estimate_light_direction(image)
    assert abs(dz - 1.0) < 0.3, "Uniform image should default to roughly overhead"
