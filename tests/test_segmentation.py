import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.wrist_pipeline.segmentation import segment_wrist, WristSegment, measure_wrist_width


def test_measure_wrist_width_horizontal():
    """A horizontal band mask should have width = band height."""
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:120, 50:250] = 255  # 40px tall horizontal band
    width = measure_wrist_width(mask, forearm_angle_rad=0.0)
    assert abs(width - 40.0) < 5.0


def test_measure_wrist_width_vertical():
    """A vertical band mask measured along vertical forearm."""
    mask = np.zeros((300, 200), dtype=np.uint8)
    mask[50:250, 80:120] = 255  # 40px wide vertical band
    import math
    width = measure_wrist_width(mask, forearm_angle_rad=math.pi / 2)
    assert abs(width - 40.0) < 5.0


def test_wrist_segment_dataclass():
    mask = np.zeros((100, 100), dtype=np.uint8)
    contour = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
    seg = WristSegment(mask=mask, contour=contour, width_px=50.0)
    assert seg.width_px == 50.0


def test_segment_wrist_returns_segment():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_mask = np.zeros((480, 640), dtype=np.uint8)
    fake_mask[200:280, 200:440] = 255

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = (
        np.array([fake_mask]),  # masks
        np.array([0.95]),       # scores
        None,                   # logits
    )

    result = segment_wrist(
        image=image,
        point_prompt=(320.0, 240.0),
        forearm_angle_rad=0.0,
        predictor=mock_predictor,
    )

    assert isinstance(result, WristSegment)
    assert result.mask.shape == (480, 640)
    assert result.width_px > 0
