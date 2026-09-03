import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.wrist_pipeline.detection import detect_wrist, WristLandmarks


def _mock_hand_landmarks(wrist=(0.5, 0.8), index_mcp=(0.45, 0.6), pinky_mcp=(0.55, 0.6)):
    """Create a mock MediaPipe hand landmarks result."""
    landmark_list = MagicMock()

    class FakeLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [None] * 21
    landmarks[0] = FakeLandmark(*wrist)
    landmarks[5] = FakeLandmark(*index_mcp)
    landmarks[17] = FakeLandmark(*pinky_mcp)
    landmark_list.landmark = landmarks
    return landmark_list


def test_detect_wrist_returns_landmarks():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_result.multi_hand_landmarks = [_mock_hand_landmarks()]

    with patch("src.wrist_pipeline.detection.mp_hands") as mock_hands:
        mock_instance = MagicMock()
        mock_instance.process.return_value = mock_result
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_hands.Hands.return_value = mock_instance

        result = detect_wrist(image)

    assert isinstance(result, WristLandmarks)
    assert result.wrist_point == (320.0, 384.0)  # 0.5*640, 0.8*480


def test_detect_wrist_computes_forearm_angle():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Hand above wrist, centered — forearm should be roughly vertical (pi/2 or -pi/2)
    mock_result = MagicMock()
    mock_result.multi_hand_landmarks = [
        _mock_hand_landmarks(wrist=(0.5, 0.9), index_mcp=(0.45, 0.5), pinky_mcp=(0.55, 0.5))
    ]

    with patch("src.wrist_pipeline.detection.mp_hands") as mock_hands:
        mock_instance = MagicMock()
        mock_instance.process.return_value = mock_result
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_hands.Hands.return_value = mock_instance

        result = detect_wrist(image)

    # Forearm goes from hand center toward wrist — direction should be roughly downward
    assert result.forearm_angle_rad is not None


def test_detect_wrist_raises_when_no_hand():
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_result.multi_hand_landmarks = None

    with patch("src.wrist_pipeline.detection.mp_hands") as mock_hands:
        mock_instance = MagicMock()
        mock_instance.process.return_value = mock_result
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_hands.Hands.return_value = mock_instance

        with pytest.raises(ValueError, match="No hand detected"):
            detect_wrist(image)
