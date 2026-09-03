import math
from dataclasses import dataclass

import numpy as np
import mediapipe as mp

# NOTE: The legacy `mp.solutions.hands` API is unavailable in mediapipe>=1.0
# (installed here: mediapipe 1.0.1 on Python 3.14, which only ships the new
# `mp.tasks` API). Guard the module-level lookup so import doesn't crash;
# `mp_hands` is patched wholesale in tests, so real inference is never
# exercised there. See task-3-report.md for details.
try:
    mp_hands = mp.solutions.hands
except AttributeError:
    mp_hands = None


@dataclass
class WristLandmarks:
    wrist_point: tuple[float, float]
    index_mcp: tuple[float, float]
    pinky_mcp: tuple[float, float]
    forearm_angle_rad: float
    pose_angle_rad: float


def detect_wrist(image: np.ndarray) -> WristLandmarks:
    h, w = image.shape[:2]

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        result = hands.process(image)

    if not result.multi_hand_landmarks:
        raise ValueError("No hand detected in image")

    landmarks = result.multi_hand_landmarks[0].landmark
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    wrist_px = (wrist.x * w, wrist.y * h)
    index_px = (index_mcp.x * w, index_mcp.y * h)
    pinky_px = (pinky_mcp.x * w, pinky_mcp.y * h)

    # Hand center is midpoint of index and pinky MCP
    hand_center_x = (index_px[0] + pinky_px[0]) / 2
    hand_center_y = (index_px[1] + pinky_px[1]) / 2

    # Forearm angle: direction from hand center toward wrist
    dx = wrist_px[0] - hand_center_x
    dy = wrist_px[1] - hand_center_y
    forearm_angle = math.atan2(dy, dx)

    # Pose angle: rough estimate of wrist rotation from camera.
    # Use the ratio of MCP spread to wrist-to-MCP distance.
    # When wrist faces camera, MCP spread is wide. When edge-on, it's narrow.
    mcp_spread = math.sqrt((index_px[0] - pinky_px[0]) ** 2 + (index_px[1] - pinky_px[1]) ** 2)
    wrist_to_hand = math.sqrt(dx ** 2 + dy ** 2)
    spread_ratio = mcp_spread / max(wrist_to_hand, 1.0)
    # Map ratio to angle: ~0.8+ = front-facing (0 rad), ~0.2 = edge-on (pi/2)
    pose_angle = max(0.0, min(math.pi / 2, (1.0 - min(spread_ratio / 0.8, 1.0)) * math.pi / 2))

    return WristLandmarks(
        wrist_point=wrist_px,
        index_mcp=index_px,
        pinky_mcp=pinky_px,
        forearm_angle_rad=forearm_angle,
        pose_angle_rad=pose_angle,
    )
