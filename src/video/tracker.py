import math
from pathlib import Path

import cv2
import numpy as np


class OneEuroFilter:
    """1-Euro filter for smoothing noisy real-time signals."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _smoothing_factor(t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t: float, x: float) -> float:
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        t_e = t - self._t_prev
        if t_e <= 0:
            t_e = 1e-6

        # Derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


def track_wrist_video(
    video_path: Path,
    initial_point: tuple[float, float],
    sam_video_predictor,
) -> list[np.ndarray]:
    """Track wrist across all frames using SAM 2 video propagation."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    # Initialize SAM 2 video predictor with first frame + point prompt
    state = sam_video_predictor.init_state(video_path=str(video_path))
    sam_video_predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=np.array([[initial_point[0], initial_point[1]]]),
        labels=np.array([1]),
    )

    # Propagate through video
    masks = []
    for frame_idx, obj_ids, mask_logits in sam_video_predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
        masks.append(mask)

    return masks
