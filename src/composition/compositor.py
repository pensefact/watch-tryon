import cv2
import numpy as np

from src.models import PlacementSpec, RenderPasses


def place_and_rotate(
    foreground: np.ndarray,
    background_shape: tuple[int, int],
    placement: PlacementSpec,
) -> np.ndarray:
    """Place and rotate a foreground RGBA image onto a canvas matching the background size."""
    fh, fw = foreground.shape[:2]
    bh, bw = background_shape

    canvas = np.zeros((bh, bw, 4), dtype=np.uint8)

    # Rotation matrix around the foreground center
    center_fg = (fw / 2, fh / 2)
    rot_mat = cv2.getRotationMatrix2D(center_fg, -placement.rotation_deg, 1.0)

    # Shift so that foreground center lands at placement center
    rot_mat[0, 2] += placement.center_x - fw / 2
    rot_mat[1, 2] += placement.center_y - fh / 2

    cv2.warpAffine(foreground, rot_mat, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
    return canvas


def composite_watch(
    background: np.ndarray,
    passes: RenderPasses,
    placement: PlacementSpec,
) -> np.ndarray:
    bh, bw = background.shape[:2]
    result = background.copy()

    # Place color pass
    placed_color = place_and_rotate(passes.color, (bh, bw), placement)
    alpha = placed_color[:, :, 3].astype(np.float32) / 255.0

    # Apply wrist occlusion mask — where wrist_mask is white, the wrist is in front of the watch
    if placement.wrist_mask is not None and placement.wrist_mask.shape == (bh, bw):
        occlusion = 1.0 - (placement.wrist_mask.astype(np.float32) / 255.0)
        alpha *= occlusion

    # Feather edges with slight Gaussian blur on alpha
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)

    # Place shadow pass
    shadow_rgba = np.zeros((passes.shadow.shape[0], passes.shadow.shape[1], 4), dtype=np.uint8)
    shadow_val = (255 - passes.shadow)  # invert: dark where shadow is strong
    shadow_rgba[:, :, 3] = shadow_val
    placed_shadow = place_and_rotate(shadow_rgba, (bh, bw), placement)
    shadow_alpha = placed_shadow[:, :, 3].astype(np.float32) / 255.0 * 0.4  # subtle shadow

    # Darken background where shadow falls
    for c in range(3):
        result[:, :, c] = (result[:, :, c].astype(np.float32) * (1.0 - shadow_alpha)).astype(np.uint8)

    # Alpha-blend watch onto result
    for c in range(3):
        fg = placed_color[:, :, c].astype(np.float32)
        bg = result[:, :, c].astype(np.float32)
        result[:, :, c] = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)

    return result
