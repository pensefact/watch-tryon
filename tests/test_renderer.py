import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.composition.renderer import (
    build_render_script,
    parse_render_output,
    render_watch,
)
from src.models import PlacementSpec, RenderPasses


def test_build_render_script_contains_mesh_path():
    script = build_render_script(
        mesh_path=Path("/tmp/watch.glb"),
        output_dir=Path("/tmp/renders"),
        image_width=640,
        image_height=480,
        light_direction=(0.5, -0.7, 0.5),
        camera_angle_rad=1.57,
    )
    assert "/tmp/watch.glb" in script
    assert "/tmp/renders" in script
    assert "color.png" in script
    assert "shadow.png" in script


def test_parse_render_output(tmp_path):
    # Create fake render output images
    import cv2
    color = np.zeros((480, 640, 4), dtype=np.uint8)
    color[:, :, 3] = 255
    shadow = np.zeros((480, 640), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)

    cv2.imwrite(str(tmp_path / "color.png"), color)
    cv2.imwrite(str(tmp_path / "shadow.png"), shadow)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)

    passes = parse_render_output(tmp_path)
    assert isinstance(passes, RenderPasses)
    assert passes.color.shape[:2] == (480, 640)
    assert passes.shadow.shape == (480, 640)
    assert passes.mask.shape == (480, 640)


def test_render_watch_calls_blender(tmp_path):
    mesh_path = tmp_path / "watch.glb"
    mesh_path.write_bytes(b"fake")
    output_dir = tmp_path / "renders"
    output_dir.mkdir()

    # Create fake output files that Blender "would" produce
    import cv2
    color = np.zeros((480, 640, 4), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "color.png"), color)
    cv2.imwrite(str(output_dir / "shadow.png"), np.zeros((480, 640), dtype=np.uint8))
    cv2.imwrite(str(output_dir / "mask.png"), np.zeros((480, 640), dtype=np.uint8))

    mask_arr = np.zeros((480, 640), dtype=np.uint8)
    placement = PlacementSpec(
        center_x=320.0, center_y=240.0, rotation_deg=15.0,
        px_per_mm=5.0, light_direction=(0.5, -0.7, 0.5), wrist_mask=mask_arr,
    )

    with patch("src.composition.renderer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Render complete", stderr="")

        passes = render_watch(mesh_path, placement, output_dir, image_size=(640, 480))

    assert isinstance(passes, RenderPasses)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "blender" in cmd[0]
    assert "--background" in cmd
