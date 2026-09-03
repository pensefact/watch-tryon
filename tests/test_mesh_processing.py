import numpy as np
import trimesh
import pytest
from pathlib import Path

from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.models import WatchSpec


def _make_unit_box_glb(path: Path) -> Path:
    """Create a 1x1x1 unit box mesh and save as GLB."""
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh.export(str(path))
    return path


def _make_spec(**overrides) -> WatchSpec:
    defaults = dict(
        reference="test",
        name="Test Watch",
        case_diameter_mm=42.0,
        lug_to_lug_mm=48.0,
        thickness_mm=13.0,
        band_width_mm=20.0,
        case_shape="round",
        photo_paths=[],
    )
    defaults.update(overrides)
    return WatchSpec(**defaults)


def test_scales_to_spec_dimensions(tmp_path):
    glb_path = _make_unit_box_glb(tmp_path / "box.glb")
    spec = _make_spec(case_diameter_mm=42.0, lug_to_lug_mm=48.0, thickness_mm=13.0)

    mesh = load_and_scale_mesh(glb_path, spec)
    extents = mesh.bounding_box.extents

    assert abs(extents[0] - 42.0) < 0.1, f"X (diameter) should be 42mm, got {extents[0]}"
    assert abs(extents[1] - 48.0) < 0.1, f"Y (lug-to-lug) should be 48mm, got {extents[1]}"
    assert abs(extents[2] - 13.0) < 0.1, f"Z (thickness) should be 13mm, got {extents[2]}"


def test_centers_mesh_at_origin(tmp_path):
    glb_path = _make_unit_box_glb(tmp_path / "box.glb")
    spec = _make_spec()

    mesh = load_and_scale_mesh(glb_path, spec)
    centroid = mesh.bounding_box.centroid

    assert abs(centroid[0]) < 0.1
    assert abs(centroid[1]) < 0.1
    assert abs(centroid[2]) < 0.1


def test_preserves_mesh_validity(tmp_path):
    glb_path = _make_unit_box_glb(tmp_path / "box.glb")
    spec = _make_spec()

    mesh = load_and_scale_mesh(glb_path, spec)
    assert mesh.is_volume or len(mesh.faces) > 0
