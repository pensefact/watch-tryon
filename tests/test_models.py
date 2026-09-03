import numpy as np
from src.models import WatchSpec, PlacementSpec, RenderPasses
from pathlib import Path


def test_watch_spec_creation():
    spec = WatchSpec(
        reference="126710BLNR",
        name="Rolex GMT-Master II",
        case_diameter_mm=42.3,
        lug_to_lug_mm=48.7,
        thickness_mm=13.1,
        band_width_mm=20.0,
        case_shape="round",
        photo_paths=[Path("watch_front.png")],
    )
    assert spec.case_diameter_mm == 42.3
    assert spec.case_shape == "round"


def test_watch_spec_validates_positive_dimensions():
    try:
        WatchSpec(
            reference="test",
            name="test",
            case_diameter_mm=-1.0,
            lug_to_lug_mm=48.0,
            thickness_mm=13.0,
            band_width_mm=20.0,
            case_shape="round",
            photo_paths=[],
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_placement_spec_creation():
    mask = np.zeros((100, 100), dtype=np.uint8)
    spec = PlacementSpec(
        center_x=50.0,
        center_y=50.0,
        rotation_deg=15.0,
        px_per_mm=5.0,
        light_direction=(0.5, -0.7, 0.5),
        wrist_mask=mask,
    )
    assert spec.px_per_mm == 5.0
    assert spec.wrist_mask.shape == (100, 100)


def test_render_passes_creation():
    color = np.zeros((100, 100, 4), dtype=np.uint8)
    shadow = np.zeros((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=np.uint8)
    passes = RenderPasses(color=color, shadow=shadow, mask=mask)
    assert passes.color.shape == (100, 100, 4)
