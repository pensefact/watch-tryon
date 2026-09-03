import numpy as np
from src.composition.compositor import composite_watch, place_and_rotate
from src.models import PlacementSpec, RenderPasses


def _make_placement(**overrides) -> PlacementSpec:
    defaults = dict(
        center_x=150.0,
        center_y=100.0,
        rotation_deg=0.0,
        px_per_mm=3.0,
        light_direction=(0.0, 0.0, 1.0),
        wrist_mask=np.zeros((200, 300), dtype=np.uint8),
    )
    defaults.update(overrides)
    return PlacementSpec(**defaults)


def _make_passes(h=50, w=50) -> RenderPasses:
    color = np.zeros((h, w, 4), dtype=np.uint8)
    color[:, :, 0] = 128  # blue channel (BGR+A)
    color[:, :, 3] = 255  # fully opaque
    shadow = np.full((h, w), 200, dtype=np.uint8)
    mask = np.full((h, w), 255, dtype=np.uint8)
    return RenderPasses(color=color, shadow=shadow, mask=mask)


def test_place_and_rotate_centers_on_target():
    """Placed image center should land near placement center."""
    foreground = np.full((40, 40, 4), 255, dtype=np.uint8)
    background_shape = (200, 300)
    placement = _make_placement(center_x=150.0, center_y=100.0, rotation_deg=0.0)

    result = place_and_rotate(foreground, background_shape, placement)
    assert result.shape == (200, 300, 4)

    # The placed content should be nonzero near the center
    region = result[80:120, 130:170]
    assert region[:, :, 3].sum() > 0, "Content should be placed near center"


def test_composite_produces_correct_shape():
    background = np.full((200, 300, 3), 180, dtype=np.uint8)
    passes = _make_passes()
    placement = _make_placement()

    result = composite_watch(background, passes, placement, watch_diameter_mm=50.0 / 3.0)
    assert result.shape == (200, 300, 3)
    assert result.dtype == np.uint8


def test_composite_modifies_background():
    background = np.full((200, 300, 3), 180, dtype=np.uint8)
    passes = _make_passes()
    placement = _make_placement()

    result = composite_watch(background, passes, placement, watch_diameter_mm=50.0 / 3.0)
    # Result should differ from uniform background where watch was placed
    diff = np.abs(result.astype(int) - 180)
    assert diff.sum() > 0, "Composite should modify the background"


def test_composite_scales_watch_to_px_per_mm():
    """Watch render should be resized so its width matches watch_diameter_mm * px_per_mm."""
    background = np.full((200, 300, 3), 0, dtype=np.uint8)
    passes = _make_passes(h=50, w=50)
    placement = _make_placement(px_per_mm=4.0)

    result = composite_watch(background, passes, placement, watch_diameter_mm=20.0)
    # Expected target width = 20mm * 4 px/mm = 80px, larger than the source 50px render.
    row = result[int(placement.center_y), :, 0]
    nonzero_cols = np.nonzero(row)[0]
    assert nonzero_cols.size > 0
    width = nonzero_cols.max() - nonzero_cols.min() + 1
    assert 70 <= width <= 90


def test_composite_respects_wrist_mask_occlusion():
    """Pixels where wrist_mask is white should show background (wrist), not watch."""
    background = np.full((200, 300, 3), 180, dtype=np.uint8)
    passes = _make_passes()

    wrist_mask = np.zeros((200, 300), dtype=np.uint8)
    wrist_mask[90:110, 140:160] = 255  # occlude center of watch placement

    placement = _make_placement(wrist_mask=wrist_mask)
    result = composite_watch(background, passes, placement, watch_diameter_mm=50.0 / 3.0)

    # In the occluded region, result should be close to original background
    occluded = result[95:105, 145:155]
    assert np.mean(np.abs(occluded.astype(int) - 180)) < 30
