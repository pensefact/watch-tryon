# Watch Virtual Try-On POC — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a POC that composites a 3D-reconstructed watch onto a wrist photo at physically accurate scale, given a watch reference, wrist photo, and wrist circumference.

**Architecture:** Two independent pipelines (watch + wrist) feed a composition engine. The watch pipeline reconstructs a 3D mesh from photos via PiAPI Trellis 2 and scales it to real-world dimensions. The wrist pipeline detects/segments the wrist via MediaPipe + SAM 2 and computes a placement spec. Blender renders the watch headlessly; OpenCV composites it onto the original photo. Gradio provides the UI.

**Tech Stack:** Python 3.11+, trimesh, mediapipe, segment-anything-2, bpy (Blender Python), OpenCV, numpy, httpx, gradio

**Spec:** `2026-09-03-watch-tryon-design.md`

## Global Constraints

- Python 3.11+
- All heavy models (SAM 2, Blender) must work headless — no display server required
- Watch dimensions are always in millimeters
- Image coordinates use OpenCV convention (origin top-left, y increases downward)
- Meshes use right-handed coordinate system: X = width (diameter), Y = height (lug-to-lug), Z = depth (thickness), dial face = +Z
- API keys stored in `.env`, loaded via `python-dotenv`, never committed
- Blender 4.x required (for bpy Python module)
- Cache 3D reconstruction results to avoid redundant API calls (keyed by watch reference)

## File Structure

```
watch-tryon/
├── src/
│   ├── __init__.py
│   ├── models.py              # WatchSpec, PlacementSpec, RenderPasses dataclasses
│   ├── watch_pipeline/
│   │   ├── __init__.py
│   │   ├── reconstruction.py  # PiAPI Trellis 2 client — image→3D mesh
│   │   └── mesh_processing.py # Load, scale, orient mesh with trimesh
│   ├── wrist_pipeline/
│   │   ├── __init__.py
│   │   ├── detection.py       # MediaPipe hand landmark detection
│   │   ├── segmentation.py    # SAM 2 point-prompt segmentation
│   │   └── calibration.py     # Geometry math: circumference + pixels → px/mm
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── renderer.py        # Blender headless multi-pass rendering
│   │   └── compositor.py      # OpenCV compositing: layers, shadows, feathering
│   ├── video/
│   │   ├── __init__.py
│   │   ├── tracker.py         # SAM 2 video propagation + 1-Euro smoothing
│   │   └── pipeline.py        # Per-frame render → ffmpeg encode
│   ├── pipeline.py            # End-to-end orchestration (still + video)
│   └── app.py                 # Gradio UI
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_reconstruction.py
│   ├── test_mesh_processing.py
│   ├── test_detection.py
│   ├── test_segmentation.py
│   ├── test_calibration.py
│   ├── test_renderer.py
│   ├── test_compositor.py
│   └── fixtures/              # Sample images, meshes for tests
│       └── README.md
├── cache/                     # Cached 3D meshes (gitignored)
├── outputs/                   # Generated composites (gitignored)
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

### Task 1: Project Scaffolding + Data Models + Scale Calibration

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/models.py`
- Create: `src/wrist_pipeline/__init__.py`
- Create: `src/wrist_pipeline/calibration.py`
- Create: `tests/__init__.py`
- Create: `tests/test_models.py`
- Create: `tests/test_calibration.py`
- Create: `tests/fixtures/README.md`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `WatchSpec(reference: str, name: str, case_diameter_mm: float, lug_to_lug_mm: float, thickness_mm: float, band_width_mm: float, case_shape: str, photo_paths: list[Path])` dataclass
  - `PlacementSpec(center_x: float, center_y: float, rotation_deg: float, px_per_mm: float, light_direction: tuple[float, float, float], wrist_mask: np.ndarray)` dataclass
  - `RenderPasses(color: np.ndarray, shadow: np.ndarray, mask: np.ndarray)` dataclass
  - `compute_px_per_mm(wrist_width_px: float, circumference_mm: float, pose_angle_rad: float) -> float`
  - `estimate_visible_width(circumference_mm: float, pose_angle_rad: float) -> float`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "watch-tryon"
version = "0.1.0"
description = "Watch virtual try-on POC"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Create requirements.txt**

```
numpy>=1.26
opencv-python>=4.9
trimesh>=4.0
mediapipe>=0.10
httpx>=0.27
Pillow>=10.0
gradio>=4.0
python-dotenv>=1.0
pytest>=8.0
scipy>=1.12
```

Note: `bpy` (Blender Python), `segment-anything-2`, and `torch` are installed separately — they have complex GPU dependencies. Add install instructions in `.env.example`.

- [ ] **Step 3: Create .env.example**

```
# PiAPI Trellis 2
PIAPI_API_KEY=your_piapi_key_here

# Tripo AI (optional comparison)
TRIPO_API_KEY=your_tripo_key_here

# Install notes:
# pip install -r requirements.txt
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install segment-anything-2
# For Blender Python: pip install bpy==4.1.0
#   OR use system Blender with: blender --background --python script.py
```

- [ ] **Step 4: Create src/__init__.py and tests/__init__.py**

Both empty files. Also create `tests/fixtures/README.md`:

```markdown
# Test Fixtures

Place sample images here for integration tests:
- `wrist_front.jpg` — wrist photo facing camera
- `wrist_angled.jpg` — wrist at ~30 degree angle
- `watch_front.png` — front-facing watch dial shot
```

- [ ] **Step 5: Write failing tests for data models**

`tests/test_models.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 7: Implement data models**

`src/models.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WatchSpec:
    reference: str
    name: str
    case_diameter_mm: float
    lug_to_lug_mm: float
    thickness_mm: float
    band_width_mm: float
    case_shape: str
    photo_paths: list[Path] = field(default_factory=list)

    def __post_init__(self):
        for dim_name in ("case_diameter_mm", "lug_to_lug_mm", "thickness_mm", "band_width_mm"):
            if getattr(self, dim_name) <= 0:
                raise ValueError(f"{dim_name} must be positive, got {getattr(self, dim_name)}")
        if self.case_shape not in ("round", "square", "tonneau"):
            raise ValueError(f"Unknown case_shape: {self.case_shape}")


@dataclass
class PlacementSpec:
    center_x: float
    center_y: float
    rotation_deg: float
    px_per_mm: float
    light_direction: tuple[float, float, float]
    wrist_mask: np.ndarray


@dataclass
class RenderPasses:
    color: np.ndarray
    shadow: np.ndarray
    mask: np.ndarray
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: all 4 tests PASS

- [ ] **Step 9: Write failing tests for scale calibration**

`tests/test_calibration.py`:

```python
import math
import pytest
from src.wrist_pipeline.calibration import compute_px_per_mm, estimate_visible_width


def test_visible_width_front_facing():
    """When wrist faces camera straight on (angle=0), visible width = diameter = circumference/pi."""
    circumference_mm = 175.0
    width = estimate_visible_width(circumference_mm, pose_angle_rad=0.0)
    expected_diameter = circumference_mm / math.pi
    assert abs(width - expected_diameter) < 0.01


def test_visible_width_angled():
    """At 45 degrees, visible width should be less than the full diameter."""
    circumference_mm = 175.0
    front_width = estimate_visible_width(circumference_mm, pose_angle_rad=0.0)
    angled_width = estimate_visible_width(circumference_mm, pose_angle_rad=math.pi / 4)
    assert angled_width < front_width
    assert angled_width > 0


def test_visible_width_edge_on():
    """At 90 degrees (edge-on), visible width approaches the minor axis of the ellipse."""
    circumference_mm = 175.0
    width = estimate_visible_width(circumference_mm, pose_angle_rad=math.pi / 2)
    assert width > 0
    front_width = estimate_visible_width(circumference_mm, pose_angle_rad=0.0)
    assert width < front_width


def test_px_per_mm_basic():
    """If wrist is 200px wide in image and 55.7mm actual width, scale = 200/55.7."""
    circumference_mm = 175.0
    wrist_width_px = 200.0
    px_per_mm = compute_px_per_mm(wrist_width_px, circumference_mm, pose_angle_rad=0.0)
    expected_diameter = circumference_mm / math.pi
    expected_scale = wrist_width_px / expected_diameter
    assert abs(px_per_mm - expected_scale) < 0.01


def test_px_per_mm_larger_wrist():
    """Larger wrist at same pixel width = smaller px/mm (further from camera or thinner wrist)."""
    small_wrist = compute_px_per_mm(200.0, 160.0, 0.0)
    large_wrist = compute_px_per_mm(200.0, 200.0, 0.0)
    assert small_wrist > large_wrist


def test_px_per_mm_rejects_invalid():
    with pytest.raises(ValueError):
        compute_px_per_mm(0.0, 175.0, 0.0)
    with pytest.raises(ValueError):
        compute_px_per_mm(200.0, 0.0, 0.0)
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 11: Implement scale calibration**

`src/wrist_pipeline/__init__.py`: empty file.

`src/wrist_pipeline/calibration.py`:

```python
import math


# Wrist cross-section modeled as an ellipse.
# Major axis (front-facing) = diameter = circumference / pi.
# Minor axis (depth) estimated at ~0.7× major axis (typical wrist is flatter than it is wide).
_WRIST_ECCENTRICITY = 0.7


def estimate_visible_width(circumference_mm: float, pose_angle_rad: float) -> float:
    """Estimate the visible wrist width in mm given circumference and viewing angle.

    angle=0 means camera faces the top of the wrist (dial side).
    angle=pi/2 means camera sees the wrist edge-on (from the thumb/pinky side).
    """
    major_axis = circumference_mm / math.pi
    minor_axis = major_axis * _WRIST_ECCENTRICITY

    # Visible width is the ellipse projection at the given angle
    a = major_axis / 2
    b = minor_axis / 2
    visible_half_width = math.sqrt(
        (a * math.cos(pose_angle_rad)) ** 2 + (b * math.sin(pose_angle_rad)) ** 2
    )
    return 2 * visible_half_width


def compute_px_per_mm(
    wrist_width_px: float, circumference_mm: float, pose_angle_rad: float
) -> float:
    """Compute pixels-per-millimeter at the wrist surface."""
    if wrist_width_px <= 0:
        raise ValueError(f"wrist_width_px must be positive, got {wrist_width_px}")
    if circumference_mm <= 0:
        raise ValueError(f"circumference_mm must be positive, got {circumference_mm}")

    visible_width_mm = estimate_visible_width(circumference_mm, pose_angle_rad)
    return wrist_width_px / visible_width_mm
```

- [ ] **Step 12: Run all tests**

Run: `pytest tests/test_models.py tests/test_calibration.py -v`
Expected: all 10 tests PASS

- [ ] **Step 13: Commit**

```bash
git add src/__init__.py src/models.py src/wrist_pipeline/__init__.py src/wrist_pipeline/calibration.py \
       tests/__init__.py tests/test_models.py tests/test_calibration.py tests/fixtures/README.md \
       pyproject.toml requirements.txt .env.example
git commit -m "feat: project scaffolding, data models, scale calibration"
```

---

### Task 2: Watch Pipeline — 3D Reconstruction + Mesh Processing

**Files:**
- Create: `src/watch_pipeline/__init__.py`
- Create: `src/watch_pipeline/reconstruction.py`
- Create: `src/watch_pipeline/mesh_processing.py`
- Create: `tests/test_reconstruction.py`
- Create: `tests/test_mesh_processing.py`

**Interfaces:**
- Consumes: `WatchSpec` from `src/models.py`
- Produces:
  - `reconstruct_watch(photo_path: Path, api_key: str, cache_dir: Path) -> Path` — calls PiAPI, returns path to downloaded `.glb` mesh file
  - `load_and_scale_mesh(mesh_path: Path, spec: WatchSpec) -> trimesh.Trimesh` — loads mesh, scales to real-world mm, orients consistently
  - `get_cached_mesh(reference: str, cache_dir: Path) -> Path | None` — returns cached mesh path if it exists

- [ ] **Step 1: Write failing tests for reconstruction client**

`tests/test_reconstruction.py`:

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from src.watch_pipeline.reconstruction import reconstruct_watch, get_cached_mesh


def test_get_cached_mesh_returns_none_when_missing(tmp_path):
    result = get_cached_mesh("nonexistent-ref", cache_dir=tmp_path)
    assert result is None


def test_get_cached_mesh_returns_path_when_exists(tmp_path):
    mesh_file = tmp_path / "test-ref.glb"
    mesh_file.write_bytes(b"fake mesh data")
    result = get_cached_mesh("test-ref", cache_dir=tmp_path)
    assert result == mesh_file


@pytest.mark.asyncio
async def test_reconstruct_watch_calls_api(tmp_path):
    fake_glb = b"\x67\x6c\x54\x46"  # glTF magic bytes

    mock_response_create = MagicMock()
    mock_response_create.status_code = 200
    mock_response_create.json.return_value = {"task_id": "task-123"}

    mock_response_poll = MagicMock()
    mock_response_poll.status_code = 200
    mock_response_poll.json.return_value = {
        "status": "completed",
        "output": {"model_url": "https://example.com/mesh.glb"},
    }

    mock_response_download = MagicMock()
    mock_response_download.status_code = 200
    mock_response_download.content = fake_glb

    with patch("src.watch_pipeline.reconstruction.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response_create)
        mock_client.get = AsyncMock(side_effect=[mock_response_poll, mock_response_download])
        mock_client_cls.return_value = mock_client

        result = await reconstruct_watch(
            photo_path=Path("tests/fixtures/watch_front.png"),
            api_key="test-key",
            cache_dir=tmp_path,
            reference="test-watch",
        )

    assert result.exists()
    assert result.suffix == ".glb"
    assert result.read_bytes() == fake_glb


def test_reconstruct_watch_uses_cache(tmp_path):
    mesh_file = tmp_path / "cached-ref.glb"
    mesh_file.write_bytes(b"cached mesh")
    result = get_cached_mesh("cached-ref", cache_dir=tmp_path)
    assert result == mesh_file
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reconstruction.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement reconstruction client**

`src/watch_pipeline/__init__.py`: empty file.

`src/watch_pipeline/reconstruction.py`:

```python
import asyncio
import base64
import time
from pathlib import Path

import httpx

PIAPI_BASE_URL = "https://api.piapi.ai/api/v1"


def get_cached_mesh(reference: str, cache_dir: Path) -> Path | None:
    safe_name = reference.replace("/", "_").replace(" ", "_")
    mesh_path = cache_dir / f"{safe_name}.glb"
    return mesh_path if mesh_path.exists() else None


async def reconstruct_watch(
    photo_path: Path,
    api_key: str,
    cache_dir: Path,
    reference: str,
    poll_interval: float = 5.0,
    timeout: float = 120.0,
) -> Path:
    cached = get_cached_mesh(reference, cache_dir)
    if cached is not None:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    image_b64 = base64.b64encode(photo_path.read_bytes()).decode()

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Submit reconstruction task
        resp = await client.post(
            f"{PIAPI_BASE_URL}/task",
            headers={"X-API-Key": api_key},
            json={
                "model": "trellis",
                "task_type": "image-to-3d",
                "input": {"image": f"data:image/png;base64,{image_b64}"},
            },
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]

        # Poll for completion
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            poll_resp = await client.get(
                f"{PIAPI_BASE_URL}/task/{task_id}",
                headers={"X-API-Key": api_key},
            )
            poll_resp.raise_for_status()
            data = poll_resp.json()

            if data["status"] == "completed":
                model_url = data["output"]["model_url"]
                break
            elif data["status"] == "failed":
                raise RuntimeError(f"Reconstruction failed: {data}")

            await asyncio.sleep(poll_interval)
        else:
            raise TimeoutError(f"Reconstruction timed out after {timeout}s")

        # Download mesh
        dl_resp = await client.get(model_url)
        dl_resp.raise_for_status()

        safe_name = reference.replace("/", "_").replace(" ", "_")
        out_path = cache_dir / f"{safe_name}.glb"
        out_path.write_bytes(dl_resp.content)
        return out_path
```

- [ ] **Step 4: Run reconstruction tests**

Run: `pytest tests/test_reconstruction.py -v`
Expected: PASS (all 4 tests including the async mock test)

Note: you may need `pip install pytest-asyncio` and add `asyncio_mode = "auto"` to `pyproject.toml` under `[tool.pytest.ini_options]`.

- [ ] **Step 5: Write failing tests for mesh processing**

`tests/test_mesh_processing.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_mesh_processing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 7: Implement mesh processing**

`src/watch_pipeline/mesh_processing.py`:

```python
from pathlib import Path

import numpy as np
import trimesh

from src.models import WatchSpec


def load_and_scale_mesh(mesh_path: Path, spec: WatchSpec) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(str(mesh_path))

    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No meshes found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    # Center at origin
    mesh.vertices -= mesh.bounding_box.centroid

    # Scale each axis independently to match real-world dimensions
    current_extents = mesh.bounding_box.extents
    target_extents = np.array([
        spec.case_diameter_mm,  # X = width
        spec.lug_to_lug_mm,    # Y = height
        spec.thickness_mm,     # Z = depth
    ])

    scale_factors = target_extents / current_extents
    mesh.vertices *= scale_factors

    # Re-center after scaling (floating point drift)
    mesh.vertices -= mesh.bounding_box.centroid

    return mesh
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_reconstruction.py tests/test_mesh_processing.py -v`
Expected: all tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/watch_pipeline/ tests/test_reconstruction.py tests/test_mesh_processing.py
git commit -m "feat: watch pipeline — PiAPI reconstruction client + mesh scaling"
```

---

### Task 3: Wrist Detection + Segmentation

**Files:**
- Create: `src/wrist_pipeline/detection.py`
- Create: `src/wrist_pipeline/segmentation.py`
- Create: `tests/test_detection.py`
- Create: `tests/test_segmentation.py`

**Interfaces:**
- Consumes: nothing from prior tasks (operates on raw images)
- Produces:
  - `detect_wrist(image: np.ndarray) -> WristLandmarks` — returns `WristLandmarks(wrist_point: tuple[float, float], index_mcp: tuple[float, float], pinky_mcp: tuple[float, float], forearm_angle_rad: float, pose_angle_rad: float)`
  - `segment_wrist(image: np.ndarray, point_prompt: tuple[float, float], model: Sam2) -> WristSegment` — returns `WristSegment(mask: np.ndarray, contour: np.ndarray, width_px: float)`

- [ ] **Step 1: Write failing tests for wrist detection**

`tests/test_detection.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement wrist detection**

`src/wrist_pipeline/detection.py`:

```python
import math
from dataclasses import dataclass

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands


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
```

- [ ] **Step 4: Run detection tests**

Run: `pytest tests/test_detection.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Write failing tests for segmentation**

`tests/test_segmentation.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 7: Implement wrist segmentation**

`src/wrist_pipeline/segmentation.py`:

```python
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WristSegment:
    mask: np.ndarray
    contour: np.ndarray
    width_px: float


def measure_wrist_width(mask: np.ndarray, forearm_angle_rad: float) -> float:
    """Measure wrist width perpendicular to the forearm direction."""
    # Find contour points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask")

    largest = max(contours, key=cv2.contourArea)
    points = largest.reshape(-1, 2).astype(np.float64)

    # Project contour points onto the axis perpendicular to the forearm
    perp_angle = forearm_angle_rad + math.pi / 2
    perp_dir = np.array([math.cos(perp_angle), math.sin(perp_angle)])

    projections = points @ perp_dir
    width = projections.max() - projections.min()
    return float(width)


def segment_wrist(
    image: np.ndarray,
    point_prompt: tuple[float, float],
    forearm_angle_rad: float,
    predictor,
) -> WristSegment:
    """Segment the wrist using SAM 2 with a point prompt."""
    predictor.set_image(image)

    input_point = np.array([[point_prompt[0], point_prompt[1]]])
    input_label = np.array([1])  # foreground

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    # Take highest-scoring mask
    best_idx = np.argmax(scores)
    mask = (masks[best_idx] * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else np.array([])

    width = measure_wrist_width(mask, forearm_angle_rad)

    return WristSegment(
        mask=mask,
        contour=largest_contour.reshape(-1, 2) if len(largest_contour) > 0 else np.array([]),
        width_px=width,
    )
```

- [ ] **Step 8: Run all Task 3 tests**

Run: `pytest tests/test_detection.py tests/test_segmentation.py -v`
Expected: all 7 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/wrist_pipeline/detection.py src/wrist_pipeline/segmentation.py \
       tests/test_detection.py tests/test_segmentation.py
git commit -m "feat: wrist pipeline — MediaPipe detection + SAM 2 segmentation"
```

---

### Task 4: Lighting Estimation

**Files:**
- Create: `src/wrist_pipeline/lighting.py`
- Create: `tests/test_lighting.py`

**Interfaces:**
- Consumes: raw image as `np.ndarray`
- Produces:
  - `estimate_light_direction(image: np.ndarray) -> tuple[float, float, float]` — returns normalized (x, y, z) direction of dominant light source

- [ ] **Step 1: Write failing tests for lighting estimation**

`tests/test_lighting.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lighting.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement lighting estimation**

`src/wrist_pipeline/lighting.py`:

```python
import cv2
import numpy as np


def estimate_light_direction(image: np.ndarray) -> tuple[float, float, float]:
    """Estimate dominant light direction from image brightness distribution.

    Returns a normalized (x, y, z) direction vector.
    x: positive = right, y: positive = down, z: positive = toward camera.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float64)

    h, w = gray.shape
    total_brightness = gray.sum()

    if total_brightness < 1.0:
        return (0.0, 0.0, 1.0)

    # Compute brightness-weighted centroid
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    cx = (gray * x_coords).sum() / total_brightness
    cy = (gray * y_coords).sum() / total_brightness

    # Map centroid to [-1, 1] relative to image center
    dx = (cx - w / 2) / (w / 2)
    dy = (cy - h / 2) / (h / 2)

    # How concentrated is the brightness? If uniform, default to overhead.
    brightness_std = gray.std() / max(gray.mean(), 1.0)
    concentration = min(brightness_std / 0.5, 1.0)

    # Scale lateral components by concentration — uniform light stays overhead
    dx *= concentration
    dy *= concentration
    dz = max(0.3, 1.0 - abs(dx) - abs(dy))

    # Normalize
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    if length < 1e-6:
        return (0.0, 0.0, 1.0)

    return (float(dx / length), float(dy / length), float(dz / length))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_lighting.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/wrist_pipeline/lighting.py tests/test_lighting.py
git commit -m "feat: simple dominant-light-direction estimation"
```

---

### Task 5: Blender Headless Rendering

**Files:**
- Create: `src/composition/__init__.py`
- Create: `src/composition/renderer.py`
- Create: `tests/test_renderer.py`

**Interfaces:**
- Consumes:
  - `trimesh.Trimesh` mesh (from Task 2's `load_and_scale_mesh`)
  - `PlacementSpec` (from `src/models.py`)
- Produces:
  - `render_watch(mesh_path: Path, placement: PlacementSpec, output_dir: Path, image_size: tuple[int, int]) -> RenderPasses` — renders color, shadow, and mask passes

Note: Blender's `bpy` module is difficult to import in a standard pytest environment. The rendering script runs as a **subprocess** via `blender --background --python render_script.py`. The renderer module writes a temporary Python script, invokes Blender, and reads back the rendered images.

- [ ] **Step 1: Write failing tests for renderer**

`tests/test_renderer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement renderer**

`src/composition/__init__.py`: empty file.

`src/composition/renderer.py`:

```python
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.models import PlacementSpec, RenderPasses


def build_render_script(
    mesh_path: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
    light_direction: tuple[float, float, float],
    camera_angle_rad: float,
) -> str:
    return f"""
import bpy
import math
import os

# Clear default scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = {image_width}
scene.render.resolution_y = {image_height}
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Import mesh
bpy.ops.import_scene.gltf(filepath=r"{mesh_path}")

# Get imported objects
watch_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

# Camera setup
cam_data = bpy.data.cameras.new(name='Camera')
cam_data.lens = 28  # typical phone focal length
cam_obj = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Position camera to look at watch from dial side
import mathutils
cam_obj.location = (0, 0, 80)
cam_obj.rotation_euler = (0, 0, 0)

# Lighting
light_dir = {light_direction}
light_data = bpy.data.lights.new(name='Key', type='SUN')
light_data.energy = 3.0
light_obj = bpy.data.objects.new('Key', light_data)
scene.collection.objects.link(light_obj)
light_obj.rotation_euler = (
    math.atan2(-light_dir[1], light_dir[2]),
    math.atan2(light_dir[0], light_dir[2]),
    0,
)

# Use Cycles for quality (EEVEE as fallback)
scene.render.engine = 'BLENDER_EEVEE_NEXT'

output_dir = r"{output_dir}"

# Render color pass
scene.render.filepath = os.path.join(output_dir, "color.png")
bpy.ops.render.render(write_still=True)

# Render shadow pass (matte ground plane + shadow catcher)
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -5))
plane = bpy.context.active_object
plane.is_shadow_catcher = True
scene.render.filepath = os.path.join(output_dir, "shadow.png")
bpy.ops.render.render(write_still=True)
bpy.data.objects.remove(plane)

# Render mask pass (flat white on transparent)
for obj in watch_objects:
    for slot in obj.material_slots:
        mat = bpy.data.materials.new(name="Mask")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (1, 1, 1, 1)
        emission.inputs['Strength'].default_value = 1.0
        output = nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
        slot.material = mat

scene.render.filepath = os.path.join(output_dir, "mask.png")
bpy.ops.render.render(write_still=True)

print("Render complete")
"""


def parse_render_output(output_dir: Path) -> RenderPasses:
    color_path = output_dir / "color.png"
    shadow_path = output_dir / "shadow.png"
    mask_path = output_dir / "mask.png"

    color = cv2.imread(str(color_path), cv2.IMREAD_UNCHANGED)
    if color is None:
        raise FileNotFoundError(f"Color pass not found: {color_path}")

    shadow_raw = cv2.imread(str(shadow_path), cv2.IMREAD_UNCHANGED)
    if shadow_raw is None:
        raise FileNotFoundError(f"Shadow pass not found: {shadow_path}")
    shadow = cv2.cvtColor(shadow_raw, cv2.COLOR_BGR2GRAY) if len(shadow_raw.shape) == 3 else shadow_raw

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise FileNotFoundError(f"Mask pass not found: {mask_path}")
    mask = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY) if len(mask_raw.shape) == 3 else mask_raw

    return RenderPasses(color=color, shadow=shadow, mask=mask)


def render_watch(
    mesh_path: Path,
    placement: PlacementSpec,
    output_dir: Path,
    image_size: tuple[int, int] = (1024, 1024),
) -> RenderPasses:
    output_dir.mkdir(parents=True, exist_ok=True)

    script = build_render_script(
        mesh_path=mesh_path,
        output_dir=output_dir,
        image_width=image_size[0],
        image_height=image_size[1],
        light_direction=placement.light_direction,
        camera_angle_rad=0.0,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    result = subprocess.run(
        ["blender", "--background", "--python", script_path],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Blender render failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    return parse_render_output(output_dir)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_renderer.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/composition/__init__.py src/composition/renderer.py tests/test_renderer.py
git commit -m "feat: Blender headless multi-pass renderer"
```

---

### Task 6: Compositing + End-to-End Still Pipeline

**Files:**
- Create: `src/composition/compositor.py`
- Create: `src/pipeline.py`
- Create: `tests/test_compositor.py`

**Interfaces:**
- Consumes:
  - `RenderPasses` (from Task 5)
  - `PlacementSpec` (from Task 1)
  - Original wrist image as `np.ndarray`
- Produces:
  - `composite_watch(background: np.ndarray, passes: RenderPasses, placement: PlacementSpec) -> np.ndarray` — returns final composited BGR image
  - `run_still_pipeline(watch_spec: WatchSpec, wrist_image: np.ndarray, circumference_mm: float, api_key: str, cache_dir: Path, sam_predictor) -> np.ndarray` — end-to-end orchestration

- [ ] **Step 1: Write failing tests for compositor**

`tests/test_compositor.py`:

```python
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

    result = composite_watch(background, passes, placement)
    assert result.shape == (200, 300, 3)
    assert result.dtype == np.uint8


def test_composite_modifies_background():
    background = np.full((200, 300, 3), 180, dtype=np.uint8)
    passes = _make_passes()
    placement = _make_placement()

    result = composite_watch(background, passes, placement)
    # Result should differ from uniform background where watch was placed
    diff = np.abs(result.astype(int) - 180)
    assert diff.sum() > 0, "Composite should modify the background"


def test_composite_respects_wrist_mask_occlusion():
    """Pixels where wrist_mask is white should show background (wrist), not watch."""
    background = np.full((200, 300, 3), 180, dtype=np.uint8)
    passes = _make_passes()

    wrist_mask = np.zeros((200, 300), dtype=np.uint8)
    wrist_mask[90:110, 140:160] = 255  # occlude center of watch placement

    placement = _make_placement(wrist_mask=wrist_mask)
    result = composite_watch(background, passes, placement)

    # In the occluded region, result should be close to original background
    occluded = result[95:105, 145:155]
    assert np.mean(np.abs(occluded.astype(int) - 180)) < 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compositor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement compositor**

`src/composition/compositor.py`:

```python
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
```

- [ ] **Step 4: Run compositor tests**

Run: `pytest tests/test_compositor.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Implement end-to-end still pipeline**

`src/pipeline.py`:

```python
import asyncio
from pathlib import Path

import numpy as np

from src.models import WatchSpec, PlacementSpec
from src.watch_pipeline.reconstruction import reconstruct_watch
from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction
from src.composition.renderer import render_watch
from src.composition.compositor import composite_watch


def run_still_pipeline(
    watch_spec: WatchSpec,
    wrist_image: np.ndarray,
    circumference_mm: float,
    api_key: str,
    cache_dir: Path,
    output_dir: Path,
    sam_predictor=None,
) -> np.ndarray:
    """Run the full still-image try-on pipeline."""
    # --- Watch pipeline ---
    photo_path = watch_spec.photo_paths[0]
    mesh_path = asyncio.run(
        reconstruct_watch(photo_path, api_key, cache_dir, watch_spec.reference)
    )
    mesh = load_and_scale_mesh(mesh_path, watch_spec)

    # Save scaled mesh for Blender
    scaled_mesh_path = cache_dir / f"{watch_spec.reference}_scaled.glb"
    mesh.export(str(scaled_mesh_path))

    # --- Wrist pipeline ---
    landmarks = detect_wrist(wrist_image)
    segment = segment_wrist(
        wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, sam_predictor
    )
    px_per_mm = compute_px_per_mm(
        segment.width_px, circumference_mm, landmarks.pose_angle_rad
    )
    light_dir = estimate_light_direction(wrist_image)

    placement = PlacementSpec(
        center_x=landmarks.wrist_point[0],
        center_y=landmarks.wrist_point[1],
        rotation_deg=np.degrees(landmarks.forearm_angle_rad),
        px_per_mm=px_per_mm,
        light_direction=light_dir,
        wrist_mask=segment.mask,
    )

    # --- Composition ---
    h, w = wrist_image.shape[:2]
    passes = render_watch(scaled_mesh_path, placement, output_dir / "renders", image_size=(w, h))
    result = composite_watch(wrist_image, passes, placement)

    return result
```

- [ ] **Step 6: Commit**

```bash
git add src/composition/compositor.py src/pipeline.py tests/test_compositor.py
git commit -m "feat: compositor + end-to-end still pipeline orchestration"
```

---

### Task 7: Gradio UI

**Files:**
- Create: `src/app.py`

**Interfaces:**
- Consumes: `run_still_pipeline` from `src/pipeline.py`, all data models
- Produces: Runnable Gradio web app (`python -m src.app`)

- [ ] **Step 1: Implement Gradio app**

`src/app.py`:

```python
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv

from src.models import WatchSpec
from src.pipeline import run_still_pipeline

load_dotenv()

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("outputs")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def try_on(
    wrist_image: np.ndarray,
    circumference_cm: float,
    watch_reference: str,
    watch_name: str,
    case_diameter_mm: float,
    lug_to_lug_mm: float,
    thickness_mm: float,
    band_width_mm: float,
    case_shape: str,
    watch_photo: np.ndarray,
) -> np.ndarray:
    api_key = os.environ.get("PIAPI_API_KEY", "")
    if not api_key:
        raise gr.Error("PIAPI_API_KEY not set in .env")

    # Save watch photo to a temp file for the reconstruction API
    watch_photo_path = CACHE_DIR / f"{watch_reference}_photo.png"
    cv2.imwrite(str(watch_photo_path), cv2.cvtColor(watch_photo, cv2.COLOR_RGB2BGR))

    spec = WatchSpec(
        reference=watch_reference,
        name=watch_name,
        case_diameter_mm=case_diameter_mm,
        lug_to_lug_mm=lug_to_lug_mm,
        thickness_mm=thickness_mm,
        band_width_mm=band_width_mm,
        case_shape=case_shape,
        photo_paths=[watch_photo_path],
    )

    wrist_bgr = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)

    # SAM 2 predictor — lazy-loaded
    predictor = _load_sam_predictor()

    result_bgr = run_still_pipeline(
        watch_spec=spec,
        wrist_image=wrist_bgr,
        circumference_mm=circumference_cm * 10,
        api_key=api_key,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        sam_predictor=predictor,
    )

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


_sam_predictor_cache = None


def _load_sam_predictor():
    global _sam_predictor_cache
    if _sam_predictor_cache is not None:
        return _sam_predictor_cache

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_model = build_sam2("sam2_hiera_small.yaml", "sam2_hiera_small.pt")
        _sam_predictor_cache = SAM2ImagePredictor(sam_model)
        return _sam_predictor_cache
    except ImportError:
        raise gr.Error(
            "SAM 2 not installed. Run: pip install segment-anything-2 "
            "and download sam2_hiera_small.pt"
        )


demo = gr.Interface(
    fn=try_on,
    inputs=[
        gr.Image(label="Wrist Photo", type="numpy"),
        gr.Number(label="Wrist Circumference (cm)", value=17.5, minimum=10, maximum=25),
        gr.Textbox(label="Watch Reference", placeholder="e.g. 126710BLNR"),
        gr.Textbox(label="Watch Name", placeholder="e.g. Rolex GMT-Master II"),
        gr.Number(label="Case Diameter (mm)", value=42.0, minimum=20, maximum=60),
        gr.Number(label="Lug-to-Lug (mm)", value=48.0, minimum=25, maximum=65),
        gr.Number(label="Thickness (mm)", value=13.0, minimum=5, maximum=25),
        gr.Number(label="Band Width (mm)", value=20.0, minimum=10, maximum=30),
        gr.Dropdown(label="Case Shape", choices=["round", "square", "tonneau"], value="round"),
        gr.Image(label="Watch Photo (front-facing)", type="numpy"),
    ],
    outputs=gr.Image(label="Try-On Result", type="numpy"),
    title="Watch Virtual Try-On",
    description="Upload a wrist photo and watch details to see how the watch looks on your wrist at accurate scale.",
)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Smoke test the app launches**

Run: `cd /home/junaid/Dev/watch-tryon && python -m src.app`

Expected: Gradio prints a local URL (e.g. `http://127.0.0.1:7860`). Open in browser and verify the UI renders with all input fields. Stop the server with Ctrl+C.

This is a UI integration point — the full pipeline won't work without a GPU, SAM 2 weights, and a PiAPI key. But the UI should render and accept inputs.

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat: Gradio UI for still-image try-on"
```

---

### Task 8: Video Extension (Phase 2)

**Files:**
- Create: `src/video/__init__.py`
- Create: `src/video/tracker.py`
- Create: `src/video/pipeline.py`
- Create: `tests/test_tracker.py`

**Interfaces:**
- Consumes:
  - SAM 2 video predictor
  - `WristLandmarks` from `src/wrist_pipeline/detection.py`
  - `compute_px_per_mm` from `src/wrist_pipeline/calibration.py`
  - `render_watch` from `src/composition/renderer.py`
  - `composite_watch` from `src/composition/compositor.py`
- Produces:
  - `OneEuroFilter` — smoothing filter for placement parameters
  - `track_wrist_video(video_path: Path, initial_point: tuple[float, float], sam_video_predictor) -> list[np.ndarray]` — returns per-frame wrist masks
  - `run_video_pipeline(watch_spec: WatchSpec, video_path: Path, circumference_mm: float, api_key: str, cache_dir: Path, output_path: Path, sam_predictor, sam_video_predictor) -> Path` — end-to-end video pipeline, returns path to output video

- [ ] **Step 1: Write failing tests for 1-Euro filter**

`tests/test_tracker.py`:

```python
import math
import numpy as np
import pytest

from src.video.tracker import OneEuroFilter


def test_filter_returns_first_value():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    result = f(0.0, 100.0)
    assert abs(result - 100.0) < 0.01


def test_filter_smooths_noise():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    values = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5, 100.0]
    filtered = []
    for i, v in enumerate(values):
        filtered.append(f(i * 0.033, v))  # 30fps

    # Filtered values should have less variance than input
    input_var = np.var(values)
    filtered_var = np.var(filtered)
    assert filtered_var < input_var


def test_filter_tracks_large_changes():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    # Start at 100, jump to 200
    f(0.0, 100.0)
    result = f(0.033, 200.0)
    # With beta > 0, the filter should follow large changes more aggressively
    assert result > 150.0, f"Filter should track large jump, got {result}"


def test_filter_independent_per_instance():
    f1 = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f2 = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f1(0.0, 100.0)
    f2(0.0, 200.0)
    assert abs(f1(0.033, 105.0) - f2(0.033, 205.0)) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement tracker with 1-Euro filter**

`src/video/__init__.py`: empty file.

`src/video/tracker.py`:

```python
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
```

- [ ] **Step 4: Run tracker tests**

Run: `pytest tests/test_tracker.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Implement video pipeline**

`src/video/pipeline.py`:

```python
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.models import WatchSpec, PlacementSpec
from src.watch_pipeline.reconstruction import reconstruct_watch
from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import measure_wrist_width
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction
from src.composition.renderer import render_watch
from src.composition.compositor import composite_watch
from src.video.tracker import OneEuroFilter, track_wrist_video


def run_video_pipeline(
    watch_spec: WatchSpec,
    video_path: Path,
    circumference_mm: float,
    api_key: str,
    cache_dir: Path,
    output_path: Path,
    sam_predictor,
    sam_video_predictor,
) -> Path:
    """Run the full video try-on pipeline."""
    import asyncio

    # --- Watch pipeline (same as stills, cached) ---
    photo_path = watch_spec.photo_paths[0]
    mesh_path = asyncio.run(
        reconstruct_watch(photo_path, api_key, cache_dir, watch_spec.reference)
    )
    mesh = load_and_scale_mesh(mesh_path, watch_spec)
    scaled_mesh_path = cache_dir / f"{watch_spec.reference}_scaled.glb"
    mesh.export(str(scaled_mesh_path))

    # --- Read video metadata ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames in video")

    # --- Wrist detection on first frame ---
    landmarks = detect_wrist(frames[0])
    light_dir = estimate_light_direction(frames[0])

    # --- SAM 2 video tracking ---
    masks = track_wrist_video(video_path, landmarks.wrist_point, sam_video_predictor)

    # --- Smoothing filters ---
    cx_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    cy_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.3)
    scale_filter = OneEuroFilter(min_cutoff=0.5, beta=0.2)

    # --- Per-frame render + composite ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        frame_dir = Path(tmp_dir) / "frames"
        frame_dir.mkdir()

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            t = i / fps

            # Extract placement from mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # No wrist detected this frame — use previous placement
                cv2.imwrite(str(frame_dir / f"{i:06d}.png"), frame)
                continue

            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] == 0:
                cv2.imwrite(str(frame_dir / f"{i:06d}.png"), frame)
                continue

            raw_cx = M["m10"] / M["m00"]
            raw_cy = M["m01"] / M["m00"]

            # Forearm angle from mask orientation
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                raw_angle = ellipse[2]
            else:
                raw_angle = 0.0

            raw_width = measure_wrist_width(mask, math.radians(raw_angle))
            raw_scale = compute_px_per_mm(raw_width, circumference_mm, landmarks.pose_angle_rad)

            # Smooth
            cx = cx_filter(t, raw_cx)
            cy = cy_filter(t, raw_cy)
            angle = angle_filter(t, raw_angle)
            px_per_mm = scale_filter(t, raw_scale)

            placement = PlacementSpec(
                center_x=cx,
                center_y=cy,
                rotation_deg=angle,
                px_per_mm=px_per_mm,
                light_direction=light_dir,
                wrist_mask=mask,
            )

            render_dir = Path(tmp_dir) / f"render_{i:06d}"
            passes = render_watch(scaled_mesh_path, placement, render_dir, image_size=(w, h))
            composited = composite_watch(frame, passes, placement)
            cv2.imwrite(str(frame_dir / f"{i:06d}.png"), composited)

        # --- Encode to video ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(frame_dir / "%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    return output_path
```

- [ ] **Step 6: Add video tab to Gradio app**

Modify `src/app.py` — add a second tab for video input. Add this function and update the interface:

```python
def try_on_video(
    wrist_video: str,  # Gradio returns video as file path
    circumference_cm: float,
    watch_reference: str,
    watch_name: str,
    case_diameter_mm: float,
    lug_to_lug_mm: float,
    thickness_mm: float,
    band_width_mm: float,
    case_shape: str,
    watch_photo: np.ndarray,
) -> str:
    api_key = os.environ.get("PIAPI_API_KEY", "")
    if not api_key:
        raise gr.Error("PIAPI_API_KEY not set in .env")

    watch_photo_path = CACHE_DIR / f"{watch_reference}_photo.png"
    cv2.imwrite(str(watch_photo_path), cv2.cvtColor(watch_photo, cv2.COLOR_RGB2BGR))

    spec = WatchSpec(
        reference=watch_reference,
        name=watch_name,
        case_diameter_mm=case_diameter_mm,
        lug_to_lug_mm=lug_to_lug_mm,
        thickness_mm=thickness_mm,
        band_width_mm=band_width_mm,
        case_shape=case_shape,
        photo_paths=[watch_photo_path],
    )

    from src.video.pipeline import run_video_pipeline

    predictor = _load_sam_predictor()

    # SAM 2 video predictor
    from sam2.build_sam import build_sam2_video_predictor
    video_predictor = build_sam2_video_predictor("sam2_hiera_small.yaml", "sam2_hiera_small.pt")

    output_path = OUTPUT_DIR / f"{watch_reference}_tryon.mp4"
    run_video_pipeline(
        watch_spec=spec,
        video_path=Path(wrist_video),
        circumference_mm=circumference_cm * 10,
        api_key=api_key,
        cache_dir=CACHE_DIR,
        output_path=output_path,
        sam_predictor=predictor,
        sam_video_predictor=video_predictor,
    )

    return str(output_path)
```

Update the interface to use `gr.TabbedInterface` with "Still Image" and "Video" tabs. The video tab uses `gr.Video(label="Wrist Video")` as input and `gr.Video(label="Try-On Result")` as output.

- [ ] **Step 7: Run tracker tests to verify everything passes**

Run: `pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/video/ tests/test_tracker.py src/app.py
git commit -m "feat: Phase 2 video pipeline — SAM 2 tracking + 1-Euro smoothing + ffmpeg encode"
```

---

## Integration Testing Checklist

After all tasks are complete, run the full pipeline manually with real inputs:

1. **Prepare test assets:**
   - Download a front-facing watch photo (e.g. search "Rolex GMT Master II dial")
   - Take a wrist photo with good lighting
   - Measure your wrist circumference

2. **Set up environment:**
   - Add `PIAPI_API_KEY` to `.env`
   - Install SAM 2 and download `sam2_hiera_small.pt`
   - Ensure Blender 4.x is installed and `blender` is on PATH

3. **Run still pipeline:**
   - `python -m src.app`
   - Upload wrist photo, enter watch specs, upload watch photo
   - Verify: watch appears at correct size relative to wrist

4. **Run video pipeline:**
   - Record a 3-5 second wrist video
   - Switch to video tab, upload, run
   - Verify: watch tracks wrist smoothly, no jitter

5. **Size validation:**
   - Try the same watch on two different wrist sizes (e.g. 16cm and 19cm)
   - The watch should appear proportionally different — larger on the smaller wrist
