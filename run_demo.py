"""Run the still pipeline with a proper watch mesh and corrected placement."""
import math
import os
from pathlib import Path
import cv2
import numpy as np
import trimesh
from dotenv import load_dotenv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.models import WatchSpec, PlacementSpec
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction
from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.composition.renderer import render_watch
from src.composition.compositor import composite_watch

load_dotenv()

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260907_124808.jpg"))

spec = WatchSpec(
    reference="casio-gshock-dw5600",
    name="Casio G-Shock DW-5600",
    case_diameter_mm=42.8,
    lug_to_lug_mm=48.9,
    thickness_mm=13.4,
    band_width_mm=25.0,
    case_shape="square",
    photo_paths=[],
)

wrist_image = cv2.imread(str(WRIST_IMAGE))
h, w = wrist_image.shape[:2]
print(f"Wrist image: {w}x{h}")
print(f"Watch: {spec.reference} ({spec.case_diameter_mm}mm)")

# Load SAM 2
print("Loading SAM 2 on CPU...")
sam_model = build_sam2("configs/sam2/sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device="cpu")
predictor = SAM2ImagePredictor(sam_model)
print("SAM 2 loaded")

# Detect wrist
print("Detecting wrist...")
wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
landmarks = detect_wrist(wrist_rgb)
print(f"  Wrist point: ({landmarks.wrist_point[0]:.0f}, {landmarks.wrist_point[1]:.0f})")
print(f"  Forearm angle: {np.degrees(landmarks.forearm_angle_rad):.1f} deg")

# Segment wrist
print("Segmenting wrist...")
segment = segment_wrist(wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, predictor)
print(f"  Wrist width: {segment.width_px:.0f}px")

# Scale calibration
circumference_mm = 175.0
px_per_mm = compute_px_per_mm(segment.width_px, circumference_mm, landmarks.pose_angle_rad)
print(f"  px_per_mm: {px_per_mm:.2f}")
print(f"  Watch will be {spec.case_diameter_mm * px_per_mm:.0f}px wide")

light_dir = estimate_light_direction(wrist_image)

# Shift watch placement ~30mm up the forearm from the wrist point
# (watches sit above the wrist bone, not on it)
shift_mm = 30.0
shift_px = shift_mm * px_per_mm
fore_dir = np.array([math.cos(landmarks.forearm_angle_rad), math.sin(landmarks.forearm_angle_rad)])
watch_center_x = landmarks.wrist_point[0] + fore_dir[0] * shift_px
watch_center_y = landmarks.wrist_point[1] + fore_dir[1] * shift_px
print(f"  Watch center: ({watch_center_x:.0f}, {watch_center_y:.0f}) (shifted {shift_mm}mm up arm)")

placement = PlacementSpec(
    center_x=watch_center_x,
    center_y=watch_center_y,
    rotation_deg=np.degrees(landmarks.forearm_angle_rad),
    px_per_mm=px_per_mm,
    light_direction=light_dir,
    wrist_mask=segment.mask,
)

# Build watch mesh
cache_dir = Path("cache")
output_dir = Path("outputs")
cache_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

print("Building watch mesh...")
raw_mesh_path = cache_dir / "watch_model.glb"
if not raw_mesh_path.exists():
    case = trimesh.creation.cylinder(radius=0.5, height=0.3, sections=64)
    bezel = trimesh.creation.cylinder(radius=0.55, height=0.05, sections=64)
    bezel.apply_translation([0, 0, 0.15])
    crown = trimesh.creation.cylinder(radius=0.05, height=0.1, sections=16)
    crown.apply_translation([0.55, 0, 0.05])
    crown.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,0,1]))
    lug1 = trimesh.creation.cylinder(radius=0.06, height=0.15, sections=8)
    lug1.apply_translation([0, 0.5, 0.05])
    lug2 = trimesh.creation.cylinder(radius=0.06, height=0.15, sections=8)
    lug2.apply_translation([0, -0.5, 0.05])
    watch = trimesh.util.concatenate([case, bezel, crown, lug1, lug2])
    watch.export(str(raw_mesh_path))

scaled_mesh = load_and_scale_mesh(raw_mesh_path, spec)
scaled_mesh_path = cache_dir / f"{spec.reference}_scaled.glb"
scaled_mesh.export(str(scaled_mesh_path))

# Render + composite
print("Rendering with Blender (this takes ~30s on CPU)...")
passes = render_watch(scaled_mesh_path, placement, output_dir / "renders", image_size=(1024, 1024))
print("Compositing...")
result = composite_watch(wrist_image, passes, placement, spec.case_diameter_mm)

out_path = output_dir / "demo_result.png"
cv2.imwrite(str(out_path), result)
print(f"\nResult saved to {out_path}")
