"""V3 demo: 3D watch model rendered in Blender at correct size + perspective."""
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.models import WatchSpec
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction

load_dotenv()

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260907_124808.jpg"))
WATCH_MODEL = Path("cache/gshock_5600.glb")


def build_blender_script(
    mesh_path: str,
    output_path: str,
    render_w: int,
    render_h: int,
    camera_elevation_deg: float,
    forearm_angle_deg: float,
    light_direction: tuple,
) -> str:
    """Generate a Blender Python script that renders the watch model."""
    return f"""
import bpy
import math
import mathutils
import os

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = {render_w}
scene.render.resolution_y = {render_h}
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'

# Import watch model
bpy.ops.import_scene.gltf(filepath=r"{mesh_path}")

# Get all mesh objects
watch_objects = [obj for obj in scene.objects if obj.type == 'MESH']
if not watch_objects:
    print("ERROR: No mesh objects found after import")
    exit(1)

# Compute bounding box of all watch objects
bbox_min = mathutils.Vector((1e10, 1e10, 1e10))
bbox_max = mathutils.Vector((-1e10, -1e10, -1e10))
for obj in watch_objects:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ mathutils.Vector(corner)
        for i in range(3):
            bbox_min[i] = min(bbox_min[i], world_corner[i])
            bbox_max[i] = max(bbox_max[i], world_corner[i])

center = (bbox_min + bbox_max) / 2
extent = bbox_max - bbox_min
max_extent = max(extent.x, extent.y, extent.z)
print(f"Watch bounding box: {{extent.x:.3f}} x {{extent.y:.3f}} x {{extent.z:.3f}}")
print(f"Watch center: {{center.x:.3f}}, {{center.y:.3f}}, {{center.z:.3f}}")

# Camera: perspective, viewing from above at the estimated elevation angle
cam_data = bpy.data.cameras.new(name='Camera')
cam_data.type = 'PERSP'
cam_data.lens = 50  # 50mm standard lens

elevation_rad = math.radians({camera_elevation_deg})
cam_distance = max_extent * 3.0  # far enough to see the whole watch

# Camera position: looking at center from above-front
cam_x = center.x
cam_y = center.y - cam_distance * math.cos(elevation_rad)
cam_z = center.z + cam_distance * math.sin(elevation_rad)

cam_obj = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (cam_x, cam_y, cam_z)

# Point camera at the watch center
direction = mathutils.Vector((center.x - cam_x, center.y - cam_y, center.z - cam_z))
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# Key light — sun matching the estimated scene lighting
light_dir = {light_direction}
key_data = bpy.data.lights.new(name='Key', type='SUN')
key_data.energy = 3.0
key_obj = bpy.data.objects.new('Key', key_data)
scene.collection.objects.link(key_obj)
key_obj.rotation_euler = (
    math.atan2(-light_dir[1], light_dir[2]),
    math.atan2(light_dir[0], light_dir[2]),
    0,
)

# Fill light from opposite side
fill_data = bpy.data.lights.new(name='Fill', type='SUN')
fill_data.energy = 1.5
fill_obj = bpy.data.objects.new('Fill', fill_data)
scene.collection.objects.link(fill_obj)
fill_obj.rotation_euler = (
    math.atan2(light_dir[1], light_dir[2]),
    math.atan2(-light_dir[0], light_dir[2]),
    0,
)

# Ambient light for realism
ambient = bpy.data.lights.new(name='Ambient', type='SUN')
ambient.energy = 0.5
amb_obj = bpy.data.objects.new('Ambient', ambient)
scene.collection.objects.link(amb_obj)
amb_obj.rotation_euler = (math.radians(90), 0, 0)  # straight down

# Use EEVEE for speed
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError:
    scene.render.engine = 'BLENDER_EEVEE'

# Render
scene.render.filepath = r"{output_path}"
bpy.ops.render.render(write_still=True)
print("Render complete")
"""


def render_watch_3d(
    mesh_path: Path,
    output_path: Path,
    render_size: tuple[int, int],
    camera_elevation_deg: float,
    forearm_angle_deg: float,
    light_direction: tuple,
) -> np.ndarray:
    """Render watch model in Blender and return RGBA image."""
    script = build_blender_script(
        mesh_path=str(mesh_path),
        output_path=str(output_path),
        render_w=render_size[0],
        render_h=render_size[1],
        camera_elevation_deg=camera_elevation_deg,
        forearm_angle_deg=forearm_angle_deg,
        light_direction=light_direction,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    print(f"  Running Blender render ({render_size[0]}x{render_size[1]})...")
    result = subprocess.run(
        ["blender", "--background", "--python", script_path],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Print Blender output for debugging
    for line in result.stdout.split('\n'):
        if line.startswith(('Watch ', 'Render ', 'ERROR')):
            print(f"  Blender: {line}")

    if result.returncode != 0:
        print(f"  Blender stderr: {result.stderr[-500:]}")
        raise RuntimeError("Blender render failed")

    rendered = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    if rendered is None:
        raise FileNotFoundError(f"Render output not found: {output_path}")

    return rendered


def composite_3d(background, rendered_watch, center, rotation_deg, target_width_px):
    """Composite the 3D-rendered watch RGBA onto the background."""
    bh, bw = background.shape[:2]
    rh, rw = rendered_watch.shape[:2]

    # Scale rendered watch to physical size
    # The render captures the full watch — scale so its width matches target
    # Find the actual watch extent in the render (non-transparent pixels)
    alpha = rendered_watch[:, :, 3]
    cols = np.any(alpha > 10, axis=0)
    if not cols.any():
        print("  WARNING: render is fully transparent")
        return background.copy()
    x_min, x_max = np.where(cols)[0][[0, -1]]
    watch_render_width = x_max - x_min

    scale = target_width_px / max(watch_render_width, 1)
    new_w = max(1, round(rw * scale))
    new_h = max(1, round(rh * scale))
    scaled = cv2.resize(rendered_watch, (new_w, new_h),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # Place and rotate onto canvas
    canvas = np.zeros((bh, bw, 4), dtype=np.uint8)
    fg_center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(fg_center, -rotation_deg, 1.0)
    rot_mat[0, 2] += center[0] - new_w / 2
    rot_mat[1, 2] += center[1] - new_h / 2
    cv2.warpAffine(scaled, rot_mat, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)

    alpha_f = canvas[:, :, 3].astype(np.float32) / 255.0
    alpha_f = cv2.GaussianBlur(alpha_f, (3, 3), 0.5)

    # Simple alpha blend (the 3D render already has correct lighting/perspective)
    result = background.copy()
    for c in range(3):
        fg = canvas[:, :, c].astype(np.float32)
        bg = result[:, :, c].astype(np.float32)
        result[:, :, c] = np.clip(fg * alpha_f + bg * (1.0 - alpha_f), 0, 255).astype(np.uint8)

    return result


def main():
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

    if not WATCH_MODEL.exists():
        print(f"ERROR: Watch model not found at {WATCH_MODEL}")
        print("Download from: https://sketchfab.com/3d-models/casio-g-shock-5600-b5830e99faed43549db339da6cd2fd43")
        print(f"Save as: {WATCH_MODEL}")
        return

    wrist_image = cv2.imread(str(WRIST_IMAGE))
    h, w = wrist_image.shape[:2]
    print(f"Wrist image: {w}x{h}")

    # Load SAM 2
    print("Loading SAM 2 on CPU...")
    sam_model = build_sam2("configs/sam2/sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device="cpu")
    predictor = SAM2ImagePredictor(sam_model)

    # Detect wrist
    print("Detecting wrist...")
    wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
    landmarks = detect_wrist(wrist_rgb)
    print(f"  Wrist: ({landmarks.wrist_point[0]:.0f}, {landmarks.wrist_point[1]:.0f})")
    print(f"  Forearm angle: {np.degrees(landmarks.forearm_angle_rad):.1f} deg")
    print(f"  Pose angle: {np.degrees(landmarks.pose_angle_rad):.1f} deg")

    # Segment wrist for width
    print("Segmenting wrist...")
    segment = segment_wrist(wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, predictor)
    print(f"  Wrist width: {segment.width_px:.0f}px")

    # Scale calibration
    circumference_mm = 175.0
    px_per_mm = compute_px_per_mm(segment.width_px, circumference_mm, landmarks.pose_angle_rad)
    target_width_px = spec.case_diameter_mm * px_per_mm
    print(f"  px_per_mm: {px_per_mm:.2f}")
    print(f"  Watch will be {target_width_px:.0f}px wide")

    # Estimate camera elevation from pose angle
    # pose_angle ~0 = facing camera (overhead), ~pi/2 = edge-on
    # Camera elevation: 90° - pose_angle (overhead = 90°, edge-on = 0°)
    camera_elevation_deg = 90.0 - np.degrees(landmarks.pose_angle_rad)
    camera_elevation_deg = max(30, min(85, camera_elevation_deg))  # clamp
    print(f"  Camera elevation: {camera_elevation_deg:.1f} deg")

    # Light direction
    light_dir = estimate_light_direction(wrist_image)

    # Watch placement
    shift_mm = 35.0
    shift_px = shift_mm * px_per_mm
    fore_dir = np.array([math.cos(landmarks.forearm_angle_rad), math.sin(landmarks.forearm_angle_rad)])
    watch_cx = landmarks.wrist_point[0] + fore_dir[0] * shift_px
    watch_cy = landmarks.wrist_point[1] + fore_dir[1] * shift_px
    print(f"  Watch center: ({watch_cx:.0f}, {watch_cy:.0f})")

    # Render watch in Blender
    print("Rendering watch in Blender...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    render_path = output_dir / "watch_render.png"

    rendered = render_watch_3d(
        mesh_path=WATCH_MODEL,
        output_path=render_path,
        render_size=(1024, 1024),
        camera_elevation_deg=camera_elevation_deg,
        forearm_angle_deg=np.degrees(landmarks.forearm_angle_rad),
        light_direction=light_dir,
    )
    print(f"  Render: {rendered.shape[1]}x{rendered.shape[0]}, {rendered.shape[2]} channels")

    # Composite
    forearm_deg = np.degrees(landmarks.forearm_angle_rad)
    rotation_deg = forearm_deg + 180
    print(f"  Composite rotation: {rotation_deg:.1f} deg")

    print("Compositing...")
    result = composite_3d(wrist_image, rendered, (watch_cx, watch_cy), rotation_deg, target_width_px)

    out_path = output_dir / "demo_v3.png"
    cv2.imwrite(str(out_path), result)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
