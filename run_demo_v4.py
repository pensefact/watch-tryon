"""V4 demo: Full SDD pipeline — PiAPI TRELLIS image-to-3D → Blender render → composite."""
import asyncio
import math
import os
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from src.models import WatchSpec
from src.watch_pipeline.reconstruction import reconstruct_watch, get_cached_mesh
from src.watch_pipeline.mesh_processing import load_and_scale_mesh
from src.wrist_pipeline.detection import detect_wrist
from src.wrist_pipeline.segmentation import segment_wrist
from src.wrist_pipeline.calibration import compute_px_per_mm
from src.wrist_pipeline.lighting import estimate_light_direction

load_dotenv()

WRIST_IMAGE = Path(os.path.expanduser("~/Downloads/20260908_092226.jpg"))
GROUND_TRUTH = Path(os.path.expanduser("~/Downloads/20260908_092120.jpg"))
WATCH_REF = Path(os.path.expanduser(
    "~/Downloads/G-Shock-x-Charles-Darwin-Foundation-GAB2100DF-1A-Watch-Black-front-1024x1024.jpg"
))


def score_against_ground_truth(
    composite: np.ndarray,
    ground_truth_path: Path,
    wrist_center: tuple[float, float],
    target_width_px: float,
) -> dict:
    """Compare composite output against ground truth photo wearing the watch.

    Crops a region around the watch placement in both images and computes:
    - SSIM (structural similarity) on the watch region
    - Histogram correlation for color match
    - Edge overlap (Canny) for shape/position accuracy
    Returns a dict of scores and a combined score 0-100.
    """
    gt = cv2.imread(str(ground_truth_path))
    if gt is None:
        raise FileNotFoundError(f"Cannot load ground truth: {ground_truth_path}")

    # Both images should be same size (same camera, same pose)
    if composite.shape[:2] != gt.shape[:2]:
        gt = cv2.resize(gt, (composite.shape[1], composite.shape[0]))

    # Crop region around watch: 1.5x the watch width centered on placement
    margin = int(target_width_px * 0.75)
    cx, cy = int(wrist_center[0]), int(wrist_center[1])
    h, w = composite.shape[:2]
    y1 = max(0, cy - margin)
    y2 = min(h, cy + margin)
    x1 = max(0, cx - margin)
    x2 = min(w, cx + margin)

    comp_crop = composite[y1:y2, x1:x2]
    gt_crop = gt[y1:y2, x1:x2]

    # 1. SSIM on grayscale
    comp_gray = cv2.cvtColor(comp_crop, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(gt_gray, comp_gray, full=True)

    # 2. Histogram correlation (color distribution match)
    hist_scores = []
    for c in range(3):
        h_comp = cv2.calcHist([comp_crop], [c], None, [64], [0, 256])
        h_gt = cv2.calcHist([gt_crop], [c], None, [64], [0, 256])
        cv2.normalize(h_comp, h_comp)
        cv2.normalize(h_gt, h_gt)
        hist_scores.append(cv2.compareHist(h_comp, h_gt, cv2.HISTCMP_CORREL))
    hist_score = float(np.mean(hist_scores))

    # 3. Edge overlap (Canny)
    edges_comp = cv2.Canny(comp_gray, 50, 150)
    edges_gt = cv2.Canny(gt_gray, 50, 150)
    # Dilate edges slightly for tolerance
    kernel = np.ones((3, 3), np.uint8)
    edges_comp_d = cv2.dilate(edges_comp, kernel, iterations=1)
    edges_gt_d = cv2.dilate(edges_gt, kernel, iterations=1)
    overlap = np.logical_and(edges_comp > 0, edges_gt_d > 0).sum()
    total = max(1, (edges_comp > 0).sum())
    edge_precision = overlap / total
    overlap_rev = np.logical_and(edges_gt > 0, edges_comp_d > 0).sum()
    total_rev = max(1, (edges_gt > 0).sum())
    edge_recall = overlap_rev / total_rev
    edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-6)

    # Combined score: weighted average
    combined = (ssim_score * 40 + hist_score * 30 + edge_f1 * 30)
    combined = max(0, min(100, combined))

    # Save debug crops
    output_dir = Path("outputs")
    cv2.imwrite(str(output_dir / "eval_comp_crop.png"), comp_crop)
    cv2.imwrite(str(output_dir / "eval_gt_crop.png"), gt_crop)
    cv2.imwrite(str(output_dir / "eval_edges_comp.png"), edges_comp)
    cv2.imwrite(str(output_dir / "eval_edges_gt.png"), edges_gt)

    return {
        "ssim": round(ssim_score, 4),
        "hist_corr": round(hist_score, 4),
        "edge_f1": round(edge_f1, 4),
        "edge_precision": round(edge_precision, 4),
        "edge_recall": round(edge_recall, 4),
        "combined": round(combined, 2),
    }


def build_blender_script(
    mesh_path: str,
    output_path: str,
    render_w: int,
    render_h: int,
    camera_elevation_deg: float,
    camera_azimuth_deg: float,
    light_direction: tuple,
    dial_texture_path: str = "",
) -> str:
    """Generate Blender script that renders the watch model with matched perspective.

    The TRELLIS mesh is hollow (no dial surface), so we add a disc cap textured
    with the product photo to show the real dial face.
    """
    return f"""
import bpy
import math
import mathutils

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

# Compute bounding box
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
print(f"Watch bbox: {{extent.x:.3f}} x {{extent.y:.3f}} x {{extent.z:.3f}}")

# Find dial face axis: thinnest dimension
dims = [extent.x, extent.y, extent.z]
dial_axis = dims.index(min(dims))  # 0=X, 1=Y, 2=Z
print(f"Dial axis: {{'XYZ'[dial_axis]}} (thinnest={{min(dims):.3f}})")

# Cap the hollow dial opening with a textured disc
if dial_axis == 0:
    disc_radius = min(extent.y, extent.z) * 0.44
    disc_loc = (bbox_max.x - extent.x * 0.08, center.y, center.z)
    disc_rot = (0, math.radians(90), 0)
elif dial_axis == 1:
    disc_radius = min(extent.x, extent.z) * 0.44
    disc_loc = (center.x, bbox_max.y - extent.y * 0.08, center.z)
    disc_rot = (math.radians(90), 0, 0)
else:
    disc_radius = min(extent.x, extent.y) * 0.44
    disc_loc = (center.x, center.y, bbox_max.z - extent.z * 0.08)
    disc_rot = (0, 0, 0)

# Create disc cap textured with the dial face
bpy.ops.mesh.primitive_circle_add(
    vertices=128, radius=disc_radius, fill_type='NGON',
    location=disc_loc, rotation=disc_rot,
)
disc = bpy.context.active_object
disc.name = 'DialCap'

dial_tex_path = r"{dial_texture_path}"
if dial_tex_path:
    mat = bpy.data.materials.new(name='DialMat')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output_node = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.3
    bsdf.inputs['Specular IOR Level'].default_value = 0.4
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])
    tex_node = nodes.new('ShaderNodeTexImage')
    try:
        tex_node.image = bpy.data.images.load(dial_tex_path)
    except Exception as e:
        print(f"Could not load dial texture: {{e}}")
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    coord = nodes.new('ShaderNodeTexCoord')
    mapping = nodes.new('ShaderNodeMapping')
    links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
    disc.data.materials.append(mat)
else:
    mat = bpy.data.materials.new(name='DialMat')
    mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1)
    disc.data.materials.append(mat)

# Camera: perspective, viewing the dial face
cam_data = bpy.data.cameras.new(name='Camera')
cam_data.type = 'PERSP'
cam_data.lens = 85

# Position camera along the dial axis
cam_distance = max(extent.x, extent.y, extent.z) * 2.5
if dial_axis == 0:
    cam_pos = (center.x + cam_distance, center.y, center.z)
elif dial_axis == 1:
    cam_pos = (center.x, center.y + cam_distance, center.z)
else:
    cam_pos = (center.x, center.y, center.z + cam_distance)

cam_obj = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = cam_pos

direction = mathutils.Vector((center.x - cam_pos[0], center.y - cam_pos[1], center.z - cam_pos[2]))
rot_quat = direction.to_track_quat('-Z', 'Y')
cam_obj.rotation_euler = rot_quat.to_euler()

# Warm scene-matched lighting
# Key: warm area light from upper-left
key = bpy.data.lights.new(name='Key', type='AREA')
key.energy = 150
key.size = 3
key.color = (1.0, 0.92, 0.82)
key_obj = bpy.data.objects.new('Key', key)
scene.collection.objects.link(key_obj)
key_obj.location = (cam_pos[0] * 0.6, center.y + 1.5, center.z + 1.5)
d = mathutils.Vector(center) - mathutils.Vector(key_obj.location)
key_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

# Fill: cooler, from opposite side
fill = bpy.data.lights.new(name='Fill', type='AREA')
fill.energy = 60
fill.size = 3
fill.color = (0.85, 0.9, 1.0)
fill_obj = bpy.data.objects.new('Fill', fill)
scene.collection.objects.link(fill_obj)
fill_obj.location = (cam_pos[0] * 0.6, center.y - 1.5, center.z + 0.3)
d = mathutils.Vector(center) - mathutils.Vector(fill_obj.location)
fill_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

# Warm bounce from below (wood desk reflection)
bounce = bpy.data.lights.new(name='Bounce', type='AREA')
bounce.energy = 30
bounce.size = 4
bounce.color = (1.0, 0.88, 0.7)
bounce_obj = bpy.data.objects.new('Bounce', bounce)
scene.collection.objects.link(bounce_obj)
bounce_obj.location = (cam_pos[0] * 0.3, center.y, center.z - 1.5)
d = mathutils.Vector(center) - mathutils.Vector(bounce_obj.location)
bounce_obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

scene.render.engine = 'BLENDER_EEVEE'
scene.render.filepath = r"{output_path}"
bpy.ops.render.render(write_still=True)
print("Render complete")
"""


def render_watch_blender(
    mesh_path: Path,
    output_path: Path,
    render_size: tuple[int, int],
    camera_elevation_deg: float,
    camera_azimuth_deg: float,
    light_direction: tuple,
    dial_texture_path: str = "",
) -> np.ndarray:
    """Render watch in Blender, return RGBA image."""
    import subprocess
    import tempfile

    script = build_blender_script(
        mesh_path=str(mesh_path),
        output_path=str(output_path),
        render_w=render_size[0],
        render_h=render_size[1],
        camera_elevation_deg=camera_elevation_deg,
        camera_azimuth_deg=camera_azimuth_deg,
        light_direction=light_direction,
        dial_texture_path=dial_texture_path,
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

    for line in result.stdout.split('\n'):
        if line.startswith(('Watch ', 'Render ', 'ERROR')):
            print(f"  Blender: {line}")

    if result.returncode != 0:
        print(f"  Blender stderr (last 500): {result.stderr[-500:]}")
        raise RuntimeError("Blender render failed")

    rendered = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    if rendered is None:
        raise FileNotFoundError(f"Render output not found: {output_path}")

    return rendered


def extract_watch_from_product(product_path: Path) -> np.ndarray:
    """Extract watch RGBA from product photo with white/light background."""
    ref = cv2.imread(str(product_path))
    if ref is None:
        raise FileNotFoundError(f"Cannot load: {product_path}")
    h, w = ref.shape[:2]

    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No watch found in product photo")
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    # Feather edges
    clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 2.0)

    # Crop to bounding box with padding
    ys, xs = np.where(clean_mask > 0)
    pad = 10
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad)

    cropped = ref[y1:y2, x1:x2]
    cropped_mask = clean_mask[y1:y2, x1:x2]

    rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = cropped_mask
    print(f"  Extracted: {rgba.shape[1]}x{rgba.shape[0]}")
    return rgba


def fill_bezel_with_dial(bezel_render: np.ndarray, product_photo_path: Path) -> np.ndarray:
    """Fill the hollow center of the 3D bezel render with the product photo's dial face.

    The bezel render has transparency in the center where the dial should be.
    We extract the watch face from the product photo and place it behind the bezel.
    """
    product = cv2.imread(str(product_photo_path), cv2.IMREAD_UNCHANGED)
    if product is None:
        return bezel_render
    if product.shape[2] == 3:
        product = cv2.cvtColor(product, cv2.COLOR_BGR2BGRA)

    ph, pw = product.shape[:2]
    rh, rw = bezel_render.shape[:2]

    # Extract watch from product photo (dark watch on light/white background)
    gray = cv2.cvtColor(product[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bezel_render
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)

    # Find bounding box of the watch in the product photo
    ys, xs = np.where(clean_mask > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    watch_w = x2 - x1
    watch_h = y2 - y1

    # Find the extent of the bezel ring in the render (non-transparent pixels)
    bezel_alpha = bezel_render[:, :, 3]
    bcols = np.any(bezel_alpha > 10, axis=0)
    brows = np.any(bezel_alpha > 10, axis=1)
    if not bcols.any() or not brows.any():
        return bezel_render
    bx1, bx2 = np.where(bcols)[0][[0, -1]]
    by1, by2 = np.where(brows)[0][[0, -1]]
    bezel_w = bx2 - bx1
    bezel_h = by2 - by1
    bezel_cx = (bx1 + bx2) // 2
    bezel_cy = (by1 + by2) // 2

    # Scale product photo watch to fit inside the bezel
    scale = max(bezel_w, bezel_h) / max(watch_w, watch_h) * 0.95
    new_pw = int(pw * scale)
    new_ph = int(ph * scale)
    product_scaled = cv2.resize(product, (new_pw, new_ph), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(clean_mask, (new_pw, new_ph), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    # Center the scaled product photo's watch center on the bezel center
    watch_cx_scaled = int((x1 + x2) / 2 * scale)
    watch_cy_scaled = int((y1 + y2) / 2 * scale)

    # Create output: product photo behind bezel
    result = np.zeros_like(bezel_render)

    # Paste product photo centered on bezel
    ox = bezel_cx - watch_cx_scaled
    oy = bezel_cy - watch_cy_scaled

    # Compute overlap region
    src_x1 = max(0, -ox)
    src_y1 = max(0, -oy)
    dst_x1 = max(0, ox)
    dst_y1 = max(0, oy)
    w_copy = min(new_pw - src_x1, rw - dst_x1)
    h_copy = min(new_ph - src_y1, rh - dst_y1)

    if w_copy > 0 and h_copy > 0:
        src_region = product_scaled[src_y1:src_y1+h_copy, src_x1:src_x1+w_copy]
        mask_region = mask_scaled[src_y1:src_y1+h_copy, src_x1:src_x1+w_copy]
        # Apply the product watch mask as alpha
        src_rgba = src_region.copy()
        src_rgba[:, :, 3] = np.minimum(src_rgba[:, :, 3], mask_region)
        result[dst_y1:dst_y1+h_copy, dst_x1:dst_x1+w_copy] = src_rgba

    # Layer bezel on top (bezel alpha takes priority)
    bezel_a = bezel_render[:, :, 3].astype(np.float32) / 255.0
    for c in range(3):
        result[:, :, c] = (bezel_render[:, :, c].astype(np.float32) * bezel_a +
                           result[:, :, c].astype(np.float32) * (1 - bezel_a)).astype(np.uint8)
    result[:, :, 3] = np.maximum(bezel_render[:, :, 3], result[:, :, 3])

    return result


def crop_to_case(rendered: np.ndarray, stub_fraction: float = 0.25) -> np.ndarray:
    """Crop render to watch case + short strap stubs, feather the strap edges."""
    alpha = rendered[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    if not rows.any():
        return rendered

    y_min, y_max = np.where(rows)[0][[0, -1]]

    # Find case region: rows where width > 70% of max width
    row_widths = np.zeros(rendered.shape[0])
    for r in range(y_min, y_max + 1):
        cols = np.where(alpha[r] > 10)[0]
        if len(cols) > 0:
            row_widths[r] = cols[-1] - cols[0]

    max_w = row_widths.max()
    case_rows = np.where(row_widths > max_w * 0.7)[0]
    case_top, case_bot = case_rows[0], case_rows[-1]
    case_h = case_bot - case_top

    # Keep case + stub_fraction of case height as strap stubs
    stub_px = int(case_h * stub_fraction)
    crop_top = max(0, case_top - stub_px)
    crop_bot = min(rendered.shape[0], case_bot + stub_px)

    cropped = rendered[crop_top:crop_bot].copy()

    # Feather the strap stub edges (fade alpha to 0 over the stub region)
    fade_top = case_top - crop_top
    fade_bot = crop_bot - case_bot
    if fade_top > 2:
        for i in range(fade_top):
            t = i / fade_top
            cropped[i, :, 3] = (cropped[i, :, 3].astype(np.float32) * t).astype(np.uint8)
    if fade_bot > 2:
        for i in range(fade_bot):
            row = cropped.shape[0] - 1 - i
            t = i / fade_bot
            cropped[row, :, 3] = (cropped[row, :, 3].astype(np.float32) * t).astype(np.uint8)

    return cropped


def composite_3d(background, rendered_watch, center, rotation_deg, target_width_px):
    """Composite 3D-rendered RGBA watch onto background at correct size and angle."""
    bh, bw = background.shape[:2]
    rh, rw = rendered_watch.shape[:2]

    # Find actual watch extent in render (non-transparent pixels)
    alpha = rendered_watch[:, :, 3]
    cols = np.any(alpha > 10, axis=0)
    rows = np.any(alpha > 10, axis=1)
    if not cols.any():
        print("  WARNING: render is fully transparent")
        return background.copy()
    x_min, x_max = np.where(cols)[0][[0, -1]]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    watch_render_width = x_max - x_min

    # Scale to physical size
    scale = target_width_px / max(watch_render_width, 1)
    new_w = max(1, round(rw * scale))
    new_h = max(1, round(rh * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(rendered_watch, (new_w, new_h), interpolation=interp)

    # Place and rotate onto canvas
    canvas = np.zeros((bh, bw, 4), dtype=np.uint8)
    fg_center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(fg_center, -rotation_deg, 1.0)
    rot_mat[0, 2] += center[0] - new_w / 2
    rot_mat[1, 2] += center[1] - new_h / 2
    cv2.warpAffine(scaled, rot_mat, (bw, bh), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)

    # Edge feathering radius proportional to watch size
    feather_r = max(3, int(target_width_px * 0.015))
    feather_k = feather_r * 2 + 1  # must be odd

    # Drop shadow: offset, blurred copy of watch alpha
    shadow_offset = max(4, int(target_width_px * 0.02))
    shadow_sigma = target_width_px * 0.04
    shadow_alpha = canvas[:, :, 3].astype(np.float32) / 255.0
    M_shadow = np.float32([[1, 0, shadow_offset], [0, 1, shadow_offset]])
    shadow_alpha = cv2.warpAffine(shadow_alpha, M_shadow, (bw, bh))
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (0, 0), shadow_sigma)
    shadow_alpha = np.clip(shadow_alpha * 0.6, 0, 1)

    result = background.copy()
    # Apply shadow (darken background)
    for c in range(3):
        bg = result[:, :, c].astype(np.float32)
        result[:, :, c] = np.clip(bg * (1.0 - shadow_alpha * 0.5), 0, 255).astype(np.uint8)

    # Color-match: sample background near watch to get scene tint + brightness
    margin = int(target_width_px * 0.3)
    sample_r = max(0, int(center[1]) - margin)
    sample_b = min(bh, int(center[1]) + margin)
    sample_l = max(0, int(center[0]) - margin)
    sample_rr = min(bw, int(center[0]) + margin)
    bg_patch = background[sample_r:sample_b, sample_l:sample_rr].astype(np.float32)
    bg_mean = bg_patch.mean(axis=(0, 1))
    # Scene brightness factor: how dark the scene is compared to studio (255)
    scene_brightness = bg_mean.mean() / 255.0
    # Apply both color tint and brightness reduction
    tint = bg_mean / max(bg_mean.max(), 1)
    tint = 0.50 + 0.50 * tint  # color tint
    tint *= min(1.0, scene_brightness * 1.5)  # darken for scene brightness

    # Alpha blend watch with feathered edges
    alpha_f = canvas[:, :, 3].astype(np.float32) / 255.0
    # Erode slightly then blur for softer edges
    alpha_u8 = (alpha_f * 255).astype(np.uint8)
    erode_k = max(1, feather_r // 2)
    alpha_u8 = cv2.erode(alpha_u8, np.ones((erode_k, erode_k), np.uint8), iterations=1)
    alpha_f = cv2.GaussianBlur(alpha_u8.astype(np.float32) / 255.0, (feather_k, feather_k), feather_r * 0.5)

    for c in range(3):
        fg = canvas[:, :, c].astype(np.float32) * tint[c]
        bg = result[:, :, c].astype(np.float32)
        result[:, :, c] = np.clip(fg * alpha_f + bg * (1.0 - alpha_f), 0, 255).astype(np.uint8)

    return result


def main():
    api_key = os.getenv("PIAPI_API_KEY")
    if not api_key:
        print("ERROR: Set PIAPI_API_KEY in .env")
        return

    spec = WatchSpec(
        reference="casio-gshock-gab2100df",
        name="Casio G-Shock GA-B2100DF-1A",
        case_diameter_mm=45.4,
        lug_to_lug_mm=48.5,
        thickness_mm=11.8,
        band_width_mm=22.0,
        case_shape="round",
        photo_paths=[str(WATCH_REF)],
    )

    wrist_image = cv2.imread(str(WRIST_IMAGE))
    if wrist_image is None:
        print(f"ERROR: Cannot load wrist image: {WRIST_IMAGE}")
        return
    h, w = wrist_image.shape[:2]
    print(f"Wrist image: {w}x{h}")

    # --- Watch pipeline: image → 3D mesh ---
    cache_dir = Path("cache")
    output_dir = Path("outputs")
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print("\n=== Watch Pipeline ===")
    cached = get_cached_mesh(spec.reference, cache_dir)
    if cached:
        print(f"  Using cached mesh: {cached}")
        mesh_path = cached
    else:
        print("  Converting product photo to 3D model via PiAPI TRELLIS...")
        mesh_path = asyncio.run(
            reconstruct_watch(
                photo_path=Path(spec.photo_paths[0]),
                api_key=api_key,
                cache_dir=cache_dir,
                reference=spec.reference,
            )
        )
    # Use original mesh — compositor handles physical sizing via target_width_px
    print(f"  Mesh: {mesh_path}")

    # --- Wrist pipeline ---
    print("\n=== Wrist Pipeline ===")
    print("Loading SAM 2 on CPU...")
    sam_model = build_sam2("configs/sam2/sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt", device="cpu")
    predictor = SAM2ImagePredictor(sam_model)

    print("Detecting wrist...")
    wrist_rgb = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
    landmarks = detect_wrist(wrist_rgb)
    forearm_deg = np.degrees(landmarks.forearm_angle_rad)
    print(f"  Wrist: ({landmarks.wrist_point[0]:.0f}, {landmarks.wrist_point[1]:.0f})")
    print(f"  Forearm angle: {forearm_deg:.1f} deg")
    print(f"  Pose angle: {np.degrees(landmarks.pose_angle_rad):.1f} deg")

    print("Segmenting wrist...")
    segment = segment_wrist(wrist_image, landmarks.wrist_point, landmarks.forearm_angle_rad, predictor)
    print(f"  Wrist width: {segment.width_px:.0f}px")

    # Scale calibration
    circumference_mm = 175.0
    px_per_mm = compute_px_per_mm(segment.width_px, circumference_mm, landmarks.pose_angle_rad)
    target_width_px = spec.case_diameter_mm * px_per_mm
    print(f"  px_per_mm: {px_per_mm:.2f}")
    print(f"  Watch target width: {target_width_px:.0f}px")

    # Camera elevation from pose angle
    camera_elevation_deg = 90.0 - np.degrees(landmarks.pose_angle_rad)
    camera_elevation_deg = max(30, min(85, camera_elevation_deg))
    print(f"  Camera elevation: {camera_elevation_deg:.1f} deg")

    light_dir = estimate_light_direction(wrist_image)

    # Watch placement — shifted up forearm from wrist point
    shift_mm = 25.0
    shift_px = shift_mm * px_per_mm
    fore_dir = np.array([math.cos(landmarks.forearm_angle_rad), math.sin(landmarks.forearm_angle_rad)])
    watch_cx = landmarks.wrist_point[0] + fore_dir[0] * shift_px
    watch_cy = landmarks.wrist_point[1] + fore_dir[1] * shift_px
    print(f"  Watch center: ({watch_cx:.0f}, {watch_cy:.0f})")

    # --- Blender 3D Render ---
    print("\n=== 3D Rendering ===")
    render_path = output_dir / "watch_render_v4.png"

    # Prepare dial texture: crop the watch face from the product photo
    dial_texture_path = cache_dir / "dial_face_flat.jpg"
    if not dial_texture_path.exists():
        print("  Preparing dial texture from product photo...")
        prod_img = cv2.imread(str(WATCH_REF))
        gray = cv2.cvtColor(prod_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        ys, xs = np.where(clean_mask > 0)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = prod_img[y1:y2, x1:x2]
        size = max(crop.shape[:2])
        square = np.zeros((size, size, 3), dtype=np.uint8)
        oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
        square[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop
        square = cv2.resize(square, (512, 512))
        cv2.imwrite(str(dial_texture_path), square)
        print(f"  Dial texture: {dial_texture_path}")

    rendered = render_watch_blender(
        mesh_path=mesh_path,
        output_path=render_path,
        render_size=(1024, 1024),
        camera_elevation_deg=camera_elevation_deg,
        camera_azimuth_deg=0.0,
        light_direction=light_dir,
        dial_texture_path=str(dial_texture_path),
    )
    print(f"  Render: {rendered.shape[1]}x{rendered.shape[0]}, channels={rendered.shape[2]}")
    cv2.imwrite(str(output_dir / "watch_3d_render.png"), rendered)

    # Crop to case region (remove long strap stubs if present)
    cropped = crop_to_case(rendered, stub_fraction=0.15)
    print(f"  Cropped: {cropped.shape[1]}x{cropped.shape[0]}")
    cv2.imwrite(str(output_dir / "watch_3d_cropped.png"), cropped)

    # --- Composite ---
    print("\n=== Compositing ===")
    rotation_deg = forearm_deg + 90
    print(f"  Rotation: {rotation_deg:.1f} deg")

    result = composite_3d(wrist_image, cropped, (watch_cx, watch_cy), rotation_deg, target_width_px)

    out_path = output_dir / "demo_v4.png"
    cv2.imwrite(str(out_path), result)
    print(f"\nResult saved to {out_path}")

    # --- Ground Truth Evaluation ---
    if GROUND_TRUTH.exists():
        print("\n=== Ground Truth Evaluation ===")
        scores = score_against_ground_truth(
            composite=result,
            ground_truth_path=GROUND_TRUTH,
            wrist_center=(watch_cx, watch_cy),
            target_width_px=target_width_px,
        )
        print(f"  SSIM:           {scores['ssim']:.4f}")
        print(f"  Histogram corr: {scores['hist_corr']:.4f}")
        print(f"  Edge F1:        {scores['edge_f1']:.4f}  (P={scores['edge_precision']:.4f} R={scores['edge_recall']:.4f})")
        print(f"  Combined score: {scores['combined']:.1f} / 100")
        print(f"  Debug crops saved to outputs/eval_*.png")
    else:
        print(f"\nSkipping eval — ground truth not found: {GROUND_TRUTH}")


if __name__ == "__main__":
    main()
