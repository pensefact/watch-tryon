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

# Apply a dark metallic material to watch objects
for obj in watch_objects:
    mat = bpy.data.materials.new(name="WatchMetal")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.06, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = 0.3
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# Camera setup — orthographic, framed tight on the watch
cam_data = bpy.data.cameras.new(name='Camera')
cam_data.type = 'ORTHO'
# Ortho scale = world units visible. Watch is ~50mm lug-to-lug, add margin.
bbox = None
for obj in watch_objects:
    b = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    for v in b:
        if bbox is None:
            bbox = [list(v), list(v)]
        else:
            for i in range(3):
                bbox[0][i] = min(bbox[0][i], v[i])
                bbox[1][i] = max(bbox[1][i], v[i])
import mathutils
if bbox:
    extent = max(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1])
    cam_data.ortho_scale = extent * 1.2
else:
    cam_data.ortho_scale = 60
cam_obj = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, 0, 100)
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

# Use EEVEE for fast rendering (compatible across Blender versions)
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError:
    scene.render.engine = 'BLENDER_EEVEE'

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
