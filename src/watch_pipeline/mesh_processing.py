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
