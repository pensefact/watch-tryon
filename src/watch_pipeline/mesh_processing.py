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

    current_extents = mesh.bounding_box.extents

    # Sort mesh axes by extent size and match to watch dimensions sorted by size:
    # thinnest mesh axis → thickness, middle → diameter, tallest → lug-to-lug
    axis_order = np.argsort(current_extents)  # [thinnest, middle, tallest]
    target_sorted = np.sort([spec.thickness_mm, spec.case_diameter_mm, spec.lug_to_lug_mm])

    target_extents = np.zeros(3)
    for rank, axis_idx in enumerate(axis_order):
        target_extents[axis_idx] = target_sorted[rank]

    scale_factors = target_extents / current_extents
    mesh.vertices *= scale_factors

    # Re-center after scaling
    mesh.vertices -= mesh.bounding_box.centroid

    return mesh
