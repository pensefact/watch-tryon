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
