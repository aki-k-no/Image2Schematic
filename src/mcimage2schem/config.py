from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ImageConfig:
    target_width: int = 256
    target_height: int = 144


@dataclass(slots=True)
class DepthConfig:
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"
    invert: bool = True
    gamma: float = 1.0
    max_depth_layers: int = 48


@dataclass(slots=True)
class SegmentationConfig:
    model_id: str = "facebook/sam-vit-base"
    device: str = "auto"
    points_per_batch: int = 64
    points_per_side: int = 16
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 128
    max_masks: int = 24


@dataclass(slots=True)
class BuildConfig:
    placement_mode: str = "direct_valid_points"
    fill_mode: str = "surface"
    background_block: str = "minecraft:air"
    default_block: str = "minecraft:stone"
    y_scale: int = 1
    target_width: int = 128
    target_height: int = 72
    target_length: int = 48
    point_radius: int = 0
    connect_neighbors: bool = True
    max_connection_gap: int = 8
    max_surface_depth_gap: int = 2
    fill_triangles: bool = True
    near_width_fill_ratio: float = 0.92
    near_depth_percentile: float = 0.98
    forward_distance_scale: float = 1.5
    far_distance_pivot: float = 0.6
    far_distance_boost: float = 0.9
    far_distance_power: float = 1.4
    shell_enabled: bool = True
    shell_edge_suppression: float = 0.7
    fill_column_gaps: bool = True
    max_column_gap: int = 3
    fill_enclosed_holes: bool = True
    enclosed_hole_iterations: int = 1
    enclosed_hole_min_neighbors: int = 4


@dataclass(slots=True)
class ClassificationConfig:
    water_depth_threshold: float = 0.52
    sky_brightness_threshold: float = 0.65
    vegetation_hue_min: int = 60
    vegetation_hue_max: int = 170
    sky_top_portion: float = 0.45
    water_bottom_portion: float = 0.7
    dark_shadow_threshold: float = 0.2


@dataclass(slots=True)
class WorkflowConfig:
    image: ImageConfig
    depth: DepthConfig
    segmentation: SegmentationConfig
    build: BuildConfig
    classification: ClassificationConfig

    @classmethod
    def from_path(cls, path: Path) -> "WorkflowConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            image=ImageConfig(**raw.get("image", {})),
            depth=DepthConfig(**raw.get("depth", {})),
            segmentation=SegmentationConfig(**raw.get("segmentation", {})),
            build=BuildConfig(**raw.get("build", {})),
            classification=ClassificationConfig(**raw.get("classification", {})),
        )
