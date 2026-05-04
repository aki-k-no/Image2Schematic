from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .blocks import BlockSelector
from .camera import CameraEstimate, CameraEstimator
from .classify import LabeledRegion, RegionClassifier
from .config import WorkflowConfig
from .depth import DepthAnythingV2Estimator
from .debug import DebugArtifactWriter
from .image_utils import load_and_resize_image, to_numpy_rgb
from .schematic import SchematicVolume, SchematicWriter
from .segment import SamSegmenter
from .voxelize import (
    ForwardDistanceFit,
    ScaleFit,
    compute_forward_distance_map,
    compute_scale_fit,
    paint_line,
    paint_triangle,
    paint_voxel,
    quantize_voxel_coords,
    scale_points_to_voxel_coords,
)


@dataclass(slots=True)
class PipelineArtifacts:
    camera: CameraEstimate
    regions: list[LabeledRegion]
    schematic: SchematicVolume
    debug_dir: Path
    scale_fit: ScaleFit
    forward_distance_fit: ForwardDistanceFit


class ImageToSchematicPipeline:
    def __init__(self, config: WorkflowConfig, cache_root: Path) -> None:
        self.config = config
        self.depth_estimator = DepthAnythingV2Estimator(config.depth, cache_root)
        self.segmenter = SamSegmenter(config.segmentation, cache_root)
        self.classifier = RegionClassifier(config.classification)
        self.block_selector = BlockSelector()
        self.writer = SchematicWriter()
        self.debug_writer = DebugArtifactWriter()
        self.camera_estimator = CameraEstimator()

    def run(self, input_path: Path, output_path: Path) -> PipelineArtifacts:
        image = load_and_resize_image(
            input_path,
            self.config.image.target_width,
            self.config.image.target_height,
        )
        rgb = to_numpy_rgb(image)
        camera = self.camera_estimator.estimate(input_path, rgb)
        depth = self.depth_estimator.estimate(image).normalized_depth
        masks = self.segmenter.generate_masks(image)
        regions = self.classifier.classify(rgb, depth, masks)
        label_map = self._compose_label_map(rgb.shape[:2], regions)
        schematic, scale_fit, forward_distance_map, forward_distance_fit = self._build_schematic(camera, rgb, depth, label_map)
        self.writer.write(schematic, output_path)
        debug_dir = self.debug_writer.write(output_path, rgb, depth, forward_distance_map, camera, regions, label_map, schematic, scale_fit, forward_distance_fit)
        return PipelineArtifacts(camera=camera, regions=regions, schematic=schematic, debug_dir=debug_dir, scale_fit=scale_fit, forward_distance_fit=forward_distance_fit)

    def _build_schematic(
        self,
        camera: CameraEstimate,
        rgb: np.ndarray,
        depth: np.ndarray,
        label_map: list[list[tuple[str, tuple[int, int, int]]]],
    ) -> tuple[SchematicVolume, ScaleFit, np.ndarray, ForwardDistanceFit]:
        configured_target_size = (
            self.config.build.target_width,
            self.config.build.target_height,
            self.config.build.target_length,
        )
        valid_points: list[np.ndarray] = []
        valid_mask = np.zeros(depth.shape, dtype=bool)
        block_grid = np.full(depth.shape, self.config.build.background_block, dtype=object)
        label_grid = np.full(depth.shape, "", dtype=object)
        rgb_grid = np.zeros((*depth.shape, 3), dtype=np.uint8)
        for y_image, row in enumerate(label_map):
            for x, (label, avg_rgb) in enumerate(row):
                if label == "sky":
                    continue
                valid_mask[y_image, x] = True
                label_grid[y_image, x] = label
                rgb_grid[y_image, x] = np.asarray(avg_rgb, dtype=np.uint8)

        forward_distance_map, forward_distance_fit = compute_forward_distance_map(
            principal_point_x=camera.principal_point_x,
            focal_length_px_x=camera.focal_length_px_x,
            relative_depth=depth.astype(np.float32),
            valid_mask=valid_mask,
            target_width=configured_target_size[0],
            target_length=configured_target_size[2],
            near_width_fill_ratio=self.config.build.near_width_fill_ratio,
            near_depth_percentile=self.config.build.near_depth_percentile,
            forward_distance_scale=self.config.build.forward_distance_scale,
            far_distance_pivot=self.config.build.far_distance_pivot,
            far_distance_boost=self.config.build.far_distance_boost,
            far_distance_power=self.config.build.far_distance_power,
        )
        points_world = camera.world_point_grid_from_forward_depth(forward_distance_map)
        for y_image, x in zip(*np.where(valid_mask)):
            valid_points.append(points_world[y_image, x])

        if not valid_points:
            target_size = configured_target_size
            block_states = np.full(
                target_size,
                self.config.build.background_block,
                dtype=object,
            )
            empty = SchematicVolume(
                width=target_size[0],
                height=target_size[1],
                length=target_size[2],
                block_states=block_states,
            )
            return empty, compute_scale_fit(np.zeros((0, 3), dtype=np.float32), target_size), forward_distance_map, forward_distance_fit

        point_array = np.asarray(valid_points, dtype=np.float32)
        if self.config.build.placement_mode == "direct_valid_points":
            int_points_world = points_world.astype(np.int32)
            local_min = int_points_world[valid_mask].min(axis=0)
            local_points = int_points_world - local_min
            local_valid_points = local_points[valid_mask]
            local_max = local_valid_points.max(axis=0)
            target_size = (
                int(local_max[0]) + 1,
                int(local_max[1]) + 1,
                int(local_max[2]) + 1,
            )
            block_states = np.full(
                target_size,
                self.config.build.background_block,
                dtype=object,
            )
            scale_fit = compute_scale_fit(
                local_valid_points.astype(np.float32),
                target_size,
                anchor_min_x_world=float(local_valid_points[:, 0].min()),
                anchor_span_x_world=float(max(local_valid_points[:, 0].max() - local_valid_points[:, 0].min(), 1e-6)),
            )
            voxel_grid = local_points.astype(np.int32)
        else:
            target_size = configured_target_size
            block_states = np.full(
                target_size,
                self.config.build.background_block,
                dtype=object,
            )
            front_points = points_world[forward_distance_fit.front_mask]
            front_min_x = float(front_points[:, 0].min()) if front_points.size else float(point_array[:, 0].min())
            front_span_x = float(max(front_points[:, 0].max() - front_min_x, 1e-6)) if front_points.size else float(max(point_array[:, 0].max() - point_array[:, 0].min(), 1e-6))
            scale_fit = compute_scale_fit(
                point_array,
                target_size,
                anchor_min_x_world=front_min_x,
                anchor_span_x_world=front_span_x,
            )
            scaled_points = scale_points_to_voxel_coords(points_world.astype(np.float32), scale_fit)
            voxel_grid = quantize_voxel_coords(scaled_points, target_size)
        height_2d, width_2d = depth.shape

        for y_image in range(height_2d):
            for x in range(width_2d):
                if not valid_mask[y_image, x]:
                    continue
                coord = voxel_grid[y_image, x]
                block_grid[y_image, x] = self.block_selector.choose(
                    label=str(label_grid[y_image, x]),
                    rgb=tuple(int(v) for v in rgb_grid[y_image, x].tolist()),
                    default_block=self.config.build.default_block,
                    image_x=x,
                    image_y=y_image,
                    voxel_x=int(coord[0]),
                    voxel_y=int(coord[1]),
                    voxel_z=int(coord[2]),
                    depth_value=float(depth[y_image, x]),
                )

        for y_image in range(height_2d):
            for x in range(width_2d):
                if not valid_mask[y_image, x]:
                    continue
                coord = tuple(int(v) for v in voxel_grid[y_image, x].tolist())
                block = str(block_grid[y_image, x])
                if self.config.build.placement_mode == "direct_valid_points":
                    paint_voxel(block_states, coord, block, radius=self.config.build.point_radius)
                elif self.config.build.fill_mode == "surface":
                    paint_voxel(block_states, coord, block, radius=self.config.build.point_radius)
                else:
                    px, py, pz = coord
                    block_states[px, py, pz:target_size[2]] = block

                if not self.config.build.connect_neighbors:
                    continue
                for dx, dy in ((1, 0), (0, 1)):
                    nx = x + dx
                    ny = y_image + dy
                    if nx >= width_2d or ny >= height_2d or not valid_mask[ny, nx]:
                        continue
                    neighbor = tuple(int(v) for v in voxel_grid[ny, nx].tolist())
                    gap = max(abs(neighbor[i] - coord[i]) for i in range(3))
                    if gap > self.config.build.max_connection_gap:
                        continue
                    if abs(neighbor[2] - coord[2]) > self.config.build.max_surface_depth_gap:
                        continue
                    paint_line(
                        block_states,
                        coord,
                        neighbor,
                        block,
                        radius=self.config.build.point_radius,
                    )

        if self.config.build.connect_neighbors and self.config.build.fill_triangles:
            for y_image in range(height_2d - 1):
                for x in range(width_2d - 1):
                    if not (
                        valid_mask[y_image, x]
                        and valid_mask[y_image, x + 1]
                        and valid_mask[y_image + 1, x]
                        and valid_mask[y_image + 1, x + 1]
                    ):
                        continue
                    p00 = tuple(int(v) for v in voxel_grid[y_image, x].tolist())
                    p10 = tuple(int(v) for v in voxel_grid[y_image, x + 1].tolist())
                    p01 = tuple(int(v) for v in voxel_grid[y_image + 1, x].tolist())
                    p11 = tuple(int(v) for v in voxel_grid[y_image + 1, x + 1].tolist())
                    max_gap = max(
                        max(abs(p00[i] - p10[i]) for i in range(3)),
                        max(abs(p00[i] - p01[i]) for i in range(3)),
                        max(abs(p11[i] - p10[i]) for i in range(3)),
                        max(abs(p11[i] - p01[i]) for i in range(3)),
                    )
                    if max_gap > self.config.build.max_connection_gap:
                        continue
                    max_depth_gap = max(
                        abs(p00[2] - p10[2]),
                        abs(p00[2] - p01[2]),
                        abs(p11[2] - p10[2]),
                        abs(p11[2] - p01[2]),
                    )
                    if max_depth_gap > self.config.build.max_surface_depth_gap:
                        continue
                    block = str(block_grid[y_image, x])
                    paint_triangle(
                        block_states,
                        p00,
                        p10,
                        p01,
                        block,
                        radius=self.config.build.point_radius,
                    )
                    paint_triangle(
                        block_states,
                        p11,
                        p10,
                        p01,
                        block,
                        radius=self.config.build.point_radius,
                    )

        volume = SchematicVolume(
            width=target_size[0],
            height=target_size[1],
            length=target_size[2],
            block_states=block_states,
        )
        return volume, scale_fit, forward_distance_map, forward_distance_fit

    def _compose_label_map(
        self,
        image_shape: tuple[int, int],
        regions: list[LabeledRegion],
    ) -> list[list[tuple[str, tuple[int, int, int]]]]:
        height, width = image_shape
        default = ("ground", (128, 128, 128))
        label_map = [[default for _ in range(width)] for _ in range(height)]
        for region in regions:
            ys, xs = np.where(region.mask)
            for y, x in zip(ys.tolist(), xs.tolist()):
                label_map[y][x] = (region.label, region.mean_rgb)
        return label_map
