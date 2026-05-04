from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScaleFit:
    scale_x: float
    scale_y: float
    scale_z: float
    min_corner_world: tuple[float, float, float]
    span_world: tuple[float, float, float]
    target_size: tuple[int, int, int]
    anchor_min_x_world: float
    anchor_span_x_world: float


@dataclass(frozen=True, slots=True)
class ForwardDistanceFit:
    near_distance: float
    far_distance: float
    front_depth_threshold: float
    front_span_unit_x: float
    target_front_width: float
    front_mask: np.ndarray


def compute_forward_distance_map(
    principal_point_x: float,
    focal_length_px_x: float,
    relative_depth: np.ndarray,
    valid_mask: np.ndarray,
    target_width: int,
    target_length: int,
    near_width_fill_ratio: float,
    near_depth_percentile: float,
    forward_distance_scale: float = 1.0,
    far_distance_pivot: float = 0.6,
    far_distance_boost: float = 0.0,
    far_distance_power: float = 1.0,
) -> tuple[np.ndarray, ForwardDistanceFit]:
    valid_depth = relative_depth[valid_mask]
    if valid_depth.size == 0:
        front_mask = np.zeros_like(relative_depth, dtype=bool)
        empty = np.zeros_like(relative_depth, dtype=np.float32)
        return empty, ForwardDistanceFit(
            near_distance=1.0,
            far_distance=float(max(target_length, 1)),
            front_depth_threshold=1.0,
            front_span_unit_x=1.0,
            target_front_width=float(max(target_width - 1, 1)),
            front_mask=front_mask,
        )

    percentile = float(np.clip(near_depth_percentile, 0.0, 1.0))
    threshold = float(np.quantile(valid_depth, 1.0 - percentile))
    front_mask = valid_mask & (relative_depth <= threshold)
    if int(front_mask.sum()) < 8:
        front_mask = valid_mask.copy()

    xs = np.arange(relative_depth.shape[1], dtype=np.float32)
    unit_x = (xs - float(principal_point_x)) / float(max(focal_length_px_x, 1e-6))
    front_x_values = unit_x[np.where(front_mask)[1]]
    span_unit_x = float(max(front_x_values.max() - front_x_values.min(), 1e-6))
    target_front_width = float(max((target_width - 1) * near_width_fill_ratio, 1.0))
    near_distance = target_front_width / span_unit_x
    far_distance = near_distance + float(max(target_length - 1, 1))
    distance_scale = max(float(forward_distance_scale), 1e-6)
    near_distance *= distance_scale
    far_distance *= distance_scale
    relative_depth = relative_depth.astype(np.float32)
    span = far_distance - near_distance
    forward_distance = near_distance + relative_depth * span
    pivot = float(np.clip(far_distance_pivot, 0.0, 0.999999))
    boost = max(float(far_distance_boost), 0.0)
    power = max(float(far_distance_power), 1e-6)
    if boost > 0.0 and pivot < 1.0:
        tail = np.clip((relative_depth - pivot) / max(1.0 - pivot, 1e-6), 0.0, 1.0)
        forward_distance = forward_distance + (tail ** power) * span * boost
    return forward_distance.astype(np.float32), ForwardDistanceFit(
        near_distance=float(near_distance),
        far_distance=float(far_distance),
        front_depth_threshold=threshold,
        front_span_unit_x=span_unit_x,
        target_front_width=target_front_width,
        front_mask=front_mask,
    )


def compute_scale_fit(
    points_world: np.ndarray,
    target_size: tuple[int, int, int],
    anchor_min_x_world: float | None = None,
    anchor_span_x_world: float | None = None,
) -> ScaleFit:
    if points_world.size == 0:
        return ScaleFit(1.0, 1.0, 1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), target_size, 0.0, 1.0)
    mins = points_world.min(axis=0)
    maxs = points_world.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    limits = np.maximum(np.asarray(target_size, dtype=np.float32) - 1.0, 1.0)
    anchor_min_x = float(mins[0] if anchor_min_x_world is None else anchor_min_x_world)
    anchor_span_x = float(spans[0] if anchor_span_x_world is None else max(anchor_span_x_world, 1e-6))
    scale_x = float(limits[0] / anchor_span_x)
    scale_y = float(limits[1] / spans[1])
    scale_z = float(limits[2] / spans[2])
    return ScaleFit(
        scale_x=scale_x,
        scale_y=scale_y,
        scale_z=scale_z,
        min_corner_world=(float(mins[0]), float(mins[1]), float(mins[2])),
        span_world=(float(spans[0]), float(spans[1]), float(spans[2])),
        target_size=target_size,
        anchor_min_x_world=anchor_min_x,
        anchor_span_x_world=anchor_span_x,
    )


def scale_points_to_voxel_coords(points_world: np.ndarray, fit: ScaleFit) -> np.ndarray:
    offsets = np.array(
        [fit.anchor_min_x_world, fit.min_corner_world[1], fit.min_corner_world[2]],
        dtype=np.float32,
    )
    scales = np.array([fit.scale_x, fit.scale_y, fit.scale_z], dtype=np.float32)
    return (points_world - offsets) * scales


def quantize_voxel_coords(coords: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    limits = np.asarray(target_size, dtype=np.int32) - 1
    quantized = np.rint(coords).astype(np.int32)
    return np.clip(quantized, 0, limits)


def paint_voxel(
    block_states: np.ndarray,
    coord: tuple[int, int, int],
    block: str,
    radius: int = 0,
) -> None:
    x, y, z = coord
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    z0 = max(0, z - radius)
    x1 = min(block_states.shape[0] - 1, x + radius)
    y1 = min(block_states.shape[1] - 1, y + radius)
    z1 = min(block_states.shape[2] - 1, z + radius)
    block_states[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = block


def paint_line(
    block_states: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    block: str,
    radius: int = 0,
) -> None:
    start_arr = np.asarray(start, dtype=np.float32)
    end_arr = np.asarray(end, dtype=np.float32)
    delta = end_arr - start_arr
    steps = int(max(np.abs(delta).max(), 1))
    for index in range(steps + 1):
        t = index / steps
        point = np.rint(start_arr + delta * t).astype(np.int32)
        paint_voxel(block_states, (int(point[0]), int(point[1]), int(point[2])), block, radius=radius)


def paint_triangle(
    block_states: np.ndarray,
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    block: str,
    radius: int = 0,
) -> None:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    c_arr = np.asarray(c, dtype=np.float32)
    edge_scale = max(
        np.abs(b_arr - a_arr).max(),
        np.abs(c_arr - a_arr).max(),
        np.abs(c_arr - b_arr).max(),
        1.0,
    )
    steps = int(edge_scale)
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            u = i / steps
            v = j / steps
            w = 1.0 - u - v
            point = np.rint(a_arr * w + b_arr * u + c_arr * v).astype(np.int32)
            paint_voxel(block_states, (int(point[0]), int(point[1]), int(point[2])), block, radius=radius)


def fill_column_gaps(
    block_states: np.ndarray,
    background_block: str,
    max_gap: int = 3,
) -> None:
    width, height, _ = block_states.shape
    for x in range(width):
        for y in range(height):
            occupied = np.flatnonzero(block_states[x, y] != background_block)
            if occupied.size < 2:
                continue
            for start, end in zip(occupied[:-1], occupied[1:]):
                gap = int(end - start - 1)
                if gap <= 0 or gap > max_gap:
                    continue
                left_block = str(block_states[x, y, int(start)])
                right_block = str(block_states[x, y, int(end)])
                fill_block = left_block if left_block == right_block else left_block
                block_states[x, y, int(start) + 1 : int(end)] = fill_block


def fill_enclosed_holes(
    block_states: np.ndarray,
    background_block: str,
    iterations: int = 1,
    min_neighbors: int = 4,
) -> None:
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    width, height, length = block_states.shape
    for _ in range(max(iterations, 0)):
        updates: list[tuple[int, int, int, str]] = []
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                for z in range(1, length - 1):
                    if block_states[x, y, z] != background_block:
                        continue
                    adjacent: list[str] = []
                    for dx, dy, dz in neighbors:
                        block = str(block_states[x + dx, y + dy, z + dz])
                        if block != background_block:
                            adjacent.append(block)
                    if len(adjacent) < min_neighbors:
                        continue
                    updates.append((x, y, z, _majority_block(adjacent)))
        if not updates:
            break
        for x, y, z, block in updates:
            block_states[x, y, z] = block


def _majority_block(blocks: list[str]) -> str:
    counts: dict[str, int] = {}
    winner = blocks[0]
    best = 0
    for block in blocks:
        counts[block] = counts.get(block, 0) + 1
        if counts[block] > best:
            best = counts[block]
            winner = block
    return winner


def compute_depth_gradient_map(depth_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    depth_map = depth_map.astype(np.float32)
    grad_x = np.zeros_like(depth_map, dtype=np.float32)
    grad_y = np.zeros_like(depth_map, dtype=np.float32)
    grad_x[:, 1:] = np.abs(depth_map[:, 1:] - depth_map[:, :-1])
    grad_y[1:, :] = np.abs(depth_map[1:, :] - depth_map[:-1, :])
    gradient = np.maximum(grad_x, grad_y)
    gradient[~valid_mask] = 0.0
    valid_values = gradient[valid_mask]
    if valid_values.size == 0:
        return gradient
    scale = float(np.percentile(valid_values, 90))
    if scale <= 1e-6:
        return np.zeros_like(gradient, dtype=np.float32)
    return np.clip(gradient / scale, 0.0, 1.0).astype(np.float32)


def estimate_back_surface_coords(
    voxel_coords: np.ndarray,
    valid_mask: np.ndarray,
    label_grid: np.ndarray,
    depth_map: np.ndarray,
    camera_local: np.ndarray,
    edge_suppression: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    back_coords = np.rint(voxel_coords).astype(np.int32).copy()
    thickness_map = np.zeros(depth_map.shape, dtype=np.float32)
    gradient_map = compute_depth_gradient_map(depth_map, valid_mask)

    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            if not valid_mask[y, x]:
                continue
            label = str(label_grid[y, x])
            base_thickness = _label_base_thickness(label)
            if base_thickness <= 0.0:
                continue
            gradient_penalty = max(0.0, 1.0 - float(edge_suppression) * float(gradient_map[y, x]))
            thickness = max(1.0, base_thickness * gradient_penalty)
            thickness_map[y, x] = thickness

            normal = _estimate_local_normal(voxel_coords, valid_mask, label_grid, x, y)
            front = voxel_coords[y, x].astype(np.float32)
            view_dir = front - camera_local.astype(np.float32)
            view_norm = float(np.linalg.norm(view_dir))
            if view_norm > 1e-6:
                view_dir = view_dir / view_norm
            else:
                view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            if normal is None:
                back_dir = view_dir
            else:
                if float(np.dot(normal, view_dir)) > 0.0:
                    normal = -normal
                back_dir = (-normal * 0.7) + (view_dir * 0.3)
                back_norm = float(np.linalg.norm(back_dir))
                if back_norm <= 1e-6:
                    back_dir = view_dir
                else:
                    back_dir = back_dir / back_norm

            back_point = front + back_dir * thickness
            back_coords[y, x] = np.rint(back_point).astype(np.int32)

    return back_coords, thickness_map


def _label_base_thickness(label: str) -> float:
    return {
        "foliage": 3.0,
        "foliage_dark": 4.0,
        "foliage_light": 3.0,
        "ground": 4.0,
        "ground_grass": 4.0,
        "ground_dirt": 4.5,
        "path": 2.0,
        "rock": 5.0,
        "rock_dark": 4.0,
        "cliff": 6.0,
        "sand": 3.0,
        "snow": 2.0,
        "water": 1.5,
        "water_deep": 2.0,
        "water_shallow": 1.0,
        "wood": 3.0,
        "shadow": 2.0,
    }.get(label, 3.0)


def _estimate_local_normal(
    voxel_coords: np.ndarray,
    valid_mask: np.ndarray,
    label_grid: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray | None:
    dx = _neighbor_span(voxel_coords, valid_mask, label_grid, x, y, axis=1)
    dy = _neighbor_span(voxel_coords, valid_mask, label_grid, x, y, axis=0)
    if dx is None or dy is None:
        return None
    normal = np.cross(dx, dy).astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-6:
        return None
    return normal / norm


def _neighbor_span(
    voxel_coords: np.ndarray,
    valid_mask: np.ndarray,
    label_grid: np.ndarray,
    x: int,
    y: int,
    axis: int,
) -> np.ndarray | None:
    label = str(label_grid[y, x])
    offsets = [(-1, 0), (1, 0)] if axis == 1 else [(0, -1), (0, 1)]
    samples: list[np.ndarray] = []
    for dx, dy in offsets:
        nx = x + dx
        ny = y + dy
        if ny < 0 or ny >= voxel_coords.shape[0] or nx < 0 or nx >= voxel_coords.shape[1]:
            continue
        if not valid_mask[ny, nx] or str(label_grid[ny, nx]) != label:
            continue
        samples.append(voxel_coords[ny, nx].astype(np.float32))
    if len(samples) == 2:
        return samples[1] - samples[0]
    if len(samples) == 1:
        return samples[0] - voxel_coords[y, x].astype(np.float32)
    return None
