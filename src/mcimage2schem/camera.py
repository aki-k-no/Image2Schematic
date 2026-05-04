from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image


EXIF_TAGS = {value: key for key, value in ExifTags.TAGS.items()}


@dataclass(frozen=True, slots=True)
class CameraVector:
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [round(self.x, 6), round(self.y, 6), round(self.z, 6)]


@dataclass(frozen=True, slots=True)
class CameraEstimate:
    width: int
    height: int
    principal_point_x: float
    principal_point_y: float
    focal_length_px_x: float
    focal_length_px_y: float
    horizontal_fov_deg: float
    vertical_fov_deg: float
    pitch_down_deg: float
    roll_cw_deg: float
    horizon_center_y_px: float
    horizon_slope_px_per_x: float
    calibration_source: str
    exif_focal_length_mm: float | None
    exif_focal_length_35mm_eq_mm: float | None
    position_camera_world: CameraVector
    axis_right_world: CameraVector
    axis_up_world: CameraVector
    axis_forward_world: CameraVector
    ray_center: CameraVector
    ray_top_left: CameraVector
    ray_top_right: CameraVector
    ray_bottom_left: CameraVector
    ray_bottom_right: CameraVector
    ray_horizon_center: CameraVector
    ray_center_world: CameraVector
    ray_top_left_world: CameraVector
    ray_top_right_world: CameraVector
    ray_bottom_left_world: CameraVector
    ray_bottom_right_world: CameraVector
    ray_horizon_center_world: CameraVector

    def as_dict(self) -> dict:
        return {
            "frame": {
                "origin": "camera optical center",
                "position_camera_world": self.position_camera_world.as_list(),
                "camera_axes_world": {
                    "right": self.axis_right_world.as_list(),
                    "up": self.axis_up_world.as_list(),
                    "forward": self.axis_forward_world.as_list(),
                },
            },
            "image_size": {"width": self.width, "height": self.height},
            "intrinsics": {
                "principal_point_x": round(self.principal_point_x, 4),
                "principal_point_y": round(self.principal_point_y, 4),
                "focal_length_px_x": round(self.focal_length_px_x, 4),
                "focal_length_px_y": round(self.focal_length_px_y, 4),
                "horizontal_fov_deg": round(self.horizontal_fov_deg, 4),
                "vertical_fov_deg": round(self.vertical_fov_deg, 4),
                "calibration_source": self.calibration_source,
                "exif_focal_length_mm": self.exif_focal_length_mm,
                "exif_focal_length_35mm_eq_mm": self.exif_focal_length_35mm_eq_mm,
            },
            "orientation": {
                "pitch_down_deg": round(self.pitch_down_deg, 4),
                "roll_cw_deg": round(self.roll_cw_deg, 4),
                "horizon_center_y_px": round(self.horizon_center_y_px, 4),
                "horizon_slope_px_per_x": round(self.horizon_slope_px_per_x, 6),
            },
            "convention": {
                "camera_axes": {
                    "x": "right",
                    "y": "up",
                    "z": "forward",
                },
                "depth_interpretation": "forward_depth_along_camera_z",
                "pitch_down_deg": "positive means camera is tilted downward",
                "roll_cw_deg": "positive means image is rotated clockwise",
            },
            "ray_directions_camera": {
                "center": self.ray_center.as_list(),
                "top_left": self.ray_top_left.as_list(),
                "top_right": self.ray_top_right.as_list(),
                "bottom_left": self.ray_bottom_left.as_list(),
                "bottom_right": self.ray_bottom_right.as_list(),
                "horizon_center": self.ray_horizon_center.as_list(),
            },
            "ray_directions_world": {
                "center": self.ray_center_world.as_list(),
                "top_left": self.ray_top_left_world.as_list(),
                "top_right": self.ray_top_right_world.as_list(),
                "bottom_left": self.ray_bottom_left_world.as_list(),
                "bottom_right": self.ray_bottom_right_world.as_list(),
                "horizon_center": self.ray_horizon_center_world.as_list(),
            },
            "point_reconstruction": {
                "camera_point_formula": "[((u-cx)/fx)*z_forward, -((v-cy)/fy)*z_forward, z_forward]",
                "note": "Depth Anything V2 output is relative depth. Reconstructed points are unique in normalized camera units under the forward-depth assumption, not absolute metric world units."
            },
        }

    def camera_ray_from_pixel(self, u: float, v: float) -> CameraVector:
        x = (u - self.principal_point_x) / self.focal_length_px_x
        y = -(v - self.principal_point_y) / self.focal_length_px_y
        return CameraVector(float(x), float(y), 1.0)

    def camera_point_from_pixel(self, u: float, v: float, forward_depth: float) -> CameraVector:
        ray = self.camera_ray_from_pixel(u, v)
        return CameraVector(ray.x * forward_depth, ray.y * forward_depth, float(forward_depth))

    def world_point_from_pixel(self, u: float, v: float, forward_depth: float) -> CameraVector:
        point = self.camera_point_from_pixel(u, v, forward_depth)
        return CameraVector(
            self.position_camera_world.x + self.axis_right_world.x * point.x + self.axis_up_world.x * point.y + self.axis_forward_world.x * point.z,
            self.position_camera_world.y + self.axis_right_world.y * point.x + self.axis_up_world.y * point.y + self.axis_forward_world.y * point.z,
            self.position_camera_world.z + self.axis_right_world.z * point.x + self.axis_up_world.z * point.y + self.axis_forward_world.z * point.z,
        )

    def point_grid_from_forward_depth(self, depth_map: np.ndarray) -> np.ndarray:
        height, width = depth_map.shape
        xs = np.arange(width, dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)
        x = ((uu - self.principal_point_x) / self.focal_length_px_x) * depth_map
        y = -((vv - self.principal_point_y) / self.focal_length_px_y) * depth_map
        z = depth_map.astype(np.float32)
        return np.stack([x, y, z], axis=-1)

    def world_point_grid_from_forward_depth(self, depth_map: np.ndarray) -> np.ndarray:
        camera_points = self.point_grid_from_forward_depth(depth_map)
        right = np.array([self.axis_right_world.x, self.axis_right_world.y, self.axis_right_world.z], dtype=np.float32)
        up = np.array([self.axis_up_world.x, self.axis_up_world.y, self.axis_up_world.z], dtype=np.float32)
        forward = np.array([self.axis_forward_world.x, self.axis_forward_world.y, self.axis_forward_world.z], dtype=np.float32)
        world = (
            camera_points[..., 0:1] * right
            + camera_points[..., 1:2] * up
            + camera_points[..., 2:3] * forward
        )
        origin = np.array(
            [self.position_camera_world.x, self.position_camera_world.y, self.position_camera_world.z],
            dtype=np.float32,
        )
        return world + origin


class CameraEstimator:
    _geocalib_model = None
    _geocalib_device: str | None = None

    def estimate(self, input_path: Path, rgb_image: np.ndarray) -> CameraEstimate:
        learned_estimate = self._estimate_with_geocalib(input_path, rgb_image)
        if learned_estimate is not None:
            return learned_estimate

        height, width = rgb_image.shape[:2]
        exif_mm, exif_35mm = self._read_exif_focal_lengths(input_path)
        horizon_y, slope = self._estimate_horizon(rgb_image)
        hfov_deg, source = self._estimate_horizontal_fov(width, exif_mm, exif_35mm)
        vfov_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * (height / width)))
        fx = (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
        fy = (height * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        pitch_down_deg = math.degrees(math.atan2(horizon_y - cy, fy))
        roll_cw_deg = math.degrees(math.atan(slope))
        axis_right, axis_up, axis_forward = self._camera_axes_world(pitch_down_deg, roll_cw_deg)
        ray_center = self._ray_from_pixel(cx, cy, cx, cy, fx, fy)
        ray_top_left = self._ray_from_pixel(0.0, 0.0, cx, cy, fx, fy)
        ray_top_right = self._ray_from_pixel(width - 1.0, 0.0, cx, cy, fx, fy)
        ray_bottom_left = self._ray_from_pixel(0.0, height - 1.0, cx, cy, fx, fy)
        ray_bottom_right = self._ray_from_pixel(width - 1.0, height - 1.0, cx, cy, fx, fy)
        ray_horizon_center = self._ray_from_pixel(cx, horizon_y, cx, cy, fx, fy)
        return CameraEstimate(
            width=width,
            height=height,
            principal_point_x=cx,
            principal_point_y=cy,
            focal_length_px_x=fx,
            focal_length_px_y=fy,
            horizontal_fov_deg=hfov_deg,
            vertical_fov_deg=vfov_deg,
            pitch_down_deg=pitch_down_deg,
            roll_cw_deg=roll_cw_deg,
            horizon_center_y_px=horizon_y,
            horizon_slope_px_per_x=slope,
            calibration_source=source,
            exif_focal_length_mm=exif_mm,
            exif_focal_length_35mm_eq_mm=exif_35mm,
            position_camera_world=CameraVector(0.0, 0.0, 0.0),
            axis_right_world=axis_right,
            axis_up_world=axis_up,
            axis_forward_world=axis_forward,
            ray_center=ray_center,
            ray_top_left=ray_top_left,
            ray_top_right=ray_top_right,
            ray_bottom_left=ray_bottom_left,
            ray_bottom_right=ray_bottom_right,
            ray_horizon_center=ray_horizon_center,
            ray_center_world=self._camera_to_world(ray_center, axis_right, axis_up, axis_forward),
            ray_top_left_world=self._camera_to_world(ray_top_left, axis_right, axis_up, axis_forward),
            ray_top_right_world=self._camera_to_world(ray_top_right, axis_right, axis_up, axis_forward),
            ray_bottom_left_world=self._camera_to_world(ray_bottom_left, axis_right, axis_up, axis_forward),
            ray_bottom_right_world=self._camera_to_world(ray_bottom_right, axis_right, axis_up, axis_forward),
            ray_horizon_center_world=self._camera_to_world(ray_horizon_center, axis_right, axis_up, axis_forward),
        )

    def _estimate_with_geocalib(self, input_path: Path, rgb_image: np.ndarray) -> CameraEstimate | None:
        try:
            import torch
            from geocalib import GeoCalib
        except ImportError:
            return None
        try:
            model = self._load_geocalib_model(GeoCalib, torch)
            image_array = np.array(rgb_image, copy=True, order="C")
            image = torch.from_numpy(image_array).permute(2, 0, 1).float().div(255.0)
            image = image.to(self._geocalib_device)
            with torch.inference_mode():
                result = model.calibrate(image)
            camera = result["camera"]
            gravity = result["gravity"]
            exif_mm, exif_35mm = self._read_exif_focal_lengths(input_path)
            height, width = rgb_image.shape[:2]
            fx = float(camera.f[0, 0].item())
            fy = float(camera.f[0, 1].item())
            cx = float(camera.c[0, 0].item())
            cy = float(camera.c[0, 1].item())
            hfov_deg = math.degrees(float(camera.hfov.item()))
            vfov_deg = math.degrees(float(camera.vfov.item()))
            pitch_down_deg = math.degrees(float(gravity.pitch.item()))
            roll_cw_deg = -math.degrees(float(gravity.roll.item()))
            slope = math.tan(math.radians(roll_cw_deg))
            horizon_y = cy + math.tan(math.radians(pitch_down_deg)) * fy
            axis_right, axis_up, axis_forward = self._camera_axes_world(pitch_down_deg, roll_cw_deg)
            ray_center = self._ray_from_pixel(cx, cy, cx, cy, fx, fy)
            ray_top_left = self._ray_from_pixel(0.0, 0.0, cx, cy, fx, fy)
            ray_top_right = self._ray_from_pixel(width - 1.0, 0.0, cx, cy, fx, fy)
            ray_bottom_left = self._ray_from_pixel(0.0, height - 1.0, cx, cy, fx, fy)
            ray_bottom_right = self._ray_from_pixel(width - 1.0, height - 1.0, cx, cy, fx, fy)
            ray_horizon_center = self._ray_from_pixel(cx, horizon_y, cx, cy, fx, fy)
            return CameraEstimate(
                width=width,
                height=height,
                principal_point_x=cx,
                principal_point_y=cy,
                focal_length_px_x=fx,
                focal_length_px_y=fy,
                horizontal_fov_deg=hfov_deg,
                vertical_fov_deg=vfov_deg,
                pitch_down_deg=pitch_down_deg,
                roll_cw_deg=roll_cw_deg,
                horizon_center_y_px=horizon_y,
                horizon_slope_px_per_x=slope,
                calibration_source="geocalib_pinhole",
                exif_focal_length_mm=exif_mm,
                exif_focal_length_35mm_eq_mm=exif_35mm,
                position_camera_world=CameraVector(0.0, 0.0, 0.0),
                axis_right_world=axis_right,
                axis_up_world=axis_up,
                axis_forward_world=axis_forward,
                ray_center=ray_center,
                ray_top_left=ray_top_left,
                ray_top_right=ray_top_right,
                ray_bottom_left=ray_bottom_left,
                ray_bottom_right=ray_bottom_right,
                ray_horizon_center=ray_horizon_center,
                ray_center_world=self._camera_to_world(ray_center, axis_right, axis_up, axis_forward),
                ray_top_left_world=self._camera_to_world(ray_top_left, axis_right, axis_up, axis_forward),
                ray_top_right_world=self._camera_to_world(ray_top_right, axis_right, axis_up, axis_forward),
                ray_bottom_left_world=self._camera_to_world(ray_bottom_left, axis_right, axis_up, axis_forward),
                ray_bottom_right_world=self._camera_to_world(ray_bottom_right, axis_right, axis_up, axis_forward),
                ray_horizon_center_world=self._camera_to_world(ray_horizon_center, axis_right, axis_up, axis_forward),
            )
        except Exception:
            return None

    def _load_geocalib_model(self, geocalib_cls, torch_module):
        device = "cuda" if torch_module.cuda.is_available() else "cpu"
        if self._geocalib_model is None or self._geocalib_device != device:
            self._geocalib_model = geocalib_cls().to(device)
            self._geocalib_device = device
        return self._geocalib_model

    def _read_exif_focal_lengths(self, input_path: Path) -> tuple[float | None, float | None]:
        try:
            with Image.open(input_path) as image:
                exif = image.getexif()
        except OSError:
            return None, None
        focal_mm = self._ratio_to_float(exif.get(EXIF_TAGS.get("FocalLength")))
        focal_35mm = self._ratio_to_float(exif.get(EXIF_TAGS.get("FocalLengthIn35mmFilm")))
        return focal_mm, focal_35mm

    def _estimate_horizontal_fov(
        self,
        width: int,
        exif_focal_length_mm: float | None,
        exif_focal_length_35mm_eq_mm: float | None,
    ) -> tuple[float, str]:
        if exif_focal_length_35mm_eq_mm and exif_focal_length_35mm_eq_mm > 0:
            hfov = math.degrees(2.0 * math.atan(36.0 / (2.0 * exif_focal_length_35mm_eq_mm)))
            return float(np.clip(hfov, 20.0, 100.0)), "exif_35mm_equivalent"
        if exif_focal_length_mm and exif_focal_length_mm <= 18.0:
            return 73.0, "exif_focal_length_wide_heuristic"
        if exif_focal_length_mm and exif_focal_length_mm >= 50.0:
            return 27.0, "exif_focal_length_tele_heuristic"
        return 50.0, "landscape_default_heuristic"

    def _estimate_horizon(self, rgb_image: np.ndarray) -> tuple[float, float]:
        mask = self._estimate_sky_mask(rgb_image)
        height, width = mask.shape
        xs: list[float] = []
        ys: list[float] = []
        for x in range(width):
            column = mask[:, x]
            indices = np.flatnonzero(column)
            if indices.size == 0:
                continue
            y = float(indices.max())
            if 2.0 <= y <= height * 0.85:
                xs.append(float(x))
                ys.append(y)
        if len(xs) < max(8, width // 12):
            return height * 0.38, 0.0
        slope, intercept = np.polyfit(np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), deg=1)
        center_x = (width - 1) * 0.5
        horizon_y = float(slope * center_x + intercept)
        return float(np.clip(horizon_y, 0.0, height - 1.0)), float(slope)

    def _estimate_sky_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        normalized = rgb_image.astype(np.float32) / 255.0
        r = normalized[:, :, 0]
        g = normalized[:, :, 1]
        b = normalized[:, :, 2]
        brightness = normalized.max(axis=2)
        saturation = brightness - normalized.min(axis=2)
        blue_sky = (b > g * 0.95) & (g > r * 0.9) & (brightness > 0.35)
        bright_cloud = (brightness > 0.75) & (saturation < 0.18)
        top_bias = np.linspace(1.0, 0.0, rgb_image.shape[0], dtype=np.float32)[:, None]
        return (blue_sky | bright_cloud) & (top_bias > 0.15)

    def _ray_from_pixel(self, x: float, y: float, cx: float, cy: float, fx: float, fy: float) -> CameraVector:
        dx = (x - cx) / fx
        dy = -(y - cy) / fy
        dz = 1.0
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        return CameraVector(dx / length, dy / length, dz / length)

    def _camera_axes_world(
        self,
        pitch_down_deg: float,
        roll_cw_deg: float,
    ) -> tuple[CameraVector, CameraVector, CameraVector]:
        pitch = math.radians(pitch_down_deg)
        roll = math.radians(roll_cw_deg)

        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cr = math.cos(roll)
        sr = math.sin(roll)

        right = np.array([cr, -sr, 0.0], dtype=np.float64)
        up = np.array([sr * cp, cr * cp, -sp], dtype=np.float64)
        forward = np.array([sr * sp, cr * sp, cp], dtype=np.float64)

        return (
            CameraVector(*right.tolist()),
            CameraVector(*up.tolist()),
            CameraVector(*forward.tolist()),
        )

    def _camera_to_world(
        self,
        ray: CameraVector,
        axis_right: CameraVector,
        axis_up: CameraVector,
        axis_forward: CameraVector,
    ) -> CameraVector:
        world = (
            np.array([axis_right.x, axis_right.y, axis_right.z], dtype=np.float64) * ray.x
            + np.array([axis_up.x, axis_up.y, axis_up.z], dtype=np.float64) * ray.y
            + np.array([axis_forward.x, axis_forward.y, axis_forward.z], dtype=np.float64) * ray.z
        )
        length = float(np.linalg.norm(world))
        if length == 0.0:
            return CameraVector(0.0, 0.0, 0.0)
        world = world / length
        return CameraVector(float(world[0]), float(world[1]), float(world[2]))

    def _ratio_to_float(self, value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if numerator is not None and denominator:
            return float(numerator) / float(denominator)
        if isinstance(value, tuple) and len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        return None
