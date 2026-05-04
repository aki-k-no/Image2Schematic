from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from .camera import CameraEstimate
from .classify import LabeledRegion
from .schematic import SchematicVolume
from .voxelize import ForwardDistanceFit, ScaleFit


LABEL_COLORS = {
    "sky": "#8ed0ff",
    "water": "#2f63d3",
    "foliage": "#48a23f",
    "snow": "#f4f8ff",
    "sand": "#d9c27c",
    "rock": "#8a8a8a",
    "wood": "#8a5b33",
    "shadow": "#3a3a4a",
    "ground": "#8f6f4c",
}


class DebugArtifactWriter:
    def write(
        self,
        output_path: Path,
        rgb: np.ndarray,
        depth: np.ndarray,
        forward_distance: np.ndarray,
        camera: CameraEstimate,
        regions: list[LabeledRegion],
        label_map: list[list[tuple[str, tuple[int, int, int]]]],
        volume: SchematicVolume,
        scale_fit: ScaleFit,
        forward_distance_fit: ForwardDistanceFit,
    ) -> Path:
        debug_dir = output_path.parent / f"{output_path.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        base = Image.fromarray(rgb, mode="RGB")
        base.save(debug_dir / "01_resized_input.png")
        self._save_depth(depth, debug_dir / "02_depth_grayscale.png")
        self._save_masks_overlay(base, regions, debug_dir / "03_sam_overlay.png")
        self._save_label_overlay(base, label_map, debug_dir / "04_label_overlay.png")
        self._save_block_projection(volume, debug_dir / "05_block_projection.png")
        self._save_camera_overlay(base, camera, debug_dir / "06_camera_overlay.png")
        self._save_front_mask_overlay(base, forward_distance_fit.front_mask, debug_dir / "06b_front_mask_overlay.png")
        self._save_camera_report(camera, debug_dir / "camera.json")
        valid_mask = self._valid_mask_from_label_map(label_map)
        self._save_point_cloud(camera, forward_distance, debug_dir)
        self._save_camera_space_plot(
            camera,
            rgb,
            forward_distance,
            valid_mask,
            forward_distance_fit,
            debug_dir / "07_camera_space_plot.png",
        )
        self._save_camera_space_html(
            camera,
            rgb,
            forward_distance,
            valid_mask,
            forward_distance_fit,
            debug_dir / "08_camera_space_plot.html",
        )
        self._save_forward_distance_report(forward_distance_fit, debug_dir / "forward_distance.json")
        self._save_scale_report(scale_fit, debug_dir / "scale_fit.json")
        self._save_region_report(regions, debug_dir / "regions.json")
        return debug_dir

    def _save_depth(self, depth: np.ndarray, path: Path) -> None:
        depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(depth_u8, mode="L").save(path)

    def _save_masks_overlay(self, base: Image.Image, regions: list[LabeledRegion], path: Path) -> None:
        canvas = base.convert("RGBA")
        for index, region in enumerate(regions):
            color = self._indexed_color(index)
            overlay = np.zeros((region.mask.shape[0], region.mask.shape[1], 4), dtype=np.uint8)
            overlay[region.mask] = (*color, 90)
            canvas = Image.alpha_composite(canvas, Image.fromarray(overlay, mode="RGBA"))
        canvas.convert("RGB").save(path)

    def _save_label_overlay(
        self,
        base: Image.Image,
        label_map: list[list[tuple[str, tuple[int, int, int]]]],
        path: Path,
    ) -> None:
        overlay = np.zeros((len(label_map), len(label_map[0]), 4), dtype=np.uint8)
        for y, row in enumerate(label_map):
            for x, (label, _) in enumerate(row):
                color = ImageColor.getrgb(LABEL_COLORS.get(label, "#ff00ff"))
                overlay[y, x] = (*color, 120)
        image = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(overlay, mode="RGBA"))
        draw = ImageDraw.Draw(image)
        for label, color in LABEL_COLORS.items():
            draw.rectangle((4, 4 + 12 * list(LABEL_COLORS).index(label), 12, 12 + 12 * list(LABEL_COLORS).index(label)), fill=color)
            draw.text((16, 2 + 12 * list(LABEL_COLORS).index(label)), label, fill="white")
        image.convert("RGB").save(path)

    def _save_block_projection(self, volume: SchematicVolume, path: Path) -> None:
        width, height, length = volume.width, volume.height, volume.length
        image = np.zeros((height, width, 3), dtype=np.uint8)
        colors = {
            block: self._block_color(block)
            for block in set(volume.block_states.ravel().tolist())
        }
        for x in range(width):
            for y in range(height):
                for z in range(length):
                    block = volume.block_states[x, y, z]
                    if block != "minecraft:air":
                        image[height - 1 - y, x] = colors[block]
                        break
        Image.fromarray(image, mode="RGB").save(path)

    def _save_region_report(self, regions: list[LabeledRegion], path: Path) -> None:
        payload = [
            {
                "label": region.label,
                "mean_rgb": region.mean_rgb,
                "mean_depth": round(region.mean_depth, 4),
                "area": int(region.mask.sum()),
                "centroid_x": round(region.centroid_x, 4),
                "centroid_y": round(region.centroid_y, 4),
            }
            for region in regions
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_camera_report(self, camera: CameraEstimate, path: Path) -> None:
        path.write_text(json.dumps(camera.as_dict(), indent=2), encoding="utf-8")

    def _save_camera_overlay(self, base: Image.Image, camera: CameraEstimate, path: Path) -> None:
        image = base.convert("RGBA")
        draw = ImageDraw.Draw(image)
        width, height = base.size
        left_y = camera.horizon_center_y_px - camera.horizon_slope_px_per_x * camera.principal_point_x
        right_y = camera.horizon_center_y_px + camera.horizon_slope_px_per_x * (width - 1 - camera.principal_point_x)
        draw.line((0, left_y, width - 1, right_y), fill=(255, 120, 80, 255), width=2)
        cx = camera.principal_point_x
        cy = camera.principal_point_y
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 0, 255))
        draw.text((6, 6), f"hfov={camera.horizontal_fov_deg:.1f} pitch={camera.pitch_down_deg:.1f} roll={camera.roll_cw_deg:.1f}", fill="white")
        image.convert("RGB").save(path)

    def _save_front_mask_overlay(self, base: Image.Image, front_mask: np.ndarray, path: Path) -> None:
        overlay = np.zeros((front_mask.shape[0], front_mask.shape[1], 4), dtype=np.uint8)
        overlay[front_mask] = (255, 64, 64, 170)
        image = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(overlay, mode="RGBA"))
        draw = ImageDraw.Draw(image)
        draw.text((6, 6), f"front mask pixels={int(front_mask.sum())}", fill="white")
        image.convert("RGB").save(path)

    def _save_point_cloud(self, camera: CameraEstimate, depth: np.ndarray, debug_dir: Path) -> None:
        points_camera = camera.point_grid_from_forward_depth(depth.astype(np.float32))
        points_world = camera.world_point_grid_from_forward_depth(depth.astype(np.float32))
        np.save(debug_dir / "points_camera.npy", points_camera)
        np.save(debug_dir / "points_world.npy", points_world)

    def _save_camera_space_plot(
        self,
        camera: CameraEstimate,
        rgb: np.ndarray,
        forward_distance: np.ndarray,
        valid_mask: np.ndarray,
        forward_distance_fit: ForwardDistanceFit,
        path: Path,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        height, width = forward_distance.shape
        stride = max(1, int(math.sqrt((height * width) / 1200)))
        sampled_forward_distance = forward_distance[::stride, ::stride].astype(np.float32)
        sampled_rgb = rgb[::stride, ::stride].astype(np.float32) / 255.0
        sampled_valid_mask = valid_mask[::stride, ::stride]
        points_camera = camera.point_grid_from_forward_depth(sampled_forward_distance)

        ys = np.arange(0, height, stride, dtype=np.float32)
        xs = np.arange(0, width, stride, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)
        plane_distance = float(forward_distance_fit.far_distance)
        image_plane = camera.point_grid_from_forward_depth(
            np.full_like(sampled_forward_distance, plane_distance, dtype=np.float32)
        )

        point_xyz = points_camera[sampled_valid_mask]
        point_rgb = sampled_rgb[sampled_valid_mask]
        plane_xyz = image_plane.reshape(-1, 3)
        plane_rgb = sampled_rgb.reshape(-1, 3)
        plane_corners = np.asarray(
            [
                camera.camera_point_from_pixel(0.0, 0.0, plane_distance).as_list(),
                camera.camera_point_from_pixel(width - 1.0, 0.0, plane_distance).as_list(),
                camera.camera_point_from_pixel(width - 1.0, height - 1.0, plane_distance).as_list(),
                camera.camera_point_from_pixel(0.0, height - 1.0, plane_distance).as_list(),
            ],
            dtype=np.float32,
        )
        plane_center = np.asarray(
            camera.camera_point_from_pixel(camera.principal_point_x, camera.principal_point_y, plane_distance).as_list(),
            dtype=np.float32,
        )

        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            point_xyz[:, 0],
            point_xyz[:, 2],
            point_xyz[:, 1],
            c=point_rgb,
            s=4,
            alpha=0.75,
            depthshade=False,
            label="reconstructed terrain points",
        )
        ax.scatter(
            plane_xyz[:, 0],
            plane_xyz[:, 2],
            plane_xyz[:, 1],
            c=plane_rgb,
            s=2,
            alpha=0.12,
            depthshade=False,
            label=f"input image plane (z={plane_distance:.1f})",
        )
        plane_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for a, b in plane_edges:
            ax.plot(
                [plane_corners[a, 0], plane_corners[b, 0]],
                [plane_corners[a, 2], plane_corners[b, 2]],
                [plane_corners[a, 1], plane_corners[b, 1]],
                color="#ffffff",
                linewidth=1.4,
                alpha=0.9,
            )
        ax.scatter(
            [plane_center[0]],
            [plane_center[2]],
            [plane_center[1]],
            c=["#ffd84d"],
            s=22,
            label="image plane center",
        )
        ax.scatter([0.0], [0.0], [0.0], c=[(1.0, 0.1, 0.1)], s=48, label="camera origin")

        ray_scale = plane_distance
        for ray in (
            camera.ray_center,
            camera.ray_top_left,
            camera.ray_top_right,
            camera.ray_bottom_left,
            camera.ray_bottom_right,
        ):
            ax.plot(
                [0.0, ray.x * ray_scale],
                [0.0, ray.z * ray_scale],
                [0.0, ray.y * ray_scale],
                color="#ffcc33",
                linewidth=1.2,
                alpha=0.8,
            )

        combined_x = np.concatenate([point_xyz[:, 0], plane_xyz[:, 0], np.array([0.0], dtype=np.float32)])
        combined_z = np.concatenate([point_xyz[:, 2], plane_xyz[:, 2], np.array([0.0], dtype=np.float32)])
        combined_y = np.concatenate([point_xyz[:, 1], plane_xyz[:, 1], np.array([0.0], dtype=np.float32)])
        min_x, max_x = float(combined_x.min()), float(combined_x.max())
        min_z, max_z = float(combined_z.min()), float(combined_z.max())
        min_y, max_y = float(combined_y.min()), float(combined_y.max())
        pad_x = max((max_x - min_x) * 0.08, 1.0)
        pad_z = max((max_z - min_z) * 0.08, 1.0)
        pad_y = max((max_y - min_y) * 0.08, 1.0)
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(max(0.0, min_z - pad_z), max_z + pad_z)
        ax.set_zlim(min_y - pad_y, max_y + pad_y)
        ax.set_box_aspect((max_x - min_x + 2 * pad_x, max_z - min_z + 2 * pad_z, max_y - min_y + 2 * pad_y))
        ax.set_xlabel("camera x (right)")
        ax.set_ylabel("camera z (forward)")
        ax.set_zlabel("camera y (up)")
        ax.set_title("Pinhole Camera Reconstruction in Camera Coordinates")
        ax.view_init(elev=22, azim=-62)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _save_camera_space_html(
        self,
        camera: CameraEstimate,
        rgb: np.ndarray,
        forward_distance: np.ndarray,
        valid_mask: np.ndarray,
        forward_distance_fit: ForwardDistanceFit,
        path: Path,
    ) -> None:
        height, width = forward_distance.shape
        stride = max(1, int(math.sqrt((height * width) / 1800)))
        sampled_forward_distance = forward_distance[::stride, ::stride].astype(np.float32)
        sampled_rgb = rgb[::stride, ::stride]
        sampled_valid_mask = valid_mask[::stride, ::stride]
        points_camera = camera.point_grid_from_forward_depth(sampled_forward_distance)
        plane_distance = float(forward_distance_fit.far_distance)
        image_plane = camera.point_grid_from_forward_depth(
            np.full_like(sampled_forward_distance, plane_distance, dtype=np.float32)
        )

        terrain_points: list[list[float | int]] = []
        plane_points: list[list[float | int]] = []
        for y in range(points_camera.shape[0]):
            for x in range(points_camera.shape[1]):
                px = points_camera[y, x]
                pr = sampled_rgb[y, x]
                plane = image_plane[y, x]
                plane_points.append(
                    [float(plane[0]), float(plane[2]), float(plane[1]), int(pr[0]), int(pr[1]), int(pr[2])]
                )
                if sampled_valid_mask[y, x]:
                    terrain_points.append(
                        [float(px[0]), float(px[2]), float(px[1]), int(pr[0]), int(pr[1]), int(pr[2])]
                    )

        ray_scale = plane_distance
        rays = []
        for ray in (
            camera.ray_center,
            camera.ray_top_left,
            camera.ray_top_right,
            camera.ray_bottom_left,
            camera.ray_bottom_right,
        ):
            rays.append([0.0, 0.0, 0.0, float(ray.x * ray_scale), float(ray.z * ray_scale), float(ray.y * ray_scale)])
        plane_corners = [
            camera.camera_point_from_pixel(0.0, 0.0, plane_distance).as_list(),
            camera.camera_point_from_pixel(width - 1.0, 0.0, plane_distance).as_list(),
            camera.camera_point_from_pixel(width - 1.0, height - 1.0, plane_distance).as_list(),
            camera.camera_point_from_pixel(0.0, height - 1.0, plane_distance).as_list(),
        ]
        plane_center = camera.camera_point_from_pixel(camera.principal_point_x, camera.principal_point_y, plane_distance).as_list()

        payload = {
            "terrain": terrain_points,
            "plane": plane_points,
            "rays": rays,
            "planeDistance": plane_distance,
            "planeCorners": [[float(p[0]), float(p[2]), float(p[1])] for p in plane_corners],
            "planeCenter": [float(plane_center[0]), float(plane_center[2]), float(plane_center[1])],
        }
        json_payload = json.dumps(payload, separators=(",", ":"))
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Camera Space Plot</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    #wrap {{ display: grid; grid-template-columns: 1fr 280px; height: 100vh; }}
    #view {{ width: 100%; height: 100%; display: block; background: #16181c; }}
    #side {{ padding: 16px; background: #1d2128; border-left: 1px solid #333; }}
    .meta {{ font-size: 13px; line-height: 1.5; opacity: 0.9; }}
    .kbd {{ display: inline-block; padding: 2px 6px; border: 1px solid #555; border-radius: 4px; background: #111; }}
  </style>
</head>
<body>
  <div id="wrap">
    <canvas id="view"></canvas>
    <div id="side">
      <h2 style="margin-top:0">Camera Space Viewer</h2>
      <div class="meta">Drag: rotate</div>
      <div class="meta">Wheel: zoom</div>
      <div class="meta"><span class="kbd">R</span>: reset view</div>
      <hr style="border-color:#333">
      <div class="meta">Blue/green: reconstructed terrain points</div>
      <div class="meta">Faint points: input image plane</div>
      <div class="meta">Yellow lines: camera rays</div>
      <div class="meta">Red point: camera origin</div>
      <hr style="border-color:#333">
      <div class="meta">plane distance: <span id="plane"></span></div>
      <div class="meta">terrain points: <span id="terrainCount"></span></div>
      <div class="meta">plane points: <span id="planeCount"></span></div>
    </div>
  </div>
  <script>
    const data = {json_payload};
    document.getElementById('plane').textContent = data.planeDistance.toFixed(2);
    document.getElementById('terrainCount').textContent = String(data.terrain.length);
    document.getElementById('planeCount').textContent = String(data.plane.length);
    const canvas = document.getElementById('view');
    const ctx = canvas.getContext('2d');
    let yaw = -0.85;
    let pitch = -0.28;
    let zoom = 5.0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function resize() {{
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function rotatePoint(p) {{
      let [x, z, y] = p;
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      const y2 = cp * y - sp * z1;
      const z2 = sp * y + cp * z1;
      return [x1, y2, z2];
    }}

    function project(p) {{
      const [x, y, z] = rotatePoint(p);
      const depth = z + zoom * 40.0;
      const scale = 700 / Math.max(depth, 1);
      return [x * scale + canvas.clientWidth / 2, -y * scale + canvas.clientHeight / 2, depth];
    }}

    function drawPointCloud(points, alphaMul, radius) {{
      const projected = [];
      for (const p of points) {{
        const q = project(p);
        projected.push([q[2], q[0], q[1], p[3], p[4], p[5]]);
      }}
      projected.sort((a, b) => b[0] - a[0]);
      for (const p of projected) {{
        ctx.fillStyle = `rgba(${{p[3]}},${{p[4]}},${{p[5]}},${{alphaMul}})`;
        ctx.beginPath();
        ctx.arc(p[1], p[2], radius, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    function drawRay(ray) {{
      const a = project([ray[0], ray[1], ray[2]]);
      const b = project([ray[3], ray[4], ray[5]]);
      ctx.strokeStyle = '#f4c542';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }}

    function drawOrigin() {{
      const o = project([0, 0, 0]);
      ctx.fillStyle = '#ff4040';
      ctx.beginPath();
      ctx.arc(o[0], o[1], 5, 0, Math.PI * 2);
      ctx.fill();
    }}

    function drawPlaneFrame() {{
      const corners = data.planeCorners.map(project);
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < corners.length; i += 1) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath();
      ctx.stroke();
      const c = project(data.planeCenter);
      ctx.fillStyle = '#ffd84d';
      ctx.beginPath();
      ctx.arc(c[0], c[1], 4, 0, Math.PI * 2);
      ctx.fill();
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      ctx.fillStyle = '#16181c';
      ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      drawPointCloud(data.plane, 0.10, 1.4);
      drawPlaneFrame();
      drawPointCloud(data.terrain, 0.78, 2.0);
      for (const ray of data.rays) drawRay(ray);
      drawOrigin();
    }}

    canvas.addEventListener('mousedown', (e) => {{
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    }});
    window.addEventListener('mousemove', (e) => {{
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.01;
      pitch += (e.clientY - lastY) * 0.01;
      pitch = Math.max(-1.45, Math.min(1.45, pitch));
      lastX = e.clientX;
      lastY = e.clientY;
      draw();
    }});
    window.addEventListener('mouseup', () => dragging = false);
    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      zoom *= e.deltaY > 0 ? 1.08 : 0.92;
      zoom = Math.max(1.2, Math.min(zoom, 20));
      draw();
    }}, {{ passive: false }});
    window.addEventListener('keydown', (e) => {{
      if (e.key.toLowerCase() === 'r') {{
        yaw = -0.85;
        pitch = -0.28;
        zoom = 5.0;
        draw();
      }}
    }});
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")

    def _valid_mask_from_label_map(self, label_map: list[list[tuple[str, tuple[int, int, int]]]]) -> np.ndarray:
        mask = np.zeros((len(label_map), len(label_map[0])), dtype=bool)
        for y, row in enumerate(label_map):
            for x, (label, _) in enumerate(row):
                if label != "sky":
                    mask[y, x] = True
        return mask

    def _save_scale_report(self, scale_fit: ScaleFit, path: Path) -> None:
        payload = {
            "scale_x": scale_fit.scale_x,
            "scale_y": scale_fit.scale_y,
            "scale_z": scale_fit.scale_z,
            "min_corner_world": list(scale_fit.min_corner_world),
            "span_world": list(scale_fit.span_world),
            "target_size": list(scale_fit.target_size),
            "anchor_min_x_world": scale_fit.anchor_min_x_world,
            "anchor_span_x_world": scale_fit.anchor_span_x_world,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_forward_distance_report(self, fit: ForwardDistanceFit, path: Path) -> None:
        payload = {
            "near_distance": fit.near_distance,
            "far_distance": fit.far_distance,
            "front_depth_threshold": fit.front_depth_threshold,
            "front_span_unit_x": fit.front_span_unit_x,
            "target_front_width": fit.target_front_width,
            "front_pixel_count": int(fit.front_mask.sum()),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _block_color(self, block_state: str) -> tuple[int, int, int]:
        if ":" in block_state:
            name = block_state.split(":", 1)[1]
        else:
            name = block_state
        if name.startswith("stained_glass:3"):
            return (117, 204, 255)
        if name.startswith("stained_glass:11"):
            return (60, 68, 170)
        if name.startswith("stained_glass:0"):
            return (240, 240, 255)
        if name.startswith("water"):
            return (63, 118, 228)
        if name.startswith("wool:13") or name.startswith("leaves") or name.startswith("leaves2"):
            return (85, 120, 50)
        if name.startswith("snow_block") or name.startswith("wool:0"):
            return (240, 240, 240)
        if name.startswith("sand") or name.startswith("sandstone"):
            return (218, 204, 156)
        if name.startswith("stone") or name.startswith("cobblestone"):
            return (128, 128, 128)
        if name.startswith("log") or name.startswith("oak_planks"):
            return (140, 100, 60)
        if name.startswith("wool:7") or name.startswith("obsidian"):
            return (60, 60, 70)
        if name.startswith("dirt") or name.startswith("grass") or name.startswith("stained_hardened_clay"):
            return (120, 90, 60)
        return (255, 0, 255)

    def _indexed_color(self, index: int) -> tuple[int, int, int]:
        palette = [
            (255, 99, 71),
            (135, 206, 235),
            (124, 252, 0),
            (255, 215, 0),
            (255, 105, 180),
            (147, 112, 219),
            (64, 224, 208),
            (244, 164, 96),
        ]
        return palette[index % len(palette)]
