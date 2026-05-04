from __future__ import annotations

import colorsys
from dataclasses import dataclass

import numpy as np

from .config import ClassificationConfig
from .segment import SegmentMask


@dataclass(slots=True)
class LabeledRegion:
    mask: np.ndarray
    label: str
    mean_rgb: tuple[int, int, int]
    mean_depth: float
    centroid_y: float
    centroid_x: float


class RegionClassifier:
    def __init__(self, config: ClassificationConfig) -> None:
        self.config = config

    def classify(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        masks: list[SegmentMask],
    ) -> list[LabeledRegion]:
        regions: list[LabeledRegion] = []
        for mask in masks:
            pixels = rgb_image[mask.segmentation]
            if pixels.size == 0:
                continue
            ys, xs = np.where(mask.segmentation)
            centroid_y = float(ys.mean() / max(rgb_image.shape[0] - 1, 1))
            centroid_x = float(xs.mean() / max(rgb_image.shape[1] - 1, 1))
            mean_rgb = tuple(int(v) for v in pixels.mean(axis=0))
            mean_depth = float(depth_map[mask.segmentation].mean())
            label = self._infer_label(mean_rgb, mean_depth, centroid_y, centroid_x)
            regions.append(
                LabeledRegion(
                    mask=mask.segmentation,
                    label=label,
                    mean_rgb=mean_rgb,
                    mean_depth=mean_depth,
                    centroid_y=centroid_y,
                    centroid_x=centroid_x,
                )
            )
        return regions

    def _infer_label(
        self,
        mean_rgb: tuple[int, int, int],
        mean_depth: float,
        centroid_y: float,
        centroid_x: float,
    ) -> str:
        r, g, b = [channel / 255.0 for channel in mean_rgb]
        hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
        hue_deg = hue * 360.0
        brightness = (r + g + b) / 3.0
        blue_dominance = b - max(r, g)
        green_dominance = g - max(r, b)
        red_dominance = r - max(g, b)

        if (
            centroid_y <= self.config.sky_top_portion
            and val > self.config.sky_brightness_threshold
            and (b >= r or sat < 0.22)
        ):
            if sat < 0.12 and val > 0.82:
                return "cloud"
            return "sky"
        if (
            b > max(r, g)
            and centroid_y >= self.config.water_bottom_portion
            and sat > 0.2
        ):
            if val < 0.42 or mean_depth > self.config.water_depth_threshold:
                return "water_deep"
            return "water_shallow"
        if self.config.vegetation_hue_min <= hue_deg <= self.config.vegetation_hue_max and g >= r:
            if val < 0.34:
                return "foliage_dark"
            if val > 0.58 or green_dominance > 0.12:
                return "foliage_light"
            return "foliage"
        if val > 0.85 and sat < 0.18:
            return "snow"
        if r > 0.55 and g > 0.5 and b < 0.45:
            return "sand"
        if r > 0.42 and g > 0.34 and b < 0.28 and sat < 0.32 and centroid_y > 0.45:
            return "path"
        if abs(r - g) < 0.08 and abs(g - b) < 0.08:
            if val < 0.34:
                return "rock_dark"
            if mean_depth > 0.62 or centroid_y < 0.55:
                return "cliff"
            return "rock"
        if centroid_y > 0.58 and green_dominance > 0.06:
            if brightness > 0.48:
                return "ground_grass"
            return "ground"
        if centroid_y > 0.52 and red_dominance > 0.04 and brightness < 0.52:
            return "ground_dirt"
        if abs(r - g) < 0.08 and abs(g - b) < 0.08:
            return "rock"
        if r > g and r > b and sat > 0.35:
            return "wood"
        if val < self.config.dark_shadow_threshold:
            return "shadow"
        if centroid_y <= self.config.sky_top_portion * 0.9 and b > g and b > r:
            return "sky"
        if centroid_y > 0.55:
            return "ground_grass" if green_dominance > 0.04 else "ground_dirt"
        return "ground"
