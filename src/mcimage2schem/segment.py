from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SegmentationConfig
from .model_utils import resolve_model_path


@dataclass(slots=True)
class SegmentMask:
    segmentation: np.ndarray
    area: int
    score: float


class SamSegmenter:
    def __init__(self, config: SegmentationConfig, cache_root: Path) -> None:
        self.config = config
        self.cache_root = cache_root

    def generate_masks(self, image: Image.Image) -> list[SegmentMask]:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for SAM segmentation. "
                "Install requirements.txt before running this workflow."
            ) from exc

        model_path = resolve_model_path(self.config.model_id, self.cache_root)
        generator = pipeline(
            task="mask-generation",
            model=model_path,
            device=self._resolve_device(),
            local_files_only=True,
        )
        outputs = generator(
            image,
            points_per_batch=self.config.points_per_batch,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
        )
        masks: list[SegmentMask] = []
        scores = outputs.get("scores", [])
        for index, mask in enumerate(outputs["masks"]):
            mask_array = np.asarray(mask, dtype=bool)
            area = int(mask_array.sum())
            if area < self.config.min_mask_region_area:
                continue
            score = scores[index]
            if hasattr(score, "item"):
                score = score.item()
            masks.append(
                SegmentMask(
                    segmentation=mask_array,
                    area=area,
                    score=float(score),
                )
            )
        masks = sorted(masks, key=lambda item: (-item.score, item.area))
        return masks[: self.config.max_masks]

    def _resolve_device(self) -> int:
        if self.config.device == "cpu":
            return -1
        if self.config.device == "cuda":
            return 0
        try:
            import torch
        except ImportError:
            return -1
        if not torch.cuda.is_available():
            return -1
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0 if total_gb >= 6.0 else -1
