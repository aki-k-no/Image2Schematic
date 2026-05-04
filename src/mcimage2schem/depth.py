from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DepthConfig
from .model_utils import resolve_model_path


@dataclass(slots=True)
class DepthResult:
    normalized_depth: np.ndarray


class DepthAnythingV2Estimator:
    def __init__(self, config: DepthConfig, cache_root: Path) -> None:
        self.config = config
        self.cache_root = cache_root

    def estimate(self, image: Image.Image) -> DepthResult:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for depth estimation. "
                "Install requirements.txt before running this workflow."
            ) from exc

        model_path = resolve_model_path(self.config.model_id, self.cache_root)
        estimator = pipeline(
            task="depth-estimation",
            model=model_path,
            device=self._resolve_device(),
            local_files_only=True,
        )
        output = estimator(image)
        depth = np.asarray(output["depth"], dtype=np.float32)
        depth = depth - depth.min()
        max_value = float(depth.max())
        if max_value > 0:
            depth = depth / max_value
        if self.config.invert:
            depth = 1.0 - depth
        gamma = max(float(self.config.gamma), 1e-6)
        if gamma != 1.0:
            depth = np.power(np.clip(depth, 0.0, 1.0), gamma, dtype=np.float32)
        return DepthResult(normalized_depth=depth)

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
        return 0 if total_gb >= 2.5 else -1
