from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_and_resize_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8)
