from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import WorkflowConfig
from .pipeline import ImageToSchematicPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a landscape image into a Minecraft 1.8.9 .schematic file.")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input image.")
    parser.add_argument("--output", required=True, type=Path, help="Path to the output .schematic file.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the workflow JSON config.")
    parser.add_argument("--schem-width", type=int, help="Override output schematic width.")
    parser.add_argument("--schem-height", type=int, help="Override output schematic height.")
    parser.add_argument("--schem-length", type=int, help="Override output schematic length.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    hf_cache = project_root / ".hf-cache"
    hf_cache.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = WorkflowConfig.from_path(args.config)
    if args.schem_width is not None:
        config.build.target_width = args.schem_width
    if args.schem_height is not None:
        config.build.target_height = args.schem_height
    if args.schem_length is not None:
        config.build.target_length = args.schem_length
    pipeline = ImageToSchematicPipeline(config, hf_cache)
    artifacts = pipeline.run(args.input, args.output)
    print(
        f"Generated {args.output} "
        f"({artifacts.schematic.width}x{artifacts.schematic.height}x{artifacts.schematic.length}, "
        f"{len(artifacts.regions)} labeled regions, "
        f"debug: {artifacts.debug_dir})"
    )


if __name__ == "__main__":
    main()
