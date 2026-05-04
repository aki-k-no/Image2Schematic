from __future__ import annotations

from pathlib import Path


def resolve_model_path(model_id: str, cache_root: Path) -> str:
    model_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not model_dir.exists():
        return model_id
    snapshots = sorted(path for path in model_dir.iterdir() if path.is_dir())
    if not snapshots:
        return model_id
    return str(snapshots[-1])
