# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Load scene and object point clouds used by the local Viser tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np


def _load_object_file(path: Path) -> Any:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as loaded:
            return {key: loaded[key] for key in loaded.files}
    if path.suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
            return loaded.item()
        return loaded
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported point-cloud file type: {path}")


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _unwrap_batch_list(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        first = value[0]
        if first and isinstance(first[0], (list, tuple)):
            return first
    return value


def _coerce_points(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    points = np.asarray(_unwrap_batch_list(value), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    return points[np.isfinite(points).all(axis=1)]


def _coerce_colors(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    colors = np.asarray(_unwrap_batch_list(value))
    if colors.ndim != 2 or colors.shape[1] != 3:
        return None
    if colors.dtype.kind == "f" and colors.max(initial=0) <= 1.0:
        colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def load_scene_point_cloud(
    path: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return scene points/colors and optional object points/colors."""

    raw = _load_object_file(Path(path))
    if isinstance(raw, dict) and "scene_info" in raw:
        scene_info = raw.get("scene_info", {})
        object_info = raw.get("object_info", {})
        return (
            _coerce_points(_first_present(scene_info, ("pc", "points", "pc_color"))),
            _coerce_colors(_first_present(scene_info, ("colors", "img_color"))),
            _coerce_points(_first_present(object_info, ("pc", "points"))),
            _coerce_colors(_first_present(object_info, ("pc_color", "colors"))),
        )

    if isinstance(raw, dict):
        return (
            _coerce_points(_first_present(raw, ("scene_points", "points", "pc", "xyz"))),
            _coerce_colors(_first_present(raw, ("scene_colors", "colors", "rgb"))),
            _coerce_points(_first_present(raw, ("object_points", "object_pc"))),
            _coerce_colors(_first_present(raw, ("object_colors", "object_rgb"))),
        )

    return _coerce_points(raw), None, None, None
