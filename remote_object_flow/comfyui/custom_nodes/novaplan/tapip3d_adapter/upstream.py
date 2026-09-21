# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Bridge NovaPlan's ComfyUI node to an unmodified TAPIP3D checkout."""

from __future__ import annotations

import importlib
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch


_ADAPTER_DIR = Path(__file__).resolve().parent
_NOVAPLAN_NODE_DIR = _ADAPTER_DIR.parent
_ROOT_FILE = _ADAPTER_DIR / "tapip3d_root.txt"


def _resolve_tapip3d_root() -> Path:
    configured = os.environ.get("NOVAPLAN_TAPIP3D_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif _ROOT_FILE.is_file():
        root = Path(_ROOT_FILE.read_text().strip()).expanduser()
    else:
        root = _NOVAPLAN_NODE_DIR / ".external" / "TAPIP3D"

    root = root.resolve()
    required = (root / "models", root / "utils" / "inference_utils.py", root / "LICENSE")
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Official TAPIP3D checkout is missing or incomplete at "
            f"{root}. Run: pixi run -e object-flow-host setup-object-flow-host"
        )
    return root


def _module_belongs_to(module: object, root: Path) -> bool:
    source = getattr(module, "__file__", None)
    if not source:
        return False
    try:
        Path(source).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


_UPSTREAM_TOP_LEVEL_PACKAGES = ("models", "utils", "datasets", "training", "third_party")
_UPSTREAM_NAMESPACE = "novaplan_tapip3d_upstream"


def _is_upstream_module_name(name: str) -> bool:
    return any(name == package or name.startswith(f"{package}.") for package in _UPSTREAM_TOP_LEVEL_PACKAGES)


@contextmanager
def _isolated_upstream_imports(root: Path):
    """Temporarily reserve TAPIP3D's generic package names during import.

    ComfyUI itself owns a top-level ``utils`` package. TAPIP3D also imports
    ``utils`` (plus ``models`` and friends) as top-level packages. Load the
    pinned checkout while those names are isolated, retain aliases for its
    loaded modules, and then restore ComfyUI's modules unchanged.
    """

    root_str = str(root)
    original_path = list(sys.path)
    preserved = {
        name: module for name, module in tuple(sys.modules.items()) if _is_upstream_module_name(name)
    }
    for name in preserved:
        sys.modules.pop(name, None)

    while root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    try:
        yield
    finally:
        loaded = {
            name: module
            for name, module in tuple(sys.modules.items())
            if _is_upstream_module_name(name) and _module_belongs_to(module, root)
        }
        for name in tuple(sys.modules):
            if _is_upstream_module_name(name):
                sys.modules.pop(name, None)

        namespace = sys.modules.get(_UPSTREAM_NAMESPACE)
        if namespace is None:
            namespace = types.ModuleType(_UPSTREAM_NAMESPACE)
            namespace.__path__ = [root_str]
            sys.modules[_UPSTREAM_NAMESPACE] = namespace
        for name, module in loaded.items():
            sys.modules[f"{_UPSTREAM_NAMESPACE}.{name}"] = module

        sys.modules.update(preserved)
        sys.path[:] = original_path


@contextmanager
def _prepared_pointops_import():
    placeholders = {}
    preserved = {}
    try:
        pointops_cuda = importlib.import_module("pointops2_cuda")
        has_pointops_cuda = callable(getattr(pointops_cuda, "knnquery_cuda", None))
    except (ImportError, OSError):
        has_pointops_cuda = False

    if not has_pointops_cuda:
        # TAPIP3D imports the extension eagerly even though inference only uses
        # knnquery. A placeholder lets the vendored Python module load before
        # NovaPlan installs its slower torch-only knnquery adapter below.
        if "pointops2_cuda" in sys.modules:
            preserved["pointops2_cuda"] = sys.modules["pointops2_cuda"]
        placeholder = types.ModuleType("pointops2_cuda")
        placeholder.__all__ = ()
        sys.modules["pointops2_cuda"] = placeholder
        placeholders["pointops2_cuda"] = placeholder

    try:
        importlib.import_module("pointops2")
    except (ImportError, OSError):
        # The pinned checkout's functions/__init__.py also imports the installed
        # pointops2 Python package. A failed wheel build leaves that package
        # absent, so temporarily provide its otherwise-unused import surface.
        if "pointops2" in sys.modules:
            preserved["pointops2"] = sys.modules["pointops2"]
        placeholder = types.ModuleType("pointops2")
        placeholder.__all__ = ()
        sys.modules["pointops2"] = placeholder
        placeholders["pointops2"] = placeholder

    try:
        yield has_pointops_cuda
    finally:
        for name, placeholder in placeholders.items():
            if sys.modules.get(name) is placeholder:
                if name in preserved:
                    sys.modules[name] = preserved[name]
                else:
                    sys.modules.pop(name, None)


def _knnquery_torch(nsample, xyz, new_xyz, offset, new_offset):
    """Torch-only replacement for TAPIP3D's inference-time KNN query."""
    if new_xyz is None:
        new_xyz = xyz

    device = xyz.device
    indices = []
    distances = []
    context_start = 0
    query_start = 0

    for context_end, query_end in zip(offset.tolist(), new_offset.tolist()):
        context = xyz[context_start:context_end]
        queries = new_xyz[query_start:query_end]
        if queries.numel() == 0:
            context_start = context_end
            query_start = query_end
            continue
        if context.numel() == 0:
            raise ValueError("TAPIP3D knnquery received an empty context segment")

        k = min(int(nsample), int(context.shape[0]))
        distance = torch.cdist(queries.float(), context.float())
        knn_distance, knn_index = torch.topk(
            distance, k=k, dim=1, largest=False, sorted=True
        )
        knn_index = knn_index + context_start

        if k < int(nsample):
            pad_count = int(nsample) - k
            knn_index = torch.cat(
                [knn_index, knn_index[:, -1:].expand(-1, pad_count)], dim=1
            )
            knn_distance = torch.cat(
                [knn_distance, knn_distance[:, -1:].expand(-1, pad_count)], dim=1
            )

        indices.append(knn_index.to(device=device, dtype=torch.long))
        distances.append(knn_distance.to(device=device, dtype=xyz.dtype))
        context_start = context_end
        query_start = query_end

    if not indices:
        return (
            torch.empty((0, int(nsample)), device=device, dtype=torch.long),
            torch.empty((0, int(nsample)), device=device, dtype=xyz.dtype),
        )
    return torch.cat(indices, dim=0), torch.cat(distances, dim=0)


TAPIP3D_ROOT = _resolve_tapip3d_root()
with _isolated_upstream_imports(TAPIP3D_ROOT):
    with _prepared_pointops_import() as _HAS_POINTOPS_CUDA:
        _INFERENCE_UTILS = importlib.import_module("utils.inference_utils")

        if not _module_belongs_to(_INFERENCE_UTILS, TAPIP3D_ROOT):
            raise RuntimeError(
                f"Resolved TAPIP3D inference utilities outside {TAPIP3D_ROOT}: "
                f"{getattr(_INFERENCE_UTILS, '__file__', 'unknown')}"
            )

        _POINTOPS = importlib.import_module("third_party.pointops2.functions.pointops")
        if not _HAS_POINTOPS_CUDA:
            _POINTOPS.knnquery = _knnquery_torch

load_model = _INFERENCE_UTILS.load_model
inference = _INFERENCE_UTILS.inference
get_grid_queries = _INFERENCE_UTILS.get_grid_queries
resize_depth_bilinear = _INFERENCE_UTILS.resize_depth_bilinear


__all__ = [
    "TAPIP3D_ROOT",
    "get_grid_queries",
    "inference",
    "load_model",
    "resize_depth_bilinear",
]
