# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Import and smoke-test the deployed TAPIP3D adapter."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_adapter(adapter_path: Path, tapip3d_root: Path) -> ModuleType:
    """Load the deployed TAPIP3D adapter module."""
    adapter_path = adapter_path.resolve()
    tapip3d_root = tapip3d_root.resolve()
    os.environ["NOVAPLAN_TAPIP3D_DIR"] = str(tapip3d_root)

    spec = importlib.util.spec_from_file_location(
        "_novaplan_tapip3d_setup_check",
        adapter_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TAPIP3D adapter: {adapter_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _smoke_test_knn(module: ModuleType, pointops: ModuleType, device: str) -> None:
    torch = module.torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("pointops2 CUDA is installed, but CUDA is not available")

    xyz = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    queries = torch.tensor(
        [[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    indices, distances = pointops.knnquery(
        2,
        xyz,
        queries,
        torch.tensor([3], dtype=torch.int32, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
    )
    if indices.cpu().tolist() != [[0, 1], [2, 1]]:
        raise RuntimeError(f"TAPIP3D {device} KNN returned unexpected indices: {indices}")
    if tuple(distances.shape) != (2, 2) or not bool(
        torch.isfinite(distances).all().item()
    ):
        raise RuntimeError(f"TAPIP3D {device} KNN returned invalid distances")


def verify_adapter(module: ModuleType, require_pointops_cuda: bool = False) -> str:
    """Verify the deployed TAPIP3D adapter contract."""
    for name in ("load_model", "inference", "get_grid_queries", "resize_depth_bilinear"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"TAPIP3D adapter is missing callable {name}")

    pointops = getattr(module, "_POINTOPS", None)
    if pointops is None or not callable(getattr(pointops, "knnquery", None)):
        raise RuntimeError("TAPIP3D adapter did not load a KNN backend")

    if bool(getattr(module, "_HAS_POINTOPS_CUDA", False)):
        _smoke_test_knn(module, pointops, "cuda")
        return "pointops2 CUDA"

    if require_pointops_cuda:
        raise RuntimeError("pointops2 CUDA was required, but only the fallback loaded")

    fallback = getattr(module, "_knnquery_torch", None)
    if getattr(pointops, "knnquery", None) is not fallback:
        raise RuntimeError("TAPIP3D PyTorch KNN fallback was not installed")
    _smoke_test_knn(module, pointops, "cpu")
    return "PyTorch KNN fallback"


def main() -> int:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--tapip3d-root", required=True, type=Path)
    parser.add_argument("--require-pointops-cuda", action="store_true")
    args = parser.parse_args()

    module = load_adapter(args.adapter, args.tapip3d_root)
    backend = verify_adapter(
        module,
        require_pointops_cuda=args.require_pointops_cuda,
    )
    print(f"TAPIP3D adapter ready ({backend}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
