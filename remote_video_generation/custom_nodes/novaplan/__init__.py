# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""NovaPlan ComfyUI custom-node bundle."""

from importlib.util import module_from_spec, spec_from_file_location
import sys
import types
from pathlib import Path

import torch


class IntrinsicsNode:
    """Construct camera-intrinsics matrices for ComfyUI workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "fx": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "fy": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                "cx": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "cy": ("FLOAT", {"default": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INTRINSICS",)
    RETURN_NAMES = ("intrinsics",)
    FUNCTION = "build"
    CATEGORY = "NovaPlan/Camera"

    def build(self, fx, fy, cx, cy):
        """Construct a camera-intrinsics matrix from scalar parameters."""
        intrinsics = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        return (intrinsics,)


NODE_CLASS_MAPPINGS = {"IntrinsicsNode": IntrinsicsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"IntrinsicsNode": "Camera Intrinsics"}


def _load_node_package(name: str, init_path: Path) -> None:
    if not init_path.exists():
        print(f"[NovaPlan] Warning: missing custom-node package: {init_path}")
        return

    parent_name = "novaplan_custom_nodes"
    if parent_name not in sys.modules:
        parent = types.ModuleType(parent_name)
        parent.__path__ = [str(_ROOT)]
        sys.modules[parent_name] = parent

    spec = spec_from_file_location(
        f"{parent_name}.{name}",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        print(f"[NovaPlan] Warning: could not load custom-node package: {init_path}")
        return

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))


_ROOT = Path(__file__).resolve().parent
_PACKAGE_CANDIDATES = {
    "moge2_metric_depth": _ROOT / "moge2_metric_depth" / "comfyui_node" / "__init__.py",
    "tapip3d": _ROOT / "TAPIP3D" / "comfyui_node" / "__init__.py",
    "sam3": _ROOT / "sam3" / "comfyui_node" / "__init__.py",
    "cotracker3": _ROOT / "cotracker3" / "__init__.py",
}

for _name, _path in _PACKAGE_CANDIDATES.items():
    if not _path.exists():
        continue
    try:
        _load_node_package(_name, _path)
    except Exception as exc:
        print(f"[NovaPlan] Warning: failed to load {_name}: {exc}")

print(f"[NovaPlan] Loaded {len(NODE_CLASS_MAPPINGS)} custom nodes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
