# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""NovaPlan ComfyUI custom-node bundle."""

import os
import sys
import traceback
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .intrinsics_node import NODE_CLASS_MAPPINGS as INTRINSICS_NODES
from .intrinsics_node import NODE_DISPLAY_NAME_MAPPINGS as INTRINSICS_DISPLAY

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(INTRINSICS_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(INTRINSICS_DISPLAY)


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
    # The adapter isolates upstream TAPIP3D's generic package names from
    # ComfyUI's own top-level utils package.
    "tapip3d": _ROOT / "tapip3d_adapter" / "__init__.py",
    "moge2_metric_depth": _ROOT / "moge2_metric_depth" / "comfyui_node" / "__init__.py",
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
        if _name == "tapip3d" or os.environ.get("NOVAPLAN_VERBOSE_LOGS", "0").lower() in {"1", "true", "yes", "on"}:
            traceback.print_exc()

print(f"[NovaPlan] Loaded {len(NODE_CLASS_MAPPINGS)} custom nodes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
