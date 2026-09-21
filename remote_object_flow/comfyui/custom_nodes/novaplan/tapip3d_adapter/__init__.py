# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Register the vanilla TAPIP3D tracker only."""

import os

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

if os.environ.get("NOVAPLAN_VERBOSE_LOGS", "0").lower() in {"1", "true", "yes", "on"}:
    print(f"[TAPIP3D] Loaded nodes: {list(NODE_CLASS_MAPPINGS.keys())}", flush=True)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
