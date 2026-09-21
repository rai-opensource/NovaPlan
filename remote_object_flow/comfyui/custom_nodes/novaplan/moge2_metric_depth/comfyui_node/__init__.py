# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Register NovaPlan's focused MoGe2 metric-depth nodes."""

from .cache_nodes import NODE_CLASS_MAPPINGS as CACHE_NODES
from .cache_nodes import NODE_DISPLAY_NAME_MAPPINGS as CACHE_DISPLAY_NAMES
from .node import NODE_CLASS_MAPPINGS as MOGE2_NODES
from .node import NODE_DISPLAY_NAME_MAPPINGS as MOGE2_DISPLAY_NAMES


NODE_CLASS_MAPPINGS = {**CACHE_NODES, **MOGE2_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**CACHE_DISPLAY_NAMES, **MOGE2_DISPLAY_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
