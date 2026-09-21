# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Compatibility imports for the former generic video client module.

New code should import the explicitly named Wan backend from
``novaplan.video_generation``.
"""

from .video_generation.wan import (
    VideoGenerationClient,
    VideoGenerationResult,
    WanVideoGenerationClient,
)


__all__ = [
    "VideoGenerationClient",
    "VideoGenerationResult",
    "WanVideoGenerationClient",
]
