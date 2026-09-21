# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Compatibility imports for the former standalone Veo client module.

New code should import ``VeoVideoGenerationClient`` from
``novaplan.video_generation``.
"""

from .video_generation.veo import Veo3VideoAdapter, VeoVideoGenerationClient


__all__ = ["Veo3VideoAdapter", "VeoVideoGenerationClient"]
