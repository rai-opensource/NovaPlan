# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Public video-generation backends used by NovaPlan.

Wan is served by the remote ComfyUI queue, Veo is called through Vertex AI,
and the hybrid client runs both backends concurrently. Legacy class names are
kept as aliases for downstream integrations.
"""

from .hybrid import HybridVideoGenerationClient
from .wan import VideoGenerationResult, WanVideoGenerationClient

try:
    from .veo import VeoVideoGenerationClient
    VEO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional Vertex dependency
    VeoVideoGenerationClient = None
    VEO_IMPORT_ERROR = exc


def make_veo_video_client(*, mock: bool = False, seed: int = 0, debug_dir=None):
    """Construct Veo with one actionable error when its SDK is unavailable."""

    if VeoVideoGenerationClient is None:
        raise RuntimeError(
            "The Veo backend requires google-genai and Vertex AI credentials. "
            "Install the local-planning environment or use the Wan backend."
        ) from VEO_IMPORT_ERROR
    return VeoVideoGenerationClient(mock=mock, seed=seed, debug_dir=debug_dir)


# Compatibility aliases. Prefer the explicit backend names above in new code.
VideoGenerationClient = WanVideoGenerationClient
Veo3VideoAdapter = VeoVideoGenerationClient

__all__ = [
    "HybridVideoGenerationClient",
    "VideoGenerationClient",
    "VideoGenerationResult",
    "Veo3VideoAdapter",
    "VeoVideoGenerationClient",
    "WanVideoGenerationClient",
    "make_veo_video_client",
]
