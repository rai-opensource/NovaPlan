# Video Generation

This package is the workstation-side API for every NovaPlan video backend.

## Backends

- `wan.py`: `WanVideoGenerationClient` submits Wan 2.2 jobs to the remote
  ComfyUI video server. A `full` request returns both video and inline
  SAM3/CoTracker3 evidence.
- `veo.py`: `VeoVideoGenerationClient` calls Vertex AI directly, resamples
  native output to the configured rollout length, and returns RGB frames.
- `hybrid.py`: `HybridVideoGenerationClient` runs Wan and Veo concurrently,
  sends completed Veo clips through the remote server's `flow_only` path, and
  combines both backends into one ranked candidate pool.

Use the public package imports:

```python
from novaplan.video_generation import (
    HybridVideoGenerationClient,
    VeoVideoGenerationClient,
    WanVideoGenerationClient,
)
```

`novaplan.video_generation_client` and `novaplan.veo_client` remain available
as compatibility shims. Use the public package imports above for new integrations.

Remote Wan setup and its single live installation check are documented in
[remote_video_generation/README.md](../../remote_video_generation/README.md).
