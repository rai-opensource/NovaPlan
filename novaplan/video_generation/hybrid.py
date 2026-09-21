# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Parallel Wan/Veo candidate generation and selection-flow preparation."""

from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .wan import VideoGenerationResult, WanVideoGenerationClient
    from ..flow_extraction_client import FlowExtractionClient
except ImportError:  # pragma: no cover - direct-script compatibility
    from video_generation.wan import VideoGenerationResult, WanVideoGenerationClient
    from flow_extraction_client import FlowExtractionClient


class HybridVideoGenerationClient:
    """Generate Wan and Veo rollouts concurrently as one candidate pool."""

    handles_flow = True

    def __init__(
        self,
        wan_client: WanVideoGenerationClient,
        veo_client: Any,
        flow_client: Optional[FlowExtractionClient] = None,
    ):
        self.wan_client = wan_client
        self.veo_client = veo_client
        self.flow_client = flow_client
        self._debug_dir: Optional[Path] = None

    @property
    def debug_dir(self) -> Optional[Path]:
        """Get or set the directory used for generation diagnostics."""
        return self._debug_dir

    @debug_dir.setter
    def debug_dir(self, value: Optional[Path]) -> None:
        """Get or set the directory used for generation diagnostics."""
        self._debug_dir = value
        for client in (self.wan_client, self.veo_client):
            if hasattr(client, "debug_dir"):
                client.debug_dir = value

    def generate_rollouts(
        self,
        start_frame: np.ndarray,
        action_text: str,
        num_samples: int = 8,
        num_frames: int = 41,
        fps: int = 16,
        enable_flow: bool = False,
        mask_prompt: Optional[str] = None,
        wan_action_text: Optional[str] = None,
        veo_action_text: Optional[str] = None,
        last_frame: Optional[np.ndarray] = None,
        negative_prompt: Optional[str] = None,
        wan_negative_prompt: Optional[str] = None,
        veo_negative_prompt: Optional[str] = None,
        **wan_kwargs: Any,
    ) -> VideoGenerationResult:
        """Generate and return video rollout candidates."""
        if enable_flow:
            if self.flow_client is None:
                raise RuntimeError(
                    "Hybrid WAN/Veo rollout selection requires a FlowExtractionClient "
                    "for Veo CoTracker3 flow_only evidence."
                )
            # Check before either backend starts so an unavailable local tunnel
            # cannot spend Veo quota on candidates that cannot be ranked.
            self.flow_client.ensure_selection_flow_contract()

        print(
            f"[HybridVideoGenerationClient] Generating {num_samples} WAN + "
            f"{num_samples} Veo candidates"
        )
        wan_prompt = wan_action_text or action_text
        veo_prompt = veo_action_text or action_text

        def _run_wan() -> VideoGenerationResult:
            result = self.wan_client.generate_rollouts(
                start_frame=start_frame,
                action_text=wan_prompt,
                num_samples=num_samples,
                num_frames=num_frames,
                fps=fps,
                enable_flow=enable_flow,
                mask_prompt=mask_prompt,
                last_frame=last_frame,
                negative_prompt=wan_negative_prompt or negative_prompt,
                **wan_kwargs,
            )
            if isinstance(result, VideoGenerationResult):
                result.video_sources = result.video_sources or ["wan22"] * len(result.videos)
                return result
            videos = result if isinstance(result, list) else list(result)
            return VideoGenerationResult(videos=videos, video_sources=["wan22"] * len(videos))

        def _run_veo() -> VideoGenerationResult:
            flow_futures: Dict[Any, int] = {}
            flow_executor = None
            if enable_flow and self.flow_client is not None:
                flow_workers = max(
                    1,
                    min(num_samples, int(os.getenv("VEO_FLOW_MAX_PARALLEL", "8"))),
                )
                flow_executor = concurrent.futures.ThreadPoolExecutor(max_workers=flow_workers)
                print(
                    "[HybridVideoGenerationClient] Veo flow extraction will be pipelined "
                    f"with max_parallel={flow_workers} via remote flow_only mode."
                )

            def _queue_flow(idx: int, video: np.ndarray) -> None:
                if flow_executor is None:
                    return
                print(
                    "[HybridVideoGenerationClient] Queueing flow_only extraction "
                    f"for Veo candidate {idx + 1} as soon as its video finished."
                )
                future = flow_executor.submit(
                    self.flow_client.extract_flow,
                    videos=[video],
                    mask_prompt=mask_prompt or "object",
                    fps=fps,
                )
                flow_futures[future] = idx

            videos = self.veo_client.generate_rollouts(
                start_frame=start_frame,
                action_text=veo_prompt,
                num_samples=num_samples,
                num_frames=num_frames,
                fps=fps,
                last_frame=last_frame,
                negative_prompt=veo_negative_prompt or negative_prompt,
                seed=wan_kwargs.get("seed"),
                on_video=_queue_flow,
            )
            sample_indices = list(
                getattr(videos, "sample_indices", range(len(videos)))
            )
            if len(sample_indices) != len(videos):
                raise RuntimeError(
                    "Veo returned inconsistent sample-index metadata for its "
                    "successful videos."
                )
            sample_errors = dict(getattr(videos, "sample_errors", {}) or {})
            veo_errors = {}
            if sample_errors:
                details = "; ".join(
                    f"sample {int(index) + 1}: {error}"
                    for index, error in sorted(sample_errors.items())
                )
                veo_errors["veo3"] = (
                    f"partial batch ({len(videos)}/{num_samples} successful): {details}"
                )
            result = VideoGenerationResult(
                videos=list(videos),
                video_sources=["veo3"] * len(videos),
                backend_errors=veo_errors,
            )
            if flow_executor is not None:
                try:
                    flow_images_by_sample: Dict[int, Optional[np.ndarray]] = {}
                    flow_bytes_by_sample: Dict[int, Optional[bytes]] = {}
                    coords_3d_by_sample: Dict[int, Optional[np.ndarray]] = {}
                    visibilities_by_sample: Dict[int, Optional[np.ndarray]] = {}
                    flow_choices_by_sample: Dict[int, Any] = {}
                    for future in concurrent.futures.as_completed(flow_futures):
                        idx = flow_futures[future]
                        try:
                            flow_result = future.result()
                        except Exception as exc:
                            print(
                                "[HybridVideoGenerationClient] Veo flow_only failed "
                                f"for candidate {idx + 1}: {exc}"
                            )
                            continue
                        if flow_result.flow_images:
                            flow_images_by_sample[idx] = flow_result.flow_images[0]
                        if flow_result.flow_bytes:
                            flow_bytes_by_sample[idx] = flow_result.flow_bytes[0]
                        if flow_result.coords_3d:
                            coords_3d_by_sample[idx] = flow_result.coords_3d[0]
                        if flow_result.visibilities:
                            visibilities_by_sample[idx] = flow_result.visibilities[0]
                        if flow_result.flow_choices:
                            flow_choices_by_sample[idx] = flow_result.flow_choices[0]
                    result.flow_images = [
                        flow_images_by_sample.get(index) for index in sample_indices
                    ]
                    result.flow_bytes = [
                        flow_bytes_by_sample.get(index) for index in sample_indices
                    ]
                    result.coords_3d = [
                        coords_3d_by_sample.get(index) for index in sample_indices
                    ]
                    result.visibilities = [
                        visibilities_by_sample.get(index) for index in sample_indices
                    ]
                    result.flow_choices = [
                        flow_choices_by_sample.get(index) for index in sample_indices
                    ]
                finally:
                    flow_executor.shutdown(wait=True)
            return result

        outputs: Dict[str, VideoGenerationResult] = {}
        errors: Dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_run_wan): "wan22",
                executor.submit(_run_veo): "veo3",
            }
            for future in concurrent.futures.as_completed(futures):
                backend = futures[future]
                try:
                    outputs[backend] = future.result()
                except Exception as exc:
                    errors[backend] = str(exc)
                    print(f"[HybridVideoGenerationClient] {backend} generation failed: {exc}")

        combined = VideoGenerationResult(
            requested_backends=["wan22", "veo3"],
            requested_samples_per_backend=int(num_samples),
            backend_errors=dict(errors),
        )

        def _aligned(values: List[Any], count: int) -> List[Any]:
            return list(values[:count]) + [None] * max(0, count - len(values))

        for backend in ("wan22", "veo3"):
            result = outputs.get(backend)
            if result is None:
                continue
            for error_backend, error in result.backend_errors.items():
                combined.backend_errors.setdefault(error_backend, error)
            count = len(result.videos)
            combined.videos.extend(result.videos)
            combined.flow_images.extend(_aligned(result.flow_images, count))
            combined.video_bytes.extend(result.video_bytes)
            combined.flow_bytes.extend(_aligned(result.flow_bytes, count))
            combined.coords_3d.extend(_aligned(result.coords_3d, count))
            combined.visibilities.extend(_aligned(result.visibilities, count))
            combined.flow_choices.extend(_aligned(result.flow_choices, count))
            if result.video_sources:
                combined.video_sources.extend(
                    result.video_sources[:count]
                    + [backend] * max(0, count - len(result.video_sources))
                )
            else:
                combined.video_sources.extend([backend] * count)

        if errors and not combined.videos:
            raise RuntimeError(f"All video backends failed: {errors}")
        print(
            "[HybridVideoGenerationClient] Combined candidates: "
            f"{len(combined.videos)} videos "
            f"({combined.video_sources.count('wan22')} WAN, "
            f"{combined.video_sources.count('veo3')} Veo)"
        )
        return combined


__all__ = ["HybridVideoGenerationClient"]
