#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
NovaPlan Client for extracting optical flow from videos.

This client connects to the ComfyUI job server and submits flow-only jobs
for extracting optical flow using CoTracker/SAM3. It can be used with any
video generation backend (WAN22, Veo3, etc.).

Usage:
    from flow_extraction_client import FlowExtractionClient
    
    client = FlowExtractionClient(server_base="http://localhost:7000")
    
    # Extract flow from videos (numpy arrays or MP4 bytes)
    flow_images = client.extract_flow(
        videos=[video1, video2],
        mask_prompt="black handle on drawer"
    )
"""

from __future__ import annotations
import os
import io
import time
import base64
import tempfile
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Tuple, Union

import requests
import numpy as np
from PIL import Image

try:
    from .flow_switching import (
        DEFAULT_FLOW_SWITCH_THETA_DEG,
        FlowSwitchDecision,
        HandFlowCandidate,
        evaluate_object_flow,
        select_flow_reference,
    )
except ImportError:
    from flow_switching import (
        DEFAULT_FLOW_SWITCH_THETA_DEG,
        FlowSwitchDecision,
        HandFlowCandidate,
        evaluate_object_flow,
        select_flow_reference,
    )


def _decode_image_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes to numpy array (H, W, 3) RGB."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _encode_video_to_bytes(video: np.ndarray, fps: int = 16) -> bytes:
    """Encode numpy video (T, H, W, 3) to MP4 bytes."""
    import imageio
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp:
        writer = imageio.get_writer(
            tmp.name, format='FFMPEG', mode='I',
            fps=fps, codec='libx264', bitrate='8M',
            ffmpeg_log_level='error'
        )
        for frame in video:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            writer.append_data(frame)
        writer.close()
        with open(tmp.name, 'rb') as f:
            return f.read()


@dataclass
class FlowExtractionResult:
    """Result of flow extraction from videos."""
    flow_images: List[Optional[np.ndarray]] = field(default_factory=list)  # (H, W, 3) flow visualizations
    flow_bytes: List[Optional[bytes]] = field(default_factory=list)  # Raw PNG bytes
    coords_3d: List[Optional[np.ndarray]] = field(default_factory=list)  # Optional TAPIP3D [T, N, 3]
    visibilities: List[Optional[np.ndarray]] = field(default_factory=list)  # Optional TAPIP3D [T, N]
    flow_choices: List[Optional[FlowSwitchDecision]] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.flow_images)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.flow_images[idx]
    
    def __iter__(self):
        return iter(self.flow_images)
    
    @property
    def selected_flow_types(self) -> List[str]:
        """Return the selected flow type for each rollout candidate."""
        return [choice.selected_flow for choice in self.flow_choices if choice is not None]


class FlowExtractionClient:
    """
    Client for extracting optical flow from videos using the ComfyUI job server.
    
    This client uses the 'flow_only' mode to extract flow images from existing
    videos without generating new videos. It's designed to work with videos
    from any source (WAN22, Veo3, etc.).
    """
    
    def __init__(
        self,
        server_base: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3600,
        hand_flow_provider: Optional[Callable[..., Optional[HandFlowCandidate]]] = None,
        flow_switch_theta_deg: float = DEFAULT_FLOW_SWITCH_THETA_DEG,
    ):
        """
        Args:
            server_base: Base URL of the ComfyUI job server (e.g., "http://localhost:7000")
            timeout: HTTP request timeout in seconds
            max_retries: Maximum poll iterations for job completion
        """
        self.base = server_base or os.getenv(
            "NOVAPLAN_VIDEO_SERVER_URL", "http://127.0.0.1:7000"
        )
        self._submit_url = f"{self.base}/submit"
        self._status_url = f"{self.base}/status"
        self.timeout = timeout
        self.max_retries = max_retries
        self.hand_flow_provider = hand_flow_provider
        self.flow_switch_theta_deg = float(flow_switch_theta_deg)
        self._session = requests.Session()
        self._selection_flow_contract_verified = False

    def ensure_selection_flow_contract(self) -> None:
        """Verify that the video service can extract CoTracker selection flow."""
        if self._selection_flow_contract_verified:
            return

        health_url = f"{self.base}/health"
        try:
            response = self._session.get(health_url, timeout=min(self.timeout, 10))
            response.raise_for_status()
            health = response.json()
        except Exception as exc:
            raise RuntimeError(
                "The video-generation service required for rollout selection is "
                f"unreachable at {health_url}. WAN generation and CoTracker3 flow_only "
                "extraction for Veo both use NOVAPLAN_VIDEO_SERVER_URL. Start or "
                "port-forward the video service to that URL before generating videos "
                "(for example: kubectl port-forward pod/<video-pod> -n <namespace> "
                "7000:7000 8188:8188)."
            ) from exc

        contract_version = int(health.get("contract_version", 0))
        flow_only_available = health.get("flow_only_available")
        if contract_version < 3 or flow_only_available is False:
            raise RuntimeError(
                "The video-generation service is stale or does not support the "
                f"CoTracker3 flow_only contract required for rollout selection: {health}. "
                "Restart/deploy the current remote_video_generation service."
            )
        self._selection_flow_contract_verified = True
    
    def _submit_flow_job(
        self,
        video_bytes: bytes,
        mask_prompt: str,
    ) -> str:
        """Submit a flow-only job and return job_id."""
        video_b64 = base64.b64encode(video_bytes).decode('ascii')
        payload: Dict[str, Any] = {
            "mode": "flow_only",
            "video_base64": video_b64,
            "mask_prompt": mask_prompt,
            "num_videos": 1,  # Always 1 - we send separate parallel requests
        }
        
        r = self._session.post(self._submit_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        resp = r.json()
        if "job_id" not in resp:
            raise RuntimeError(f"Unexpected submit response: {resp}")
        return resp["job_id"]
    
    def _poll_until_done(self, job_id: str) -> Dict[str, Any]:
        """Poll until job is completed or failed."""
        sleep = 0.5
        for _ in range(self.max_retries):
            r = self._session.get(f"{self._status_url}/{job_id}", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            st = data.get("status")
            if st in ("completed", "failed"):
                return data
            time.sleep(sleep)
            if sleep < 1.0:
                sleep = min(1.0, sleep * 1.5)
        raise TimeoutError(f"Job {job_id} did not complete in time")
    
    def _fetch_flow_outputs(self, job_id: str) -> Dict[str, List[Any]]:
        """Fetch 2D CoTracker3 evidence from the video-generation server."""
        outputs: Dict[str, List[Any]] = {
            "flow_images": [],
            "coords_3d": [],
            "visibilities": [],
        }
        all_url = f"{self.base}/result/{job_id}/all"
        try:
            r = self._session.get(all_url, timeout=self.timeout)
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
            else:
                data = {}
            
            for flow in data.get("flow_images", []):
                if "data_base64" in flow:
                    outputs["flow_images"].append(base64.b64decode(flow["data_base64"]))

        except Exception as e:
            print(f"[FlowExtractionClient] /all result fetch failed: {e}")

        if not outputs["flow_images"]:
            try:
                r = self._session.get(f"{self.base}/result/{job_id}/flow", timeout=self.timeout)
                if r.status_code == 200 and r.content:
                    outputs["flow_images"].append(r.content)
            except Exception:
                pass

        return outputs

    def _fetch_flow_results(self, job_id: str) -> List[bytes]:
        """Fetch flow images from the configured server."""
        return [bytes(x) for x in self._fetch_flow_outputs(job_id)["flow_images"]]

    def _maybe_get_hand_flow(
        self,
        *,
        video: Union[np.ndarray, bytes],
        video_bytes: bytes,
        mask_prompt: str,
        job_id: str,
        index: int,
    ) -> Optional[HandFlowCandidate]:
        if self.hand_flow_provider is None:
            return None
        try:
            result = self.hand_flow_provider(
                video=video,
                video_bytes=video_bytes,
                mask_prompt=mask_prompt,
                job_id=job_id,
                index=index,
            )
        except TypeError:
            result = self.hand_flow_provider(video)
        if result is None:
            return None
        if isinstance(result, HandFlowCandidate):
            return result
        if isinstance(result, dict):
            return HandFlowCandidate(**result)
        return HandFlowCandidate(trajectory=np.asarray(result))
    
    def extract_flow(
        self,
        videos: List[Union[np.ndarray, bytes]],
        mask_prompt: str,
        fps: int = 16,
    ) -> FlowExtractionResult:
        """
        Extract optical flow from a list of videos.
        
        Args:
            videos: List of videos as numpy arrays (T, H, W, 3) or MP4 bytes
            mask_prompt: SAM3 mask prompt describing the object to track
            fps: Frames per second (used when encoding numpy videos to bytes)
            
        Returns:
            FlowExtractionResult containing flow images
        """
        if not videos:
            return FlowExtractionResult()

        self.ensure_selection_flow_contract()
        
        # Convert numpy videos to bytes if needed
        video_bytes_list: List[bytes] = []
        for vid in videos:
            if isinstance(vid, np.ndarray):
                video_bytes_list.append(_encode_video_to_bytes(vid, fps=fps))
            else:
                video_bytes_list.append(vid)
        
        # Submit all flow jobs in parallel
        submit_results: List[Tuple[int, str]] = []
        submit_errors: Dict[int, str] = {}
        lock = threading.Lock()
        
        def _submit_one(idx: int):
            try:
                job_id = self._submit_flow_job(
                    video_bytes=video_bytes_list[idx],
                    mask_prompt=mask_prompt,
                )
                with lock:
                    submit_results.append((idx, job_id))
            except Exception as e:
                with lock:
                    submit_errors[idx] = str(e)
        
        max_workers = min(len(videos), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_submit_one, i) for i in range(len(videos))]
            concurrent.futures.wait(futs)
        
        if submit_errors:
            failed = ", ".join(f"{i}:{err}" for i, err in submit_errors.items())
            print(f"[FlowExtractionClient] Some submissions failed: {failed}")
        
        submit_results.sort(key=lambda x: x[0])
        
        # Poll and fetch results
        flow_images: List[Optional[np.ndarray]] = [None] * len(videos)
        flow_bytes: List[Optional[bytes]] = [None] * len(videos)
        coords_3d: List[Optional[np.ndarray]] = [None] * len(videos)
        visibilities: List[Optional[np.ndarray]] = [None] * len(videos)
        flow_choices: List[Optional[FlowSwitchDecision]] = [None] * len(videos)
        poll_errors: Dict[int, str] = {}
        
        def _poll_one(idx: int, job_id: str):
            try:
                status = self._poll_until_done(job_id)
                if status.get("status") != "completed":
                    raise RuntimeError(f"Job {job_id} failed: {status}")
                
                flow_outputs = self._fetch_flow_outputs(job_id)
                flow_bytes_list = flow_outputs["flow_images"]
                if flow_bytes_list:
                    flow_images[idx] = _decode_image_bytes_to_numpy(flow_bytes_list[0])
                    flow_bytes[idx] = flow_bytes_list[0]

                if flow_outputs["coords_3d"]:
                    coords_3d[idx] = flow_outputs["coords_3d"][0]
                if flow_outputs["visibilities"]:
                    visibilities[idx] = flow_outputs["visibilities"][0]

                if coords_3d[idx] is not None:
                    object_eval = evaluate_object_flow(
                        coords_3d[idx],
                        visibilities[idx],
                        layout="time_major",
                        flow_switch_theta_deg=self.flow_switch_theta_deg,
                    )
                    hand_flow = None
                    if object_eval.should_switch_to_hand or not object_eval.valid:
                        hand_flow = self._maybe_get_hand_flow(
                            video=videos[idx],
                            video_bytes=video_bytes_list[idx],
                            mask_prompt=mask_prompt,
                            job_id=job_id,
                            index=idx,
                        )
                    flow_choices[idx] = select_flow_reference(object_eval, hand_flow)
            except Exception as e:
                poll_errors[idx] = str(e)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_poll_one, idx, job_id) for idx, job_id in submit_results]
            concurrent.futures.wait(futs)
        
        if poll_errors:
            failed = ", ".join(f"{i}:{err}" for i, err in poll_errors.items())
            print(f"[FlowExtractionClient] Some jobs failed: {failed}")
        
        # Keep outputs aligned with the input video list. Downstream planner
        # metadata uses list indices to match a flow result back to a rollout.
        result_images = list(flow_images)
        result_bytes = list(flow_bytes)
        result_coords = list(coords_3d)
        result_vis = list(visibilities)
        result_choices = list(flow_choices)
        
        return FlowExtractionResult(
            flow_images=result_images,
            flow_bytes=result_bytes,
            coords_3d=result_coords,
            visibilities=result_vis,
            flow_choices=result_choices,
        )
    
    def extract_flow_single(
        self,
        video: Union[np.ndarray, bytes],
        mask_prompt: str,
        fps: int = 16,
    ) -> Optional[np.ndarray]:
        """
        Extract optical flow from a single video.
        
        Args:
            video: Video as numpy array (T, H, W, 3) or MP4 bytes
            mask_prompt: SAM3 mask prompt describing the object to track
            fps: Frames per second (used when encoding numpy videos to bytes)
            
        Returns:
            Flow image as numpy array (H, W, 3) or None if extraction failed
        """
        result = self.extract_flow([video], mask_prompt, fps=fps)
        return result.flow_images[0] if result.flow_images else None
