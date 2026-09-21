#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
ComfyUI Wan2.2 video-generation client.

- Pure HTTP client for a ComfyUI Wan2.2 queue server
- Submits I2V / TI2V jobs with per-request hyperparameters
- Returns videos as NumPy arrays in memory (no files saved)
- Supports optional flow image extraction (CoTracker) alongside video generation
- Thread-safe; supports concurrent planner submissions

Dependencies:
  pixi install -e local-planning
"""

from __future__ import annotations
import os
import io
import time
import base64
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, List, Tuple, Union

import requests
import numpy as np
from PIL import Image
import imageio.v3 as iio  # imageio.v3 can read from bytes with extension hint

try:
    from ..flow_switching import (
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


@dataclass
class VideoGenerationResult:
    """Result of video generation, optionally including flow images."""
    videos: List[np.ndarray] = field(default_factory=list)  # List of (T, H, W, 3) RGB videos
    flow_images: List[Optional[np.ndarray]] = field(default_factory=list)  # List of (H, W, 3) flow visualization images
    video_bytes: List[bytes] = field(default_factory=list)  # Raw MP4 bytes for each video
    flow_bytes: List[Optional[bytes]] = field(default_factory=list)  # Raw PNG bytes for each flow image
    coords_3d: List[Optional[np.ndarray]] = field(default_factory=list)  # Optional TAPIP3D [T, N, 3]
    visibilities: List[Optional[np.ndarray]] = field(default_factory=list)  # Optional TAPIP3D [T, N]
    flow_choices: List[Optional[FlowSwitchDecision]] = field(default_factory=list)
    video_sources: List[str] = field(default_factory=list)  # Backend name for each video, e.g. wan22 or veo3.
    requested_backends: List[str] = field(default_factory=list)
    requested_samples_per_backend: Optional[int] = None
    backend_errors: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> np.ndarray:
        """For backward compatibility: indexing returns videos."""
        return self.videos[idx]

    def __iter__(self):
        """For backward compatibility: iteration yields videos."""
        return iter(self.videos)

    @property
    def selected_flow_types(self) -> List[str]:
        """Return the selected flow type for each rollout candidate."""
        return [choice.selected_flow for choice in self.flow_choices if choice is not None]


def _np_to_jpeg_bytes(arr: np.ndarray, quality: int = 95) -> bytes:
    """Convert an RGB uint8 numpy array (H,W,3) to JPEG bytes in-memory."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    img = Image.fromarray(arr, mode="RGB")
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=quality)
    return bio.getvalue()


def _decode_video_bytes_to_numpy(mp4_bytes: bytes) -> np.ndarray:
    """
    Decode MP4 bytes fully in memory → np.ndarray (T, H, W, 3) RGB.
    Uses imageio.v3; no temp files.
    """
    # imageio.v3 needs either an extension hint or a file-like object with .name
    # We provide extension via the "extension" keyword.
    vid = iio.imread(mp4_bytes, extension=".mp4", index=None)  # returns (T, H, W, 3)
    # imageio returns RGB uint8 already
    if vid.ndim != 4 or vid.shape[-1] != 3:
        # Safety: ensure (T,H,W,3)
        raise ValueError(f"Unexpected decoded shape {vid.shape}")
    return vid


def _decode_image_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    """
    Decode PNG/JPEG bytes fully in memory → np.ndarray (H, W, 3) RGB.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img, dtype=np.uint8)


class WanVideoGenerationClient:
    """
    ComfyUI Wan 2.2 HTTP client used by the NovaPlan planner.
    The planner uses the generate_rollouts() entry point.
    """

    def __init__(
        self,
        server_base: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3600,  # 3600 * 1s = 1 hour max wait
        hand_flow_provider: Optional[Callable[..., Optional[HandFlowCandidate]]] = None,
        flow_switch_theta_deg: float = DEFAULT_FLOW_SWITCH_THETA_DEG,
    ):
        """
        Args:
            server_base: Base URL of the queue server (e.g. "http://<host>:7000").
                         If None, read NOVAPLAN_VIDEO_SERVER_URL or use localhost.
            timeout: per-request HTTP timeout (seconds)
            max_retries: max poll iterations (sleep is adaptive)
        """
        self.base = server_base or os.getenv(
            "NOVAPLAN_VIDEO_SERVER_URL", "http://127.0.0.1:7000"
        )
        # Endpoints (match the queue_server you’re running)
        # Updated server uses /submit and /status/{job_id}
        self._submit_url = f"{self.base}/submit"
        self._status_url = f"{self.base}/status"
        self._result_url = f"{self.base}/result"  # kept for backward compat with older servers
        self.timeout = timeout
        self.max_retries = max_retries
        self.hand_flow_provider = hand_flow_provider
        self.flow_switch_theta_deg = float(flow_switch_theta_deg)

        # Optional debug markers used by planner diagnostics.
        self.step_counter = 0
        self.beam_counter = 0
        self.debug_dir = None
        self._video_scores = {}

        # Session for connection reuse
        self._session = requests.Session()
        self._wan_full_contract_verified = False

    def _ensure_wan_full_contract(self) -> None:
        """Fail before generation when the server cannot return paired WAN flow."""
        if self._wan_full_contract_verified:
            return
        health_url = f"{self.base}/health"
        try:
            response = self._session.get(health_url, timeout=min(self.timeout, 10))
            response.raise_for_status()
            health = response.json()
        except Exception as exc:
            raise RuntimeError(
                "The video-generation service required for WAN rollout generation is "
                f"unreachable at {health_url}. Check NOVAPLAN_VIDEO_SERVER_URL and "
                "start or port-forward the remote video service before generating "
                "candidates (for example: kubectl port-forward pod/<video-pod> "
                "-n <namespace> 7000:7000 8188:8188)."
            ) from exc
        if int(health.get("contract_version", 0)) < 3 or not health.get("wan_full_inline_flow"):
            raise RuntimeError(
                "The running video-generation server is stale or does not support paired "
                f"WAN+flow output: {health}. Restart/deploy the current server before "
                "generating candidates."
            )
        self._wan_full_contract_verified = True

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

    # -------- Low-level HTTP helpers --------

    def _submit_job(
        self,
        *,
        prompt: str,
        first_frame_jpeg_b64: str,
        frame_num: int,
        fps: int,
        size: str,                        # "1280*720"
        seed: int,
        sampling_steps: Optional[int],
        guide_scale: Optional[float],
        sample_solver: Optional[str],
        fast_mode: Optional[bool] = False, # Enable LightX2V 4-step LoRAs (14B only)
        mode: str = "generate_only",      # "full" (WAN + flow), "generate_only" (WAN only)
        mask_prompt: Optional[str] = None,  # SAM3 mask prompt for flow extraction
        last_frame_jpeg_b64: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> str:
        """
        Submit a single video generation job and return job_id.

        Args:
            mode: "full" for video generation + flow extraction,
                  "generate_only" for video generation only
            mask_prompt: SAM3 mask prompt for flow extraction (required if mode="full")
        """
        # Parse size string like "640*480" into width/height
        width = None
        height = None
        if isinstance(size, str) and "*" in size:
            try:
                w, h = size.split("*")
                width = int(w)
                height = int(h)
            except Exception:
                width = None
                height = None

        payload: Dict[str, Any] = {
            "mode": mode,
            "prompt": prompt,
            "first_frame_base64": first_frame_jpeg_b64,
            "fast_mode": bool(fast_mode),
            "seed": int(seed),
            "frames": int(frame_num),
            "fps": int(fps),
            "num_videos": 1,  # Always 1 - we send separate parallel requests
        }
        if last_frame_jpeg_b64:
            payload["last_frame_base64"] = last_frame_jpeg_b64
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if width is not None:
            payload["width"] = width
        if height is not None:
            payload["height"] = height
        if sampling_steps is not None:
            payload["sampling_steps"] = int(sampling_steps)
        if guide_scale is not None:
            payload["guide_scale"] = float(guide_scale)
        if sample_solver:
            payload["sample_solver"] = str(sample_solver)
        if mask_prompt:
            payload["mask_prompt"] = mask_prompt

        r = self._session.post(self._submit_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        resp = r.json()
        if "job_id" not in resp:
            raise RuntimeError(f"Unexpected submit response: {resp}")
        return resp["job_id"]

    def _submit_flow_only_job(
        self,
        *,
        video_bytes: bytes,
        mask_prompt: str,
    ) -> str:
        """
        Submit a flow-only job for extracting flow from an existing video.
        Used for processing Veo3-generated videos.
        """
        video_b64 = base64.b64encode(video_bytes).decode('ascii')
        payload: Dict[str, Any] = {
            "mode": "flow_only",
            "video_base64": video_b64,
            "mask_prompt": mask_prompt,
            "num_videos": 1,
        }

        r = self._session.post(self._submit_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        resp = r.json()
        if "job_id" not in resp:
            raise RuntimeError(f"Unexpected submit response: {resp}")
        return resp["job_id"]

    def _poll_until_done(self, job_id: str) -> Dict[str, Any]:
        """
        Poll /status until job is 'completed' or 'failed'.
        Returns the final status payload.
        """
        # Adaptive poll: start fast, stay responsive (max 1s sleep)
        sleep = 0.5
        for i in range(self.max_retries):
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

    def _fetch_result_bytes(self, job_id: str) -> List[bytes]:
        """
        GET /result/{job_id}/binary → raw MP4 bytes (FAST - no base64 overhead!)
        Handles both single video and multipart responses for multiple videos.
        Falls back to JSON endpoint if binary not available.
        Returns list of video bytes (even if single video for consistency).
        """
        binary_url = f"{self.base}/result/{job_id}/binary"
        try:
            r = self._session.get(binary_url, timeout=self.timeout, stream=True)
            if r.status_code == 200:
                ctype = r.headers.get("content-type", "").lower()
                if ctype.startswith("video/") or ctype.startswith("application/octet-stream"):
                    return [r.content]
                if ctype.startswith("multipart/mixed"):
                    boundary = None
                    for part in ctype.split(";"):
                        if "boundary=" in part:
                            boundary = part.split("boundary=", 1)[1].strip().strip('"')
                            break
                    if boundary:
                        payload = r.content
                        videos: List[bytes] = []
                        for part in payload.split(f"--{boundary}".encode("utf-8"))[1:-1]:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end == -1:
                                continue
                            video_data = part[header_end + 4 :]
                            if video_data.endswith(b"\r\n"):
                                video_data = video_data[:-2]
                            if video_data:
                                videos.append(bytes(video_data))
                        if videos:
                            return videos
        except Exception:
            # Older servers may not expose the binary endpoint; fall through to
            # JSON/status-based compatibility paths below.
            pass

        try:
            rj = self._session.get(f"{self._result_url}/{job_id}", timeout=self.timeout)
            if rj.status_code == 200 and rj.headers.get("content-type", "").startswith("application/json"):
                result_data = rj.json()
                if "videos_base64" in result_data:
                    return [base64.b64decode(b64) for b64 in result_data["videos_base64"]]
                b64 = result_data.get("video_base64") or result_data.get("mp4_base64") or result_data.get("data")
                if b64:
                    return [base64.b64decode(b64)]
        except Exception:
            pass

        # New server currently exposes results via the status payload. We'll
        # fetch the status and try to locate video filenames or base64 blobs.
        r = self._session.get(f"{self._status_url}/{job_id}", timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # If the server included base64-encoded videos, return those first.
        if isinstance(data, dict):
            if "videos_base64" in data:
                return [base64.b64decode(b64) for b64 in data["videos_base64"]]
            b64 = data.get("video_base64") or data.get("mp4_base64") or data.get("data")
            if b64:
                return [base64.b64decode(b64)]

            # If the server returned a path/filename for a generated video, try
            # to read it from common output locations (NFS mounts).
            wan_video = data.get("wan_video") or data.get("video_filename")
            if wan_video:
                # A shared output directory is optional; normal remote clients
                # receive bytes from the server and do not need filesystem access.
                candidates = [wan_video]
                shared_output_dir = os.environ.get("COMFYUI_OUTPUT_DIR")
                if shared_output_dir:
                    candidates.append(os.path.join(shared_output_dir, wan_video))
                candidates.append(os.path.join(os.getcwd(), wan_video))
                for p in candidates:
                    try:
                        if p and os.path.exists(p):
                            with open(p, "rb") as f:
                                return [f.read()]
                    except Exception:
                        continue

        raise RuntimeError("Unable to retrieve binary result for job")

    def _fetch_all_results(self, job_id: str) -> Tuple[List[bytes], List[bytes], List[np.ndarray], List[np.ndarray]]:
        """
        Fetch videos and 2D CoTracker3 flow images from the video server.

        Metric 3D tracks are intentionally absent here. They are computed by
        the separate object-flow server after rollout selection.
        """
        all_url = f"{self.base}/result/{job_id}/all"
        videos = []
        flow_images = []
        coords_3d = []
        visibilities = []
        try:
            r = self._session.get(all_url, timeout=self.timeout)
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith("application/json"):
                data = r.json()
            else:
                data = {}

            for vid in data.get("videos", []):
                if "data_base64" in vid:
                    videos.append(base64.b64decode(vid["data_base64"]))

            for flow in data.get("flow_images", []):
                if "data_base64" in flow:
                    flow_images.append(base64.b64decode(flow["data_base64"]))

            if not flow_images:
                flow_resp = self._session.get(f"{self.base}/result/{job_id}/flow", timeout=self.timeout)
                if flow_resp.status_code == 200 and flow_resp.content:
                    flow_images.append(flow_resp.content)

            return videos, flow_images, coords_3d, visibilities

        except Exception as e:
            print(f"[WanVideoGenerationClient] fetch_all_results failed: {e}")
            # Fall back to regular video fetch, no flow images
            try:
                videos = self._fetch_result_bytes(job_id)
                return videos, [], [], []
            except Exception:
                return [], [], [], []

    # -------- Public planner API --------

    @staticmethod
    def _complete_output_indices(
        videos: List[Optional[np.ndarray]],
        flows: List[Optional[np.ndarray]],
        *,
        require_inline_flow: bool,
    ) -> Tuple[List[int], List[int]]:
        """Select usable WAN outputs while enforcing the full-mode contract."""

        video_indices = [idx for idx, video in enumerate(videos) if video is not None]
        if not require_inline_flow:
            return video_indices, []

        complete_indices = [idx for idx in video_indices if idx < len(flows) and flows[idx] is not None]
        missing_flow_indices = [idx for idx in video_indices if idx not in complete_indices]
        if video_indices and not complete_indices:
            raise RuntimeError(
                "WAN full-mode jobs returned video data but no inline SAM3/CoTracker3 flow images. "
                "The video-generation server must return both artifacts from the same full request; "
                "flow_only is reserved for non-WAN videos such as Veo. Restart/deploy the current "
                "remote_video_generation server and run its full-mode integration test."
            )
        return complete_indices, missing_flow_indices

    def generate_rollouts(
        self,
        start_frame: np.ndarray,
        action_text: str,
        num_samples: int = 2,
        num_frames: int = 41,
        fps: int = 16,
        size: str = "640*480",
        seed: int = 42,
        sampling_steps: Optional[int] = None,
        guide_scale: Optional[float] = None,
        sample_solver: str = "euler",
        fast_mode: bool = True,     # Enable LightX2V 4-step LoRAs (14B only, ~7x faster)
        enable_flow: bool = False,  # If True, also extract flow images (returns VideoGenerationResult)
        mask_prompt: Optional[str] = None,  # SAM3 mask prompt for flow extraction
        last_frame: Optional[np.ndarray] = None,  # If set, use first-last-frame conditioning.
        negative_prompt: Optional[str] = None,
    ) -> List[np.ndarray] | VideoGenerationResult:
        """
        Submit N parallel jobs and return a list of NumPy videos (T,H,W,3) RGB.

        If enable_flow=True, returns VideoGenerationResult containing both videos and flow images.
        Otherwise, returns List[np.ndarray] for backward compatibility.

        Args:
            enable_flow: If True, run flow extraction (CoTracker) alongside video generation.
                        Requires mask_prompt to be set.
            mask_prompt: SAM3 mask prompt describing the object to track (e.g., "black handle on drawer")
        """
        # Determine mode based on enable_flow
        mode = "full" if enable_flow else "generate_only"
        if enable_flow and not mask_prompt:
            print("[WanVideoGenerationClient] Warning: enable_flow=True but no mask_prompt provided, using default")
            mask_prompt = "object"
        if enable_flow:
            self._ensure_wan_full_contract()
        # Ensure RGB uint8
        if start_frame.dtype != np.uint8:
            start_frame = start_frame.astype(np.uint8, copy=False)
        if start_frame.ndim != 3 or start_frame.shape[-1] != 3:
            raise ValueError("start_frame must be (H, W, 3) RGB uint8")

        # Encode first frame once to JPEG base64 to reduce payload size
        jpeg_bytes = _np_to_jpeg_bytes(start_frame, quality=95)
        first_frame_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        last_frame_b64 = None
        if last_frame is not None:
            if last_frame.dtype != np.uint8:
                last_frame = last_frame.astype(np.uint8, copy=False)
            if last_frame.ndim != 3 or last_frame.shape[-1] != 3:
                raise ValueError("last_frame must be (H, W, 3) RGB uint8")
            last_frame_b64 = base64.b64encode(_np_to_jpeg_bytes(last_frame, quality=95)).decode("ascii")

        # Submit separate parallel requests (one per video sample)
        # Each request goes to a different GPU worker on the server
        mode_label = f"{mode}+flf" if last_frame_b64 else mode
        print(f"[WanVideoGenerationClient] Submitting {num_samples} parallel requests (mode={mode_label})")

        # Prepare seeds for each sample (simple variation)
        seeds = [int(seed) + i * 17 for i in range(num_samples)]

        # Step 1: submit all jobs in parallel
        submit_results: List[Tuple[int, str]] = []  # (index, job_id)
        submit_errors: Dict[int, str] = {}
        lock = threading.Lock()

        def _submit_one(idx: int):
            try:
                job_id = self._submit_job(
                    prompt=action_text,
                    first_frame_jpeg_b64=first_frame_b64,
                    last_frame_jpeg_b64=last_frame_b64,
                    negative_prompt=negative_prompt,
                    frame_num=int(num_frames),
                    fps=int(fps),
                    size=size,
                    seed=seeds[idx],
                    sampling_steps=sampling_steps,
                    guide_scale=guide_scale,
                    sample_solver=sample_solver,
                    fast_mode=fast_mode,
                    mode=mode,
                    mask_prompt=mask_prompt,
                )
                with lock:
                    submit_results.append((idx, job_id))
            except Exception as e:
                with lock:
                    submit_errors[idx] = str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_samples, 8)) as ex:
            futs = [ex.submit(_submit_one, i) for i in range(num_samples)]
            concurrent.futures.wait(futs)

        if submit_errors:
            # Return as many as possible, but log the failures
            failed = ", ".join(f"{i}:{err}" for i, err in submit_errors.items())
            print(f"[WanVideoGenerationClient] Some submissions failed: {failed}")

        # Keep original index ordering
        submit_results.sort(key=lambda x: x[0])

        # Step 2: poll all jobs and fetch results in parallel
        videos_result: List[Optional[np.ndarray]] = [None] * num_samples
        flows_result: List[Optional[np.ndarray]] = [None] * num_samples
        video_bytes_result: List[Optional[bytes]] = [None] * num_samples
        flow_bytes_result: List[Optional[bytes]] = [None] * num_samples
        coords_result: List[Optional[np.ndarray]] = [None] * num_samples
        vis_result: List[Optional[np.ndarray]] = [None] * num_samples
        flow_choice_result: List[Optional[FlowSwitchDecision]] = [None] * num_samples
        poll_errors: Dict[int, str] = {}

        def _poll_and_fetch(idx: int, job_id: str):
            try:
                status = self._poll_until_done(job_id)
                if status.get("status") != "completed":
                    raise RuntimeError(f"Job {job_id} ended with status: {status}")

                if enable_flow:
                    mp4_bytes_list, flow_bytes_list, coords_list, vis_list = self._fetch_all_results(job_id)
                else:
                    mp4_bytes_list = self._fetch_result_bytes(job_id)
                    flow_bytes_list = []
                    coords_list = []
                    vis_list = []

                # Should only be one video per job in fallback mode
                if mp4_bytes_list:
                    arr = _decode_video_bytes_to_numpy(mp4_bytes_list[0])
                    videos_result[idx] = arr
                    video_bytes_result[idx] = mp4_bytes_list[0]

                if flow_bytes_list:
                    flow_arr = _decode_image_bytes_to_numpy(flow_bytes_list[0])
                    flows_result[idx] = flow_arr
                    flow_bytes_result[idx] = flow_bytes_list[0]

                if coords_list:
                    coords_result[idx] = coords_list[0]
                if vis_list:
                    vis_result[idx] = vis_list[0]

                if coords_result[idx] is not None:
                    object_eval = evaluate_object_flow(
                        coords_result[idx],
                        vis_result[idx],
                        layout="time_major",
                        flow_switch_theta_deg=self.flow_switch_theta_deg,
                    )
                    hand_flow = None
                    if object_eval.should_switch_to_hand or not object_eval.valid:
                        provider_video = videos_result[idx] if videos_result[idx] is not None else (
                            mp4_bytes_list[0] if mp4_bytes_list else b""
                        )
                        provider_video_bytes = mp4_bytes_list[0] if mp4_bytes_list else b""
                        hand_flow = self._maybe_get_hand_flow(
                            video=provider_video,
                            video_bytes=provider_video_bytes,
                            mask_prompt=mask_prompt or "object",
                            job_id=job_id,
                            index=idx,
                        )
                    flow_choice_result[idx] = select_flow_reference(object_eval, hand_flow)
            except Exception as e:
                poll_errors[idx] = str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_samples, 8)) as ex:
            futs = [ex.submit(_poll_and_fetch, idx, job_id) for idx, job_id in submit_results]
            concurrent.futures.wait(futs)

        if poll_errors:
            failed = ", ".join(f"{i}:{err}" for i, err in poll_errors.items())
            print(f"[WanVideoGenerationClient] Some jobs failed during polling/fetch: {failed}")

        # WAN full mode must return video and CoTracker3 evidence together.
        successful_indices, missing_flow_indices = self._complete_output_indices(
            videos_result,
            flows_result,
            require_inline_flow=enable_flow,
        )
        if missing_flow_indices:
            print(
                "[WanVideoGenerationClient] Dropping WAN full-mode result(s) without inline flow: "
                + ", ".join(str(idx) for idx in missing_flow_indices)
            )
        out_videos: List[np.ndarray] = [videos_result[idx] for idx in successful_indices]
        out_flows = [flows_result[idx] for idx in successful_indices]
        out_video_bytes: List[bytes] = [video_bytes_result[idx] for idx in successful_indices if video_bytes_result[idx] is not None]
        out_flow_bytes = [flow_bytes_result[idx] for idx in successful_indices]
        out_coords = [coords_result[idx] for idx in successful_indices]
        out_vis = [vis_result[idx] for idx in successful_indices]
        out_flow_choices = [flow_choice_result[idx] for idx in successful_indices]

        if enable_flow:
            return VideoGenerationResult(
                videos=out_videos,
                flow_images=out_flows,
                video_bytes=out_video_bytes,
                flow_bytes=out_flow_bytes,
                coords_3d=out_coords,
                visibilities=out_vis,
                flow_choices=out_flow_choices,
            )
        return out_videos

    def generate_first_last_frame_rollouts(
        self,
        *,
        start_frame: np.ndarray,
        last_frame: np.ndarray,
        prompt: str,
        num_samples: int = 1,
        num_frames: int = 41,
        fps: int = 16,
        size: str = "640*480",
        seed: int = 42,
        fast_mode: bool = True,
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Generate WAN first-last-frame recovery rollouts."""

        result = self.generate_rollouts(
            start_frame=start_frame,
            action_text=prompt,
            num_samples=num_samples,
            num_frames=num_frames,
            fps=fps,
            size=size,
            seed=seed,
            fast_mode=fast_mode,
            last_frame=last_frame,
            negative_prompt=negative_prompt,
            enable_flow=False,
            **kwargs,
        )
        if isinstance(result, VideoGenerationResult):
            return result.videos
        return result

    def extract_flow(
        self,
        videos: List[np.ndarray] | List[bytes],
        mask_prompt: str,
    ) -> List[np.ndarray]:
        """
        Extract flow images from existing videos using the flow-only mode.
        This is useful for processing videos generated by other backends (e.g., Veo3).

        Args:
            videos: List of videos as either numpy arrays (T,H,W,3) or MP4 bytes
            mask_prompt: SAM3 mask prompt describing the object to track

        Returns:
            List of flow images as numpy arrays (H,W,3) RGB
        """
        import tempfile

        # Convert numpy videos to bytes if needed
        video_bytes_list: List[bytes] = []
        for vid in videos:
            if isinstance(vid, np.ndarray):
                # Encode video to MP4 bytes
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp:
                    import imageio
                    writer = imageio.get_writer(tmp.name, format='FFMPEG', mode='I',
                                               fps=16, codec='libx264', bitrate='8M')
                    for frame in vid:
                        writer.append_data(frame)
                    writer.close()
                    with open(tmp.name, 'rb') as f:
                        video_bytes_list.append(f.read())
            else:
                video_bytes_list.append(vid)

        # Submit flow-only jobs for each video
        submit_results: List[Tuple[int, str]] = []
        submit_errors: Dict[int, str] = {}
        lock = threading.Lock()

        def _submit_flow_one(idx: int):
            try:
                job_id = self._submit_flow_only_job(
                    video_bytes=video_bytes_list[idx],
                    mask_prompt=mask_prompt,
                )
                with lock:
                    submit_results.append((idx, job_id))
            except Exception as e:
                with lock:
                    submit_errors[idx] = str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(videos), 8)) as ex:
            futs = [ex.submit(_submit_flow_one, i) for i in range(len(videos))]
            concurrent.futures.wait(futs)

        if submit_errors:
            failed = ", ".join(f"{i}:{err}" for i, err in submit_errors.items())
            print(f"[WanVideoGenerationClient] Some flow submissions failed: {failed}")

        submit_results.sort(key=lambda x: x[0])

        # Poll and fetch flow results
        flows_result: List[Optional[np.ndarray]] = [None] * len(videos)
        poll_errors: Dict[int, str] = {}

        def _poll_flow(idx: int, job_id: str):
            try:
                status = self._poll_until_done(job_id)
                if status.get("status") != "completed":
                    raise RuntimeError(f"Job {job_id} ended with status: {status}")

                _, flow_bytes_list, _, _ = self._fetch_all_results(job_id)
                if flow_bytes_list:
                    flow_arr = _decode_image_bytes_to_numpy(flow_bytes_list[0])
                    flows_result[idx] = flow_arr
            except Exception as e:
                poll_errors[idx] = str(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(videos), 8)) as ex:
            futs = [ex.submit(_poll_flow, idx, job_id) for idx, job_id in submit_results]
            concurrent.futures.wait(futs)

        if poll_errors:
            failed = ", ".join(f"{i}:{err}" for i, err in poll_errors.items())
            print(f"[WanVideoGenerationClient] Some flow jobs failed: {failed}")

        return [f for f in flows_result if f is not None]

    # Compatibility stubs for older workstation clients.
    def submit_video_job(self, *args, **kwargs):
        """Submit a video-generation job to the remote service."""
        raise NotImplementedError("Use generate_rollouts() in this Comfy client")

    def get_job_status(self, *args, **kwargs):
        """Return job status."""
        raise NotImplementedError("Use generate_rollouts() in this Comfy client")

    def get_system_status(self, *args, **kwargs):
        """Return system status."""
        try:
            r = self._session.get(f"{self.base}/health", timeout=self.timeout)
            if r.ok:
                return {"status": "ok", "details": r.json() if r.headers.get("content-type","").startswith("application/json") else r.text}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        return {"status": "unknown"}


# Backward-compatible import for existing integrations. New code should import
# WanVideoGenerationClient from novaplan.video_generation.
VideoGenerationClient = WanVideoGenerationClient
