#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
Diagnostic tests for Wan 2.2 video generation, optional object-flow branching,
and model caching validation.

This script validates model caching behavior where:
  1. A cold "full" mode run (WAN + CoTracker) loads all models into VRAM
  2. Subsequent "flow_only" runs should reuse loaded models
  3. Subsequent "generate_only" runs should reuse loaded models

Test sequence:
  1. Cold full run (mode="full") - loads WAN + CoTracker models
  2. Flow-only run (mode="flow_only") - should be fast, reuses CoTracker models
  3. Generate-only run (mode="generate_only") - should be fast, reuses WAN models

Usage from the repository root:

  python remote_video_generation/tests/test_video_generation_server.py --server http://localhost:7000 \
      --data_dir example_data/color_sorting

The script prints detailed server timing and node breakdowns to verify:
  - Step 1 shows model loading overhead
  - Steps 2 & 3 show fast execution without reloading
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from novaplan.cli_args import http_url, positive_int  # noqa: E402


def _decode_video_bytes_to_numpy(mp4_bytes: bytes) -> np.ndarray:
    """Decode MP4 bytes to numpy array (T, H, W, 3) RGB."""
    vid = iio.imread(mp4_bytes, extension=".mp4", index=None)
    if vid.ndim != 4 or vid.shape[-1] != 3:
        raise ValueError(f"Unexpected decoded shape {vid.shape}")
    return vid


def load_demo_data(data_dir) -> Tuple[np.ndarray, Path, Optional[np.ndarray], Optional[Path]]:
    """Load demo data for testing. For distributed setup, video may not be needed locally."""
    print(f"Loading demo data from {data_dir}...")

    data_dir_path = Path(data_dir)

    # Try to load first frame image (required for WAN generation)
    first_frame_path = data_dir_path / "start.png"
    if first_frame_path.exists():
        first_frame = iio.imread(first_frame_path)
        print(f"Loaded first frame from {first_frame_path}: shape={first_frame.shape}")
    else:
        # Try to load video and extract first frame
        video_path = data_dir / "video_gen.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Neither start.png nor video_gen.mp4 found in {data_dir}")
        video = iio.imread(video_path, index=None)
        if video.dtype != np.uint8:
            video = video.astype(np.uint8)
        first_frame = video[0]
        print(f"Extracted first frame from {video_path}: shape={first_frame.shape}")
        return first_frame, video_path, video, video_path

    # For flow testing, try to load video
    video_path = data_dir_path / "video_gen.mp4"
    if video_path.exists():
        video = iio.imread(video_path, index=None)
        if video.dtype != np.uint8:
            video = video.astype(np.uint8)
        print(f"Loaded video from {video_path}: shape={video.shape}")
        return first_frame, first_frame_path, video, video_path
    else:
        print("Warning: Video not available locally, flow testing will use generated WAN video")
        return first_frame, first_frame_path, None, None


def np_to_jpeg_b64(frame: np.ndarray, quality: int = 95) -> str:
    """Convert numpy image to JPEG base64 string."""
    import io
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('ascii')


@dataclass
class JobTiming:
    name: str
    wall_time: float
    status: Dict[str, Any]
    video_bytes: Optional[List[bytes]] = None
    flow_image_bytes: Optional[List[bytes]] = None

    def print_summary(self) -> None:
        st = self.status
        timing = st.get("timing") or {}
        print(f"\n=== {self.name} ===")
        print(f"Total wall time (client): {self.wall_time:.2f}s")
        if timing:
            parts = []
            for key in [
                "prepare_s",
                "submit_http_s",
                "comfy_processing_s",
                "gather_metadata_s",
                "fetch_videos_s",
                "store_s",
                "total_s",
            ]:
                v = timing.get(key)
                if v is not None:
                    parts.append(f"{key}={v:.3f}s")
            if parts:
                print("Server timing:", ", ".join(parts))
            node_breakdown = timing.get("node_breakdown") or []
            if node_breakdown:
                print("Node runtimes:")
                for entry in node_breakdown:
                    dur = entry.get("duration")
                    if dur is None:
                        continue
                    cls = entry.get("class_type") or "unknown"
                    nid = entry.get("node_id")
                    print(f"  - {nid}: {dur:.3f}s ({cls})")

            per_video = timing.get("per_video") or []
            if per_video:
                print("Distribution:")
                for i, meta in enumerate(per_video):
                    print(f"  - Video {i+1}: {meta.get('device', 'unknown')}")


def submit_job_raw(server: str, payload: Dict[str, Any]) -> str:
    """Submit a JobRequest-compatible payload to /submit endpoint."""
    submit_url = f"{server.rstrip('/')}/submit"
    r = requests.post(submit_url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    if "job_id" not in data:
        raise RuntimeError(f"Unexpected /submit response: {data}")
    return data["job_id"]


def poll_status(server: str, job_id: str, timeout: float = 3600.0) -> Dict[str, Any]:
    """Poll /status/{job_id} until completed or failed."""
    status_url = f"{server.rstrip('/')}/status/{job_id}"
    t0 = time.time()
    sleep = 0.5
    while True:
        r = requests.get(status_url, timeout=600)
        r.raise_for_status()
        data = r.json()
        st = data.get("status")
        if st in ("completed", "failed"):
            return data
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
        time.sleep(sleep)
        if sleep < 1.0:
            sleep = min(1.0, sleep * 1.5)

def fetch_result_videos(server: str, job_id: str) -> List[np.ndarray]:
    """Fetch result videos by inspecting the job status payload.

    This wrapper returns decoded numpy arrays like before. It relies on the
    helper `fetch_raw_result_bytes` to obtain raw MP4 bytes (either via
    base64 in the status payload or by reading local output files).
    """
    raw = fetch_raw_result_bytes(server, job_id)
    decoded: List[np.ndarray] = []
    for i, vb in enumerate(raw):
        try:
            arr = _decode_video_bytes_to_numpy(vb)
            decoded.append(arr)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  ⚠️ Failed to decode video {i}: {e}")
    return decoded


def fetch_all_results(server: str, job_id: str) -> Tuple[List[bytes], List[bytes]]:
    """Fetch all videos and flow images from the server.
    
    Returns:
        Tuple of (video_bytes_list, flow_image_bytes_list)
    """
    all_url = f"{server.rstrip('/')}/result/{job_id}/all"
    try:
        print(f"  Fetching all results from {all_url}...")
        r = requests.get(all_url, timeout=600)
        r.raise_for_status()
        data = r.json()
        
        videos = []
        flow_images = []
        
        for vid in data.get("videos", []):
            if "data_base64" in vid:
                videos.append(base64.b64decode(vid["data_base64"]))
                print(f"    ✓ Video: {vid.get('filename')} ({vid.get('size_bytes')} bytes)")
        
        for flow in data.get("flow_images", []):
            if "data_base64" in flow:
                flow_images.append(base64.b64decode(flow["data_base64"]))
                print(f"    ✓ Flow: {flow.get('filename')} ({flow.get('size_bytes')} bytes)")
        
        print(f"  ✓ Total retrieved: {len(videos)} videos, {len(flow_images)} flow images")
        return videos, flow_images
        
    except Exception as e:
        print(f"  ⚠️ fetch_all_results failed: {e}")
        return [], []


def fetch_raw_result_bytes(server: str, job_id: str) -> List[bytes]:
    """Return raw MP4 bytes for a job by inspecting /status/{job_id}.

    Preference order:
      1. status JSON fields videos_base64 / video_base64 / mp4_base64 / data
      2. status JSON filename fields (wan_video / video_filename) read from
         common output locations (including NFS path)
    """
    # First try the /result/{job_id}/binary endpoint (should work for our setup)
    binary_url = f"{server.rstrip('/')}/result/{job_id}/binary"
    try:
        r = requests.get(binary_url, timeout=600)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if ctype.startswith("video/mp4"):
            print(f"  ✓ Retrieved video from binary endpoint ({len(r.content)} bytes)")
            return [r.content]
        if ctype.startswith("multipart/mixed"):
            print("  Retrieved multipart video response from binary endpoint")
            boundary = None
            for part in ctype.split(";"):
                if "boundary=" in part:
                    boundary = part.split("boundary=")[1].strip()
                    break
            if not boundary:
                raise RuntimeError("Multipart response missing boundary")
            accumulated = bytearray()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                accumulated.extend(chunk)
            parts = accumulated.split(f"--{boundary}".encode("utf-8"))
            videos: List[bytes] = []
            for part in parts[1:-1]:
                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                video_data = part[header_end + 4 :]
                if video_data.endswith(b"\r\n"):
                    video_data = video_data[:-2]
                videos.append(bytes(video_data))
            if videos:
                print(f"  ✓ Retrieved {len(videos)} videos from multipart response")
                return videos
        print(f"  ⚠️ Binary endpoint returned unexpected content-type: {ctype}")
    except Exception as e:
        print(f"  ⚠️ Binary endpoint failed: {e}")
        # Fall back to other methods
        pass

    # Next try the JSON /result/{job_id} endpoint
    try:
        rj = requests.get(f"{server.rstrip('/')}/result/{job_id}", timeout=600)
        rj.raise_for_status()
        if rj.headers.get("Content-Type", "").startswith("application/json"):
            data = rj.json()
            if "videos_base64" in data:
                return [base64.b64decode(b64) for b64 in data["videos_base64"]]
            b64 = data.get("video_base64") or data.get("mp4_base64") or data.get("data")
            if b64:
                return [base64.b64decode(b64)]
    except Exception:
        pass

    # Fallback: inspect job status and try binary endpoint again with status info
    status_url = f"{server.rstrip('/')}/status/{job_id}"
    r = requests.get(status_url, timeout=600)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        if "videos_base64" in data:
            return [base64.b64decode(b64) for b64 in data["videos_base64"]]
        b64 = data.get("video_base64") or data.get("mp4_base64") or data.get("data")
        if b64:
            return [base64.b64decode(b64)]

        # For distributed setup, we can't read local files on client
        # The binary endpoint should have worked, so this indicates a server issue
        wan_video = data.get("wan_video")
        if wan_video:
            print(f"  Server reports wan_video: {wan_video}, but binary endpoint failed")

    raise RuntimeError("Unable to retrieve binary result for job")


def build_generate_only_payload(
    jpeg_b64: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """WAN2.2 generation-only payload (mode='generate_only')."""
    payload: Dict[str, Any] = {
        "mode": "generate_only",
        "prompt": args.prompt,
        "first_frame_base64": jpeg_b64,
        "fast_mode": args.fast_mode,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": args.num_frames,
        "num_videos": args.num_videos,
    }
    return payload


def build_flow_only_payload(
    video_data: bytes,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """CoTracker flow-only payload (mode='flow_only')."""
    payload: Dict[str, Any] = {
        "mode": "flow_only",
        "video_base64": base64.b64encode(video_data).decode('ascii'),
        "mask_prompt": args.mask_prompt,
        "num_videos": args.num_videos,  # Use same num_videos for distributed flow
    }
    return payload


def build_full_payload(
    jpeg_b64: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Full WAN2.2 + CoTracker payload (mode='full').

    WAN generates the video, then CoTracker analyzes it in sequence.
    """
    payload: Dict[str, Any] = {
        "mode": "full",
        "prompt": args.prompt,
        "first_frame_base64": jpeg_b64,
        "fast_mode": args.fast_mode,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": args.num_frames,
        "num_videos": args.num_videos,
        "mask_prompt": args.mask_prompt,
    }
    return payload

def run_single_test(
    name: str,
    payload: Dict[str, Any],
    server: str,
    fetch_result: bool = True,
    output_dir: Optional[str] = None,
) -> JobTiming:
    """Run a single test job and return timing information."""
    t0 = time.time()
    job_id = submit_job_raw(server, payload)
    status = poll_status(server, job_id)
    wall = time.time() - t0

    # Initialize for return value
    video_bytes: Optional[List[bytes]] = None
    flow_image_bytes: Optional[List[bytes]] = None

    if fetch_result:
        # Ensure output dir exists
        out_dir = Path(output_dir or "runs/verification/video_generation")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use the new fetch_all_results to get both videos and flow images
        try:
            videos_raw, flows_raw = fetch_all_results(server, job_id)
            video_bytes = videos_raw
            flow_image_bytes = flows_raw
        except Exception as e:
            print(f"  ⚠️ Could not fetch all results: {e}")
            videos_raw = []
            flows_raw = []
            video_bytes = []
            flow_image_bytes = []

        # Save all videos
        vids = []
        for i, raw in enumerate(videos_raw):
            mp4_path = out_dir / f"{job_id}_video_{i+1}.mp4"
            try:
                mp4_path.write_bytes(raw)
                print(f"  💾 Saved video to {mp4_path}")
            except Exception as e:
                print(f"  ⚠️ Failed to save video bytes to {mp4_path}: {e}")
            try:
                arr = _decode_video_bytes_to_numpy(raw)
                vids.append(arr)
            except Exception as e:
                print(f"  ⚠️ Failed to decode saved video {i}: {e}")

        if vids:
            print(f"  📹 Received {len(vids)} video(s), first shape={vids[0].shape}")

        # Save all flow images
        for i, flow_raw in enumerate(flows_raw):
            flow_path = out_dir / f"{job_id}_flow_{i+1}.png"
            try:
                flow_path.write_bytes(flow_raw)
                print(f"  💾 Saved flow image to {flow_path}")
            except Exception as e:
                print(f"  ⚠️ Failed to save flow image to {flow_path}: {e}")

        if flows_raw:
            print(f"  🌊 Received {len(flows_raw)} flow image(s)")

    jt = JobTiming(name=name, wall_time=wall, status=status, video_bytes=video_bytes, flow_image_bytes=flow_image_bytes)
    jt.print_summary()
    return jt


def run_wan_and_flow_tests(
    args: argparse.Namespace,
    first_frame: np.ndarray,
    first_frame_path: Path,
    video: Optional[np.ndarray],
    video_path: Optional[Path],
) -> None:
    """
    Progressive test sequence to validate model caching:

    1. Full mode (WAN + CoTracker) - cold run, loads all models
    2. Flow-only mode (CoTracker) - warm run, should reuse models
    3. Generate-only mode (WAN) - warm run, should reuse models

    This validates that models remain resident in VRAM after initial load.
    """
    # Prepare first frame encoding
    jpeg_b64 = np_to_jpeg_b64(first_frame, quality=95)

    print("\n" + "=" * 80)
    print("MODEL CACHING VALIDATION TEST")
    print("=" * 80)
    print("Test sequence:")
    print("  1. FULL mode (cold) - loads WAN + CoTracker models")
    if not args.full_only:
        print("  2. FLOW_ONLY mode - should reuse CoTracker models (fast)")
    print("=" * 80)

    # Step 1: Full mode (cold run) - loads all models
    # print("\n=== STEP 1: Full (WAN + CoTracker) - COLD ===")
    full_payload = build_full_payload(jpeg_b64, args)
    result1 = run_single_test(
        name="Full (WAN + CoTracker) COLD",
        payload=full_payload,
        server=args.server,
        fetch_result=True,
        output_dir=args.output_dir,
    )
    if not result1.flow_image_bytes:
        raise RuntimeError(
            "WAN full-mode contract failed: the server returned no inline CoTracker3 flow image. "
            "Restart the current video_generation_server.py and inspect its ComfyUI worker log."
        )
    if len(result1.flow_image_bytes) != len(result1.video_bytes or []):
        raise RuntimeError(
            "WAN full-mode contract failed: expected one inline CoTracker3 flow image per video, "
            f"received {len(result1.flow_image_bytes)} flow image(s) for "
            f"{len(result1.video_bytes or [])} video(s)."
        )

    if args.full_only:
        print(
            f"Warm-up complete: {len(result1.video_bytes or [])} worker pipeline(s) "
            "returned WAN video and inline flow outputs."
        )
        return

    # # Get the generated video data for flow-only test
    wan_video_bytes = result1.video_bytes
    if wan_video_bytes and len(wan_video_bytes) > 0:
        print(f"✓ Using generated WAN video ({len(wan_video_bytes[0])} bytes) for flow test")
        flow_video_data = wan_video_bytes[0]
    elif video_path is not None and video_path.exists():
        print("⚠️  No WAN video retrieved, using local video file for flow test")
        # Read the original video file as bytes
        try:
            with open(video_path, "rb") as f:
                flow_video_data = f.read()
            print(f"✓ Read original video file ({len(flow_video_data)} bytes)")
        except Exception as e:
            print(f"❌ Failed to read original video file: {e}")
            return
    else:
        print("❌ No video data available for flow test")
        return

    print(f"Video data prepared ({len(flow_video_data)} bytes), proceeding to Step 2...")

    # Step 2: Flow-only mode (should be warm)
    print("\n=== STEP 2: Flow-only (CoTracker) - WARM ===")
    flow_payload = build_flow_only_payload(flow_video_data, args)
    print(f"Flow payload created with video data ({len(flow_video_data)} bytes)")
    result2 = run_single_test(
        name="Flow-only (CoTracker) WARM",
        payload=flow_payload,
        server=args.server,
        fetch_result=True,
        output_dir=args.output_dir,
    )
    print(f"Step 2 completed with status: {result2.status.get('status')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Test Wan video generation server and model caching.")
    p.add_argument(
        "--server",
        type=http_url,
        default=os.environ.get(
            "NOVAPLAN_VIDEO_SERVER_URL", "http://127.0.0.1:7000"
        ),
        help="Base URL of video_generation_server.py (e.g. http://localhost:7000)",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="example_data/color_sorting/step_001",
        help="Path to example data directory with start.png and optional video_gen.mp4.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/verification/video_generation"),
        help="Downloaded test artifacts (default: runs/verification/video_generation).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="A human hand reaches to the yellow cube on the left side of the tabletop in the scene, grasps it from the top, lifts it clear of the surface, drives it straight above the yellow cup opening, lowers it down into the cup until it is fully seated inside, and releases without contacting the rim. The human hand should try its best not to occlude the yellow cube during the action and keep the yellow cube fully in frame. The hand fully exits the scene after the action is done and the whole scene stays still.",
        help="WAN2.2 text prompt.",
    )
    p.add_argument(
        "--mask_prompt",
        type=str,
        default="yellow cube",
        help="SAM3 mask prompt.",
    )
    p.add_argument("--num_frames", type=positive_int, default=41, help="Number of frames to generate.")
    p.add_argument("--width", type=positive_int, default=1280, help="Video width.")
    p.add_argument("--height", type=positive_int, default=720, help="Video height.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--num_videos", type=positive_int, default=8, help="Number of videos to generate per job (batch_size).")
    p.add_argument(
        "--full_only",
        action="store_true",
        help="Run only the full WAN plus inline-flow batch, for worker warm-up.",
    )
    p.add_argument(
        "--fast_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast 4-step LoRA inference; pass --no-fast_mode for 20-step inference.",
    )

    args = p.parse_args()

    data_dir_path = Path(args.data_dir).resolve()
    first_frame, first_frame_path, video, video_path = load_demo_data(data_dir_path)

    run_wan_and_flow_tests(args, first_frame, first_frame_path, video, video_path)

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE - Inspect node timings above")
    print("=" * 80)
    print("Expected behavior:")
    print()
    print("STEP 1 (Cold Full mode):")
    print("  - Should show model loading overhead (UNETLoader, CLIPLoader, VAELoader)")
    print("  - Both WAN and CoTracker nodes execute")
    print("  - Slower execution time")
    print()
    if not args.full_only:
        print("STEP 2 (Warm Flow-only mode):")
        print("  - Should be FASTER than Step 1 (CoTracker models cached)")
        print("  - Should NOT show WAN generation nodes (KSampler, VAEDecode, CreateVideo)")
        print("  - Should ONLY show CoTracker nodes (LoadVideo, Sam3VideoNode, CoTracker3Node)")
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
