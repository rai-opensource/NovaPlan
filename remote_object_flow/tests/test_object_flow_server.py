#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Workstation diagnostic for the remote NovaPlan object-flow server.

Expected example layout:

    example_data/color_sorting/
      intrinsics.json
      start.png
      start_depth.png
      step_002/
        rgb_video_16fps.mp4
        start_depth.png
        config.json

Run the production fast path:

    python remote_object_flow/tests/test_object_flow_server.py --server http://localhost:7001 --example_dir example_data/color_sorting --step step_002 --mask_prompt "yellow cube"

Run with full debug artifacts:

    python remote_object_flow/tests/test_object_flow_server.py --server http://localhost:7001 --example_dir example_data/color_sorting --step step_002 --mask_prompt "yellow cube" --debug_artifacts

Use `--dry_run` to check local inputs without contacting the server.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = SCRIPT_DIR.parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from novaplan.cli_args import (  # noqa: E402
    http_url,
    nonnegative_float,
    nonnegative_int,
    positive_float,
    positive_int,
    rotation_degrees,
    unit_interval,
)

DEFAULT_SERVER = os.environ.get(
    "NOVAPLAN_OBJECT_FLOW_SERVER_URL",
    "http://127.0.0.1:7001",
)
DEFAULT_EXAMPLE_DIR = RELEASE_ROOT / "example_data" / "color_sorting"
DEFAULT_VIDEO_DIR = RELEASE_ROOT / "runs" / "verification" / "video_generation"
DEFAULT_OUTPUT_DIR = RELEASE_ROOT / "runs" / "verification" / "object_flow"
DEFAULT_DEPTH_FACTOR = 0.001
DEFAULT_STEP = "step_002"
VIDEO_NAME = "rgb_video_16fps.mp4"
INTRINSICS_KIND = "depth"
START_INDEX = 0
TARGET_SIZE = 518
DEPTH_MODEL = "moge2"
CALIBRATION_MODE = "affine"
TAPIP3D_CHECKPOINT = "checkpoints/tapip3d_final.pth"
RESOLUTION_FACTOR = 2
QUERY_GRID_SIZE = 32
VIS_THRESHOLD = 0.9
TAPIP3D_NODE = "vanilla"
REQUEST_TIMEOUT_S = 600.0
JOB_TIMEOUT_S = 3600.0
POLL_S = 2.0
RESULT_FETCH_RETRIES = 4
RESULT_FETCH_RETRY_S = 3.0
DEFAULT_DEBUG_ARTIFACTS = os.environ.get("NOVAPLAN_DEBUG_FLOW_ARTIFACTS", "0").lower() in {"1", "true", "yes", "on"}

CORE_ARRAY_ENDPOINTS = (
    "coords_3d",
    "visibilities",
)

DEBUG_ARRAY_ENDPOINTS = (
    "query_points",
    "depths",
    "video_npy",
    "intrinsics",
    "extrinsics",
    "mask_scaled",
)


@dataclass(frozen=True)
class VideoSelection:
    path: Path
    source: str


@dataclass(frozen=True)
class ResolvedInputs:
    video: Path
    video_source: str
    depth_prior: Path
    intrinsics: list[float]
    intrinsics_source: str
    depth_factor: float
    depth_factor_source: str


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_intrinsics(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 4:
        raise ValueError("camera intrinsics must be four comma-separated values: fx,fy,cx,cy")
    if not all(np.isfinite(values)):
        raise ValueError(f"camera intrinsics must be finite: {values}")
    return values


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def newest_mp4(video_dir: Path) -> Path | None:
    if not video_dir.exists():
        return None
    candidates = sorted(video_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resolve_video(args: argparse.Namespace, example_dir: Path) -> VideoSelection:
    if args.video:
        video = resolve_path(args.video)
        if not video.exists():
            raise FileNotFoundError(f"Input video does not exist: {video}")
        return VideoSelection(video, "cli --video")

    step_video = example_dir / args.step / args.video_name
    if step_video.exists():
        return VideoSelection(step_video, f"{args.step}/{args.video_name}")

    video_dir = resolve_path(args.video_dir)
    generated_video = newest_mp4(video_dir)
    if generated_video:
        return VideoSelection(generated_video, f"newest mp4 in {video_dir}")

    raise FileNotFoundError(
        "Could not find an input video.\n"
        f"Expected {step_video}, or an MP4 under {video_dir}, or pass --video /path/to/video.mp4."
    )


def resolve_depth_prior(args: argparse.Namespace, example_dir: Path, video: VideoSelection) -> Path:
    if args.depth_prior:
        depth_prior = resolve_path(args.depth_prior)
        if not depth_prior.exists():
            raise FileNotFoundError(f"Depth prior does not exist: {depth_prior}")
        return depth_prior

    step_dir = example_dir / args.step
    root_candidates = [example_dir / "start_depth.png"]
    step_candidates = [step_dir / "start_depth.png"]
    candidates = step_candidates + root_candidates if video.path.parent == step_dir else root_candidates + step_candidates
    depth_prior = first_existing(candidates)
    if depth_prior:
        return depth_prior

    expected = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find a depth prior PNG aligned with frame 0 of the selected video.\n"
        f"Looked for:\n{expected}\n"
        "Pass --depth_prior /path/to/start_depth.png if your data uses a different name."
    )


def intrinsics_from_dict(data: dict[str, Any], kind: str) -> list[float]:
    sections: list[Any] = []
    camera_block = data.get("camera_intrinsics_and_extrinsics")
    if isinstance(camera_block, dict):
        sections.extend([camera_block.get(f"{kind}_intrinsics"), camera_block.get("intrinsics")])
    sections.extend([data.get(f"{kind}_intrinsics"), data.get("intrinsics"), data])

    for section in sections:
        if isinstance(section, dict) and all(key in section for key in ("fx", "fy", "ppx", "ppy")):
            return [float(section["fx"]), float(section["fy"]), float(section["ppx"]), float(section["ppy"])]
        if isinstance(section, dict) and all(key in section for key in ("fx", "fy", "cx", "cy")):
            return [float(section["fx"]), float(section["fy"]), float(section["cx"]), float(section["cy"])]
        if isinstance(section, list) and len(section) == 4:
            return [float(value) for value in section]

    raise ValueError(f"Could not find {kind} camera intrinsics in JSON")


def resolve_intrinsics(args: argparse.Namespace, example_dir: Path) -> tuple[list[float], str, Path | None]:
    if args.intrinsics:
        return parse_intrinsics(args.intrinsics), "cli --intrinsics", None

    if args.intrinsics_json:
        intrinsics_path = resolve_path(args.intrinsics_json)
        if not intrinsics_path.exists():
            raise FileNotFoundError(f"Camera intrinsics JSON does not exist: {intrinsics_path}")
    else:
        intrinsics_path = first_existing(
            [
                example_dir / args.step / "config.json",
                example_dir / "intrinsics.json",
                example_dir / "config.json",
            ]
        )
        if intrinsics_path is None:
            raise FileNotFoundError(
                "Could not find camera intrinsics.\n"
                f"Expected {example_dir / args.step / 'config.json'}, {example_dir / 'intrinsics.json'}, "
                f"or {example_dir / 'config.json'}.\n"
                "Pass --intrinsics_json /path/to/intrinsics.json or --intrinsics fx,fy,cx,cy."
            )

    data = load_json(intrinsics_path)
    intrinsics = intrinsics_from_dict(data, args.intrinsics_kind)
    return intrinsics, f"{intrinsics_path} ({args.intrinsics_kind}_intrinsics)", intrinsics_path


def depth_factor_from_json(path: Path | None) -> tuple[float | None, str | None]:
    if path is None:
        return None, None

    data = load_json(path)
    camera_block = data.get("camera_intrinsics_and_extrinsics")
    if isinstance(camera_block, dict) and camera_block.get("depth_scale") is not None:
        return float(camera_block["depth_scale"]), f"{path} (camera_intrinsics_and_extrinsics.depth_scale)"
    if data.get("depth_scale") is not None:
        return float(data["depth_scale"]), f"{path} (depth_scale)"
    return None, None


def resolve_inputs(args: argparse.Namespace) -> ResolvedInputs:
    example_dir = resolve_path(args.example_dir)
    video = resolve_video(args, example_dir)
    depth_prior = resolve_depth_prior(args, example_dir, video)
    intrinsics, intrinsics_source, intrinsics_path = resolve_intrinsics(args, example_dir)

    if args.depth_factor is not None:
        depth_factor = args.depth_factor
        depth_factor_source = "cli --depth_factor"
    else:
        depth_factor, depth_factor_source = depth_factor_from_json(intrinsics_path)
        if depth_factor is None:
            depth_factor = DEFAULT_DEPTH_FACTOR
            depth_factor_source = f"default {DEFAULT_DEPTH_FACTOR}"

    return ResolvedInputs(
        video=video.path,
        video_source=video.source,
        depth_prior=depth_prior,
        intrinsics=intrinsics,
        intrinsics_source=intrinsics_source,
        depth_factor=float(depth_factor),
        depth_factor_source=depth_factor_source,
    )


async def submit_job(client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any]) -> str:
    response = await client.post(endpoint, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"{endpoint} failed with HTTP {response.status_code}:\n{response.text}")
    data = response.json()
    if "job_id" not in data:
        raise RuntimeError(f"Unexpected {endpoint} response: {data}")
    return str(data["job_id"])


async def wait_for_job(
    client: httpx.AsyncClient,
    job_id: str,
    label: str,
    timeout_s: float,
    poll_s: float,
) -> dict[str, Any]:
    start = time.time()
    next_log_s = 0.0

    while True:
        response = await client.get(f"/status/{job_id}")
        if response.status_code != 200:
            raise RuntimeError(f"Status check failed for {job_id}: HTTP {response.status_code} {response.text}")

        data = response.json()
        status = data.get("status")
        elapsed = time.time() - start

        if status == "completed":
            print(f"{label} completed: job_id={job_id}, elapsed_s={elapsed:.1f}")
            return data
        if status == "failed":
            raise RuntimeError(f"{label} failed: {data.get('error', data)}")
        if elapsed > timeout_s:
            raise TimeoutError(f"{label} timed out after {timeout_s:.0f}s: job_id={job_id}")

        if elapsed >= next_log_s:
            print(f"{label} status={status}, elapsed_s={elapsed:.0f}")
            next_log_s += 30.0

        await asyncio.sleep(poll_s)


async def fetch_result(
    client: httpx.AsyncClient,
    job_id: str,
    endpoint: str,
    *,
    retries: int = RESULT_FETCH_RETRIES,
    retry_s: float = RESULT_FETCH_RETRY_S,
) -> bytes | None:
    last_error: httpx.HTTPError | None = None
    attempts = max(1, int(retries) + 1)
    for attempt in range(1, attempts + 1):
        try:
            response = await client.get(f"/result/{job_id}/{endpoint}", headers={"Connection": "close"})
            if response.status_code == 404 or not response.content:
                return None
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download {endpoint} for {job_id}: HTTP {response.status_code} {response.text}"
                )
            return response.content
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                f"download {endpoint} for {job_id} failed on attempt {attempt}/{attempts}: {exc}; retrying",
                flush=True,
            )
            await asyncio.sleep(float(retry_s))
    raise RuntimeError(
        f"Failed to download {endpoint} for {job_id} after {attempts} attempts: {last_error}"
    ) from last_error


async def save_result(
    client: httpx.AsyncClient,
    job_id: str,
    endpoint: str,
    output_path: Path,
    *,
    retries: int = RESULT_FETCH_RETRIES,
    retry_s: float = RESULT_FETCH_RETRY_S,
) -> Path | None:
    content = await fetch_result(client, job_id, endpoint, retries=retries, retry_s=retry_s)
    if content is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


async def fetch_npy_result(
    client: httpx.AsyncClient,
    job_id: str,
    endpoint: str,
    *,
    retries: int = RESULT_FETCH_RETRIES,
    retry_s: float = RESULT_FETCH_RETRY_S,
) -> np.ndarray | None:
    content = await fetch_result(client, job_id, endpoint, retries=retries, retry_s=retry_s)
    if content is None:
        return None
    return np.load(io.BytesIO(content))


def save_npz_outputs(arrays: dict[str, np.ndarray], video_stem: str, output_dir: Path) -> dict[str, Path]:
    missing = [name for name in CORE_ARRAY_ENDPOINTS + DEBUG_ARRAY_ENDPOINTS if name not in arrays]
    if "mask_orig" not in arrays:
        missing.append("mask_orig")
    if missing:
        raise RuntimeError(f"Missing arrays required for compact object-flow output: {', '.join(missing)}")

    video = arrays["video_npy"]
    if video.ndim == 4 and video.shape[-1] == 3:
        video = np.transpose(video, (0, 3, 1, 2))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    npz_path = output_dir / f"tapip3d_output_{video_stem}.npz"
    np.savez(
        npz_path,
        coords=arrays["coords_3d"],
        depths=arrays["depths"],
        visibs=arrays["visibilities"],
        query_points=arrays["query_points"],
        intrinsics=arrays["intrinsics"],
        extrinsics=arrays["extrinsics"],
        video=video,
    )
    saved["tapip3d_npz"] = npz_path

    mask_path = output_dir / f"segmentation_masks_full_{video_stem}.npz"
    np.savez(mask_path, masks=arrays["mask_orig"])
    saved["segmentation_masks_full"] = mask_path

    mask_scaled_path = output_dir / f"segmentation_masks_scaled_{video_stem}.npz"
    np.savez(mask_scaled_path, masks=arrays["mask_scaled"])
    saved["segmentation_masks_scaled"] = mask_scaled_path

    return saved


def _resize_depth_stack(depths: np.ndarray, height: int, width: int) -> np.ndarray:
    if depths.ndim != 3:
        raise ValueError(f"Expected depth stack [T,H,W], got {depths.shape}")
    if depths.shape[1] == height and depths.shape[2] == width:
        return depths.astype(np.float32, copy=False)
    return np.stack(
        [cv2.resize(frame.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) for frame in depths],
        axis=0,
    ).astype(np.float32)


def copy_step_inputs(example_dir: Path, step: str, output_dir: Path) -> dict[str, Path]:
    step_dir = example_dir / step
    copied: dict[str, Path] = {}
    for name in ("start.png", "start_depth.png"):
        src = first_existing([step_dir / name, example_dir / name])
        if src is None:
            continue
        dst = output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[name] = dst
    return copied


def save_legacy_step_outputs(
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    *,
    mask_stem: str,
) -> dict[str, Path]:
    required = (
        "coords_3d",
        "visibilities",
        "query_points",
        "depths",
        "video_npy",
        "intrinsics",
        "extrinsics",
        "mask_orig",
        "mask_scaled",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise RuntimeError(f"Missing arrays required for legacy step output: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    video = arrays["video_npy"]
    if video.ndim == 4 and video.shape[-1] == 3:
        video = np.transpose(video, (0, 3, 1, 2))

    mask_orig = arrays["mask_orig"].astype(np.float32, copy=False)
    mask_scaled = arrays["mask_scaled"].astype(np.float32, copy=False)
    depths = arrays["depths"].astype(np.float32, copy=False)
    depths_video_res = _resize_depth_stack(depths, mask_orig.shape[1], mask_orig.shape[2])

    tapip_path = output_dir / "tapip3d_output.npz"
    np.savez(
        tapip_path,
        coords=arrays["coords_3d"].astype(np.float32, copy=False),
        depths=depths,
        visibilities=arrays["visibilities"],
        query_points=arrays["query_points"].astype(np.float32, copy=False),
        intrinsics=arrays["intrinsics"].astype(np.float32, copy=False),
        extrinsics=arrays["extrinsics"].astype(np.float32, copy=False),
        video=video.astype(np.float32, copy=False),
        depths_video_res=depths_video_res,
    )
    saved["tapip3d_output"] = tapip_path

    depth_est_path = output_dir / "depth_est.npy"
    np.save(depth_est_path, depths_video_res)
    saved["depth_est"] = depth_est_path

    mask_full_path = output_dir / f"segmentation_masks_full_{mask_stem}.npz"
    np.savez(mask_full_path, masks=mask_orig)
    saved["segmentation_masks_full"] = mask_full_path

    mask_scaled_path = output_dir / f"segmentation_masks_{mask_stem}.npz"
    np.savez(mask_scaled_path, masks=mask_scaled)
    saved["segmentation_masks_scaled"] = mask_scaled_path

    return saved


def format_seconds(value: Any) -> str:
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "n/a"


def print_test_summary(
    title: str,
    module_timings: list[tuple[str, float, dict[str, Any]]],
    saved_paths: dict[str, Path],
    total_wall_s: float,
) -> None:
    print("\nTest summary")
    print(f"test: {title}")
    print("timing:")
    for name, wall_s, status in module_timings:
        timing = status.get("timing") or {}
        print(f"  {name}: client_wall={wall_s:.2f}s, server_total={format_seconds(timing.get('total_s'))}")
        for key in ("prepare_s", "total_s"):
            if key in timing:
                print(f"    {key}: {format_seconds(timing.get(key))}")
        object_flow_eval = status.get("object_flow_evaluation")
        if object_flow_eval:
            print(f"    object_flow_evaluation: {object_flow_eval.get('reason', object_flow_eval)}")
    print(f"total: client_wall={total_wall_s:.2f}s")
    print("saved:")
    if saved_paths:
        for label in sorted(saved_paths):
            print(f"  {label}: {saved_paths[label]}")
    else:
        print("  none")


def print_inputs(
    args: argparse.Namespace,
    inputs: ResolvedInputs,
    output_dir: Path,
    debug_output_dir: Path,
) -> None:
    effective_auxiliary_arrays = args.debug_artifacts or args.auxiliary_arrays or args.legacy_step_layout
    effective_depth_video = args.debug_artifacts or args.depth_debug_video
    effective_sam3_video = args.debug_artifacts or args.sam3_debug_video
    effective_tapip3d_video = args.debug_artifacts or args.tapip3d_debug_video
    effective_flow_image = not args.no_flow_image

    print(f"server: {args.server}")
    print(f"video: {inputs.video}")
    print(f"video_source: {inputs.video_source}")
    print(f"depth_prior: {inputs.depth_prior}")
    if args.depth_source_job_id:
        print(f"depth_source_job_id: {args.depth_source_job_id}")
    print(f"intrinsics: {inputs.intrinsics}")
    print(f"intrinsics_source: {inputs.intrinsics_source}")
    print(f"depth_factor: {inputs.depth_factor}")
    print(f"depth_factor_source: {inputs.depth_factor_source}")
    print(f"cvd_enabled: {not args.disable_cvd}")
    print(f"cvd_w_grad: {args.cvd_w_grad}")
    print(f"cvd_w_normal: {args.cvd_w_normal}")
    print(f"cvd_resolution: {args.cvd_resolution}")
    print(f"mask_prompt: {args.mask_prompt}")
    print(f"auxiliary_arrays: {effective_auxiliary_arrays}")
    print(f"depth_video: {effective_depth_video}")
    print(f"sam3_debug_video: {effective_sam3_video}")
    print(f"tapip3d_debug_video: {effective_tapip3d_video}")
    print(f"flow_image: {effective_flow_image}")
    print(f"output_dir: {output_dir}")
    print(f"debug_output_dir: {debug_output_dir}")


async def run(args: argparse.Namespace) -> None:
    total_start = time.time()
    inputs = resolve_inputs(args)
    output_dir = resolve_path(args.output_dir)
    if args.debug_output_dir is not None:
        debug_output_dir = resolve_path(args.debug_output_dir)
    elif args.legacy_step_layout:
        debug_output_dir = output_dir / "debug"
    else:
        debug_output_dir = output_dir
    print_inputs(args, inputs, output_dir, debug_output_dir)

    effective_auxiliary_arrays = args.debug_artifacts or args.auxiliary_arrays or args.legacy_step_layout
    effective_depth_video = args.debug_artifacts or args.depth_debug_video
    effective_sam3_video = args.debug_artifacts or args.sam3_debug_video
    effective_tapip3d_video = args.debug_artifacts or args.tapip3d_debug_video
    effective_flow_image = not args.no_flow_image

    if args.dry_run:
        print("dry_run: local inputs resolved; no server request sent")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    video_base64 = encode_file(inputs.video)
    depth_prior_base64 = encode_file(inputs.depth_prior)

    depth_request = {
        "task": "depth_estimation",
        "video_mp4_base64": video_base64,
        "depth_prior_base64": depth_prior_base64,
        "intrinsics": inputs.intrinsics,
        "target_size": args.target_size,
        "depth_factor": inputs.depth_factor,
        "depth_model": args.depth_model,
        "calibration_mode": args.calibration_mode,
        "return_depth_video": effective_depth_video,
        "enable_cvd": not args.disable_cvd,
        "cvd_w_grad": args.cvd_w_grad,
        "cvd_w_normal": args.cvd_w_normal,
        "cvd_resolution": args.cvd_resolution,
    }

    flow_request = {
        "task": "flow_extraction",
        "video_mp4_base64": video_base64,
        "start_index": args.start_index,
        "intrinsics": inputs.intrinsics,
        "mask_prompt": args.mask_prompt,
        "checkpoint": args.checkpoint,
        "resolution_factor": args.resolution_factor,
        "query_grid_size": args.query_grid_size,
        "vis_threshold": args.vis_threshold,
        "flow_switch_theta_deg": args.flow_switch_theta_deg,
        "tapip3d_node": args.tapip3d_node,
        "return_debug_media": effective_tapip3d_video,
        "return_sam3_video": effective_sam3_video,
        "return_flow_image": effective_flow_image,
        "return_auxiliary_arrays": effective_auxiliary_arrays,
    }

    async with httpx.AsyncClient(base_url=args.server.rstrip("/"), timeout=args.request_timeout_s) as client:
        health = await client.get("/docs")
        if health.status_code not in (200, 404):
            raise RuntimeError(f"Server health check returned HTTP {health.status_code}: {health.text}")
        fetch_kwargs = {
            "retries": args.result_fetch_retries,
            "retry_s": args.result_fetch_retry_s,
        }

        depth_video = None
        if args.depth_source_job_id:
            depth_job_id = args.depth_source_job_id
            depth_status = {"timing": {}, "source": "cli --depth_source_job_id"}
            depth_wall_s = 0.0
            print(f"reusing cached depth estimation job: {depth_job_id}")
        else:
            print("submitting depth estimation job")
            depth_start = time.time()
            depth_job_id = await submit_job(client, "/jobs/depth_estimation", depth_request)
            depth_status = await wait_for_job(client, depth_job_id, "depth estimation", args.job_timeout_s, args.poll_s)
            depth_wall_s = time.time() - depth_start

        if effective_depth_video:
            depth_video_path = (
                debug_output_dir / f"{depth_job_id}_depth_video_{args.legacy_debug_depth_suffix}.mp4"
                if args.legacy_step_layout
                else debug_output_dir / f"depth_video_{inputs.video.stem}.mp4"
            )
            depth_video = await save_result(
                client,
                depth_job_id,
                "depth_video",
                depth_video_path,
                **fetch_kwargs,
            )
            if depth_video is None and not args.depth_source_job_id:
                raise RuntimeError("Depth estimation completed but did not return depth_video")
            if depth_video is None and args.depth_source_job_id:
                print(f"depth_video not available from reused depth job: {depth_job_id}")

        flow_request["depth_source_job_id"] = depth_job_id

        print("submitting object-flow extraction job")
        flow_start = time.time()
        flow_job_id = await submit_job(client, "/jobs/flow_extraction", flow_request)
        flow_status = await wait_for_job(client, flow_job_id, "object-flow extraction", args.job_timeout_s, args.poll_s)
        flow_wall_s = time.time() - flow_start

        flow_png = None
        if effective_flow_image:
            flow_png = await save_result(
                client,
                flow_job_id,
                "flow",
                (
                    debug_output_dir / f"{flow_job_id}_flow_{args.legacy_debug_flow_suffix}.png"
                    if args.legacy_step_layout
                    else debug_output_dir / f"object_flow_{inputs.video.stem}.png"
                ),
                **fetch_kwargs,
            )
            if flow_png is None:
                raise RuntimeError("Object-flow extraction completed but did not return flow PNG")

        sam3_video = None
        tapip3d_video = None
        if effective_sam3_video:
            sam3_video = await save_result(
                client,
                flow_job_id,
                "sam3_video",
                (
                    debug_output_dir / f"{flow_job_id}_sam3_video_segmentation_full.mp4"
                    if args.legacy_step_layout
                    else debug_output_dir / f"sam3_segmentation_{inputs.video.stem}.mp4"
                ),
                **fetch_kwargs,
            )
            if sam3_video is None:
                raise RuntimeError(
                    "Object-flow extraction completed but did not return sam3_video. "
                    "If the server was already running, restart the object-flow server and ComfyUI worker so it uses "
                    "the updated return_sam3_video-only code; older servers ignore this lean SAM3-video request."
                )

        if effective_tapip3d_video:
            tapip3d_video_path = (
                debug_output_dir / f"{flow_job_id}_tapip3d_video_{args.legacy_debug_flow_suffix}.mp4"
                if args.legacy_step_layout
                else debug_output_dir / f"tapip3d_tracking_{inputs.video.stem}.mp4"
            )
            tapip3d_video = await save_result(
                client,
                flow_job_id,
                "tapip3d_video",
                tapip3d_video_path,
                **fetch_kwargs,
            )
            if tapip3d_video is None:
                tapip3d_video = await save_result(
                    client,
                    flow_job_id,
                    "tracking_video",
                    tapip3d_video_path,
                    **fetch_kwargs,
                )
            if tapip3d_video is None:
                raise RuntimeError("Object-flow extraction completed but did not return tapip3d_video or tracking_video")

        arrays: dict[str, np.ndarray] = {}
        array_endpoints = CORE_ARRAY_ENDPOINTS + (DEBUG_ARRAY_ENDPOINTS if effective_auxiliary_arrays else ())
        for endpoint in array_endpoints:
            array = await fetch_npy_result(client, flow_job_id, endpoint, **fetch_kwargs)
            if array is not None:
                arrays[endpoint] = array
            elif endpoint in CORE_ARRAY_ENDPOINTS:
                raise RuntimeError(f"Object-flow extraction completed but did not return {endpoint}")

        if effective_auxiliary_arrays:
            mask_orig = await fetch_npy_result(client, flow_job_id, "mask_orig", **fetch_kwargs)
            if mask_orig is not None:
                arrays["mask_orig"] = mask_orig

    compact_outputs: dict[str, Path] = {}
    if effective_auxiliary_arrays:
        if args.legacy_step_layout:
            compact_outputs.update(save_legacy_step_outputs(arrays, output_dir, mask_stem=args.legacy_mask_stem))
            compact_outputs.update(copy_step_inputs(resolve_path(args.example_dir), args.step, output_dir))
            if "depths" in arrays and "mask_orig" in arrays:
                debug_depth_path = debug_output_dir / f"{depth_job_id}_raw_depth_{args.legacy_debug_depth_suffix}.npy"
                debug_depth_path.parent.mkdir(parents=True, exist_ok=True)
                depth_video_res = _resize_depth_stack(
                    arrays["depths"].astype(np.float32, copy=False),
                    arrays["mask_orig"].shape[1],
                    arrays["mask_orig"].shape[2],
                )
                np.save(debug_depth_path, depth_video_res)
                compact_outputs["debug_raw_depth"] = debug_depth_path
            if "coords_3d" in arrays:
                debug_coords_path = debug_output_dir / f"{flow_job_id}_coords_3d_{args.legacy_debug_flow_suffix}.npy"
                debug_coords_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(debug_coords_path, arrays["coords_3d"])
                compact_outputs["debug_coords_3d"] = debug_coords_path
        else:
            compact_outputs.update(save_npz_outputs(arrays, inputs.video.stem, output_dir))

    job_ids_path = output_dir / "flow_job_ids.json"
    job_ids_path.write_text(
        json.dumps(
            {
                "depth_job_id": depth_job_id,
                "flow_job_id": flow_job_id,
                "reused_depth_source": bool(args.depth_source_job_id),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    compact_outputs["flow_job_ids"] = job_ids_path

    saved_paths: dict[str, Path | None] = {
        "depth_video": depth_video,
        "object_flow": flow_png,
        "sam3_segmentation": sam3_video,
        "tapip3d_tracking": tapip3d_video,
    }
    saved_paths.update(compact_outputs)
    saved_paths = {name: path for name, path in saved_paths.items() if path is not None}

    print_test_summary(
        "remote_object_flow",
        [
            ("depth_estimation", depth_wall_s, depth_status),
            ("object_flow_extraction", flow_wall_s, flow_status),
        ],
        saved_paths,
        time.time() - total_start,
    )

    flags = sorted(key for key, value in flow_status.items() if key.startswith("has_") and value)
    if flags:
        print("server_result_flags:", ", ".join(flags))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", type=http_url, default=DEFAULT_SERVER, help="Object-flow server URL.")
    parser.add_argument(
        "--example_dir",
        type=Path,
        default=Path(os.environ.get("NOVAPLAN_EXAMPLE_DIR", DEFAULT_EXAMPLE_DIR)),
        help="Example-data folder containing intrinsics and per-step inputs.",
    )
    parser.add_argument("--step", default=os.environ.get("NOVAPLAN_EXAMPLE_STEP", DEFAULT_STEP))
    parser.add_argument("--video_name", default=VIDEO_NAME)
    parser.add_argument("--video", type=Path, help="Specific MP4 to process.")
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path(os.environ.get("NOVAPLAN_VIDEO_TEST_OUTPUT_DIR", DEFAULT_VIDEO_DIR)),
        help="Fallback directory containing videos from the video-generation server test.",
    )
    parser.add_argument("--depth_prior", type=Path, help="First-frame metric depth PNG aligned with the selected video.")
    parser.add_argument("--intrinsics_json", type=Path, help="JSON containing depth/color intrinsics.")
    parser.add_argument("--intrinsics", help="Override camera intrinsics as fx,fy,cx,cy.")
    parser.add_argument("--intrinsics_kind", choices=["depth", "color"], default=INTRINSICS_KIND)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("NOVAPLAN_TEST_RESULTS_DIR", DEFAULT_OUTPUT_DIR)),
        help="Directory for downloaded flow outputs.",
    )
    parser.add_argument(
        "--debug_output_dir",
        type=Path,
        help="Optional separate directory for flow images and depth/SAM3/TAPIP3D debug videos.",
    )
    parser.add_argument("--mask_prompt", default="blue cube", help="Object prompt for SAM3 tracking.")
    parser.add_argument(
        "--depth_source_job_id",
        default=os.environ.get("NOVAPLAN_DEPTH_SOURCE_JOB_ID"),
        help=(
            "Reuse calibrated metric depth cached in a live object-flow server from a previous depth-estimation job. "
            "This skips the depth model and runs only the prompt-specific object-flow extraction."
        ),
    )
    parser.add_argument("--start_index", type=nonnegative_int, default=START_INDEX, help="Depth-video start frame for this flow sub-video.")
    parser.add_argument("--target_size", type=positive_int, default=TARGET_SIZE, help="Depth model target size.")
    parser.add_argument("--depth_factor", type=positive_float, help="Scale raw depth-prior pixels to meters.")
    parser.add_argument(
        "--depth_model",
        choices=("moge2",),
        default=DEPTH_MODEL,
        help="Depth model configured on the object-flow host.",
    )
    parser.add_argument("--calibration_mode", default=CALIBRATION_MODE, choices=["affine"])
    parser.add_argument("--disable_cvd", action="store_true", help="Disable optional CVD refinement; enabled by default.")
    parser.add_argument("--cvd_w_grad", type=nonnegative_float, default=2.0)
    parser.add_argument("--cvd_w_normal", type=nonnegative_float, default=5.0)
    parser.add_argument("--cvd_resolution", type=positive_int, default=196608)
    parser.add_argument("--checkpoint", default=TAPIP3D_CHECKPOINT, help="TAPIP3D checkpoint path on the host.")
    parser.add_argument("--resolution_factor", type=positive_int, default=RESOLUTION_FACTOR)
    parser.add_argument("--query_grid_size", type=positive_int, default=QUERY_GRID_SIZE)
    parser.add_argument("--vis_threshold", type=unit_interval, default=VIS_THRESHOLD)
    parser.add_argument(
        "--flow_switch_theta_deg",
        type=rotation_degrees,
        default=45.0,
    )
    parser.add_argument("--tapip3d_node", default=TAPIP3D_NODE, choices=["vanilla"])
    parser.add_argument("--request_timeout_s", type=positive_float, default=REQUEST_TIMEOUT_S)
    parser.add_argument("--job_timeout_s", type=positive_float, default=JOB_TIMEOUT_S)
    parser.add_argument("--poll_s", type=positive_float, default=POLL_S)
    parser.add_argument(
        "--result_fetch_retries",
        type=nonnegative_int,
        default=RESULT_FETCH_RETRIES,
        help="Retry count for downloading completed flow/depth result artifacts.",
    )
    parser.add_argument(
        "--result_fetch_retry_s",
        type=nonnegative_float,
        default=RESULT_FETCH_RETRY_S,
        help="Sleep seconds between completed-result download retries.",
    )
    parser.add_argument(
        "--debug_artifacts",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG_ARTIFACTS,
        help="Request the full debug bundle: depth video, SAM3 video, TAPIP3D video, flow image, and auxiliary arrays.",
    )
    parser.add_argument(
        "--auxiliary_arrays",
        action="store_true",
        help="Request auxiliary arrays needed for compact/legacy npz outputs without requiring debug videos.",
    )
    parser.add_argument(
        "--depth_debug_video",
        action="store_true",
        help="Request and save the depth visualization video.",
    )
    parser.add_argument(
        "--sam3_debug_video",
        action="store_true",
        help="Request and save the SAM3 segmentation visualization video.",
    )
    parser.add_argument(
        "--tapip3d_debug_video",
        action="store_true",
        help="Request and save the TAPIP3D tracking visualization video.",
    )
    parser.add_argument(
        "--no_flow_image",
        action="store_true",
        help="Do not request or save the static flow visualization PNG.",
    )
    parser.add_argument(
        "--legacy_step_layout",
        action="store_true",
        help=(
            "Save outputs in the legacy per-step layout used by directories such as "
            "example_data/.../test_results_human_hand."
        ),
    )
    parser.add_argument(
        "--legacy_mask_stem",
        default="color_block_720p_test",
        help="Filename stem for legacy segmentation mask npz files.",
    )
    parser.add_argument(
        "--legacy_debug_depth_suffix",
        default="depth_step",
        help="Suffix for legacy depth debug files.",
    )
    parser.add_argument(
        "--legacy_debug_flow_suffix",
        default="red_step",
        help="Suffix for legacy flow debug files.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Resolve inputs and exit without contacting the server.")
    return parser


def main() -> None:
    try:
        asyncio.run(run(build_parser().parse_args()))
    except (FileNotFoundError, TimeoutError, RuntimeError, ValueError, httpx.HTTPError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None


if __name__ == "__main__":
    main()
