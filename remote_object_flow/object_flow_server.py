# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# --- Dependencies ---
# Runtime dependencies are managed by pixi: pixi install -e object-flow-host
"""Serve asynchronous NovaPlan object-flow and metric-depth jobs."""

import uvicorn
import httpx
import asyncio
import aiofiles
import itertools
import uuid
import os
import json
import base64
import time
import shutil
import traceback
import sys
from pathlib import Path
import numpy as np
import io
import glob

try:
    import websockets
except ImportError:
    websockets = None

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional, List, Tuple, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from novaplan.flow_switching import DEFAULT_FLOW_SWITCH_THETA_DEG, evaluate_object_flow

# --- Configuration ---
USE_DISTRIBUTED_MODE = True

API_PORT = int(os.environ.get("PORT", "7001"))
if not 1 <= API_PORT <= 65535:
    raise ValueError("PORT must be between 1 and 65535")
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1").rstrip("/")
MASTER_PORT = int(os.environ.get("COMFYUI_MASTER_PORT", os.environ.get("COMFYUI_START_PORT", "8187")))
if not 1 <= MASTER_PORT <= 65535:
    raise ValueError("COMFYUI_MASTER_PORT must be between 1 and 65535")
COMFYUI_MASTER_URL = f"{COMFYUI_HOST}:{MASTER_PORT}"

NUM_GPUS = int(os.environ.get("COMFYUI_NUM_WORKERS", "1"))
if NUM_GPUS < 1:
    raise ValueError("COMFYUI_NUM_WORKERS must be at least 1")
START_PORT = int(os.environ.get("COMFYUI_START_PORT", str(MASTER_PORT)))
if not 1 <= START_PORT <= 65535 or START_PORT + NUM_GPUS - 1 > 65535:
    raise ValueError("ComfyUI worker ports must be between 1 and 65535")
COMFYUI_URLS = [f"{COMFYUI_HOST}:{START_PORT + i}" for i in range(NUM_GPUS)]

DEFAULT_COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", REPO_ROOT / ".runtime" / "comfyui"))
COMFYUI_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", str(DEFAULT_COMFYUI_DIR / "input"))
COMFYUI_OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", str(DEFAULT_COMFYUI_DIR / "output"))

SAVE_DEBUG_WORKFLOWS = bool(int(os.environ.get("SAVE_DEBUG_WORKFLOWS", "0")))
PREFER_DIRECT_OUTPUT_READ = bool(int(os.environ.get("PREFER_DIRECT_OUTPUT_READ", "1")))
SAVE_RESULT_VIDEOS = bool(int(os.environ.get("SAVE_RESULT_VIDEOS", "0")))
RESULT_OUTPUT_DIR = os.environ.get("RESULT_OUTPUT_DIR", str(DEFAULT_COMFYUI_DIR / "saved_results"))

SAVEVIDEO_FORMAT = "mp4"
SAVEVIDEO_CODEC = "h264"
FLOW_SWITCH_THETA_DEG = float(os.environ.get("FLOW_SWITCH_THETA_DEG", DEFAULT_FLOW_SWITCH_THETA_DEG))
if not 0.0 <= FLOW_SWITCH_THETA_DEG <= 180.0:
    raise ValueError("FLOW_SWITCH_THETA_DEG must be between 0 and 180 inclusive")
VERBOSE_FLOW_LOGS = os.environ.get("NOVAPLAN_VERBOSE_LOGS", "0").lower() in {"1", "true", "yes", "on"}


def _log_debug(*args, **kwargs):
    if VERBOSE_FLOW_LOGS:
        print(*args, **kwargs)


def _log_warning(*args, **kwargs):
    print(*args, **kwargs)


def _log_timing(module: str, total_s: float, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[TIMING] {module}: total={total_s:.2f}s{suffix}", flush=True)

# --- NovaPlan defaults ---
COMFYUI_DIR = str(DEFAULT_COMFYUI_DIR)
NOVAPLAN_DEMO_ROOT = os.environ.get(
    "NOVAPLAN_DEMO_ROOT",
    os.path.join(COMFYUI_DIR, "custom_nodes", "novaplan", "demo_data"),
)
NOVAPLAN_DEMO_DEPTH = os.path.join(NOVAPLAN_DEMO_ROOT, "depth.png")
NOVAPLAN_DEMO_INTRINSICS = os.path.join(NOVAPLAN_DEMO_ROOT, "intrinsics.json")
NOVAPLAN_INTRINSICS_CACHE: Optional[Tuple[float, float, float, float]] = None

# --- Globals ---
app = FastAPI(title="NovaPlan 3D Object-Flow Server")
client = httpx.AsyncClient(timeout=1800.0)

jobs: Dict[str, Dict[str, Any]] = {}
job_lock = asyncio.Lock()
url_cycler = itertools.cycle(COMFYUI_URLS)
instance_lock = asyncio.Lock()
rr_index = 0
rr_lock = asyncio.Lock()

# Cache for depth: job_id -> worker-local cache metadata.  ``worker_url`` is
# required because each ComfyUI worker is a separate process with its own
# in-memory depth cache.
depth_file_cache: Dict[str, Dict[str, Optional[str]]] = {}

MODEL_PINNING_ENABLED = bool(int(os.environ.get("MODEL_PINNING_ENABLED", "1")))
pinned_models_lock = asyncio.Lock()
pinned_models_initialized = False
pinned_depth_estimation = False
pinned_flow_extraction = False

# --- Request Models ---

class DepthEstimationRequest(BaseModel):
    """Request for MoGe2 depth estimation."""
    task: Literal["depth_estimation"] = Field(default="depth_estimation", description="Task type")
    video_mp4_base64: str = Field(..., min_length=1, description="Input video as base64 encoded MP4")
    depth_prior_base64: str = Field(..., min_length=1, description="First frame depth image as base64 encoded PNG")
    intrinsics: List[float] = Field(..., min_length=4, max_length=4, description="Camera intrinsics [fx, fy, cx, cy]")
    target_size: int = Field(default=518, gt=0, description="Compatibility field retained for older clients; MoGe2 uses native sizing.")
    depth_factor: float = Field(default=0.001, gt=0, description="Depth factor")

    return_depth_video: bool = Field(default=True, description="Return the colorized depth video. Disable for faster depth-only cache priming.")

    # Fixed production depth model.
    depth_model: Literal["moge2"] = Field(
        default="moge2",
        description="The production depth model. CVD is controlled independently."
    )

    # Robust affine calibration is applied to the complete MoGe2 sequence.
    calibration_mode: Literal["affine"] = Field(
        default="affine",
        description="Fixed production calibration: depth_metric = s * depth_moge2 + t.",
    )

    # Robust calibration mask controls.
    mask_erosion_px: int = Field(default=2, ge=0, description="Erosion pixels for calibration mask (default reduced from 4 to 2)")
    planar_grad_thresh: float = Field(default=0.02, ge=0, description="Planar gradient threshold for plane fitting")
    use_planar_mask: bool = Field(default=False, description="Fit dominant plane only. DISABLED by default to use ALL surfaces (table + objects).")
    use_ratio_consistency: bool = Field(default=False, description="Enable ratio consistency filter (can be too aggressive, disabled by default)")
    ratio_thresh: float = Field(default=0.15, ge=0, description="Ratio consistency threshold (only used if use_ratio_consistency=True)")
    min_calib_pixels: int = Field(default=500, gt=0, description="Minimum pixels required for calibration")
    valid_depth_min: float = Field(default=0.6, ge=0, description="Minimum valid depth (meters) for calibration")
    valid_depth_max: float = Field(default=6.0, gt=0, description="Maximum valid depth (meters) for calibration")

    # Affine solver method (only used when calibration_mode='affine')
    affine_method: Literal["irls", "ransac"] = Field(default="ransac", description="Affine solver: 'irls' (all points, weighted) or 'ransac' (2-point consensus)")
    ransac_iters: int = Field(default=1000, gt=0, description="RANSAC iterations (only if affine_method='ransac')")
    irls_iters: int = Field(default=3, gt=0, description="IRLS refinement iterations after median scale estimate (only if affine_method='irls')")

    # Pepper filtering options
    filter_pepper: bool = Field(default=True, description="Enable pepper noise filtering (removes isolated bad pixels)")
    min_region_size: int = Field(default=50, gt=0, description="Min connected region size (pixels) to keep after pepper filter")

    # RANSAC tuning
    inlier_threshold: float = Field(default=0.15, gt=0, description="Absolute RANSAC depth-residual threshold in meters")

    # CVD (Consistent Video Depth) is enabled by default and can be explicitly disabled.
    enable_cvd: bool = Field(default=True, description="[CVD] Enable CVD temporal optimization for MoGe2.")
    cvd_w_grad: float = Field(default=2.0, ge=0, description="[CVD] Gradient loss weight for temporal consistency")
    cvd_w_normal: float = Field(default=5.0, ge=0, description="[CVD] Normal consistency weight for surface smoothness")
    cvd_resolution: int = Field(default=196608, gt=0, description="[CVD] Processing resolution in pixels (default 384*512=196608)")

    @model_validator(mode="after")
    def validate_depth_range(self):
        """Validate the requested depth bounds."""
        if self.valid_depth_min >= self.valid_depth_max:
            raise ValueError("valid_depth_min must be less than valid_depth_max")
        return self

class FlowExtractionRequest(BaseModel):
    """Request for 3D object-flow extraction using SAM3 and TAPIP3D"""
    task: Literal["flow_extraction"] = Field(default="flow_extraction", description="Task type")
    video_mp4_base64: str = Field(..., min_length=1, description="Input RGB SUB-video as base64 encoded MP4")

    # Three modes for depth input (in order of priority):
    # 1. depth_npy_base64 - calibrated metric depth numpy array
    # 2. depth_source_job_id - reuse cached depth from a completed Depth Estimation job
    # 3. depth_video_mp4_base64 - depth video (WARNING: only for grayscale depth, colorized will fail)
    depth_npy_base64: Optional[str] = Field(None, description="Calibrated metric depth as a base64-encoded .npy file [T,H,W] or [T,H,W,1].")
    depth_video_mp4_base64: Optional[str] = Field(None, description="Depth video as base64 MP4. WARNING: Only works for grayscale depth videos, NOT colorized!")
    depth_source_job_id: Optional[str] = Field(None, description="Job ID of a completed Depth Estimation task to reuse cached depth")

    # Depth scaling - only used for depth_npy_base64 and depth_video_mp4_base64
    depth_scale: float = Field(default=1.0, gt=0, description="Scale factor for depth values. For .npy: final_depth = npy_value * depth_scale. For video: final_depth = pixel_value * depth_scale")
    depth_channel: Literal["mean", "red", "green", "blue", "luminance"] = Field(default="mean", description="For depth video: channel reduction.")

    # THIS IS THE CRITICAL FIELD - only used when depth_source_job_id is provided
    start_index: int = Field(default=0, ge=0, description="Start index in the FULL depth video corresponding to the sub-video (only for depth_source_job_id mode)")

    intrinsics: List[float] = Field(..., min_length=4, max_length=4, description="Camera intrinsics [fx, fy, cx, cy]")
    mask_prompt: str = Field(..., min_length=1, description="Object prompt for tracking (e.g., 'yellow block', 'black handle')")
    checkpoint: str = Field(default="checkpoints/tapip3d_final.pth", description="TAPIP3D checkpoint path")
    resolution_factor: int = Field(default=2, gt=0, description="Resolution factor")
    query_grid_size: int = Field(default=32, gt=0, description="Query grid size")
    vis_threshold: float = Field(default=0.9, ge=0, le=1, description="Visibility threshold")
    flow_switch_theta_deg: float = Field(
        default=FLOW_SWITCH_THETA_DEG,
        ge=0,
        le=180,
        description="Adjacent-frame object rotation threshold before hand-flow fallback.",
    )

    tapip3d_node: Literal["vanilla"] = Field(
        default="vanilla",
        description="Fixed production tracker: vanilla TAPIP3DNode.",
    )

    # Vanilla node safe sampling parameters (only used when tapip3d_node="vanilla")
    # These allow mask erosion and depth gradient filtering for initial query sampling
    vanilla_mask_erosion: int = Field(default=3, ge=0, description="[Vanilla] Erode mask pixels to avoid sampling near edges (0=disabled)")
    vanilla_depth_grad_thresh: float = Field(default=0.05, ge=0, description="[Vanilla] Max depth gradient for safe sampling (0=disabled)")
    vanilla_depth_laplacian_thresh: float = Field(default=0.0, ge=0, description="[Vanilla] Max depth Laplacian (2nd derivative) to filter boundary ramps. 0=disabled. Try 0.02-0.05.")

    @model_validator(mode="after")
    def validate_depth_source(self):
        """Validate that a supported initial-depth source is provided."""
        sources = (self.depth_npy_base64, self.depth_source_job_id, self.depth_video_mp4_base64)
        if sum(value is not None for value in sources) != 1:
            raise ValueError("provide exactly one depth source")
        return self
    vanilla_multiscale_grad: bool = Field(default=False, description="[Vanilla] Use multi-scale gradient check (catches wider boundary artifacts)")

    return_debug_media: bool = Field(default=False, description="Return all debug media. Disabled by default because video encoding dominates production latency.")
    return_sam3_video: bool = Field(default=False, description="Return only the SAM3 mask visualization video without TAPIP3D debug video.")
    return_flow_image: bool = Field(default=True, description="Return the static flow visualization image.")
    return_auxiliary_arrays: bool = Field(default=False, description="Return large debug arrays such as video_npy, depths, intrinsics, extrinsics, masks, and query_points.")

# --- Helpers ---

def _sanitize_job_id(job_id: str) -> str:
    return job_id.replace("-", "_")

def _load_default_intrinsics():
    global NOVAPLAN_INTRINSICS_CACHE
    if NOVAPLAN_INTRINSICS_CACHE is None:
        if os.path.exists(NOVAPLAN_DEMO_INTRINSICS):
            with open(NOVAPLAN_DEMO_INTRINSICS, "r") as f:
                d = json.load(f).get("camera_intrinsics_and_extrinsics", {}).get("depth_intrinsics", {})
                NOVAPLAN_INTRINSICS_CACHE = (float(d.get("fx", 642.5)), float(d.get("fy", 641.9)), float(d.get("ppx", 649.8)), float(d.get("ppy", 365.9)))
        else:
            NOVAPLAN_INTRINSICS_CACHE = (642.5, 641.9, 649.8, 365.9)
    return NOVAPLAN_INTRINSICS_CACHE

async def _read_file_async(path: str) -> bytes:
    async with aiofiles.open(path, "rb") as f: return await f.read()

async def _write_file_async(path: str, data: bytes):
    async with aiofiles.open(path, "wb") as f: await f.write(data)

async def _copy_file_async(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    await asyncio.get_running_loop().run_in_executor(None, shutil.copy, src, dst)

async def try_load_video_from_disk(fname, subfolder):
    """Try to load a generated video directly from shared storage."""
    paths = [
        os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname),
        os.path.join(COMFYUI_OUTPUT_DIR, fname),
        os.path.join(COMFYUI_OUTPUT_DIR, "video", fname)
    ]
    for p in paths:
        if os.path.exists(p): return await _read_file_async(p), p
    return None, None

async def fetch_videos_from_outputs(job_id, base_url, outputs, expected):
    """Fetch videos from outputs."""
    video_nodes = []
    for nid, out in outputs.items():
        if out.get("videos"):
            video_nodes.append({
                "node_id": nid,
                "type": "videos",
                "fn": out["videos"][0]["filename"],
                "sub": out["videos"][0]["subfolder"]
            })
        elif out.get("images") and out["images"][0]["filename"].endswith(".mp4"):
            video_nodes.append({
                "node_id": nid,
                "type": "images",
                "fn": out["images"][0]["filename"],
                "sub": out["images"][0]["subfolder"]
            })

    video_nodes.sort(key=lambda x: x["node_id"])
    results = []
    for i, node in enumerate(video_nodes[:expected]):
        data, path = await try_load_video_from_disk(node["fn"], node["sub"])
        if not data:
            resp = await client.get(f"{base_url}/view", params={
                "filename": node["fn"],
                "subfolder": node["sub"],
                "type": "output"
            })
            if resp.status_code == 200:
                data = resp.content
                # Warning: we don't have a path if we downloaded it via API
        if data:
            results.append({"data": data, "filename": node["fn"], "subfolder": node["sub"], "path": path})
    return results, len(video_nodes)


async def fetch_video_by_prefix(base_url: str, outputs: dict, filename_prefix: str) -> tuple[bytes | None, str | None]:
    """Fetch video by prefix."""
    for nid, out in outputs.items():
        if not isinstance(out, dict):
            continue
        candidates = []
        candidates.extend(out.get("videos") or [])
        candidates.extend(
            img for img in (out.get("images") or [])
            if str(img.get("filename", "")).endswith(".mp4")
        )
        for item in candidates:
            filename = item.get("filename", "")
            if filename_prefix not in filename:
                continue
            subfolder = item.get("subfolder", "")
            data, path = await try_load_video_from_disk(filename, subfolder)
            if not data:
                resp = await client.get(
                    f"{base_url}/view",
                    params={"filename": filename, "subfolder": subfolder, "type": "output"},
                )
                if resp.status_code == 200:
                    data = resp.content
            if data:
                return data, path
    return None, None

async def fetch_job_history(client, url, prompt_id, attempts=30, delay=0.5):
    """Fetch job history."""
    for _ in range(attempts):
        try:
            res = await client.get(url)
            if res.status_code == 200 and prompt_id in res.json():
                return res.json()[prompt_id]
        except Exception:
            pass
        await asyncio.sleep(delay)
    return None


def _validate_depth_history(
    history: Dict[str, Any],
    cache_node_id: str,
    expected_cache_id: str,
) -> Dict[str, Any]:
    """Require a successful Comfy execution and cache-node acknowledgement."""
    if not isinstance(history, dict):
        raise RuntimeError("ComfyUI returned an invalid depth-job history record")

    status = history.get("status")
    status_str = status.get("status_str") if isinstance(status, dict) else None
    completed = status.get("completed") if isinstance(status, dict) else None
    if status_str != "success" or completed is not True:
        detail = None
        messages = status.get("messages", []) if isinstance(status, dict) else []
        for message in reversed(messages):
            if not isinstance(message, (list, tuple)) or len(message) < 2:
                continue
            if message[0] != "execution_error" or not isinstance(message[1], dict):
                continue
            payload = message[1]
            detail = payload.get("exception_message") or payload.get("exception_type")
            break
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"ComfyUI depth execution did not complete successfully "
            f"(status={status_str!r}, completed={completed!r}){suffix}"
        )

    outputs = history.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("ComfyUI depth execution returned no output map")
    cache_output = outputs.get(cache_node_id)
    cached_ids = cache_output.get("cached") if isinstance(cache_output, dict) else None
    if isinstance(cached_ids, str):
        cached_ids = [cached_ids]
    if not isinstance(cached_ids, (list, tuple)) or expected_cache_id not in cached_ids:
        raise RuntimeError(
            "ComfyUI depth execution completed without acknowledging the "
            f"expected calibrated-depth cache {expected_cache_id!r}"
        )
    return outputs

async def pin_models_to_gpu(target_url: str, workflow_type: str = "both"):
    """Run the model-pinning workflow on a ComfyUI worker."""
    global pinned_models_initialized, pinned_depth_estimation, pinned_flow_extraction
    if not MODEL_PINNING_ENABLED: return
    async with pinned_models_lock:
        if workflow_type in ("depth_estimation", "both") and not pinned_depth_estimation:
            _log_debug("[Model Pinning] MoGe2 depth node will be cached on first request")
            pinned_depth_estimation = True
        if workflow_type in ("flow_extraction", "both") and not pinned_flow_extraction:
            _log_debug("[Model Pinning] Sam3VideoNode and vanilla TAPIP3DNode will be cached on first request")
            pinned_flow_extraction = True

# --- Workflow Generators ---

def get_depth_estimation_workflow(
    job_id: str,
    video_filename: str,
    depth_filename: str,
    intrinsics: List[float],
    target_size: int = 518,  # Compatibility field; MoGe2 runs at its native sizing.
    depth_factor: float = 0.001,
    depth_model: str = "moge2",
    calibration_mode: str = "affine",
    mask_erosion_px: int = 2,
    planar_grad_thresh: float = 0.02,
    use_planar_mask: bool = False,  # DISABLED: use ALL surfaces (table + objects)
    use_ratio_consistency: bool = False,  # Disabled by default - can be too aggressive
    ratio_thresh: float = 0.15,
    min_calib_pixels: int = 500,
    valid_depth_min: float = 0.6,
    valid_depth_max: float = 6.0,
    affine_method: str = "ransac",  # "irls" or "ransac"
    ransac_iters: int = 1000,
    irls_iters: int = 3,  # IRLS refinement iterations
    filter_pepper: bool = True,
    min_region_size: int = 50,
    inlier_threshold: float = 0.15,
    return_depth_video: bool = True,
    # CVD parameters for MoGe2 variants
    enable_cvd: bool = True,
    cvd_w_grad: float = 2.0,
    cvd_w_normal: float = 5.0,
    cvd_resolution: int = 196608,
) -> Dict[str, Any]:
    """Build MoGe2 -> optional CVD -> first-frame calibration -> cache."""
    if not depth_filename:
        raise ValueError("A first-frame metric depth prior is required for calibration")
    calibration_mode = calibration_mode.lower().strip()
    if calibration_mode != "affine":
        raise ValueError("calibration_mode must be 'affine'")

    safe_id = _sanitize_job_id(job_id)
    workflow = {}

    video_loader_id = f"load_video_{safe_id}"
    workflow[video_loader_id] = {
        "class_type": "LoadVideo",
        "inputs": {"file": video_filename}
    }

    intrinsics_id = f"intrinsics_{safe_id}"
    workflow[intrinsics_id] = {
        "class_type": "IntrinsicsNode",
        "inputs": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3])
        }
    }

    # The exported pipeline has one depth implementation: MoGe2 with optional CVD.
    depth_node_id = f"depth_estimation_{safe_id}"
    dm = (depth_model or "moge2").lower().strip()
    if dm != "moge2":
        raise ValueError(f"Unsupported depth_model {depth_model!r}; expected 'moge2'")
    requested_cvd = bool(enable_cvd)

    depth_node_class = "MoGe2CVDMetricDepthNode"
    depth_inputs = {
        "video": [video_loader_id, 0],
        "depth_factor": depth_factor,
        "calibration_mode": "Affine (s * d + t)",
        "intrinsics": [intrinsics_id, 0],
        "enable_cvd": "enabled" if requested_cvd else "disabled",
        "cvd_w_grad": cvd_w_grad,
        "cvd_w_normal": cvd_w_normal,
        "cvd_resolution": cvd_resolution,
        "mask_erosion_px": mask_erosion_px,
        "planar_grad_thresh": planar_grad_thresh,
        "use_planar_mask": use_planar_mask,
        "use_ratio_consistency": use_ratio_consistency,
        "ratio_thresh": ratio_thresh,
        "min_calib_pixels": min_calib_pixels,
        "valid_depth_min": valid_depth_min,
        "valid_depth_max": valid_depth_max,
        "affine_method": affine_method,
        "ransac_iters": ransac_iters,
        "irls_iters": irls_iters,
        "filter_pepper": filter_pepper,
        "min_region_size": min_region_size,
        "inlier_threshold": inlier_threshold,
    }
    depth_inputs["depth_source"] = depth_filename
    _log_debug(f"[Workflow] Using MoGe2CVDMetricDepthNode (cvd={requested_cvd})")

    workflow[depth_node_id] = {
        "class_type": depth_node_class,
        "inputs": depth_inputs
    }

    if return_depth_video:
        # Save colorized depth VIDEO for visualization (output 1: depth_video)
        save_video_id = f"save_depth_video_{safe_id}"
        workflow[save_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [depth_node_id, 1],  # Colorized depth_video for visualization
                "filename_prefix": f"depth_estimation_{safe_id}",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    # Cache only the calibrated metric depth in memory for downstream tracking.
    cache_raw_depth_id = f"cache_raw_depth_{safe_id}"
    workflow[cache_raw_depth_id] = {
        "class_type": "CacheRawDepthNode",
        "inputs": {
            "raw_depth": [depth_node_id, 2],  # calibrated metric depth
            "cache_id": f"depth_{safe_id}"  # Use job_id as cache key
        }
    }

    return {"prompt": workflow}

def get_flow_extraction_workflow(
    job_id: str,
    video_filename: str,
    raw_depth_cache_id: str,  # Memory cache ID for raw depth (from depth estimation job)
    intrinsics: List[float],
    mask_prompt: str,
    checkpoint: str,
    resolution_factor: int,
    query_grid_size: int,
    vis_threshold: float,
    start_index: int = 0,  # <--- CRITICAL: Receiving start_index here
    # Compatibility field; workflow generation always uses vanilla TAPIP3DNode.
    tapip3d_node: str = "vanilla",
    # Vanilla node safe sampling parameters
    vanilla_mask_erosion: int = 3,
    vanilla_depth_grad_thresh: float = 0.05,
    vanilla_depth_laplacian_thresh: float = 0.0,
    vanilla_multiscale_grad: bool = False,
    return_debug_media: bool = False,
    return_sam3_video: bool = False,
    return_flow_image: bool = True,
    return_auxiliary_arrays: bool = False,
) -> Dict[str, Any]:
    """Build the ComfyUI workflow for object-flow extraction."""
    safe_id = _sanitize_job_id(job_id)
    workflow = {}

    video_loader_id = f"load_video_{safe_id}"
    workflow[video_loader_id] = {
        "class_type": "LoadVideo",
        "inputs": {"file": video_filename}
    }

    # Load RAW depth from memory cache (fast, no disk I/O!)
    depth_loader_id = f"load_cached_depth_{safe_id}"
    workflow[depth_loader_id] = {
        "class_type": "LoadCachedRawDepthNode",
        "inputs": {
            "cache_id": raw_depth_cache_id  # Memory cache ID from depth estimation job
        }
    }

    intrinsics_id = f"intrinsics_{safe_id}"
    workflow[intrinsics_id] = {
        "class_type": "IntrinsicsNode",
        "inputs": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3])
        }
    }

    sam_id = f"sam3_{safe_id}"
    workflow[sam_id] = {
        "class_type": "Sam3VideoNode",
        "inputs": {
            "video": [video_loader_id, 0],
            "prompt": mask_prompt,
            "return_visualization": return_debug_media or return_sam3_video,
        }
    }

    tapip_id = f"tapip3d_{safe_id}"
    tapip_save_prefix = f"flow_extraction_{safe_id}"

    node_type = (tapip3d_node or "vanilla").lower().strip()
    if node_type != "vanilla":
        _log_warning(f"[Workflow] TAPIP3D node {tapip3d_node!r} is disabled; using vanilla TAPIP3DNode", flush=True)
    node_type = "vanilla"
    tapip_class = "TAPIP3DNode"
    _log_debug("[Workflow] Using TAPIP3DNode (vanilla)")

    tapip_inputs = {
            "video": [video_loader_id, 0],
            "depth": [depth_loader_id, 0],
            "intrinsics": [intrinsics_id, 0],
            "checkpoint": checkpoint,
            "resolution_factor": resolution_factor,
            "vis_threshold": vis_threshold,
            "mask": [sam_id, 1],
            "start_index": start_index,      # <--- CRITICAL: Passing index to Node
            "save_prefix": tapip_save_prefix, # <--- CRITICAL: Passing prefix to Node
            "return_debug_media": return_debug_media,
            "return_auxiliary_arrays": return_auxiliary_arrays,
        }

    tapip_inputs["query_grid_size"] = query_grid_size
    tapip_inputs["mask_erosion"] = vanilla_mask_erosion
    tapip_inputs["depth_grad_thresh"] = vanilla_depth_grad_thresh
    tapip_inputs["depth_laplacian_thresh"] = vanilla_depth_laplacian_thresh
    tapip_inputs["multiscale_grad"] = vanilla_multiscale_grad

    workflow[tapip_id] = {
        "class_type": tapip_class,
        "inputs": tapip_inputs
    }

    if return_flow_image:
        save_flow_id = f"save_flow_{safe_id}"
        workflow[save_flow_id] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": [tapip_id, 3],
                "filename_prefix": f"{tapip_save_prefix}_flow"
            }
        }

    if return_debug_media or return_sam3_video:
        # Save SAM3 mask tracking visualization video (for debugging)
        save_sam3_video_id = f"save_sam3_video_{safe_id}"
        workflow[save_sam3_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [sam_id, 0],  # Sam3VideoNode output 0 = visualized_video
                "filename_prefix": f"{tapip_save_prefix}_sam3_tracking",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    if return_debug_media:
        # Save TAPIP3D point tracking visualization video (for debugging)
        save_tapip_video_id = f"save_tapip_video_{safe_id}"
        workflow[save_tapip_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [tapip_id, 0],  # TAPIP3DNode output 0 = output_video (tracking)
                "filename_prefix": f"{tapip_save_prefix}_tapip3d_tracking",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    return {"prompt": workflow}


def get_flow_extraction_workflow_with_depth_npy(
    job_id: str,
    video_filename: str,
    depth_npy_filename: str,  # .npy file in ComfyUI input directory
    intrinsics: List[float],
    mask_prompt: str,
    checkpoint: str,
    resolution_factor: int,
    query_grid_size: int,
    vis_threshold: float,
    depth_scale: float = 1.0,
    # Compatibility field; workflow generation always uses vanilla TAPIP3DNode.
    tapip3d_node: str = "vanilla",
    # Vanilla node safe sampling parameters
    vanilla_mask_erosion: int = 3,
    vanilla_depth_grad_thresh: float = 0.05,
    vanilla_depth_laplacian_thresh: float = 0.0,
    vanilla_multiscale_grad: bool = False,
    return_debug_media: bool = False,
    return_sam3_video: bool = False,
    return_flow_image: bool = True,
    return_auxiliary_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Workflow for object-flow extraction using a precomputed metric-depth NumPy file.
    This is the RECOMMENDED path for direct depth upload as it preserves metric depth values.
    """
    safe_id = _sanitize_job_id(job_id)
    workflow = {}

    video_loader_id = f"load_video_{safe_id}"
    workflow[video_loader_id] = {
        "class_type": "LoadVideo",
        "inputs": {"file": video_filename}
    }

    # Load raw depth from .npy file
    depth_loader_id = f"load_depth_npy_{safe_id}"
    workflow[depth_loader_id] = {
        "class_type": "LoadRawDepthFromFileNode",
        "inputs": {
            "depth_file": depth_npy_filename,
            "depth_scale": depth_scale
        }
    }

    intrinsics_id = f"intrinsics_{safe_id}"
    workflow[intrinsics_id] = {
        "class_type": "IntrinsicsNode",
        "inputs": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3])
        }
    }

    sam_id = f"sam3_{safe_id}"
    workflow[sam_id] = {
        "class_type": "Sam3VideoNode",
        "inputs": {
            "video": [video_loader_id, 0],
            "prompt": mask_prompt,
            "return_visualization": return_debug_media or return_sam3_video,
        }
    }

    tapip_id = f"tapip3d_{safe_id}"
    tapip_save_prefix = f"flow_extraction_{safe_id}"

    node_type = (tapip3d_node or "vanilla").lower().strip()
    if node_type != "vanilla":
        _log_warning(f"[Workflow] TAPIP3D node {tapip3d_node!r} is disabled; using vanilla TAPIP3DNode", flush=True)
    node_type = "vanilla"
    tapip_class = "TAPIP3DNode"
    _log_debug("[Workflow] Using TAPIP3DNode (vanilla)")

    tapip_inputs = {
            "video": [video_loader_id, 0],
            "depth": [depth_loader_id, 0],  # Direct from .npy file
            "intrinsics": [intrinsics_id, 0],
            "checkpoint": checkpoint,
            "resolution_factor": resolution_factor,
            "vis_threshold": vis_threshold,
            "mask": [sam_id, 1],
            "start_index": 0,  # No slicing needed - depth already matches video length
            "save_prefix": tapip_save_prefix,
            "return_debug_media": return_debug_media,
            "return_auxiliary_arrays": return_auxiliary_arrays,
        }

    tapip_inputs["query_grid_size"] = query_grid_size
    tapip_inputs["mask_erosion"] = vanilla_mask_erosion
    tapip_inputs["depth_grad_thresh"] = vanilla_depth_grad_thresh
    tapip_inputs["depth_laplacian_thresh"] = vanilla_depth_laplacian_thresh
    tapip_inputs["multiscale_grad"] = vanilla_multiscale_grad

    workflow[tapip_id] = {
        "class_type": tapip_class,
        "inputs": tapip_inputs
    }

    if return_flow_image:
        save_flow_id = f"save_flow_{safe_id}"
        workflow[save_flow_id] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": [tapip_id, 3],
                "filename_prefix": f"{tapip_save_prefix}_flow"
            }
        }

    if return_debug_media or return_sam3_video:
        # Save SAM3 mask tracking visualization video
        save_sam3_video_id = f"save_sam3_video_{safe_id}"
        workflow[save_sam3_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [sam_id, 0],
                "filename_prefix": f"{tapip_save_prefix}_sam3_tracking",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    if return_debug_media:
        # Save TAPIP3D point tracking visualization video
        save_tapip_video_id = f"save_tapip_video_{safe_id}"
        workflow[save_tapip_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [tapip_id, 0],
                "filename_prefix": f"{tapip_save_prefix}_tapip3d_tracking",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    return {"prompt": workflow}


def get_flow_extraction_workflow_with_depth_video(
    job_id: str,
    video_filename: str,
    depth_video_filename: str,  # MP4 file in ComfyUI input directory
    intrinsics: List[float],
    mask_prompt: str,
    checkpoint: str,
    resolution_factor: int,
    query_grid_size: int,
    vis_threshold: float,
    depth_scale: float = 10.0,
    depth_channel: str = "mean",
    # Compatibility field; workflow generation always uses vanilla TAPIP3DNode.
    tapip3d_node: str = "vanilla",
    # Vanilla node safe sampling parameters
    vanilla_mask_erosion: int = 3,
    vanilla_depth_grad_thresh: float = 0.05,
    vanilla_depth_laplacian_thresh: float = 0.0,
    vanilla_multiscale_grad: bool = False,
    return_debug_media: bool = False,
    return_sam3_video: bool = False,
    return_flow_image: bool = True,
    return_auxiliary_arrays: bool = False,
) -> Dict[str, Any]:
    """
    Workflow for object-flow extraction using a depth video file.
    WARNING: Only works for GRAYSCALE depth videos. Colorized depth (e.g., magma) will fail!
    """
    safe_id = _sanitize_job_id(job_id)
    workflow = {}

    video_loader_id = f"load_video_{safe_id}"
    workflow[video_loader_id] = {
        "class_type": "LoadVideo",
        "inputs": {"file": video_filename}
    }

    # Load depth video
    depth_video_loader_id = f"load_depth_video_{safe_id}"
    workflow[depth_video_loader_id] = {
        "class_type": "LoadVideo",
        "inputs": {"file": depth_video_filename}
    }

    # Convert depth video to raw depth format
    depth_converter_id = f"convert_depth_{safe_id}"
    workflow[depth_converter_id] = {
        "class_type": "LoadDepthVideoAsRawNode",
        "inputs": {
            "depth_video": [depth_video_loader_id, 0],
            "depth_scale": depth_scale,
            "channel": depth_channel
        }
    }

    intrinsics_id = f"intrinsics_{safe_id}"
    workflow[intrinsics_id] = {
        "class_type": "IntrinsicsNode",
        "inputs": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3])
        }
    }

    sam_id = f"sam3_{safe_id}"
    workflow[sam_id] = {
        "class_type": "Sam3VideoNode",
        "inputs": {
            "video": [video_loader_id, 0],
            "prompt": mask_prompt,
            "return_visualization": return_debug_media or return_sam3_video,
        }
    }

    tapip_id = f"tapip3d_{safe_id}"
    tapip_save_prefix = f"flow_extraction_{safe_id}"

    node_type = (tapip3d_node or "vanilla").lower().strip()
    if node_type != "vanilla":
        _log_warning(f"[Workflow] TAPIP3D node {tapip3d_node!r} is disabled; using vanilla TAPIP3DNode", flush=True)
    node_type = "vanilla"
    tapip_class = "TAPIP3DNode"
    _log_debug("[Workflow] Using TAPIP3DNode (vanilla)")

    tapip_inputs = {
            "video": [video_loader_id, 0],
            "depth": [depth_converter_id, 0],  # Converted from video
            "intrinsics": [intrinsics_id, 0],
            "checkpoint": checkpoint,
            "resolution_factor": resolution_factor,
            "vis_threshold": vis_threshold,
            "mask": [sam_id, 1],
            "start_index": 0,  # No slicing needed - depth video should match RGB video length
            "save_prefix": tapip_save_prefix,
            "return_debug_media": return_debug_media,
            "return_auxiliary_arrays": return_auxiliary_arrays,
        }

    tapip_inputs["query_grid_size"] = query_grid_size
    tapip_inputs["mask_erosion"] = vanilla_mask_erosion
    tapip_inputs["depth_grad_thresh"] = vanilla_depth_grad_thresh
    tapip_inputs["depth_laplacian_thresh"] = vanilla_depth_laplacian_thresh
    tapip_inputs["multiscale_grad"] = vanilla_multiscale_grad

    workflow[tapip_id] = {
        "class_type": tapip_class,
        "inputs": tapip_inputs
    }

    if return_flow_image:
        save_flow_id = f"save_flow_{safe_id}"
        workflow[save_flow_id] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": [tapip_id, 3],
                "filename_prefix": f"{tapip_save_prefix}_flow"
            }
        }

    if return_debug_media or return_sam3_video:
        # Save SAM3 mask tracking visualization video (depth video workflow)
        save_sam3_video_id = f"save_sam3_video_{safe_id}"
        workflow[save_sam3_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [sam_id, 0],
                "filename_prefix": f"{tapip_save_prefix}_sam3_tracking",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    if return_debug_media:
        # Save TAPIP3D point tracking visualization video (depth video workflow)
        save_tapip_video_id = f"save_tapip_video_{safe_id}"
        workflow[save_tapip_video_id] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [tapip_id, 0],
                "filename_prefix": f"{tapip_save_prefix}_tapip3d_tracking",
                "format": SAVEVIDEO_FORMAT,
                "codec": SAVEVIDEO_CODEC
            }
        }

    return {"prompt": workflow}


# --- Job Processing ---

async def _get_all_workers() -> List[Dict[str, Any]]:
    workers = []
    for i in range(NUM_GPUS):
        port = START_PORT + i
        workers.append({"id": i, "url": f"{COMFYUI_HOST}:{port}", "name": "Master" if i == 0 else f"GPU {i}"})
    return workers

async def _get_least_loaded_worker() -> str:
    try:
        workers = await _get_all_workers()
        async def get_load(w):
            try:
                r = await client.get(f"{w['url']}/queue", timeout=0.5)
                if r.status_code == 200:
                    q = r.json()
                    return len(q.get("queue_running",[])) + len(q.get("queue_pending",[]))
            except Exception:
                pass
            return 999
        loads = await asyncio.gather(*[get_load(w) for w in workers])
        global rr_index
        stats = []
        for i, w in enumerate(workers):
            stats.append({"url": w["url"], "load": loads[i], "rr": (int(w["id"])-rr_index) % len(workers)})
        stats.sort(key=lambda x: (x["load"], x["rr"]))
        async with rr_lock: rr_index = (rr_index + 1) % len(workers)
        return stats[0]["url"]
    except Exception:
        return COMFYUI_MASTER_URL


async def _select_flow_worker(depth_mode: str, cached_worker_url: Optional[str]) -> str:
    """Keep worker-local cached depth on the worker that produced it."""
    if depth_mode == "cached":
        if not cached_worker_url:
            raise RuntimeError("Cached depth metadata is missing its originating ComfyUI worker")
        return cached_worker_url
    return await _get_least_loaded_worker()

async def run_depth_estimation_job(job_id: str, req: DepthEstimationRequest):
    """Run depth estimation job."""
    t0 = time.perf_counter()
    timing = {"prepare_s": 0, "total_s": 0}
    cleanup_paths = []

    try:
        safe_id = _sanitize_job_id(job_id)

        video_name = f"{safe_id}_input.mp4"
        video_path = os.path.join(COMFYUI_INPUT_DIR, video_name)
        await _write_file_async(video_path, base64.b64decode(req.video_mp4_base64))
        cleanup_paths.append(video_path)

        depth_name = f"{safe_id}_depth_prior.png"
        depth_path = os.path.join(COMFYUI_INPUT_DIR, depth_name)
        await _write_file_async(depth_path, base64.b64decode(req.depth_prior_base64))
        cleanup_paths.append(depth_path)

        intrinsics = req.intrinsics if len(req.intrinsics) >= 4 else list(_load_default_intrinsics())
        timing["prepare_s"] = time.perf_counter() - t0

        target_url = await _get_least_loaded_worker()
        await pin_models_to_gpu(target_url, "depth_estimation")

        _log_debug(f"[Depth Job {job_id}] Request: depth_model={req.depth_model}, calibration={req.calibration_mode}")

        workflow = get_depth_estimation_workflow(
            job_id, video_name, depth_name, intrinsics, req.target_size, req.depth_factor,
            depth_model=req.depth_model,
            calibration_mode=req.calibration_mode,
            mask_erosion_px=req.mask_erosion_px,
            planar_grad_thresh=req.planar_grad_thresh,
            use_planar_mask=req.use_planar_mask,
            use_ratio_consistency=req.use_ratio_consistency,
            ratio_thresh=req.ratio_thresh,
            min_calib_pixels=req.min_calib_pixels,
            valid_depth_min=req.valid_depth_min,
            valid_depth_max=req.valid_depth_max,
            affine_method=req.affine_method,  # "irls" or "ransac"
            ransac_iters=req.ransac_iters,
            irls_iters=req.irls_iters,  # IRLS refinement iterations
            filter_pepper=req.filter_pepper,
            min_region_size=req.min_region_size,
            inlier_threshold=req.inlier_threshold,
            return_depth_video=req.return_depth_video,
            # CVD parameters for MoGe2 variants
            enable_cvd=req.enable_cvd,
            cvd_w_grad=req.cvd_w_grad,
            cvd_w_normal=req.cvd_w_normal,
            cvd_resolution=req.cvd_resolution,
        )
        workflow["client_id"] = f"job-{job_id}"

        resp = await client.post(f"{target_url}/prompt", json=workflow)
        if resp.status_code != 200: raise RuntimeError(f"Failed to submit: {resp.text}")

        pid = resp.json()["prompt_id"]
        data = await fetch_job_history(client, f"{target_url}/history/{pid}", pid, attempts=3600)
        if not data: raise RuntimeError("Job timeout")

        raw_depth_cache_id = f"depth_{safe_id}"
        cache_node_id = f"cache_raw_depth_{safe_id}"
        outputs = _validate_depth_history(
            data,
            cache_node_id=cache_node_id,
            expected_cache_id=raw_depth_cache_id,
        )
        depth_videos = []
        depth_video_bytes = None
        cache_video_filename = None
        if req.return_depth_video:
            depth_videos, _ = await fetch_videos_from_outputs(job_id, target_url, outputs, 1)
            if not depth_videos:
                raise RuntimeError("No depth video found")

            # Cache colorized video for visualization (optional download)
            generated_path = depth_videos[0].get("path")
            depth_video_bytes = depth_videos[0]["data"]
            if generated_path and os.path.exists(generated_path):
                cache_video_filename = f"cached_depth_{safe_id}.mp4"
                cache_video_path = os.path.join(COMFYUI_INPUT_DIR, cache_video_filename)
                await _copy_file_async(generated_path, cache_video_path)
                _log_debug(f"[Depth Job {job_id}] Cached video to {cache_video_filename}")

        # Raw depth is now cached IN MEMORY by CacheRawDepthNode (no disk I/O needed!)
        # The cache_id matches what we used in the workflow
        _log_debug(f"[Depth Job {job_id}] Raw depth cached in memory with ID: {raw_depth_cache_id}")

        # Store cache info
        async with job_lock:
            depth_file_cache[job_id] = {
                "video": cache_video_filename,
                "cache_id": raw_depth_cache_id,
                "worker_url": target_url,
            }
            _log_debug(f"[Depth Job {job_id}] Cache entry: {depth_file_cache[job_id]}")

        timing["total_s"] = time.perf_counter() - t0
        _log_timing(
            "DepthEstimation",
            timing["total_s"],
            job=job_id[:8],
            prepare=f"{timing['prepare_s']:.2f}s",
            model=req.depth_model,
        )

        async with job_lock:
            job_update = {
                "status": "completed",
                "raw_depth_cache_id": raw_depth_cache_id,  # Memory cache ID for object-flow extraction
                "timing": timing
            }
            if depth_video_bytes is not None:
                job_update["depth_video_bytes"] = depth_video_bytes
            jobs[job_id].update(job_update)

    except Exception as e:
        traceback.print_exc()
        async with job_lock: jobs[job_id].update({"status": "failed", "error": str(e)})
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

async def run_flow_extraction_job(job_id: str, req: FlowExtractionRequest):
    """Run flow extraction job."""
    t0 = time.perf_counter()
    timing = {"prepare_s": 0, "total_s": 0}
    cleanup_paths = []

    try:
        safe_id = _sanitize_job_id(job_id)

        video_name = f"{safe_id}_input.mp4"
        video_path = os.path.join(COMFYUI_INPUT_DIR, video_name)
        await _write_file_async(video_path, base64.b64decode(req.video_mp4_base64))
        cleanup_paths.append(video_path)

        intrinsics = req.intrinsics if len(req.intrinsics) >= 4 else list(_load_default_intrinsics())

        # Determine depth source mode (in priority order):
        # 1. depth_npy_base64 - raw numpy array (RECOMMENDED)
        # 2. depth_source_job_id - reuse cached depth from previous job
        # 3. depth_video_mp4_base64 - depth video (WARNING: only grayscale)

        workflow = None
        depth_mode = None
        cached_worker_url = None

        # Determine TAPIP3D node type. Only vanilla is enabled; older values are normalized.
        requested_tapip3d_node = (req.tapip3d_node or "vanilla").lower().strip()
        if requested_tapip3d_node != "vanilla":
            _log_warning(f"[Flow Job {job_id}] TAPIP3D node {req.tapip3d_node!r} is disabled; using vanilla", flush=True)
        tapip3d_node = "vanilla"
        tapip3d_display = {"vanilla": "Node (vanilla)"}

        # MODE 1: Raw depth numpy array (RECOMMENDED - most accurate)
        if req.depth_npy_base64:
            depth_mode = "npy"
            depth_npy_name = f"{safe_id}_depth.npy"
            depth_npy_path = os.path.join(COMFYUI_INPUT_DIR, depth_npy_name)
            await _write_file_async(depth_npy_path, base64.b64decode(req.depth_npy_base64))
            cleanup_paths.append(depth_npy_path)

            _log_debug(f"[Flow Job {job_id}] Using direct depth numpy upload (RECOMMENDED)")
            _log_debug(f"[Flow Job {job_id}] TAPIP3D node: {tapip3d_display.get(tapip3d_node, tapip3d_node)}")
            workflow = get_flow_extraction_workflow_with_depth_npy(
                job_id, video_name, depth_npy_name, intrinsics,
                req.mask_prompt, req.checkpoint, req.resolution_factor,
                req.query_grid_size, req.vis_threshold,
                depth_scale=req.depth_scale,
                tapip3d_node=tapip3d_node,
                vanilla_mask_erosion=req.vanilla_mask_erosion,
                vanilla_depth_grad_thresh=req.vanilla_depth_grad_thresh,
                vanilla_depth_laplacian_thresh=req.vanilla_depth_laplacian_thresh,
                vanilla_multiscale_grad=req.vanilla_multiscale_grad,
                return_debug_media=req.return_debug_media,
                return_sam3_video=req.return_sam3_video,
                return_flow_image=req.return_flow_image,
                return_auxiliary_arrays=req.return_auxiliary_arrays,
            )

        # MODE 2: Reuse cached depth from previous depth estimation job
        elif req.depth_source_job_id:
            depth_mode = "cached"
            raw_depth_cache_id = None
            async with job_lock:
                if req.depth_source_job_id in depth_file_cache:
                    cache_entry = depth_file_cache[req.depth_source_job_id]
                    raw_depth_cache_id = cache_entry.get("cache_id") if isinstance(cache_entry, dict) else None
                    cached_worker_url = cache_entry.get("worker_url") if isinstance(cache_entry, dict) else None
                    _log_debug(f"[Flow Job {job_id}] Reusing cached raw depth from memory: {raw_depth_cache_id}")

            if not raw_depth_cache_id:
                raise ValueError(f"Could not resolve depth. Job ID {req.depth_source_job_id} not found in cache. Run depth estimation first.")
            if not cached_worker_url:
                raise ValueError(
                    f"Depth job {req.depth_source_job_id} has no worker affinity; "
                    "rerun depth estimation before object-flow extraction."
                )

            _log_debug(f"[Flow Job {job_id}] TAPIP3D node: {tapip3d_display.get(tapip3d_node, tapip3d_node)}")
            workflow = get_flow_extraction_workflow(
                job_id, video_name, raw_depth_cache_id, intrinsics,
                req.mask_prompt, req.checkpoint, req.resolution_factor,
                req.query_grid_size, req.vis_threshold,
                start_index=req.start_index,  # Only used for cached mode
                tapip3d_node=tapip3d_node,
                vanilla_mask_erosion=req.vanilla_mask_erosion,
                vanilla_depth_grad_thresh=req.vanilla_depth_grad_thresh,
                vanilla_depth_laplacian_thresh=req.vanilla_depth_laplacian_thresh,
                vanilla_multiscale_grad=req.vanilla_multiscale_grad,
                return_debug_media=req.return_debug_media,
                return_sam3_video=req.return_sam3_video,
                return_flow_image=req.return_flow_image,
                return_auxiliary_arrays=req.return_auxiliary_arrays,
            )

        # MODE 3: Depth video (WARNING: only works for grayscale)
        elif req.depth_video_mp4_base64:
            depth_mode = "video"
            depth_video_name = f"{safe_id}_depth.mp4"
            depth_video_path = os.path.join(COMFYUI_INPUT_DIR, depth_video_name)
            await _write_file_async(depth_video_path, base64.b64decode(req.depth_video_mp4_base64))
            cleanup_paths.append(depth_video_path)

            _log_warning(f"[Flow Job {job_id}] WARNING: using depth video upload; only grayscale depth videos are supported")
            _log_debug(f"[Flow Job {job_id}] TAPIP3D node: {tapip3d_display.get(tapip3d_node, tapip3d_node)}")
            workflow = get_flow_extraction_workflow_with_depth_video(
                job_id, video_name, depth_video_name, intrinsics,
                req.mask_prompt, req.checkpoint, req.resolution_factor,
                req.query_grid_size, req.vis_threshold,
                depth_scale=req.depth_scale,
                depth_channel=req.depth_channel,
                tapip3d_node=tapip3d_node,
                vanilla_mask_erosion=req.vanilla_mask_erosion,
                vanilla_depth_grad_thresh=req.vanilla_depth_grad_thresh,
                vanilla_depth_laplacian_thresh=req.vanilla_depth_laplacian_thresh,
                vanilla_multiscale_grad=req.vanilla_multiscale_grad,
                return_debug_media=req.return_debug_media,
                return_sam3_video=req.return_sam3_video,
                return_flow_image=req.return_flow_image,
                return_auxiliary_arrays=req.return_auxiliary_arrays,
            )

        else:
            raise ValueError("No depth source provided. Use one of: depth_npy_base64, depth_source_job_id, or depth_video_mp4_base64")

        timing["prepare_s"] = time.perf_counter() - t0
        _log_debug(f"[Flow Job {job_id}] Depth mode: {depth_mode}, prepare time: {timing['prepare_s']:.2f}s")

        target_url = await _select_flow_worker(depth_mode, cached_worker_url)
        await pin_models_to_gpu(target_url, "flow_extraction")

        workflow["client_id"] = f"job-{job_id}"

        resp = await client.post(f"{target_url}/prompt", json=workflow)
        if resp.status_code != 200: raise RuntimeError(f"Failed to submit: {resp.text}")

        pid = resp.json()["prompt_id"]
        data = await fetch_job_history(client, f"{target_url}/history/{pid}", pid, attempts=3600)
        if not data: raise RuntimeError("Job timeout")

        outputs = data.get("outputs", {})
        flow_bytes = None
        coords_bytes, vis_bytes = None, None
        query_points_bytes, depths_bytes, video_bytes_npy = None, None, None
        intrinsics_bytes, extrinsics_bytes = None, None
        mask_orig_bytes, mask_scaled_bytes = None, None

        # TAPIP3D saves NPY files to RAM disk for fastest I/O
        RAMDISK_OUTPUT_DIR = "/dev/shm/comfyui_tapip3d"
        coords_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_coords_3d.npy")
        vis_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_visibilities.npy")
        query_points_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_query_points.npy")
        depths_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_depths.npy")
        video_npy_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_video.npy")
        intrinsics_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_intrinsics.npy")
        extrinsics_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_extrinsics.npy")
        mask_orig_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_mask_orig.npy")
        mask_scaled_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_mask_scaled.npy")
        tracking_video_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_tracking_video.mp4")
        flow_debug_path = os.path.join(RAMDISK_OUTPUT_DIR, f"flow_extraction_{safe_id}_flow_debug.png")

        # Poll for files
        elapsed = 0
        while elapsed < 20 and not os.path.exists(coords_path):
            await asyncio.sleep(0.5)
            elapsed += 0.5

        if os.path.exists(coords_path):
            coords_bytes = await _read_file_async(coords_path)
            cleanup_paths.append(coords_path)
        else: raise RuntimeError(f"coords_3d.npy missing at {coords_path}")

        if os.path.exists(vis_path):
            vis_bytes = await _read_file_async(vis_path)
            cleanup_paths.append(vis_path)

        object_flow_evaluation = None
        try:
            coords_arr = np.load(io.BytesIO(coords_bytes), allow_pickle=False)
            vis_arr = np.load(io.BytesIO(vis_bytes), allow_pickle=False) if vis_bytes else None
            object_flow_evaluation = evaluate_object_flow(
                coords_arr,
                vis_arr,
                layout="time_major",
                flow_switch_theta_deg=req.flow_switch_theta_deg,
            ).to_dict()
            _log_debug(f"[Flow Job {job_id}] Object flow evaluation: {object_flow_evaluation.get('reason')}")
        except Exception as eval_error:
            object_flow_evaluation = {
                "valid": False,
                "should_switch_to_hand": False,
                "flow_switch_theta_deg": req.flow_switch_theta_deg,
                "reason": f"object flow evaluation failed: {eval_error}",
            }

        # Large TAPIP3D debug arrays are opt-in; the production path only needs coords/vis.
        if req.return_auxiliary_arrays:
            if os.path.exists(query_points_path):
                query_points_bytes = await _read_file_async(query_points_path)
                cleanup_paths.append(query_points_path)
                _log_debug(f"[Flow Job {job_id}] Loaded query_points: {len(query_points_bytes)} bytes")

            if os.path.exists(depths_path):
                depths_bytes = await _read_file_async(depths_path)
                cleanup_paths.append(depths_path)
                _log_debug(f"[Flow Job {job_id}] Loaded depths: {len(depths_bytes)} bytes")

            if os.path.exists(video_npy_path):
                video_bytes_npy = await _read_file_async(video_npy_path)
                cleanup_paths.append(video_npy_path)
                _log_debug(f"[Flow Job {job_id}] Loaded video.npy: {len(video_bytes_npy)} bytes")

            if os.path.exists(intrinsics_path):
                intrinsics_bytes = await _read_file_async(intrinsics_path)
                cleanup_paths.append(intrinsics_path)
                _log_debug(f"[Flow Job {job_id}] Loaded intrinsics: {len(intrinsics_bytes)} bytes")

            if os.path.exists(extrinsics_path):
                extrinsics_bytes = await _read_file_async(extrinsics_path)
                cleanup_paths.append(extrinsics_path)
                _log_debug(f"[Flow Job {job_id}] Loaded extrinsics: {len(extrinsics_bytes)} bytes")

            if os.path.exists(mask_orig_path):
                mask_orig_bytes = await _read_file_async(mask_orig_path)
                cleanup_paths.append(mask_orig_path)
                _log_debug(f"[Flow Job {job_id}] Loaded mask_orig: {len(mask_orig_bytes)} bytes")

            if os.path.exists(mask_scaled_path):
                mask_scaled_bytes = await _read_file_async(mask_scaled_path)
                cleanup_paths.append(mask_scaled_path)
                _log_debug(f"[Flow Job {job_id}] Loaded mask_scaled: {len(mask_scaled_bytes)} bytes")

        elif req.return_debug_media:
            for debug_mask_path in (mask_orig_path, mask_scaled_path):
                if os.path.exists(debug_mask_path):
                    cleanup_paths.append(debug_mask_path)

        if req.return_flow_image:
            flow_prefix = f"flow_extraction_{safe_id}_flow"
            for nid, out in outputs.items():
                if not isinstance(out, dict): continue
                for img in out.get("images") or []:
                    if flow_prefix in img.get("filename", ""):
                        paths = [os.path.join(COMFYUI_OUTPUT_DIR, img.get("subfolder",""), img["filename"]),
                                 os.path.join(COMFYUI_OUTPUT_DIR, img["filename"])]
                        for p in paths:
                            if os.path.exists(p):
                                flow_bytes = await _read_file_async(p)
                                cleanup_paths.append(p)
                                _log_debug(f"[Flow Job {job_id}] Found flow PNG from outputs: {len(flow_bytes)} bytes")
                                break
                        if not flow_bytes:
                            r = await client.get(f"{target_url}/view", params={"filename": img["filename"], "subfolder": img.get("subfolder",""), "type": "output"})
                            if r.status_code == 200:
                                flow_bytes = r.content
                        break
                if flow_bytes:
                    break

            if not flow_bytes:
                flow_pattern = os.path.join(COMFYUI_OUTPUT_DIR, f"{flow_prefix}*.png")
                flow_matches = glob.glob(flow_pattern)
                if flow_matches:
                    flow_bytes = await _read_file_async(flow_matches[0])
                    cleanup_paths.append(flow_matches[0])
                    _log_debug(f"[Flow Job {job_id}] Found flow PNG via glob: {len(flow_bytes)} bytes")

        sam3_video_bytes = None
        tapip3d_video_bytes = None
        tracking_video_bytes = None
        flow_debug_bytes = None
        if req.return_debug_media or req.return_sam3_video:
            # Fetch SAM3 mask tracking video (saved by SaveVideo node)
            sam3_video_prefix = f"flow_extraction_{safe_id}_sam3_tracking"
            sam3_video_bytes, sam3_video_path = await fetch_video_by_prefix(target_url, outputs, sam3_video_prefix)
            if sam3_video_bytes:
                _log_debug(f"[Flow Job {job_id}] Found SAM3 tracking video from outputs: {len(sam3_video_bytes)} bytes")
                if sam3_video_path:
                    cleanup_paths.append(sam3_video_path)

            # Fallback: Search directly in output directory (SaveVideo appends _00001_)
            if not sam3_video_bytes:
                sam3_pattern = os.path.join(COMFYUI_OUTPUT_DIR, "**", f"{sam3_video_prefix}*.mp4")
                sam3_matches = glob.glob(sam3_pattern, recursive=True)
                if sam3_matches:
                    sam3_video_bytes = await _read_file_async(sam3_matches[0])
                    cleanup_paths.append(sam3_matches[0])
                    _log_debug(f"[Flow Job {job_id}] Found SAM3 tracking video via glob: {len(sam3_video_bytes)} bytes")

        if req.return_debug_media:
            # Fetch TAPIP3D point tracking video (saved by SaveVideo node)
            tapip3d_video_prefix = f"flow_extraction_{safe_id}_tapip3d_tracking"
            tapip3d_video_bytes, tapip3d_video_path = await fetch_video_by_prefix(target_url, outputs, tapip3d_video_prefix)
            if tapip3d_video_bytes:
                _log_debug(f"[Flow Job {job_id}] Found TAPIP3D tracking video from outputs: {len(tapip3d_video_bytes)} bytes")
                if tapip3d_video_path:
                    cleanup_paths.append(tapip3d_video_path)

            # Fallback: Search directly in output directory (SaveVideo appends _00001_)
            if not tapip3d_video_bytes:
                tapip3d_pattern = os.path.join(COMFYUI_OUTPUT_DIR, "**", f"{tapip3d_video_prefix}*.mp4")
                tapip3d_matches = glob.glob(tapip3d_pattern, recursive=True)
                if tapip3d_matches:
                    tapip3d_video_bytes = await _read_file_async(tapip3d_matches[0])
                    cleanup_paths.append(tapip3d_matches[0])
                    _log_debug(f"[Flow Job {job_id}] Found TAPIP3D tracking video via glob: {len(tapip3d_video_bytes)} bytes")

            if os.path.exists(tracking_video_path):
                tracking_video_bytes = await _read_file_async(tracking_video_path)
                cleanup_paths.append(tracking_video_path)
                tapip3d_video_bytes = tracking_video_bytes
                _log_debug(f"[Flow Job {job_id}] Found original-format TAPIP3D tracking video: {len(tracking_video_bytes)} bytes")

            if os.path.exists(flow_debug_path):
                flow_debug_bytes = await _read_file_async(flow_debug_path)
                cleanup_paths.append(flow_debug_path)
                _log_debug(f"[Flow Job {job_id}] Found flow debug image: {len(flow_debug_bytes)} bytes")

        timing["total_s"] = time.perf_counter() - t0
        _log_timing(
            "FlowExtraction",
            timing["total_s"],
            job=job_id[:8],
            prepare=f"{timing['prepare_s']:.2f}s",
            depth_mode=depth_mode,
            tapip3d=tapip3d_node,
        )

        async with job_lock:
            jobs[job_id].update({
                "status": "completed",
                "flow_image_bytes": flow_bytes,
                "coords_3d_bytes": coords_bytes,
                "visibilities_bytes": vis_bytes,
                "query_points_bytes": query_points_bytes,
                "depths_bytes": depths_bytes,
                "video_npy_bytes": video_bytes_npy,
                "intrinsics_bytes": intrinsics_bytes,
                "extrinsics_bytes": extrinsics_bytes,
                "mask_orig_bytes": mask_orig_bytes,
                "mask_scaled_bytes": mask_scaled_bytes,
                "tracking_video_bytes": tracking_video_bytes,
                "flow_debug_bytes": flow_debug_bytes,
                "sam3_video_bytes": sam3_video_bytes,
                "tapip3d_video_bytes": tapip3d_video_bytes,
                "object_flow_evaluation": object_flow_evaluation,
                "timing": timing
            })

    except Exception as e:
        traceback.print_exc()
        async with job_lock: jobs[job_id].update({"status": "failed", "error": str(e)})
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                try:
                    if os.path.isdir(p): shutil.rmtree(p)
                    else: os.remove(p)
                except OSError:
                    pass

# --- API Endpoints ---

@app.post("/jobs/depth_estimation")
async def submit_depth_estimation_job(req: DepthEstimationRequest, bg: BackgroundTasks):
    """Submit depth estimation job."""
    jid = str(uuid.uuid4())
    async with job_lock: jobs[jid] = {"status": "queued"}
    bg.add_task(run_depth_estimation_job, jid, req)
    return {"status": "queued", "job_id": jid}

@app.post("/jobs/flow_extraction")
async def submit_flow_extraction_job(req: FlowExtractionRequest, bg: BackgroundTasks):
    """Submit flow extraction job."""
    jid = str(uuid.uuid4())
    async with job_lock: jobs[jid] = {"status": "queued"}
    bg.add_task(run_flow_extraction_job, jid, req)
    return {"status": "queued", "job_id": jid}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Return the current status of a queued job."""
    async with job_lock: job = jobs.get(job_id)
    if not job: raise HTTPException(404, "Job not found")
    res = {"status": job.get("status"), "timing": job.get("timing"), "error": job.get("error")}
    if "depth_video_bytes" in job: res["has_depth_video"] = True
    if job.get("flow_image_bytes"): res["has_flow_image"] = True
    if "coords_3d_bytes" in job: res["has_coords_3d"] = True
    if job.get("query_points_bytes"): res["has_query_points"] = True
    if job.get("depths_bytes"): res["has_depths"] = True
    if job.get("video_npy_bytes"): res["has_video_npy"] = True
    if job.get("intrinsics_bytes"): res["has_intrinsics"] = True
    if job.get("extrinsics_bytes"): res["has_extrinsics"] = True
    if job.get("sam3_video_bytes"): res["has_sam3_video"] = True
    if job.get("tapip3d_video_bytes"): res["has_tapip3d_video"] = True
    if job.get("tracking_video_bytes"): res["has_tracking_video"] = True
    if job.get("flow_debug_bytes"): res["has_flow_debug"] = True
    if job.get("mask_orig_bytes"): res["has_mask_orig"] = True
    if job.get("mask_scaled_bytes"): res["has_mask_scaled"] = True
    if job.get("object_flow_evaluation"): res["object_flow_evaluation"] = job.get("object_flow_evaluation")
    return res

@app.get("/result/{job_id}/depth_video")
async def get_depth_video_result(job_id: str):
    """Return a completed job's metric-depth video."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    return Response(content=job.get("depth_video_bytes"), media_type="video/mp4")

@app.get("/result/{job_id}/flow")
async def get_flow_result(job_id: str):
    """Get the saved object-flow trajectory PNG."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("flow_image_bytes")
    if not data: raise HTTPException(404, "Flow PNG not available")
    return Response(content=data, media_type="image/png")

@app.get("/result/{job_id}/coords_3d")
async def get_coords_3d_result(job_id: str):
    """Return a completed job's tracked 3D coordinates."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    return Response(content=job.get("coords_3d_bytes"), media_type="application/octet-stream")

@app.get("/result/{job_id}/visibilities")
async def get_visibilities_result(job_id: str):
    """Return a completed job's point visibilities."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    return Response(content=job.get("visibilities_bytes"), media_type="application/octet-stream")

@app.get("/result/{job_id}/query_points")
async def get_query_points_result(job_id: str):
    """Get query points used for tracking (shape: [num_points, 4])"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("query_points_bytes")
    if not data: raise HTTPException(404, "Query points not available")
    return Response(content=data, media_type="application/octet-stream")

@app.get("/result/{job_id}/depths")
async def get_depths_result(job_id: str):
    """Get depth frames resized to original video resolution (shape: [T, H, W])"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("depths_bytes")
    if not data: raise HTTPException(404, "Depths not available")
    return Response(content=data, media_type="application/octet-stream")

@app.get("/result/{job_id}/video_npy")
async def get_video_npy_result(job_id: str):
    """Get video frames as numpy array (shape: [T, H, W, 3], values in [0,1])"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("video_npy_bytes")
    if not data: raise HTTPException(404, "Video NPY not available")
    return Response(content=data, media_type="application/octet-stream")

@app.get("/result/{job_id}/intrinsics")
async def get_intrinsics_result(job_id: str):
    """Get intrinsics matrices (shape: [T, 3, 3])"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("intrinsics_bytes")
    if not data: raise HTTPException(404, "Intrinsics not available")
    return Response(content=data, media_type="application/octet-stream")

@app.get("/result/{job_id}/extrinsics")
async def get_extrinsics_result(job_id: str):
    """Get extrinsics matrices (shape: [T, 4, 4], identity matrices)"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("extrinsics_bytes")
    if not data: raise HTTPException(404, "Extrinsics not available")
    return Response(content=data, media_type="application/octet-stream")

@app.get("/result/{job_id}/tracking_video")
async def get_tracking_video_result(job_id: str):
    """Get the original-format TAPIP3D tracking visualization video."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    video_bytes = job.get("tracking_video_bytes")
    if not video_bytes: raise HTTPException(404, "Tracking video not available")
    return Response(content=video_bytes, media_type="video/mp4")

@app.get("/result/{job_id}/flow_debug")
async def get_flow_debug_result(job_id: str):
    """Get the TAPIP3D flow debug image."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("flow_debug_bytes")
    if not data: raise HTTPException(404, "Flow debug image not available")
    return Response(content=data, media_type="image/png")

@app.get("/result/{job_id}/sam3_video")
async def get_sam3_video_result(job_id: str):
    """Get the SAM3 mask tracking visualization video"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    video_bytes = job.get("sam3_video_bytes")
    if not video_bytes: raise HTTPException(404, "SAM3 tracking video not available")
    return Response(content=video_bytes, media_type="video/mp4")

@app.get("/result/{job_id}/tapip3d_video")
async def get_tapip3d_video_result(job_id: str):
    """Get the TAPIP3D point tracking visualization video"""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    video_bytes = job.get("tapip3d_video_bytes")
    if not video_bytes: raise HTTPException(404, "TAPIP3D tracking video not available")
    return Response(content=video_bytes, media_type="video/mp4")

@app.get("/result/{job_id}/mask_orig")
async def get_mask_orig_result(job_id: str):
    """Get the original full-size mask (shape: [T, H_orig, W_orig], float32 in [0, 1])."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("mask_orig_bytes")
    if not data: raise HTTPException(404, "Original mask not available")
    return Response(content=data, media_type="application/octet-stream")

@app.get("/result/{job_id}/mask_scaled")
async def get_mask_scaled_result(job_id: str):
    """Get the TAPIP3D inference-resolution mask (shape: [T, H_scaled, W_scaled], float32 in [0, 1])."""
    async with job_lock: job = jobs.get(job_id)
    if not job or job["status"] != "completed": raise HTTPException(404, "Not ready")
    data = job.get("mask_scaled_bytes")
    if not data: raise HTTPException(404, "Scaled mask not available")
    return Response(content=data, media_type="application/octet-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
