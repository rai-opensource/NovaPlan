# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Queue Wan video-generation and selection-flow jobs across ComfyUI workers."""

import uvicorn
import httpx
import asyncio
import aiofiles
import uuid
import os
import base64
import time
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional

# ================= CONFIGURATION =================
REPO_ROOT = Path(__file__).resolve().parents[1]
API_PORT = int(os.environ.get("PORT", "7000"))
if not 1 <= API_PORT <= 65535:
    raise ValueError("PORT must be between 1 and 65535")
WORKER_START_PORT = int(os.environ.get("COMFYUI_WORKER_START_PORT", "8188"))
WORKER_COUNT = int(os.environ.get("COMFYUI_NUM_WORKERS", "8"))
WORKER_END_PORT = int(os.environ.get("COMFYUI_WORKER_END_PORT", str(WORKER_START_PORT + WORKER_COUNT)))
if not 1 <= WORKER_START_PORT <= 65535 or not 1 <= WORKER_END_PORT - 1 <= 65535:
    raise ValueError("ComfyUI worker ports must be between 1 and 65535")
if WORKER_COUNT < 1 or WORKER_END_PORT - WORKER_START_PORT != WORKER_COUNT:
    raise ValueError("COMFYUI worker range must contain exactly COMFYUI_NUM_WORKERS ports")
WORKER_PORTS = range(WORKER_START_PORT, WORKER_END_PORT)
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1").rstrip("/")

# Worker and queue-server processes must share these directories. They may be
# local disk on one host or shared storage on a multi-host deployment.
DEFAULT_COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", REPO_ROOT / ".runtime" / "comfyui"))
COMFYUI_INPUT_DIR = os.environ.get("COMFYUI_INPUT_DIR", str(DEFAULT_COMFYUI_DIR / "input"))
COMFYUI_OUTPUT_DIR = os.environ.get("COMFYUI_OUTPUT_DIR", str(DEFAULT_COMFYUI_DIR / "output"))

# ================= HELPER FUNCTIONS =================

async def _read_file_async(path: str) -> bytes:
    async with aiofiles.open(path, "rb") as f:
        return await f.read()

async def try_load_video_from_disk(fname, subfolder=""):
    """Try to load video from various possible locations"""
    paths = [
        os.path.join(COMFYUI_OUTPUT_DIR, subfolder, fname),
        os.path.join(COMFYUI_OUTPUT_DIR, fname),
        os.path.join(COMFYUI_OUTPUT_DIR, "video", fname)
    ]
    for p in paths:
        if os.path.exists(p):
            return await _read_file_async(p), p
    return None, None


async def _read_comfy_output(filename: str, subfolder: str = ""):
    """Read a ComfyUI output descriptor from the shared output tree."""

    if not filename:
        return None, None
    candidates = []
    if os.path.isabs(filename):
        candidates.append(filename)
    if subfolder:
        candidates.append(os.path.join(COMFYUI_OUTPUT_DIR, subfolder, filename))
    candidates.append(os.path.join(COMFYUI_OUTPUT_DIR, filename))
    for path in candidates:
        if os.path.isfile(path):
            return await _read_file_async(path), path
    return None, None


async def _collect_job_results(job_id: str) -> Dict[str, Any]:
    """Collect videos and flow images from recorded ComfyUI metadata first."""

    job = jobs.get(job_id, {})
    result: Dict[str, Any] = {"job_id": job_id, "videos": [], "flow_images": []}
    seen_paths = set()

    async def _append(kind: str, filename: str, subfolder: str = "") -> None:
        data, path = await _read_comfy_output(filename, subfolder)
        if not data or not path or path in seen_paths:
            return
        seen_paths.add(path)
        result[kind].append({
            "filename": os.path.basename(path),
            "subfolder": subfolder,
            "size_bytes": len(data),
            "data_base64": base64.b64encode(data).decode("ascii"),
        })

    for entry in job.get("video_outputs", []) or []:
        if isinstance(entry, dict):
            await _append("videos", entry.get("filename", ""), entry.get("subfolder", ""))
    for entry in job.get("all_videos", []) or []:
        if isinstance(entry, dict):
            await _append("videos", entry.get("filename", ""), entry.get("subfolder", ""))
        elif entry:
            await _append("videos", str(entry))
    if job.get("wan_video"):
        await _append("videos", str(job["wan_video"]))

    for entry in job.get("flow_images", []) or []:
        if isinstance(entry, dict):
            await _append("flow_images", entry.get("filename", ""), entry.get("subfolder", ""))

    # Compatibility with jobs created by older server processes that did not
    # retain output descriptors. Walk subfolders as well as the output root.
    if not result["videos"] or not result["flow_images"]:
        need_videos = not result["videos"]
        need_flows = not result["flow_images"]
        try:
            for root, _, filenames in os.walk(COMFYUI_OUTPUT_DIR):
                for filename in sorted(filenames):
                    if job_id not in filename:
                        continue
                    relative_root = os.path.relpath(root, COMFYUI_OUTPUT_DIR)
                    subfolder = "" if relative_root == "." else relative_root
                    if filename.endswith(".mp4") and need_videos:
                        await _append("videos", filename, subfolder)
                    elif filename.endswith(".png") and "flow" in filename and need_flows:
                        await _append("flow_images", filename, subfolder)
        except OSError:
            pass

    return result

# Model Filenames
MODELS = {
    "14B": {
        "unet_high": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", 
        "unet_low": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "vae": "wan_2.1_vae.safetensors",
        "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "lora_high": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "lora_low": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    }
}
ACTIVE_MODEL = "14B"
SERVER_CONTRACT_VERSION = 3
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
SAMPLER_ALIASES = {"unipc": "uni_pc", "dpm++": "dpmpp_2m"}
WAN_WORKFLOW_CLASS_TYPES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoaderModelOnly",
    "CLIPTextEncode",
    "WanImageToVideo",
    "WanFirstLastFrameToVideo",
    "ModelSamplingSD3",
    "KSamplerAdvanced",
    "VAEDecode",
    "CreateVideo",
    "SaveVideo",
}
# =================================================

app = FastAPI(title="NovaPlan Wan 2.2 Video Cluster")
http_client: Optional[httpx.AsyncClient] = None
jobs: Dict[str, Dict[str, Any]] = {}


class AffinityWorkerPool:
    """Route jobs toward workers that already hold the required model family."""

    def __init__(self, worker_urls):
        self._condition = asyncio.Condition()
        self._workers = {}
        self._reset_states(worker_urls)

    def _reset_states(self, worker_urls) -> None:
        self._workers = {
            url: {
                "order": order,
                "busy": False,
                "generation_warm": False,
                "flow_warm": False,
                "completed_jobs": 0,
                "last_mode": None,
            }
            for order, url in enumerate(worker_urls)
        }

    async def reset(self, worker_urls) -> None:
        """Reset worker-affinity state."""
        async with self._condition:
            self._reset_states(worker_urls)
            self._condition.notify_all()

    @staticmethod
    def _score(state: Dict[str, Any], mode: str):
        generation_warm = state["generation_warm"]
        flow_warm = state["flow_warm"]

        if mode == "full":
            role_score = (
                0 if generation_warm and flow_warm
                else 1 if generation_warm
                else 2 if not flow_warm
                else 3
            )
        elif mode == "generate_only":
            role_score = (
                0 if generation_warm and not flow_warm
                else 1 if generation_warm
                else 2 if not flow_warm
                else 3
            )
        else:
            role_score = (
                0 if flow_warm and not generation_warm
                else 1 if flow_warm
                else 2 if not generation_warm
                else 3
            )
        return role_score, state["completed_jobs"], state["order"]

    @staticmethod
    def _residency_label(state: Dict[str, Any]) -> str:
        if state["generation_warm"] and state["flow_warm"]:
            return "generation+flow"
        if state["generation_warm"]:
            return "generation"
        if state["flow_warm"]:
            return "flow"
        return "cold"

    async def acquire(self, mode: str) -> str:
        """Acquire the next observation bundle."""
        async with self._condition:
            while True:
                idle = [
                    (url, state)
                    for url, state in self._workers.items()
                    if not state["busy"]
                ]
                if idle:
                    worker_url, state = min(idle, key=lambda item: self._score(item[1], mode))
                    residency = self._residency_label(state)
                    state["busy"] = True
                    print(
                        f"[worker-pool] mode={mode} worker={worker_url} "
                        f"residency_before={residency}"
                    )
                    return worker_url
                await self._condition.wait()

    async def release(self, worker_url: str, mode: str, *, succeeded: bool) -> None:
        """Release a worker back to the available pool."""
        async with self._condition:
            state = self._workers[worker_url]
            if succeeded:
                if mode in {"full", "generate_only"}:
                    state["generation_warm"] = True
                if mode in {"full", "flow_only"}:
                    state["flow_warm"] = True
                state["completed_jobs"] += 1
                state["last_mode"] = mode
            state["busy"] = False
            self._condition.notify_all()

    def snapshot(self):
        """Return the current worker-affinity state."""
        return [
            {
                "worker": url,
                "busy": state["busy"],
                "estimated_residency": self._residency_label(state),
                "completed_jobs": state["completed_jobs"],
                "last_mode": state["last_mode"],
            }
            for url, state in self._workers.items()
        ]


WORKER_URLS = [f"{COMFYUI_HOST}:{port}" for port in WORKER_PORTS]
worker_pool = AffinityWorkerPool(WORKER_URLS)


def _get_http_client() -> httpx.AsyncClient:
    global http_client
    if http_client is None or http_client.is_closed:
        http_client = httpx.AsyncClient(timeout=1800.0)
    return http_client

class JobRequest(BaseModel):
    """Validate a queued video-generation request."""
    mode: Literal["full", "generate_only", "flow_only"] = "full"
    
    # WAN Parameters
    prompt: Optional[str] = "A single human right hand performs the requested action."
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT
    first_frame_base64: Optional[str] = None
    last_frame_base64: Optional[str] = None # Enables first-last-frame conditioning when provided.
    fast_mode: bool = True # True=4steps+LoRA (default), False=20steps without LoRA
    seed: int = 0
    width: int = Field(default=1280, gt=0)
    height: int = Field(default=720, gt=0)
    frames: int = Field(default=41, gt=0)
    fps: int = Field(default=16, gt=0)
    num_videos: int = Field(default=1, gt=0) # Number of videos to generate per job (batch_size)
    sampling_steps: Optional[int] = Field(default=None, ge=2)
    guide_scale: Optional[float] = Field(default=None, gt=0)
    sample_solver: Optional[Literal["unipc", "uni_pc", "dpm++", "dpmpp_2m", "euler"]] = None
    
    # Flow Parameters
    video_filename: Optional[str] = None # Required if flow_only (server-side file)
    video_base64: Optional[str] = None # Alternative: send video data directly
    mask_prompt: str = Field(default="object", min_length=1)
    flow_num_vis_points: int = Field(default=0, ge=0) # 0 = visualize all tracked points

@app.get("/health")
async def health():
    """Return service health and runtime readiness information."""
    return {
        "status": "ok",
        "contract_version": SERVER_CONTRACT_VERSION,
        "wan_full_inline_flow": True,
        "flow_only_available": True,
        "worker_count": len(WORKER_PORTS),
        "mode_aware_worker_affinity": True,
        "workers": worker_pool.snapshot(),
    }

@app.on_event("startup")
async def startup_event():
    """Initialize HTTP and worker-pool state during service startup."""
    _get_http_client()
    await worker_pool.reset(WORKER_URLS)
    print(f"🚀 Cluster Online: {len(WORKER_PORTS)} GPUs Registered.")


@app.on_event("shutdown")
async def shutdown_event():
    """Close the shared HTTP client during service shutdown."""
    global http_client
    if http_client is not None and not http_client.is_closed:
        await http_client.aclose()
    http_client = None

# ================= UNIFIED WORKFLOW GENERATOR =================

def generate_unified_workflow(job_id, img_path, req: JobRequest, video_path=None, end_img_path=None):
    """
    Minimal workflow generator - only loads what's needed for each mode.
    
    ComfyUI owns the WAN model lifecycle. Do not use ``--disable-smart-memory``
    as a pinning switch: ComfyUI defines it as aggressive CPU offloading. The
    NovaPlan SAM3 and CoTracker3 nodes keep their own process-local model caches
    so warm requests reuse those model instances.
    
    Args:
        job_id: Unique job identifier
        img_path: Path to input image (for WAN generation)
        req: JobRequest with mode specification
        video_path: Path to video (for flow_only mode)
        end_img_path: Path to final image (for first-last-frame mode)
    """
    conf = MODELS[ACTIVE_MODEL]
    wf = {}
    wan_video_output = None
    neg_prompt = req.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    
    # ========== CONDITIONAL WAN LOADING & EXECUTION ==========
    if req.mode in ["generate_only", "full"]:
        # Load WAN models only when needed
        wf["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": conf["unet_high"], "weight_dtype": "default"}}
        wf["2"] = {"class_type": "UNETLoader", "inputs": {"unet_name": conf["unet_low"], "weight_dtype": "default"}}
        wf["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": conf["clip"], "type": "wan", "weight_dtype": "default"}}
        wf["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": conf["vae"]}}
        
        # Fast mode: 4 steps + LoRAs
        if req.fast_mode:
            wf["lora_h"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": conf["lora_high"], "strength_model": 1.0}}
            wf["lora_l"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": conf["lora_low"], "strength_model": 1.0}}
            high_model, low_model = ["lora_h", 0], ["lora_l", 0]
            shift, steps, split, cfg = 5.0, 4, 2, 1.0
        else:
            high_model, low_model = ["1", 0], ["2", 0]
            shift, steps, split, cfg = 8.0, 20, 10, 3.5

        if req.sampling_steps is not None:
            steps = req.sampling_steps
            split = max(1, steps // 2)
        if req.guide_scale is not None:
            cfg = req.guide_scale
        sampler_name = SAMPLER_ALIASES.get(req.sample_solver or "euler", req.sample_solver or "euler")
        
        # Load input image
        wf["5"] = {"class_type": "LoadImage", "inputs": {"image": img_path}}
        
        # Text encoding
        wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": req.prompt}}
        wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": neg_prompt}}
        
        # WAN I2V node, or first-last-frame conditioning when an end image is supplied.
        if req.last_frame_base64 and end_img_path:
            wf["5_end"] = {"class_type": "LoadImage", "inputs": {"image": end_img_path}}
            wf["8"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {
                "positive": ["6", 0], "negative": ["7", 0], "vae": ["4", 0],
                "start_image": ["5", 0], "end_image": ["5_end", 0],
                "width": req.width, "height": req.height,
                "length": req.frames, "batch_size": 1
            }}
        else:
            wf["8"] = {"class_type": "WanImageToVideo", "inputs": {
                "positive": ["6", 0], "negative": ["7", 0], "vae": ["4", 0],
                "start_image": ["5", 0], "width": req.width, "height": req.height,
                "length": req.frames, "batch_size": 1  # Always 1 for distributed execution
            }}
        
        # Sampling chain
        wf["9"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": high_model, "shift": shift}}
        wf["10"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": low_model, "shift": shift}}
        wf["11"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["9", 0], "add_noise": "enable", "noise_seed": req.seed,"control_after_generate":"randomize",
            "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": "simple",
            "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["8", 2],
            "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"
        }}
        wf["12"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["10", 0], "add_noise": "disable", "noise_seed": 0,"control_after_generate":"fixed",
            "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": "simple",
            "positive": ["8", 0], "negative": ["8", 1], "latent_image": ["11", 0],
            "start_at_step": split, "end_at_step": steps, "return_with_leftover_noise": "disable"
        }}
        
        # Decode and create video
        wf["13"] = {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}}
        wf["14"] = {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "fps": req.fps}}
        wf["15"] = {"class_type": "SaveVideo", "inputs": {
            "video": ["14", 0],
            "filename_prefix": f"novaplan_wan_{job_id}",
            "format": "mp4",
            "codec": "h264"
        }}
        wan_video_output = ["14", 0]
    
    # ========== CONDITIONAL FLOW EXECUTION ==========
    if req.mode in ["flow_only", "full"]:
        # Determine video input
        if req.mode == "full" and wan_video_output:
            # Use WAN-generated video
            flow_video_input = wan_video_output
        elif video_path:
            # Load video from input/output directory
            if os.path.isabs(video_path):
                video_full_path = video_path
            else:
                video_full_path = os.path.join(COMFYUI_OUTPUT_DIR, video_path)
            wf["100"] = {"class_type": "LoadVideo", "inputs": {
                "file": video_full_path
            }}
            flow_video_input = ["100", 0]
        else:
            raise ValueError("flow_only mode requires video_path")
        
        # SAM3 + CoTracker execution
        wf["39"] = {"class_type": "Sam3VideoNode", "inputs": {
            "video": flow_video_input,
            "prompt": req.mask_prompt,
            # Production rollout selection only consumes the SAM3 mask. Avoid
            # rendering a second visualization video inside the flow branch.
            "return_visualization": False,
        }}
        wf["97"] = {"class_type": "CoTracker3Node", "inputs": {
            "video": flow_video_input, "mask": ["39", 1],
            "grid_size": 30, "checkpoint": "scaled_offline.pth",
            "num_vis_points": req.flow_num_vis_points,
        }}
        wf["98"] = {"class_type": "SaveImage", "inputs": {
            "images": ["97", 0], "filename_prefix": f"novaplan_flow_{job_id}"
        }}

    if req.mode == "flow_only":
        wan_nodes = {
            node_id: node.get("class_type")
            for node_id, node in wf.items()
            if node.get("class_type") in WAN_WORKFLOW_CLASS_TYPES
        }
        if wan_nodes:
            raise RuntimeError(
                "flow_only workflow must not include WAN generation nodes: "
                f"{wan_nodes}"
            )
    
    return wf

# ================= EXECUTION ENGINE =================

async def execute_on_worker(worker_url, workflow, job_id, stage_label):
    """Execute on worker."""
    print(f"[{job_id}] ⚡ Running {stage_label} on {worker_url}")
    try:
        client = _get_http_client()
        resp = await client.post(f"{worker_url}/prompt", json={"prompt": workflow, "client_id": job_id})
        resp.raise_for_status()
        prompt_id = resp.json()['prompt_id']

        while True:
            await asyncio.sleep(1)
            hist_resp = await client.get(f"{worker_url}/history/{prompt_id}")
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    return hist[prompt_id]
    except Exception as e:
        print(f"[{job_id}] ❌ Error on {worker_url}: {e}")
        raise e

async def process_single_video(sub_job_id: str, req: JobRequest, img_name: str, worker_url: str, video_path=None, end_img_name=None):
    """Process a single video/flow on a specific worker"""
    try:
        wf = generate_unified_workflow(sub_job_id, img_name, req, video_path=video_path, end_img_path=end_img_name)
        stage_label = f"{req.mode} (Fast={req.fast_mode})" if req.mode != "flow_only" else "flow_only"
        result = await execute_on_worker(worker_url, wf, sub_job_id, stage_label)
        
        response = {"worker": worker_url, "history": result}
        
        # Extract WAN video if generated
        if req.mode in ["generate_only", "full"]:
            outputs = result.get('outputs', {}).get('15', {}).get('videos', [])
            if outputs:
                response["video_filename"] = outputs[0]['filename']
                response["video_outputs"] = outputs
        
        # Extract flow image if generated
        if req.mode in ["flow_only", "full"]:
            outputs = result.get('outputs', {}).get('98', {}).get('images', [])
            if outputs:
                response["flow_images"] = outputs

        missing = []
        if req.mode in ["generate_only", "full"] and not response.get("video_outputs"):
            missing.append("SaveVideo node 15")
        if req.mode in ["flow_only", "full"] and not response.get("flow_images"):
            missing.append("SaveImage flow node 98")
        if missing:
            status = result.get("status", {})
            output_nodes = sorted(result.get("outputs", {}).keys())
            raise RuntimeError(
                f"ComfyUI {req.mode} workflow did not produce {', '.join(missing)}; "
                f"status={status}; output_nodes={output_nodes}"
            )
        
        return response
    except Exception as e:
        print(f"[{sub_job_id}] Failed on {worker_url}: {e}")
        return None

async def process_job(job_id: str, req: JobRequest):
    """Process job."""
    jobs[job_id]["status"] = "processing"
    t_start = time.time()
    timing = {"prepare_s": 0, "comfy_processing_s": 0, "total_s": 0, "node_breakdown": []}
    
    try:
        # Prepare input image if needed
        img_name = None
        end_img_name = None
        if req.mode in ["generate_only", "full"]:
            if not req.first_frame_base64:
                raise Exception("Missing first_frame_base64")
            t_prep = time.time()
            img_name = f"{job_id}_in.png"
            with open(os.path.join(COMFYUI_INPUT_DIR, img_name), "wb") as f:
                f.write(base64.b64decode(req.first_frame_base64))
            if req.last_frame_base64:
                end_img_name = f"{job_id}_end.png"
                with open(os.path.join(COMFYUI_INPUT_DIR, end_img_name), "wb") as f:
                    f.write(base64.b64decode(req.last_frame_base64))
            timing["prepare_s"] += time.time() - t_prep

        # Prepare video for flow_only mode
        video_path = None
        if req.mode == "flow_only":
            if req.video_base64:
                # Decode base64 video and save to temp file
                video_data = base64.b64decode(req.video_base64)
                video_filename = f"flow_input_{job_id}.mp4"
                video_path = os.path.join(COMFYUI_INPUT_DIR, video_filename)
                with open(video_path, "wb") as f:
                    f.write(video_data)
                print(f"[{job_id}] Saved flow video to {video_path}")
            elif req.video_filename:
                # Use existing file (server-side)
                video_path = req.video_filename
            else:
                raise ValueError("flow_only mode requires either video_base64 or video_filename")

        # Distributed execution for multiple videos
        if req.num_videos > 1:
            print(f"[{job_id}] Distributing {req.num_videos} jobs across GPUs...")
            t_comfy = time.time()

            # Acquire workers using model-residency affinity.
            workers = []
            for _ in range(min(req.num_videos, len(WORKER_PORTS))):
                workers.append(await worker_pool.acquire(req.mode))

            # Dispatch sub-jobs in parallel
            tasks = []
            for i in range(req.num_videos):
                sub_job_id = f"{job_id}_v{i+1}"
                import copy
                sub_req = copy.deepcopy(req)
                sub_req.num_videos = 1
                sub_req.seed = req.seed + i
                worker_url = workers[i % len(workers)]
                tasks.append(process_single_video(sub_job_id, sub_req, img_name, worker_url, video_path, end_img_name))

            results = []
            successful_workers = set()
            try:
                results = await asyncio.gather(*tasks)
                successful_workers = {
                    result["worker"]
                    for result in results
                    if result is not None and result.get("worker")
                }
            finally:
                for worker_url in workers:
                    await worker_pool.release(
                        worker_url,
                        req.mode,
                        succeeded=worker_url in successful_workers,
                    )

            timing["comfy_processing_s"] += time.time() - t_comfy

            # Process results
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                raise Exception(
                    "All distributed jobs failed. Inspect the worker logs for the preceding "
                    "ComfyUI output-node errors."
                )

            # Collect outputs
            if req.mode in ["generate_only", "full"]:
                video_files = [r.get("video_filename") for r in valid_results if r.get("video_filename")]
                if video_files:
                    jobs[job_id]["wan_video"] = video_files[0]
                    jobs[job_id]["all_videos"] = video_files
                    jobs[job_id]["generated_videos"] = len(video_files)
                jobs[job_id]["video_outputs"] = [
                    item
                    for result in valid_results
                    for item in (result.get("video_outputs") or [])
                ]

            if req.mode in ["flow_only", "full"]:
                flow_outputs = [r.get("flow_images") for r in valid_results if r.get("flow_images")]
                if flow_outputs:
                    jobs[job_id]["flow_images"] = [
                        item
                        for output_group in flow_outputs
                        for item in output_group
                    ]

                # Aggregate node timings from first result
                first_wf = generate_unified_workflow(f"{job_id}_v1", img_name, req, video_path, end_img_name)
                for node_id, node_info in valid_results[0]["history"].get('outputs', {}).items():
                    if 'execution_time' in node_info:
                        timing["node_breakdown"].append({
                            "node_id": node_id,
                            "class_type": first_wf.get(node_id, {}).get('class_type', 'unknown'),
                            "duration": node_info['execution_time']
                        })
        else:
            # Single execution
            worker = await worker_pool.acquire(req.mode)
            worker_succeeded = False
            try:
                t_comfy = time.time()
                wf = generate_unified_workflow(job_id, img_name, req, video_path, end_img_name)
                stage_label = f"{req.mode} (Fast={req.fast_mode})" if req.mode != "flow_only" else "flow_only"
                res = await execute_on_worker(worker, wf, job_id, stage_label)
                worker_succeeded = True
                timing["comfy_processing_s"] += time.time() - t_comfy

                # Extract node timings
                for node_id, node_info in res.get('outputs', {}).items():
                    if 'execution_time' in node_info:
                        timing["node_breakdown"].append({
                            "node_id": node_id,
                            "class_type": wf.get(node_id, {}).get('class_type', 'unknown'),
                            "duration": node_info['execution_time']
                        })

                # Collect outputs
                if req.mode in ["generate_only", "full"]:
                    # Parse workflow outputs to find video files (robust approach)
                    all_outputs = res.get('outputs', {})
                    video_files = []
                    video_outputs = []

                    for node_id, node_output in all_outputs.items():
                        # Check for videos output (SaveVideo primary output)
                        if node_output.get('videos'):
                            for vid_info in node_output['videos']:
                                if isinstance(vid_info, dict) and 'filename' in vid_info:
                                    video_files.append(vid_info['filename'])
                                    video_outputs.append(vid_info)
                        # Also check images output in case SaveVideo outputs there
                        elif node_output.get('images'):
                            for img_info in node_output['images']:
                                if isinstance(img_info, dict) and img_info.get('filename', '').endswith('.mp4'):
                                    video_files.append(img_info['filename'])

                    print(f"[{job_id}] Debug: Found video files: {video_files}")

                    if video_files:
                        jobs[job_id]["wan_video"] = video_files[0]  # Take first video
                        jobs[job_id]["all_videos"] = video_files
                        jobs[job_id]["video_outputs"] = video_outputs
                        print(f"[{job_id}] Debug: Set wan_video to: {video_files[0]}")

                if req.mode in ["flow_only", "full"]:
                    outputs = res.get('outputs', {}).get('98', {}).get('images', [])
                    if outputs:
                        jobs[job_id]["flow_images"] = outputs
                    else:
                        status = res.get("status", {})
                        output_nodes = sorted(res.get("outputs", {}).keys())
                        raise RuntimeError(
                            "ComfyUI workflow did not produce SaveImage flow node 98; "
                            f"status={status}; output_nodes={output_nodes}"
                        )
            finally:
                await worker_pool.release(
                    worker,
                    req.mode,
                    succeeded=worker_succeeded,
                )

        # Job completed successfully
        jobs[job_id]["status"] = "completed"
        timing["total_s"] = time.time() - t_start
        jobs[job_id]["timing"] = timing

        # FINAL FALLBACK: Find video files on disk if not already set
        if req.mode in ["generate_only", "full"] and not jobs[job_id].get("wan_video"):
            print(f"[{job_id}] Looking for video files on disk...")
            try:
                files = os.listdir(COMFYUI_OUTPUT_DIR)
                video_files = [f for f in files if f.endswith('.mp4') and job_id in f]
                video_files.sort()
                if video_files:
                    jobs[job_id]["wan_video"] = video_files[0]
                    jobs[job_id]["all_videos"] = video_files
                    print(f"[{job_id}] Found {len(video_files)} video files on disk: {video_files}")
                else:
                    print(f"[{job_id}] No video files found on disk matching job_id")
            except Exception as e:
                print(f"[{job_id}] Error searching for video files: {e}")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["timing"] = timing

@app.post("/submit")
async def submit(req: JobRequest, bg: BackgroundTasks):
    """Queue a new video-generation request."""
    if req.mode in {"full", "generate_only"} and not req.first_frame_base64:
        raise HTTPException(status_code=422, detail="first_frame_base64 is required for video generation")
    if req.mode == "flow_only" and not req.video_base64 and not req.video_filename:
        raise HTTPException(status_code=422, detail="flow_only mode requires video_base64 or video_filename")
    jid = uuid.uuid4().hex
    jobs[jid] = {"status": "queued"}
    bg.add_task(process_job, jid, req)
    return {"job_id": jid}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Return the current status of a queued job."""
    job = jobs.get(job_id, {"status": "not_found"})
    # Add summary info for multiple videos
    if "generated_videos" in job:
        status_copy = dict(job)
        status_copy["num_videos_generated"] = job["generated_videos"]
    else:
        status_copy = job
    return status_copy

@app.get("/result/{job_id}/binary")
async def get_result_binary(job_id: str):
    """Return video(s) as raw MP4 bytes for WAN jobs"""
    from fastapi.responses import Response, StreamingResponse
    print(f"Binary endpoint called for job_id: {job_id}")
    job = jobs.get(job_id)

    # If job not found in memory, still try to find video on disk
    if not job or job.get("status") != "completed":
        print(f"Binary endpoint: Job {job_id} not in memory or not completed, searching disk...")
        try:
            files = os.listdir(COMFYUI_OUTPUT_DIR)
            matching = sorted([f for f in files if f.endswith('.mp4') and job_id in f])
            if matching:
                print(f"Binary endpoint: Found {len(matching)} videos on disk for job {job_id}")
                
                # Handle multiple videos
                if len(matching) > 1:
                    boundary = f"----VideoBoundary{job_id}"
                    print(f"Binary endpoint: Returning {len(matching)} videos as multipart")
                    async def multipart_generator():
                        for i, video_file in enumerate(matching):
                            video_data, actual_path = await try_load_video_from_disk(video_file)
                            if video_data:
                                header = (f"--{boundary}\r\n"
                                f"Content-Type: video/mp4\r\n"
                                f"Content-Disposition: inline; filename=video_{i+1}.mp4\r\n"
                                f"X-Video-Index: {i}\r\n\r\n").encode("utf-8")
                                yield header
                                yield video_data
                                yield b"\r\n"
                                print(f"Binary endpoint: Yielded video {i+1}: {video_file} ({len(video_data)} bytes)")
                        yield f"--{boundary}--\r\n".encode("utf-8")
                    return StreamingResponse(multipart_generator(),
                                            media_type=f"multipart/mixed; boundary={boundary}")
                else:
                    # Single video
                    video_data, actual_path = await try_load_video_from_disk(matching[0])
                    if video_data:
                        print(f"Binary endpoint: Loaded video from {actual_path} ({len(video_data)} bytes)")
                        return Response(content=video_data, media_type="video/mp4")
                    else:
                        return Response(status_code=404, content=b"Video file could not be loaded")
            else:
                print(f"Binary endpoint: No videos found on disk for job {job_id}")
                return Response(status_code=404, content=b"No video found")
        except Exception as e:
            print(f"Binary endpoint: Error: {e}")
            return Response(status_code=500, content=f"Error: {str(e)}".encode())

    print(f"Binary endpoint: Job {job_id} status: {job.get('status')}")
    print(f"Binary endpoint: Job {job_id} data keys: {list(job.keys())}")
    wan_video = job.get('wan_video')
    print(f"Binary endpoint: wan_video: {wan_video}")

    # If wan_video not set, try to find it on disk (fallback)
    if not wan_video:
        print("Binary endpoint: wan_video not in job data, searching disk...")
        try:
            files = os.listdir(COMFYUI_OUTPUT_DIR)
            matching = [f for f in files if f.endswith('.mp4') and job_id in f]
            matching.sort()
            if matching:
                wan_video = matching[0]
                job['wan_video'] = wan_video  # Update job data for future requests
                job['all_videos'] = matching
                print(f"Binary endpoint: Found {len(matching)} videos on disk: {matching}")
            else:
                print(f"Binary endpoint: No videos found on disk for job {job_id}")
                return Response(status_code=404, content=b"No video found for job")
        except Exception as e:
            print(f"Binary endpoint: Error searching disk: {e}")
            return Response(status_code=500, content=f"Error: {str(e)}".encode())

    # Check if multiple videos were generated (check all_videos first)
    all_videos = job.get("all_videos", [])
    
    # If all_videos not set but we found files on disk, use those
    if not all_videos and wan_video:
        try:
            files = os.listdir(COMFYUI_OUTPUT_DIR)
            all_videos = sorted([f for f in files if f.endswith('.mp4') and job_id in f])
            job['all_videos'] = all_videos
            print(f"Binary endpoint: Populated all_videos from disk: {len(all_videos)} files")
        except Exception as e:
            print(f"Binary endpoint: Error populating all_videos: {e}")
            all_videos = [wan_video] if wan_video else []

    print(f"Binary endpoint: all_videos count = {len(all_videos)}")
    
    if len(all_videos) > 1:
        # Return multipart response for multiple videos
        boundary = f"----VideoBoundary{job_id}"
        print(f"Binary endpoint: Returning {len(all_videos)} videos as multipart")
        async def multipart_generator():
            for i, video_file in enumerate(all_videos):
                video_data, actual_path = await try_load_video_from_disk(video_file)
                if video_data:
                    header = (f"--{boundary}\r\n"
                    f"Content-Type: video/mp4\r\n"
                    f"Content-Disposition: inline; filename=video_{i+1}.mp4\r\n"
                    f"X-Video-Index: {i}\r\n\r\n").encode("utf-8")
                    yield header
                    yield video_data
                    yield b"\r\n"
                    print(f"Binary endpoint: Yielded video {i+1}: {video_file} ({len(video_data)} bytes)")
            yield f"--{boundary}--\r\n".encode("utf-8")
        return StreamingResponse(multipart_generator(),
                                media_type=f"multipart/mixed; boundary={boundary}")
    
    # Single video case
    if wan_video:
        video_data, actual_path = await try_load_video_from_disk(wan_video)
        if video_data:
            print(f"Binary endpoint: Successfully loaded single video from {actual_path} ({len(video_data)} bytes)")
            return Response(content=video_data, media_type="video/mp4")
        else:
            print(f"Binary endpoint: Could not load video file: {wan_video}")
            return Response(status_code=404, content=b"Video file not found")
    else:
        print("Binary endpoint: No wan_video filename found in job data")
        return Response(status_code=404, content=b"No video filename found")

@app.get("/result/{job_id}/all")
async def get_all_results(job_id: str):
    """Return all videos and flow images as JSON with base64 data."""
    from fastapi.responses import JSONResponse
    print(f"All results endpoint called for job_id: {job_id}")
    try:
        result = await _collect_job_results(job_id)
        print(f"All results: Returning {len(result['videos'])} videos, {len(result['flow_images'])} flow images")
        return JSONResponse(content=result)
    except Exception as e:
        print(f"All results endpoint error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/result/{job_id}/flow")
async def get_flow_result(job_id: str):
    """Return the first inline/flow-only CoTracker3 visualization as PNG."""
    from fastapi.responses import Response

    result = await _collect_job_results(job_id)
    flow_images = result.get("flow_images", [])
    if not flow_images:
        return Response(status_code=404, content=b"Flow image not found")
    return Response(
        content=base64.b64decode(flow_images[0]["data_base64"]),
        media_type="image/png",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
