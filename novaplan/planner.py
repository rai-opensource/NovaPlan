#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Task assessment, beam search, reactive planning, and rollout ranking."""

from __future__ import annotations
import argparse, os, json, time, concurrent.futures, threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from .cli_args import (
        http_url,
        image_size,
        positive_float,
        positive_int,
        rotation_degrees,
        validate_horizon_bounds,
    )
    from .video_generation import (
        HybridVideoGenerationClient,
        VideoGenerationResult,
        VeoVideoGenerationClient,
        WanVideoGenerationClient,
        make_veo_video_client,
    )
    from .llm_client import DEFAULT_OPENAI_VLM_MODEL, VLMAdapter
    from .flow_extraction_client import FlowExtractionClient
    from .vlm_prompts import build_direct_video_prompt, build_video_negative_prompt
except ImportError:
    from cli_args import (
        http_url,
        image_size,
        positive_float,
        positive_int,
        rotation_degrees,
        validate_horizon_bounds,
    )
    from video_generation import (
        HybridVideoGenerationClient,
        VideoGenerationResult,
        VeoVideoGenerationClient,
        WanVideoGenerationClient,
        make_veo_video_client,
    )
    from llm_client import DEFAULT_OPENAI_VLM_MODEL, VLMAdapter
    from flow_extraction_client import FlowExtractionClient
    from vlm_prompts import build_direct_video_prompt, build_video_negative_prompt

try:
    from .flow_switching import DEFAULT_FLOW_SWITCH_THETA_DEG
except ImportError:
    from flow_switching import DEFAULT_FLOW_SWITCH_THETA_DEG

def load_image_as_array(path: Optional[Path], w: int = 512, h: int = 288) -> np.ndarray:
    """Load an RGB image into a NumPy array."""
    if path is None or (path and not path.exists()):
        img = Image.new("RGB", (w, h), (240, 240, 240))
        d = ImageDraw.Draw(img)
        for x in range(0, w, 32): d.line([(x, 0), (x, h)], fill=(200, 200, 200))
        for y in range(0, h, 32): d.line([(0, y), (w, y)], fill=(200, 200, 200))
        d.rectangle([w//3-30, h//2-30, w//3+30, h//2+30], outline=(220, 50, 50), width=4)
        d.ellipse([2*w//3-20, h//2-20, 2*w//3+20, h//2+20], outline=(30, 120, 220), width=4)
        return np.array(img, dtype=np.uint8)
    else:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)


def _make_veo_adapter(*, mock: bool, seed: int):
    """Backward-compatible factory for the public Veo video client."""

    return make_veo_video_client(mock=mock, seed=seed)


def overlay_text(frame: np.ndarray, text: str, pos: str = "bottom", pad: int = 8) -> np.ndarray:
    """Overlay text."""
    H, W, _ = frame.shape
    img = Image.fromarray(frame); draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
    text = text or ""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x0, y0 = (pad, H - th - 2*pad) if pos == "bottom" else (pad, pad)
    draw.rectangle([x0-pad, y0-pad, x0+tw+pad, y0+th+pad], fill=(255, 255, 255))
    draw.text((x0, y0), text, fill=(0, 0, 0), font=font)
    return np.array(img, dtype=np.uint8)


def write_video(path: Path, video: np.ndarray, fps: int = 15) -> None:
    """Write video."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); ext = path.suffix.lower()
    if ext in {".mp4", ".m4v", ".mov"}:
        try: import imageio_ffmpeg  # noqa: F401
        except Exception: pass
        writer = imageio.get_writer(str(path), format="FFMPEG", mode="I",
                                    fps=fps, codec="libx264", bitrate="8M",
                                    macro_block_size=None, ffmpeg_log_level="error")
        with writer:
            for frame in video: writer.append_data(frame)
    elif ext == ".gif":
        imageio.mimsave(str(path), video, duration=1.0/max(1, fps))
    else:
        return write_video(path.with_suffix(".mp4"), video, fps=fps)


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


DEFAULT_VIDEO_SERVER_URL = "http://127.0.0.1:7000"
UNRANKED_ROLLOUT_SCORE = -1.0


@dataclass(order=True)
class Beam:
    """Represent one action-and-video hypothesis in planner beam search."""
    score: float
    frame: np.ndarray = field(compare=False)
    actions: List[str] = field(default_factory=list, compare=False)
    track_objects: List[str] = field(default_factory=list, compare=False)
    video: List[np.ndarray] = field(default_factory=list, compare=False)
    constraint_context: Optional[str] = field(default=None, compare=False)
    task_complete: bool = field(default=False, compare=False)

    def stitched(self) -> np.ndarray:
        """Return the beam video with rollout segments concatenated."""
        return np.expand_dims(self.frame, 0) if not self.video else np.concatenate(self.video, axis=0)


class NovaPlanPlanner:
    """
    Visual Language Planning using beam search over video rollouts.
    """

    def __init__(self, vlm: VLMAdapter, t2v: Any, beam_size: int = 2,
                 num_action_per_beam: int = 2, num_video_per_action: int = 4,
                 execution_num_action_per_step: int = 1,
                 execution_num_video_per_action: int = 8,
                 segment_T: int = 41, fps: int = 16, exploit_filter: bool = True,
                 replace_worst_rate: float = 0.3, seed: int = 0,
                 save_debug_videos: bool = False,
                 wan_overrides: Optional[dict] = None,
                 serial_debug: bool = False, horizon: int = 5,
                 use_prompt_extension: bool = True,
                 use_horizon_count: bool = True,
                 allow_horizon_fallback: bool = False,
                 horizon_count_min: int = 1,
                 horizon_count_max: int = 8,
                 enable_flow: bool = False,
                 mask_prompt: Optional[str] = None,
                 use_flow_for_scoring: bool = False,
                 flow_switch_theta_deg: float = DEFAULT_FLOW_SWITCH_THETA_DEG,
                 debug_dir: Optional[Path] = None,
                 flow_extraction_client: Optional[Any] = None):
        self.vlm = vlm
        self.video_model = t2v
        self.beam_size = beam_size
        self.num_action_per_beam = num_action_per_beam
        self.num_video_per_action = num_video_per_action
        self.execution_num_action_per_step = int(execution_num_action_per_step)
        self.execution_num_video_per_action = int(execution_num_video_per_action)
        self.segment_T = segment_T
        self.fps = fps
        self.exploit_filter = exploit_filter
        self.replace_worst_rate = replace_worst_rate
        self.rng = np.random.default_rng(seed)
        self.save_debug_videos = save_debug_videos
        self.wan_overrides = wan_overrides or {}
        self.serial_debug = serial_debug  # If True, forces max_workers=1 for debugging
        self.use_prompt_extension = use_prompt_extension
        self.use_horizon_count = use_horizon_count
        self.allow_horizon_fallback = bool(allow_horizon_fallback)
        self.horizon_count_min = int(horizon_count_min)
        self.horizon_count_max = int(horizon_count_max)
        self.requested_horizon = int(horizon)
        self.horizon_count_result: Optional[dict] = None
        
        # Flow extraction options
        self.enable_flow = enable_flow
        # Fallback only. Normal planning uses the VLM-proposed per-action
        # `track_object` as the SAM3/flow mask prompt.
        self.mask_prompt = mask_prompt or "object"
        self.use_flow_for_scoring = use_flow_for_scoring  # If True, pass flow images to VLM for scoring
        self.flow_switch_theta_deg = float(flow_switch_theta_deg)

        if debug_dir is None:
            from datetime import datetime
            debug_dir = (
                REPO_ROOT
                / "runs"
                / "planner"
                / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.step_counter = 0
        self.beam_counter = 0
        self.video_index = 0  # Counter for video index per step
        self._video_index_lock = threading.Lock()
        print(f"Run artifacts: {self.debug_dir}")
        if hasattr(self.video_model, 'debug_dir'):
            self.video_model.debug_dir = self.debug_dir
        self.horizon = horizon
        # Initialize comprehensive debug log for tracking all actions, videos, and scores
        self.debug_log = []

        # Timing tracking for performance analysis
        self.timing_stats = {
            'vlm_propose_actions': [],
            'vlm_score_rollout': [],
            'vlm_batch_ranking': [],  # New: batch ranking timing
            'vlm_horizon_count': [],
            'prompt_extension': [],
            'video_generation': [],
            'step_total': [],
            'planning_total': 0.0
        }

        # Selection-time flow_only extraction is used for non-WAN videos.
        # WAN returns its CoTracker3 evidence from the same full-mode job.
        self.flow_extraction_client = flow_extraction_client

    def _emit_status(self, event: str, **payload: Any) -> None:
        callback = getattr(self, "status_callback", None)
        if callable(callback):
            callback(event, payload)

    def _next_video_index(self) -> int:
        """Reserve one rollout artifact ID for the current planning step."""

        with self._video_index_lock:
            self.video_index += 1
            return self.video_index

    def _selection_debug_dir(self, step: int) -> Path:
        logical_step = int(getattr(self, "_debug_step_index_override", step))
        path = (
            self.debug_dir
            / f"step_{logical_step:03d}"
            / "debug_artifacts"
            / "video_rollout_selection"
        )
        scope = str(getattr(self, "_debug_selection_scope_override", "")).strip()
        if scope:
            path /= scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_debug_video(self, video: np.ndarray, filename: str, metadata: str = ""):
        if not self.save_debug_videos:
            return
        debug_dir = self._selection_debug_dir(self.step_counter)
        video_path = debug_dir / filename
        write_video(video_path, video, self.fps)
        with open(debug_dir / f"{filename}.txt","w") as f:
            f.write(f"Video: {filename}\nShape: {video.shape}\nFPS: {self.fps}\nMetadata: {metadata}\n")

    def _filter_exploits(self, rollout: np.ndarray) -> bool:
        diffs = np.mean(np.abs(np.diff(rollout.astype(np.float32), axis=0)))
        return 0.1 <= diffs <= 50.0

    def _process_action_complete(self, task: dict, goal: str) -> List[dict]:
        beam = task['beam']; action = task['action']; step_counter = task['step_counter']; beam_counter = task['beam_counter']
        original_step = getattr(self.video_model, 'step_counter', None)
        original_beam = getattr(self.video_model, 'beam_counter', None)
        if hasattr(self.video_model, 'step_counter'): self.video_model.step_counter = step_counter
        if hasattr(self.video_model, 'beam_counter'): self.video_model.beam_counter = beam_counter

        original_action = action
        generation_prompt = action
        wan_generation_prompt = action
        veo_generation_prompt = action
        try:
            # Merge WAN request overrides (task/model/size/sampling etc.)
            wan_kwargs = dict(self.wan_overrides)  # shallow copy
            generation_attempt = max(0, int(task.get("generation_attempt", 0)))
            if generation_attempt:
                wan_kwargs["seed"] = int(wan_kwargs.get("seed", 0)) + 1009 * generation_attempt
            # Determine effective values (CLI overrides win; otherwise use planner settings)
            effective_num_frames = wan_kwargs.get("num_frames", self.segment_T)
            effective_fps = wan_kwargs.get("fps", self.fps)
            effective_num_samples = wan_kwargs.get("num_samples", self.num_video_per_action)
            # Remove to avoid duplicate keyword/positional passing
            for _dup_key in (
                "num_frames",
                "fps",
                "num_samples",
                "start_frame",
                "action_text",
                "wan_action_text",
                "veo_action_text",
            ):
                wan_kwargs.pop(_dup_key, None)

            # Call the HTTP client: returns list[np.ndarray] or VideoGenerationResult
            # Use the VLM-proposed object for this action; planner mask_prompt is
            # only a legacy/fallback value if the proposal omits track_object.
            track_object = task.get('track_object', self.mask_prompt)
            is_wan_client = isinstance(self.video_model, WanVideoGenerationClient)
            is_hybrid_client = isinstance(self.video_model, HybridVideoGenerationClient)
            is_veo_client = (
                VeoVideoGenerationClient is not None
                and isinstance(self.video_model, VeoVideoGenerationClient)
            )
            uses_wan_generation = is_wan_client or is_hybrid_client
            uses_veo_generation = is_veo_client or is_hybrid_client

            # Keep a deterministic constrained prompt as the explicit
            # no-extension fallback. The normal path below follows main and
            # expands Veo prompts in English from the current observation.
            if uses_veo_generation and not self.use_prompt_extension:
                veo_generation_prompt = build_direct_video_prompt(
                    backend="veo3",
                    action=action,
                )
                if is_veo_client:
                    generation_prompt = veo_generation_prompt
                elif is_hybrid_client:
                    wan_kwargs["veo_action_text"] = veo_generation_prompt

            if (
                self.enable_flow
                and is_veo_client
                and self.flow_extraction_client is not None
            ):
                # Veo is generated through Vertex, but its selection evidence is
                # computed by the video service. Check first to avoid generating
                # paid candidates that cannot enter the ranking pool.
                self.flow_extraction_client.ensure_selection_flow_contract()

            # Follow the original backend-specific prompt-extension contract:
            # Chinese for Wan and English for Veo.
            if self.use_prompt_extension and uses_wan_generation:
                t0_prompt = time.time()
                wan_generation_prompt = self.vlm.extend_video_prompt(
                    image=beam.frame,
                    action=action,
                    goal=goal,
                    track_object=track_object,
                    backend="wan22",
                )
                prompt_time = time.time() - t0_prompt
                self.timing_stats['prompt_extension'].append(
                    (step_counter, beam_counter, task.get('action_idx', -1), prompt_time)
                )
                if is_wan_client:
                    generation_prompt = wan_generation_prompt
                elif is_hybrid_client:
                    wan_kwargs["wan_action_text"] = wan_generation_prompt
                print(f"  WAN prompt extension took {prompt_time:.2f}s")

            if self.use_prompt_extension and uses_veo_generation:
                t0_prompt = time.time()
                veo_generation_prompt = self.vlm.extend_video_prompt(
                    image=beam.frame,
                    action=action,
                    goal=goal,
                    track_object=track_object,
                    backend="veo3",
                )
                prompt_time = time.time() - t0_prompt
                self.timing_stats['prompt_extension'].append(
                    (step_counter, beam_counter, task.get('action_idx', -1), prompt_time)
                )
                if is_veo_client:
                    generation_prompt = veo_generation_prompt
                elif is_hybrid_client:
                    wan_kwargs["veo_action_text"] = veo_generation_prompt
                print(f"  Veo prompt extension took {prompt_time:.2f}s")

            # Main's generation contract always carries the backend-specific
            # negative prompt, even when positive-prompt extension is disabled.
            if is_hybrid_client:
                wan_kwargs.setdefault("wan_negative_prompt", build_video_negative_prompt("wan22"))
                wan_kwargs.setdefault("veo_negative_prompt", build_video_negative_prompt("veo3"))
            elif is_wan_client:
                wan_kwargs.setdefault("negative_prompt", build_video_negative_prompt("wan22"))
            elif is_veo_client:
                wan_kwargs.setdefault("negative_prompt", build_video_negative_prompt("veo3"))

            # Timing: Video generation
            t0_video = time.time()

            backends = []
            if uses_wan_generation:
                backends.append("wan22")
            if uses_veo_generation:
                backends.append("veo3")
            self._emit_status(
                "video_generation_started",
                action=action,
                track_object=track_object,
                num_samples=int(effective_num_samples),
                backends=backends,
            )
            print(f"Generating {effective_num_samples} videos with action='{action}', track_object='{track_object}'")
            if wan_generation_prompt != action:
                prompt_short = wan_generation_prompt[:120] + "..." if len(wan_generation_prompt) > 120 else wan_generation_prompt
                print(f"  WAN generation prompt: '{prompt_short}'")
            
            # Determine if we should extract 2D flow alongside video generation
            handles_flow_inline = is_wan_client or getattr(self.video_model, "handles_flow", False)
            
            if handles_flow_inline and self.enable_flow:
                # WAN22 and the hybrid WAN+Veo wrapper can return flow with generated videos.
                gen_result = self.video_model.generate_rollouts(
                    start_frame=beam.frame,
                    action_text=generation_prompt,
                    num_samples=effective_num_samples,
                    num_frames=effective_num_frames,
                    fps=effective_fps,
                    enable_flow=True,
                    mask_prompt=track_object,
                    **wan_kwargs,
                )
                if isinstance(gen_result, VideoGenerationResult):
                    rollouts = gen_result.videos
                    flow_images = gen_result.flow_images
                    flow_choices = gen_result.flow_choices
                    coords_3d = gen_result.coords_3d
                    visibilities = gen_result.visibilities
                else:
                    rollouts = gen_result
                    flow_images = []
                    flow_choices = []
                    coords_3d = []
                    visibilities = []
                video_sources = getattr(gen_result, "video_sources", []) if isinstance(gen_result, VideoGenerationResult) else []
            else:
                # Standard video generation, then optional 2D flow-only pass for non-WAN backends.
                gen_result = self.video_model.generate_rollouts(
                    start_frame=beam.frame,
                    action_text=generation_prompt,
                    num_samples=effective_num_samples,
                    num_frames=effective_num_frames,
                    fps=effective_fps,
                    **wan_kwargs,
                )
                if isinstance(gen_result, VideoGenerationResult):
                    rollouts = gen_result.videos
                    flow_images = gen_result.flow_images
                    flow_choices = gen_result.flow_choices
                    coords_3d = gen_result.coords_3d
                    visibilities = gen_result.visibilities
                else:
                    rollouts = gen_result if isinstance(gen_result, list) else list(gen_result)
                    flow_images = []
                    flow_choices = []
                    coords_3d = []
                    visibilities = []
                video_sources = getattr(gen_result, "video_sources", []) if isinstance(gen_result, VideoGenerationResult) else []
                
                if self.enable_flow and not is_wan_client and self.flow_extraction_client is not None:
                    print(f"  Extracting 2D flow for {len(rollouts)} videos using FlowExtractionClient...")
                    t0_flow = time.time()
                    flow_result = self.flow_extraction_client.extract_flow(
                        videos=rollouts,
                        mask_prompt=track_object,
                        fps=effective_fps,
                    )
                    flow_images = flow_result.flow_images
                    flow_choices = flow_result.flow_choices
                    coords_3d = flow_result.coords_3d
                    visibilities = flow_result.visibilities
                    flow_time = time.time() - t0_flow
                    print(f"  2D flow extraction took {flow_time:.2f}s, got {len(flow_images)} flow images")

            if not video_sources:
                if isinstance(self.video_model, HybridVideoGenerationClient):
                    video_sources = ["hybrid"] * len(rollouts)
                elif isinstance(self.video_model, WanVideoGenerationClient):
                    video_sources = ["wan22"] * len(rollouts)
                elif (
                    VeoVideoGenerationClient is not None
                    and isinstance(self.video_model, VeoVideoGenerationClient)
                ):
                    video_sources = ["veo3"] * len(rollouts)
                else:
                    video_sources = ["unknown"] * len(rollouts)

            requested_backends = list(getattr(gen_result, "requested_backends", []) or backends)
            requested_samples = getattr(gen_result, "requested_samples_per_backend", None)
            if requested_samples is None:
                requested_samples = int(effective_num_samples)
            backend_errors = dict(getattr(gen_result, "backend_errors", {}) or {})
            backend_counts: Dict[str, int] = {}
            for source in video_sources:
                backend_counts[source] = backend_counts.get(source, 0) + 1
            for backend in requested_backends:
                backend_counts.setdefault(backend, 0)
            self._emit_status(
                "video_generation_completed",
                action=action,
                track_object=track_object,
                requested_backends=requested_backends,
                requested_samples_per_backend=int(requested_samples),
                backend_counts=backend_counts,
                backend_errors=backend_errors,
                generated_total=len(rollouts),
            )

            video_gen_time = time.time() - t0_video
            self.timing_stats['video_generation'].append((step_counter, beam_counter, task.get('action_idx', -1), self.num_video_per_action, video_gen_time))
            requested_total = int(requested_samples) * max(1, len(requested_backends))
            print(
                f"  Video generation ({len(rollouts)} videos generated, "
                f"{requested_total} requested total) took {video_gen_time:.2f}s"
            )
            flow_image_count = sum(1 for flow in flow_images if flow is not None)
            if flow_image_count:
                print(f"  Got {flow_image_count} flow images alongside {len(rollouts)} videos")
            if flow_choices:
                selected_counts = {}
                for choice in flow_choices:
                    if choice is None:
                        continue
                    selected_counts[choice.selected_flow] = selected_counts.get(choice.selected_flow, 0) + 1
                if selected_counts:
                    print(f"  Flow switch decisions: {selected_counts}")
            print(f"  Action: '{action[:60]}...' (action_idx={task.get('action_idx', 0)})")

            processed = []; video_scores = {}

            # Save detailed debug info for each rollout
            action_debug_entry = {
                'step': step_counter,
                'beam': beam_counter,
                'original_action': original_action,
                'extended_action': action,
                'generation_prompt': generation_prompt,
                'wan_generation_prompt': wan_generation_prompt if uses_wan_generation else None,
                'veo_generation_prompt': veo_generation_prompt if uses_veo_generation else None,
                'prompt_extension_enabled': bool(
                    self.use_prompt_extension and uses_wan_generation
                ),
                'prompt_extension_backend': 'wan22' if self.use_prompt_extension and uses_wan_generation else None,
                'track_object': track_object,  # Object being tracked for flow extraction
                'goal': goal,
                'num_rollouts_generated': len(rollouts),
                'requested_backends': requested_backends,
                'requested_samples_per_backend': int(requested_samples),
                'backend_counts': backend_counts,
                'backend_errors': backend_errors,
                'rollouts': []
            }

            # Collect rollouts without scoring - scoring will be done in batch later
            default_backend = (
                "both" if isinstance(self.video_model, HybridVideoGenerationClient)
                else "wan22" if isinstance(self.video_model, WanVideoGenerationClient)
                else "veo3"
            )
            
            for i in range(len(rollouts)):
                source_backend = video_sources[i] if i < len(video_sources) else default_backend
                if source_backend == "wan22":
                    source_generation_prompt = wan_generation_prompt
                elif source_backend == "veo3":
                    source_generation_prompt = veo_generation_prompt
                else:
                    source_generation_prompt = generation_prompt
                # Get flow image for this rollout if available
                current_flow = flow_images[i] if i < len(flow_images) else None
                current_choice = flow_choices[i] if i < len(flow_choices) else None
                current_coords_3d = coords_3d[i] if i < len(coords_3d) else None
                current_visibilities = visibilities[i] if i < len(visibilities) else None
                current_choice_dict = current_choice.to_dict() if current_choice is not None else None
                selected_flow = current_choice.selected_flow if current_choice is not None else None
                
                # Use simpler format: step_{step_index}_video_{video_index}.mp4
                video_index = self._next_video_index()
                fn = f"step_{step_counter}_video_{video_index}.mp4"
                
                # Check exploit filter
                if self.exploit_filter and not self._filter_exploits(rollouts[i]):
                    rollout_entry = {
                        'sample_id': i+1,
                        'video_shape': tuple(rollouts[i].shape),
                        'filtered_out': True,
                        'filter_reason': 'Failed exploit filter',
                        'score': None,
                        'video_file': None,
                        'flow_file': None,
                        'flow_arrays_file': None,
                        'has_flow': current_flow is not None,
                        'has_flow_arrays': current_coords_3d is not None,
                        'backend': source_backend,
                        'generation_prompt': source_generation_prompt,
                        'selected_flow': selected_flow,
                        'flow_switch': current_choice_dict,
                    }
                    
                    # Still save filtered videos for debugging
                    if self.save_debug_videos:
                        try:
                            self._save_debug_video(rollouts[i], fn, metadata=f"goal={goal} action={action} prompt={source_generation_prompt} [FILTERED]")
                            rollout_entry['video_file'] = fn
                            print(f"  💾 Saved filtered video: {fn}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to save filtered video {fn}: {e}")
                    
                    action_debug_entry['rollouts'].append(rollout_entry)
                    processed.append(None)
                    continue
                
                rollout_entry = {
                    'sample_id': i+1,
                    'video_shape': tuple(rollouts[i].shape),
                    'filtered_out': False,
                    'score': None,  # Will be set during batch ranking
                    'video_file': None,
                    'flow_file': None,
                    'flow_arrays_file': None,
                    'has_flow': current_flow is not None,
                    'has_flow_arrays': current_coords_3d is not None,
                    'backend': source_backend,
                    'generation_prompt': source_generation_prompt,
                    'selected_flow': selected_flow,
                    'flow_switch': current_choice_dict,
                    'video_index': video_index,  # Store video index for candidate mapping
                }

                # Save debug video if enabled
                if self.save_debug_videos:
                    try:
                        action_short = action[:50] + "..." if len(action) > 50 else action
                        self._save_debug_video(rollouts[i], fn, metadata=f"goal={goal} action={action} prompt={source_generation_prompt}")
                        rollout_entry['video_file'] = fn
                        rollout_entry['action'] = action  # Store action in rollout entry
                        print(f"  💾 Saved video: {fn} | action: '{action_short}' | shape={rollouts[i].shape}")
                    except Exception as e:
                        print(f"  ⚠️ Failed to save video {fn}: {e}")
                    
                    # Save flow image if available
                    if current_flow is not None:
                        flow_fn = f"step_{step_counter}_video_{video_index}_flow.png"
                        flow_path = self._selection_debug_dir(step_counter) / flow_fn
                        try:
                            Image.fromarray(current_flow).save(flow_path)
                            rollout_entry['flow_file'] = flow_fn
                            print(f"  💾 Saved flow image: {flow_fn}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to save flow image {flow_fn}: {e}")

                    if current_coords_3d is not None:
                        flow_arrays_fn = f"step_{step_counter}_video_{video_index}_flow_arrays.npz"
                        flow_arrays_path = self._selection_debug_dir(step_counter) / flow_arrays_fn
                        try:
                            payload = {
                                "coords": current_coords_3d,
                                "coords_3d": current_coords_3d,
                                "flow_layout": np.array("time_major"),
                            }
                            if current_visibilities is not None:
                                payload["visibilities"] = current_visibilities
                                payload["visibs"] = current_visibilities
                            np.savez(flow_arrays_path, **payload)
                            rollout_entry['flow_arrays_file'] = flow_arrays_fn
                            print(f"  💾 Saved flow arrays: {flow_arrays_fn}")
                        except Exception as e:
                            print(f"  ⚠️ Failed to save flow arrays {flow_arrays_fn}: {e}")

                action_debug_entry['rollouts'].append(rollout_entry)
                
                # Return unscored candidate for batch ranking
                # Store metadata to map scores back to debug_log
                processed.append({
                    'rollout': rollouts[i],
                    'score': None,  # Will be set during batch ranking
                    'new_frame': rollouts[i][-1],
                    'flow_image': current_flow,
                    'coords_3d': current_coords_3d,
                    'visibilities': current_visibilities,
                    'flow_arrays_file': rollout_entry.get('flow_arrays_file'),
                    'flow_choice': current_choice,
                    'selected_flow': selected_flow,
                    'action': action,
                    'generation_prompt': source_generation_prompt,
                    'track_object': track_object,
                    'backend': source_backend,
                    'video_index': video_index,  # Store video index for candidate mapping
                    '_debug_key': (step_counter, beam_counter, action, i+1),  # For mapping scores back
                })

            self.debug_log.append(action_debug_entry)
            if hasattr(self.video_model, '_video_scores'): self.video_model._video_scores = video_scores
            
            # Summary of saved videos
            saved_count = sum(1 for r in action_debug_entry['rollouts'] if r.get('video_file'))
            filtered_count = sum(1 for r in action_debug_entry['rollouts'] if r.get('filtered_out', False))
            action_short = action[:60] + "..." if len(action) > 60 else action
            print(f"  ✓ Action complete: '{action_short}' | {saved_count} videos saved, {filtered_count} filtered out, {len(rollouts)} total generated")
            
            return processed
        finally:
            if original_step is not None: self.video_model.step_counter = original_step
            if original_beam is not None: self.video_model.beam_counter = original_beam

    def step(self, beams: List[Beam], goal: str) -> List[Beam]:
        """Advance the planner by one action-selection step."""
        t0_step = time.time()
        self.step_counter += 1
        self.video_index = 0  # Reset video index for this step
        print(f"\n=== STEP {self.step_counter} with goal: {goal} ===")
        candidates: List[Beam] = []; tasks=[]
        completed_beams = [beam for beam in beams if beam.task_complete]
        # Log all proposed actions for this step
        step_actions_log = {
            'step': self.step_counter,
            'num_beams': len(beams),
            'proposed_actions_per_beam': []
        }

        # Parallel propose actions
        def _propose_one(bidx, beam):
            t0_propose = time.time()
            previous_action = beam.actions[-1] if beam.actions else None 
            constraint_context = beam.constraint_context or (
                "Previously manipulated/tracked objects in execution order: "
                + json.dumps(beam.track_objects)
                if beam.track_objects
                else None
            )
            
            # VLM returns list of dicts: [{"action": str, "track_object": str}, ...]
            proposals = self.vlm.propose_actions(
                image=beam.frame, 
                goal=goal, 
                num_actions=self.num_action_per_beam, 
                previous_action=previous_action,
                action_history=list(beam.actions),
                constraint_context=constraint_context,
                steps_remaining=max(
                    1,
                    int(getattr(self, "_reactive_steps_remaining_override", self.horizon - (self.step_counter - 1))),
                ),
            )
            propose_time = time.time() - t0_propose
            return bidx, proposals, propose_time

        print(f"Proposing actions for {len(beams)} beams in parallel...")
        results_map = {}
        max_w = 1 if self.serial_debug else len(beams)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as ex:
            futures = {
                ex.submit(_propose_one, i, b): i
                for i, b in enumerate(beams)
                if not b.task_complete
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    results_map[res[0]] = res
                except Exception as e:
                    raise RuntimeError(
                        f"VLM action proposal failed for beam {futures[fut]}; "
                        "NovaPlan will not substitute an ungrounded motion command."
                    ) from e

        # Process results in order
        for bidx, beam in enumerate(beams):
            if beam.task_complete:
                continue
            self.beam_counter = bidx
            _, proposals, propose_time = results_map[bidx]
            
            self.timing_stats['vlm_propose_actions'].append((self.step_counter, bidx, propose_time))
            
            # Extract action strings for logging
            action_strs = [p['action'] if isinstance(p, dict) else p for p in proposals]
            print(f"  VLM propose_actions (beam {bidx}) took {propose_time:.2f}s. Actions: {action_strs}")

            # Log proposed actions with track objects
            step_actions_log['proposed_actions_per_beam'].append({
                'beam_id': bidx,
                'proposals': proposals,  # Full proposals with track_object
                'beam_previous_actions': beam.actions,
                'beam_score': beam.score
            })
    
            for action_idx, prop in enumerate(proposals):
                # Handle both new format (dict) and legacy format (str)
                if isinstance(prop, dict):
                    action = str(prop.get('action') or '').strip()
                    track_object = prop.get('track_object', self.mask_prompt)
                    proposal_constraint_context = prop.get('constraint_context') or beam.constraint_context
                    is_finish = bool(prop.get('is_finish'))
                else:
                    action = str(prop or '').strip()
                    track_object = self.mask_prompt
                    proposal_constraint_context = beam.constraint_context
                    is_finish = False
                if not action:
                    raise RuntimeError(
                        f"VLM returned an empty action for beam {bidx}, proposal {action_idx}."
                    )
                
                if is_finish:
                    completed_beams.append(
                        Beam(
                            score=max(float(beam.score), 1.0),
                            frame=beam.frame,
                            actions=list(beam.actions),
                            track_objects=list(beam.track_objects),
                            video=list(beam.video),
                            constraint_context=proposal_constraint_context,
                            task_complete=True,
                        )
                    )
                    continue

                tasks.append({
                    'beam_idx': bidx, 
                    'beam': beam, 
                    'action': action,
                    'track_object': track_object,  # Object to track with SAM3
                    'constraint_context': proposal_constraint_context,
                    'step_counter': self.step_counter, 
                    'beam_counter': bidx,
                    'action_idx': action_idx,  # Index of this action within the beam's proposals
                })
                self._emit_status(
                    "action_proposed",
                    action=action,
                    track_object=track_object,
                )

        # Save proposed actions log
        actions_log_path = self.debug_dir / f"step_{self.step_counter}_proposed_actions.json"
        with open(actions_log_path, 'w') as f:
            json.dump(step_actions_log, f, indent=2)

        print(f"Processing {len(tasks)} actions in parallel...")
        # Parallelize action branches - generate all videos first (no scoring yet)
        max_w = 1 if self.serial_debug else max(1, len(tasks))
        
        # Collect all unscored candidates with their beam context
        unscored_candidates = []  # List of (task, rollout_dict) tuples
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as ex:
            fut2task = {ex.submit(self._process_action_complete, t, goal): t for t in tasks}
            for fut in concurrent.futures.as_completed(fut2task):
                t = fut2task[fut]
                try:
                    prolls = fut.result()
                    for d in prolls:
                        if d is None: 
                            continue
                        # Store unscored candidate with beam context
                        unscored_candidates.append((t, d))
                except Exception as e:
                    print(f"Action '{t['action']}' failed: {e}")

        if not unscored_candidates:
            if completed_beams:
                completed_beams.sort(key=lambda beam: beam.score, reverse=True)
                return completed_beams[: self.beam_size]
            raise RuntimeError(
                "Video generation produced no rankable candidates; refusing to reuse the prior beam as execution."
            )

        print(f"\n=== BATCH RANKING: {len(unscored_candidates)} candidates ===")
        
        # Prepare candidates for batch ranking - only need action and flow image
        # (flow image already has the initial frame as base with motion trajectories drawn on top)
        # Also store reference to rollout and original index for later video saving
        ranking_input = []
        for idx, (task, rollout_dict) in enumerate(unscored_candidates):
            ranking_input.append({
                'candidate_id': idx,
                'action': rollout_dict.get('action', task['action']),
                'track_object': rollout_dict.get('track_object', task.get('track_object', self.mask_prompt)),
                'generation_prompt': rollout_dict.get('generation_prompt'),
                'flow_image': rollout_dict.get('flow_image'),
                'rollout': rollout_dict.get('rollout'),
                'backend': rollout_dict.get('backend', 'unknown'),
            })
        
        # Call VLM to rank all candidates in one batch
        t0_rank = time.time()
        ranked_candidates = self.vlm.rank_rollouts_batch(
            goal=goal,
            candidates=ranking_input,
            top_n=self.beam_size,  # Only need top beam_size candidates
            debug_dir=self._selection_debug_dir(self.step_counter),
            step=self.step_counter  # Include step number in filenames
        )
        if not ranked_candidates:
            raise RuntimeError(
                "No rollout had both motion-flow and final-frame evidence required by the paper ranker."
            )
        rank_time = time.time() - t0_rank
        self.timing_stats['vlm_batch_ranking'].append((self.step_counter, len(unscored_candidates), rank_time))
        print(f"  Batch ranking took {rank_time:.2f}s for {len(unscored_candidates)} candidates")
        print(f"  Ranked {len(ranked_candidates)} candidates (top {min(self.beam_size, len(ranked_candidates))} selected)")
        
        # Algorithm 2 uses stable IDs from the candidate set through ranking and
        # beam construction. Keep this mapping in memory; debug JSON is output,
        # never an internal data channel.
        candidate_id_to_rollout = {
            candidate_id: pair
            for candidate_id, pair in enumerate(unscored_candidates)
        }
        
        # Create mapping file and symlinks with candidate IDs in filenames
        print("\n  💾 Creating candidate to video mapping and symlinks...")
        candidate_mapping = {}
        selection_debug_dir = self._selection_debug_dir(self.step_counter)
        for candidate_id, (task, rollout_dict) in candidate_id_to_rollout.items():
            video_index = rollout_dict.get('video_index')
            action_text = rollout_dict.get('action', task['action'])
            if video_index is not None:
                original_video = f"step_{self.step_counter}_video_{video_index}.mp4"
                candidate_video = f"step_{self.step_counter}_video_{video_index}_candidate_{candidate_id}.mp4"
                
                # Create symlink with candidate ID in filename
                original_path = selection_debug_dir / original_video
                candidate_path = selection_debug_dir / candidate_video
                if original_path.exists():
                    try:
                        if candidate_path.exists() or candidate_path.is_symlink():
                            candidate_path.unlink()
                        candidate_path.symlink_to(original_video)
                        print(f"    ✓ Created symlink: {candidate_video}")
                    except Exception as e:
                        print(f"    ⚠️  Failed to create symlink {candidate_video}: {e}")
                
                candidate_mapping[candidate_id] = {
                    'video_index': video_index,
                    'video_file': original_video,
                    'candidate_video_file': candidate_video,
                    'flow_arrays_file': rollout_dict.get('flow_arrays_file'),
                    'action': action_text,
                    'generation_prompt': rollout_dict.get('generation_prompt'),
                    'backend': rollout_dict.get('backend', 'unknown'),
                    'score': None,  # Will be updated after ranking
                }
        
        # Save mapping file
        mapping_path = selection_debug_dir / f"step_{self.step_counter}_candidate_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(candidate_mapping, f, indent=2)
        print(f"    ✓ Saved candidate mapping to {mapping_path.name}")
        
        # Print scores for all ranked candidates with video and action info
        print("\n  📊 Batch Ranking Scores (with video files and actions):")
        for i, ranked in enumerate(ranked_candidates):
            score = ranked.get('score', 0.0)
            action = ranked.get('action', 'unknown')
            reason = ranked.get('rank_reason', '')[:80]
            candidate_id = int(ranked.get('candidate_id', -1))
            mapping = candidate_mapping.get(candidate_id, {})
            video_idx = mapping.get('video_index')
            candidate_file = mapping.get(
                'candidate_video_file',
                f"step_{self.step_counter}_video_{video_idx}.mp4" if video_idx is not None else "",
            )
            video_info = f" | {candidate_file}" if candidate_file else ""
            backend_info = ranked.get('backend', '')
            backend_info = mapping.get('backend', backend_info)
            action_short = action[:50] + "..." if len(action) > 50 else action
            backend_suffix = f" | backend={backend_info}" if backend_info else ""
            print(f"    Rank {i+1}: score={score:.4f}{video_info}{backend_suffix} | action='{action_short}' | reason='{reason}...'")
        
        # Map ranked candidates back to Beams and update debug_log with scores
        ranked_scores = {}
        rollout_to_score_map = {}  # Maps (step, beam, action, sample_id) -> (score, reason)
        
        for ranked in ranked_candidates:
            if 'score' not in ranked:
                raise RuntimeError("Validated rollout ranking omitted a score")
            candidate_id = int(ranked.get('candidate_id', -1))
            if candidate_id not in candidate_id_to_rollout:
                raise RuntimeError(f"Validated rollout ranking returned unknown candidate_id={candidate_id}")
            score = float(ranked['score'])
            reason = ranked.get('rank_reason', '')
            ranked_scores[candidate_id] = (score, reason)
            _, rollout_dict = candidate_id_to_rollout[candidate_id]
            debug_key = rollout_dict.get('_debug_key')
            if debug_key:
                rollout_to_score_map[debug_key] = (score, reason)
        
        # Update debug_log with scores for all rollouts
        for entry in self.debug_log:
            if entry['step'] == self.step_counter:
                action_key = entry.get('extended_action', entry.get('original_action', ''))
                for rollout in entry['rollouts']:
                    if not rollout.get('filtered_out', False):
                        # Match using (step, beam, action, sample_id)
                        debug_key = (entry['step'], entry['beam'], action_key, rollout['sample_id'])
                        if debug_key in rollout_to_score_map:
                            score, reason = rollout_to_score_map[debug_key]
                            rollout['score'] = float(score)
                            rollout['rank_reason'] = reason
                            print(f"  📊 Rollout score: step={entry['step']}, beam={entry['beam']}, sample={rollout['sample_id']}, score={score:.3f}")
                        else:
                            rollout['score'] = UNRANKED_ROLLOUT_SCORE
                            rollout['rank_reason'] = "Not ranked in batch evaluation"
                            print(f"  ⚠️  Rollout not ranked: step={entry['step']}, beam={entry['beam']}, sample={rollout['sample_id']}")
        
        # Build Beam candidates with scores and track candidate IDs
        candidates: List[Beam] = []
        rollout_to_candidate_id = {}  # Maps rollout array -> candidate_id (for matching)
        for j, (task, rollout_dict) in enumerate(unscored_candidates):
            if j in ranked_scores:
                score, reason = ranked_scores[j]
            else:
                # Algorithm 2 initializes omitted candidates to s_min. Ranking
                # scores are in [0,1], so -1 cannot outrank a returned result.
                score = UNRANKED_ROLLOUT_SCORE
                reason = "Not in top-N"
            
            # Candidate IDs are the stable enumeration used in the review grid.
            candidate_id = j
            rollout_video = rollout_dict.get('rollout')
            
            beam = Beam(
                score=score, 
                frame=rollout_dict['new_frame'],
                actions=task['beam'].actions + [task['action']],
                track_objects=task['beam'].track_objects + [task.get('track_object', self.mask_prompt)],
                video=task['beam'].video + [rollout_dict['rollout']],
                constraint_context=task.get('constraint_context') or task['beam'].constraint_context,
            )
            candidates.append(beam)
            if rollout_video is not None:
                rollout_to_candidate_id[id(rollout_video)] = candidate_id  # Use id() for hashing
            
            print(f"  Candidate {candidate_id}: score={score:.3f}, action='{task['action'][:50]}...', reason='{reason[:50]}...'")
        
        candidates.extend(completed_beams)
        candidates.sort(key=lambda b: b.score, reverse=True)
        selected = candidates[:self.beam_size]
        if selected:
            selected_beam = selected[0]
            selected_rollout = selected_beam.video[-1] if selected_beam.video else None
            selected_candidate_id = (
                rollout_to_candidate_id.get(id(selected_rollout))
                if selected_rollout is not None
                else None
            )
            selected_metadata = (
                ranking_input[selected_candidate_id]
                if selected_candidate_id is not None
                else {}
            )
            self._emit_status(
                "video_selected",
                action=selected_beam.actions[-1] if selected_beam.actions else "",
                score=float(selected_beam.score),
                candidate_id=selected_candidate_id,
                candidate_count=len(ranking_input),
                backend=selected_metadata.get("backend", "unknown"),
            )
        
        # Save last frames of selected videos (top N) and update mapping with scores
        print("\n  💾 Saving last frames of selected videos...")
        from PIL import Image as PILImage
        for rank_idx, beam in enumerate(selected):
            if beam.task_complete:
                continue
            if beam.video and len(beam.video) > 0:
                # Get the last video in the beam (most recent action)
                last_video = beam.video[-1]
                if last_video is not None and len(last_video) > 0:
                    last_frame = last_video[-1]  # Last frame of last video
                    if last_frame is not None:
                        # Find the candidate_id and video_index for this beam by matching the rollout
                        candidate_id = rollout_to_candidate_id.get(id(last_video))
                        video_index = None
                        
                        if candidate_id is None:
                            # Fallback: try to match by comparing arrays
                            for cid, (task, rollout_dict) in candidate_id_to_rollout.items():
                                rd_video = rollout_dict.get('rollout')
                                if rd_video is not None and np.array_equal(rd_video, last_video):
                                    candidate_id = cid
                                    video_index = rollout_dict.get('video_index')
                                    break
                        else:
                            # Get video_index from mapping
                            if candidate_id in candidate_mapping:
                                video_index = candidate_mapping[candidate_id]['video_index']
                        
                        if video_index is None:
                            # Final fallback: use rank index
                            video_index = rank_idx + 1
                        
                        # Update mapping with score
                        if candidate_id is not None and candidate_id in candidate_mapping:
                            candidate_mapping[candidate_id]['score'] = float(beam.score)
                        
                        last_frame_path = (
                            selection_debug_dir
                            / f"step_{self.step_counter}_video_{video_index}_last_frame.jpg"
                        )
                        try:
                            frame_img = PILImage.fromarray(last_frame)
                            frame_img.save(last_frame_path, quality=95)
                            print(f"    ✓ Saved step_{self.step_counter}_video_{video_index}_last_frame.jpg (rank {rank_idx+1}, score={beam.score:.3f})")
                        except Exception as e:
                            print(f"    ⚠️  Failed to save last frame for video_{video_index}: {e}")
        
        # Update mapping file with scores
        if candidate_mapping:
            mapping_path = self.debug_dir / f"step_{self.step_counter}_candidate_mapping.json"
            with open(mapping_path, 'w') as f:
                json.dump(candidate_mapping, f, indent=2)
            print("    ✓ Updated candidate mapping with scores")

        # Log selected beams
        selected_log = {
            'step': self.step_counter,
            'num_candidates': len(candidates),
            'selected_beams': []
        }
        for i, b in enumerate(selected):
            print(f" Beam {i+1}: score={b.score:.3f}, actions={b.actions}")
            selected_log['selected_beams'].append({
                'rank': i+1,
                'score': float(b.score),
                'actions': b.actions
            })

        selected_log_path = self.debug_dir / f"step_{self.step_counter}_selected_beams.json"
        with open(selected_log_path, 'w') as f:
            json.dump(selected_log, f, indent=2)

        step_time = time.time() - t0_step
        self.timing_stats['step_total'].append((self.step_counter, step_time))
        print(f"=== STEP {self.step_counter} completed in {step_time:.2f}s ===\n")

        return selected

    def assess_task_structure(self, start_frame: np.ndarray, goal: str) -> Dict[str, Any]:
        """Assess task coupling and horizon before choosing strategic vs reactive execution."""
        self.requested_horizon = int(self.requested_horizon or self.horizon)
        self.horizon_count_result = None

        if self.use_horizon_count:
            print("\n=== TASK STRUCTURE / HORIZON COUNT ===")
            t0_horizon = time.time()
            result = self.vlm.estimate_horizon(
                image=start_frame,
                goal=goal,
                fallback_horizon=self.requested_horizon,
                min_horizon=self.horizon_count_min,
                max_horizon=self.horizon_count_max,
            )
            if result.get("horizon_source") == "manual_fallback_error" and not self.allow_horizon_fallback:
                error = result.get("error", "unknown horizon-counting error")
                raise RuntimeError(
                    "Automatic task-structure/horizon counting failed, so NovaPlan refused to use "
                    f"the fallback --horizon={self.requested_horizon}. Fix the VLM/API issue or pass "
                    "--allow_horizon_fallback for smoke tests. "
                    f"Original error: {error}"
                )
            horizon_time = time.time() - t0_horizon
            generated_horizon = int(result.get("horizon", self.requested_horizon))
            self.timing_stats['vlm_horizon_count'].append((generated_horizon, horizon_time))
        else:
            generated_horizon = int(self.requested_horizon)
            result = {
                "horizon": generated_horizon,
                "planner_horizon": generated_horizon,
                "reactive_execution_horizon": None,
                "plan_in_advance_allowed": True,
                "is_coupled": True,
                "execution_mode": "plan_in_advance_manual",
                "horizon_source": "manual_planner_horizon",
                "fallback_horizon": generated_horizon,
                "reasoning": "Horizon counting disabled; using configured planning horizon.",
            }
            horizon_time = 0.0

        planner_horizon = int(result.get("planner_horizon", result.get("horizon", self.requested_horizon)) or 0)
        self.horizon_count_result = result
        self.horizon = planner_horizon
        with open(self.debug_dir / "horizon_count.json", 'w') as f:
            json.dump(self.horizon_count_result, f, indent=2)

        is_coupled = self.horizon_count_result.get("is_coupled")
        plan_in_advance_allowed = self.horizon_count_result.get("plan_in_advance_allowed")
        execution_mode = self.horizon_count_result.get("execution_mode", "unknown")
        reactive_execution_horizon = self.horizon_count_result.get("reactive_execution_horizon")
        horizon_source = self.horizon_count_result.get("horizon_source", "unknown")
        print(
            f"VLM task coupling: is_coupled={is_coupled}, "
            f"plan_in_advance_allowed={plan_in_advance_allowed}"
        )
        print(
            f"Generated horizon={generated_horizon}; planner_horizon={self.horizon}; "
            f"execution_mode={execution_mode}; source={horizon_source}; "
            f"requested fallback={self.requested_horizon}; elapsed={horizon_time:.2f}s"
        )
        if reactive_execution_horizon is not None:
            print(f"Reactive execution horizon={reactive_execution_horizon} step(s)")
        if self.horizon_count_result.get("coupling_reason"):
            print(f"Coupling reason: {self.horizon_count_result.get('coupling_reason')}")
        if self.horizon_count_result.get("reasoning"):
            print(f"Horizon reasoning: {self.horizon_count_result.get('reasoning')}")
        return self.horizon_count_result

    def _write_planning_config(self, goal: str) -> None:
        config = {
            'goal': goal,
            'requested_horizon': self.requested_horizon,
            'horizon': self.horizon,
            'horizon_count_enabled': self.use_horizon_count,
            'allow_horizon_fallback': self.allow_horizon_fallback,
            'horizon_count': self.horizon_count_result,
            'execution_mode': self.horizon_count_result.get('execution_mode') if self.horizon_count_result else 'plan_in_advance_manual',
            'generated_horizon': self.horizon_count_result.get('horizon') if self.horizon_count_result else self.horizon,
            'reactive_execution_horizon': self.horizon_count_result.get('reactive_execution_horizon') if self.horizon_count_result else None,
            'beam_size': self.beam_size,
            'num_action_per_beam': self.num_action_per_beam,
            'num_video_per_action': self.num_video_per_action,
            'execution_beam_size': 1,
            'execution_num_action_per_step': self.execution_num_action_per_step,
            'execution_num_video_per_action': self.execution_num_video_per_action,
            'video_backend': (
                "both" if isinstance(self.video_model, HybridVideoGenerationClient)
                else "wan22" if isinstance(self.video_model, WanVideoGenerationClient)
                else "veo3"
            ),
            'videos_per_action_total': (
                self.num_video_per_action * 2
                if isinstance(self.video_model, HybridVideoGenerationClient)
                else self.num_video_per_action
            ),
            'segment_T': self.segment_T,
            'fps': self.fps,
            'exploit_filter': self.exploit_filter,
            'wan_overrides': self.wan_overrides,
            'prompt_extension_enabled': bool(
                self.use_prompt_extension
                and isinstance(
                    self.video_model,
                    (WanVideoGenerationClient, HybridVideoGenerationClient),
                )
            ),
            'prompt_extension_backend': 'wan22',
            'vlm_model': getattr(self.vlm, 'model', None),
            'vlm_reasoning_effort': getattr(self.vlm, 'reasoning_effort', None),
            'vlm_task_models': getattr(self.vlm, 'task_models', {}),
            'vlm_task_reasoning_efforts': getattr(self.vlm, 'task_reasoning_efforts', {}),
            'flow_switch_theta_deg': self.flow_switch_theta_deg,
        }
        with open(self.debug_dir / "planning_config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def run_strategic_beam_search(
        self,
        start_frame: np.ndarray,
        goal: str,
        task_structure: Optional[Dict[str, Any]] = None,
    ) -> Beam:
        """Run the plan-in-advance beam search used for coupled/strategic tasks."""
        if task_structure is not None:
            self.horizon_count_result = task_structure
            self.horizon = int(task_structure.get("planner_horizon", task_structure.get("horizon", self.horizon)) or 0)

        t0_planning = time.time()
        print(f"\n=== START STRATEGIC PLAN: '{goal}', depth={self.horizon} ===")
        beams = [Beam(score=0.0, frame=start_frame, actions=[], video=[])]
        self.step_counter = 0
        self._write_planning_config(goal)

        for _ in range(self.horizon):
            beams = self.step(beams, goal)
            if beams and all(beam.task_complete for beam in beams):
                break
        beams.sort(key=lambda b: b.score, reverse=True)
        best = beams[0]

        self.timing_stats['planning_total'] = time.time() - t0_planning
        print(f"\n=== DONE === best score={best.score:.3f} actions={best.actions}")
        print(f"Total planning time: {self.timing_stats['planning_total']:.2f}s")
        print(f"Logs: {self.debug_dir}")

        self._save_final_summary(beams, best, goal, self.horizon)
        return best

    def plan_next_reactive_step(
        self,
        current_frame: np.ndarray,
        goal: str,
        *,
        history: Optional[List[str]] = None,
        steps_remaining: int = 1,
        constraint_context: Optional[str] = None,
        debug_step_index: Optional[int] = None,
    ) -> Beam:
        """Plan one execution command from the latest real observation.

        Reactive execution is not beam search: each real execution step proposes
        one action, generates the configured video pool for that action, and
        returns the single best rollout.
        """
        previous_horizon = self.horizon
        previous_beam_size = self.beam_size
        previous_num_action_per_beam = self.num_action_per_beam
        previous_num_video_per_action = self.num_video_per_action
        previous_override = getattr(self, "_reactive_steps_remaining_override", None)
        previous_debug_step = getattr(self, "_debug_step_index_override", None)
        previous_debug_scope = getattr(self, "_debug_selection_scope_override", None)
        self.horizon = max(1, int(steps_remaining))
        self.beam_size = 1
        self.num_action_per_beam = self.execution_num_action_per_step
        self.num_video_per_action = self.execution_num_video_per_action
        self._reactive_steps_remaining_override = self.horizon
        if debug_step_index is not None:
            self._debug_step_index_override = int(debug_step_index)
        self._debug_selection_scope_override = "selected_rollout"
        initial_beam = Beam(
            score=0.0,
            frame=current_frame,
            actions=list(history or []),
            track_objects=[],
            video=[],
            constraint_context=constraint_context,
        )
        try:
            selected = self.step([initial_beam], goal)
            selected.sort(key=lambda b: b.score, reverse=True)
            return selected[0]
        finally:
            self.horizon = previous_horizon
            self.beam_size = previous_beam_size
            self.num_action_per_beam = previous_num_action_per_beam
            self.num_video_per_action = previous_num_video_per_action
            if previous_override is None:
                self.__dict__.pop("_reactive_steps_remaining_override", None)
            else:
                self._reactive_steps_remaining_override = previous_override
            if previous_debug_step is None:
                self.__dict__.pop("_debug_step_index_override", None)
            else:
                self._debug_step_index_override = previous_debug_step
            if previous_debug_scope is None:
                self.__dict__.pop("_debug_selection_scope_override", None)
            else:
                self._debug_selection_scope_override = previous_debug_scope

    def generate_execution_rollout(
        self,
        current_frame: np.ndarray,
        goal: str,
        *,
        action: str,
        track_object: str,
        history: Optional[List[str]] = None,
        generation_attempt: int = 0,
        debug_step_index: Optional[int] = None,
    ) -> Beam:
        """Regenerate and select a rollout for one already-chosen action.

        Strategic planning fixes the action sequence, but the paper requires
        execution-time visual dynamics to be conditioned on the latest real
        observation.  This method deliberately does not propose or alter the
        action: it generates the configured WAN/Veo pool from ``current_frame``,
        ranks every candidate with the standard batch critic, and returns the
        best visual transition for geometric grounding.
        """
        action = str(action or "").strip()
        track_object = str(track_object or self.mask_prompt).strip()
        if not action:
            raise ValueError("A non-empty fixed action is required for execution-time rollout generation.")

        previous_num_video_per_action = self.num_video_per_action
        previous_beam_size = self.beam_size
        previous_debug_step = getattr(self, "_debug_step_index_override", None)
        previous_debug_scope = getattr(self, "_debug_selection_scope_override", None)
        if debug_step_index is not None:
            self._debug_step_index_override = int(debug_step_index)
        generation_attempt = max(0, int(generation_attempt))
        self._debug_selection_scope_override = (
            "selected_rollout" if generation_attempt == 0 else f"regeneration_{generation_attempt:03d}"
        )
        self.num_video_per_action = self.execution_num_video_per_action
        self.beam_size = 1
        self.video_index = 0
        if debug_step_index is None:
            self.step_counter += 1
            execution_step = self.step_counter
        else:
            # Regeneration retries belong to the same logical execution step.
            # Keep their filenames and ranking grids aligned with step_XXX.
            execution_step = int(debug_step_index)
            self.step_counter = max(self.step_counter, execution_step)
        source_beam = Beam(
            score=0.0,
            frame=np.asarray(current_frame, dtype=np.uint8),
            actions=list(history or []),
            track_objects=[],
            video=[],
        )
        task = {
            "beam_idx": 0,
            "beam": source_beam,
            "action": action,
            "track_object": track_object,
            "step_counter": execution_step,
            "beam_counter": 0,
            "action_idx": 0,
            "execution_time": True,
            "generation_attempt": generation_attempt,
        }
        try:
            rollouts = [item for item in self._process_action_complete(task, goal) if item is not None]
            if not rollouts:
                raise RuntimeError(
                    f"Execution-time generation produced no usable rollout for fixed action {action!r}."
                )

            ranking_input = [
                {
                    "candidate_id": idx,
                    "action": action,
                    "track_object": track_object,
                    "generation_prompt": item.get("generation_prompt"),
                    "flow_image": item.get("flow_image"),
                    "rollout": item.get("rollout"),
                    "backend": item.get("backend", "unknown"),
                }
                for idx, item in enumerate(rollouts)
            ]
            ranked = self.vlm.rank_rollouts_batch(
                goal=goal,
                candidates=ranking_input,
                top_n=1,
                debug_dir=self._selection_debug_dir(execution_step),
                step=execution_step,
            )
            if not ranked:
                raise RuntimeError(
                    f"Execution-time ranking returned no candidate for fixed action {action!r}."
                )
            winner = ranked[0]
            rollout = winner.get("rollout")
            if not isinstance(rollout, np.ndarray) or rollout.ndim != 4 or len(rollout) == 0:
                raise RuntimeError("The selected execution-time candidate has no valid video rollout.")
            self._emit_status(
                "video_selected",
                action=action,
                score=float(winner.get("score", 0.0)),
                candidate_id=int(winner.get("candidate_id", 0)),
                candidate_count=len(ranking_input),
                backend=str(winner.get("backend", "unknown")),
            )
            return Beam(
                score=float(winner.get("score", 0.0)),
                frame=np.asarray(rollout[-1], dtype=np.uint8),
                actions=list(history or []) + [action],
                track_objects=[track_object],
                video=[rollout],
            )
        finally:
            self.num_video_per_action = previous_num_video_per_action
            self.beam_size = previous_beam_size
            if previous_debug_step is None:
                self.__dict__.pop("_debug_step_index_override", None)
            else:
                self._debug_step_index_override = previous_debug_step
            if previous_debug_scope is None:
                self.__dict__.pop("_debug_selection_scope_override", None)
            else:
                self._debug_selection_scope_override = previous_debug_scope

    def plan(self, start_frame: np.ndarray, goal: str,) -> Beam:
        """Run beam search and return the selected action sequence."""
        task_structure = self.assess_task_structure(start_frame, goal)
        return self.run_strategic_beam_search(start_frame, goal, task_structure)

    def _save_final_summary(self, final_beams: List[Beam], best_beam: Beam, goal: str, horizon: int):
        final_summary = {
            'goal': goal,
            'requested_horizon': self.requested_horizon,
            'horizon': self.horizon,
            'horizon_count_enabled': self.use_horizon_count,
            'horizon_count': self.horizon_count_result,
            'execution_mode': self.horizon_count_result.get('execution_mode') if self.horizon_count_result else 'plan_in_advance_manual',
            'generated_horizon': self.horizon_count_result.get('horizon') if self.horizon_count_result else self.horizon,
            'reactive_execution_horizon': self.horizon_count_result.get('reactive_execution_horizon') if self.horizon_count_result else None,
            'best_beam': {
                'score': float(best_beam.score),
                'actions': best_beam.actions,
                'track_objects': best_beam.track_objects,
                'num_frames': int(best_beam.stitched().shape[0]) if best_beam.video else 0
            },
            'all_final_beams': []
        }

        for i, beam in enumerate(final_beams):
            final_summary['all_final_beams'].append({
                'rank': i+1,
                'score': float(beam.score),
                'actions': beam.actions,
                'track_objects': beam.track_objects,
            })

        with open(self.debug_dir / "final_summary.json", 'w') as f:
            json.dump(final_summary, f, indent=2)

        with open(self.debug_dir / "complete_debug_log.json", 'w') as f:
            json.dump(self.debug_log, f, indent=2)

        # CSV - include candidate information
        csv_path = self.debug_dir / "scores_summary.csv"
        # Load candidate mappings for each step to include candidate_id
        step_candidate_mappings = {}
        for step_num in range(1, self.step_counter + 1):
            mapping_path = self.debug_dir / f"step_{step_num}_candidate_mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    step_candidate_mappings[step_num] = json.load(f)
        
        with open(csv_path, 'w') as f:
            f.write("step,beam,original_action,extended_action,generation_prompt,track_object,selected_flow,flow_switch_reason,sample_id,video_index,candidate_id,candidate_video_file,score,rank_reason,filtered,video_file,flow_file,error\n")
            for entry in self.debug_log:
                step_num = entry['step']
                candidate_mapping = step_candidate_mappings.get(step_num, {})
                video_index_to_candidate = {}
                for cid, mapping in candidate_mapping.items():
                    vid_idx = mapping.get('video_index')
                    if vid_idx:
                        video_index_to_candidate[vid_idx] = {
                            'candidate_id': cid,
                            'candidate_file': mapping.get('candidate_video_file', 'N/A')
                        }
                
                for rollout in entry['rollouts']:
                    score = rollout.get('score')
                    score_str = f"{score:.6f}" if score is not None else "N/A"
                    rank_reason = rollout.get('rank_reason', '')
                    video_idx = rollout.get('video_index', 'N/A')
                    candidate_info = video_index_to_candidate.get(video_idx, {}) if video_idx != 'N/A' else {}
                    candidate_id = candidate_info.get('candidate_id', 'N/A')
                    candidate_file = candidate_info.get('candidate_file', 'N/A')
                    generation_prompt_csv = (
                        rollout.get('generation_prompt')
                        or entry.get('generation_prompt')
                        or 'N/A'
                    ).replace('"', '""')
                    
                    f.write(f"{entry['step']},{entry['beam']},\"{entry['original_action']}\",\"{entry['extended_action']}\",\"{generation_prompt_csv}\",")
                    flow_switch = rollout.get('flow_switch') or {}
                    selected_flow = rollout.get('selected_flow') or 'N/A'
                    flow_switch_reason = flow_switch.get('reason', '')
                    f.write(f"\"{entry.get('track_object', 'N/A')}\",\"{selected_flow}\",\"{flow_switch_reason}\",{rollout['sample_id']},{video_idx},{candidate_id},\"{candidate_file}\",")
                    f.write(f"{score_str},\"{rank_reason}\",{rollout['filtered_out']},")
                    f.write(f"{rollout.get('video_file', 'N/A')},{rollout.get('flow_file', 'N/A')},\"{rollout.get('error', '')}\"\n")

        # Human-readable summary
        summary_text_path = self.debug_dir / "SUMMARY.txt"
        with open(summary_text_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VISUAL LANGUAGE PLANNING - DEBUG SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Goal: {goal}\n")
            f.write(f"Requested Horizon: {self.requested_horizon} steps\n")
            f.write(f"Planning Horizon: {horizon} steps\n")
            if self.horizon_count_result:
                f.write(f"Execution Mode: {self.horizon_count_result.get('execution_mode', 'N/A')}\n")
                f.write(f"Generated Horizon: {self.horizon_count_result.get('horizon', 'N/A')}\n")
                f.write(f"Reactive Execution Horizon: {self.horizon_count_result.get('reactive_execution_horizon', 'N/A')}\n")
                f.write(f"Horizon Count Reasoning: {self.horizon_count_result.get('reasoning', 'N/A')}\n")
            f.write(f"Beam Size: {self.beam_size}\n")
            f.write(f"Actions per Beam: {self.num_action_per_beam}\n")
            f.write(f"Videos per Action: {self.num_video_per_action}\n\n")
            f.write("=" * 80 + "\n")
            f.write("BEST PLAN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Score: {best_beam.score:.6f}\n")
            f.write(f"Action Sequence ({len(best_beam.actions)} actions):\n")
            for i, act in enumerate(best_beam.actions, 1):
                track = best_beam.track_objects[i - 1] if i - 1 < len(best_beam.track_objects) else "object"
                f.write(f"  {i}. {act} [track_object={track}]\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("ALL ACTIONS AND SCORES\n")
            f.write("=" * 80 + "\n\n")
            # Load candidate mappings for each step
            step_candidate_mappings = {}
            for step_num in range(1, self.step_counter + 1):
                mapping_path = self.debug_dir / f"step_{step_num}_candidate_mapping.json"
                if mapping_path.exists():
                    with open(mapping_path, 'r') as f2:
                        step_candidate_mappings[step_num] = json.load(f2)
            
            for entry in self.debug_log:
                step_num = entry['step']
                candidate_mapping = step_candidate_mappings.get(step_num, {})
                video_index_to_candidate = {}
                for cid, mapping in candidate_mapping.items():
                    vid_idx = mapping.get('video_index')
                    if vid_idx:
                        video_index_to_candidate[vid_idx] = {
                            'candidate_id': cid,
                            'candidate_file': mapping.get('candidate_video_file', 'N/A')
                        }
                
                f.write(f"Step {entry['step']}, Beam {entry['beam']}\n")
                f.write(f"  Original Action: {entry['original_action']}\n")
                f.write(f"  Extended Action: {entry['extended_action']}\n")
                if entry.get('wan_generation_prompt') and entry.get('wan_generation_prompt') != entry.get('original_action'):
                    f.write(f"  WAN Generation Prompt: {entry['wan_generation_prompt']}\n")
                if entry.get('veo_generation_prompt') and entry.get('veo_generation_prompt') != entry.get('original_action'):
                    f.write(f"  Veo Generation Prompt: {entry['veo_generation_prompt']}\n")
                f.write(f"  Track Object: {entry.get('track_object', 'N/A')}\n")
                f.write("  Rollouts:\n")
                for rollout in entry['rollouts']:
                    flow_switch = rollout.get('flow_switch') or {}
                    selected_flow = rollout.get('selected_flow') or 'N/A'
                    if rollout['filtered_out']:
                        f.write(f"    Sample {rollout['sample_id']}: FILTERED OUT ({rollout.get('filter_reason', 'unknown')}), Selected Flow={selected_flow}\n")
                    else:
                        score = rollout.get('score')
                        score_str = f"{score:.6f}" if score is not None else "N/A"
                        rank_reason = rollout.get('rank_reason', '')
                        video_idx = rollout.get('video_index', 'N/A')
                        candidate_info = video_index_to_candidate.get(video_idx, {}) if video_idx != 'N/A' else {}
                        candidate_id = candidate_info.get('candidate_id', 'N/A')
                        candidate_file = candidate_info.get('candidate_file', 'N/A')
                        
                        candidate_str = f", Candidate ID={candidate_id} ({candidate_file})" if candidate_id != 'N/A' else ""
                        f.write(f"    Sample {rollout['sample_id']} (video_{video_idx}): Score={score_str}, Video={rollout['video_file']}{candidate_str}, Flow={rollout.get('flow_file', 'N/A')}, Selected Flow={selected_flow}\n")
                        if flow_switch.get('reason'):
                            f.write(f"      Flow Switch: {flow_switch.get('reason')}\n")
                        if rank_reason:
                            f.write(f"      Rank Reason: {rank_reason}\n")
                        if rollout.get('generation_prompt') and rollout.get('generation_prompt') != entry.get('original_action'):
                            f.write(f"      Generation Prompt: {rollout['generation_prompt']}\n")
                        if rollout.get('error'):
                            f.write(f"      Error: {rollout['error']}\n")
                f.write("\n")

        self._save_timing_stats()
        print("\n📊 Debug summaries saved:")
        print(f"   - {self.debug_dir}/SUMMARY.txt")
        print(f"   - {self.debug_dir}/scores_summary.csv")
        print(f"   - {self.debug_dir}/complete_debug_log.json")
        print(f"   - {self.debug_dir}/timing_stats.json")
        print(f"   - {self.debug_dir}/timing_summary.txt")
        print(f"   - {self.debug_dir}/final_summary.json")

    def _save_timing_stats(self):
        import statistics
        timing_analysis = {
            'planning_total_sec': self.timing_stats['planning_total'],
            'vlm_propose_actions': {},
            'vlm_score_rollout': {},
            'vlm_batch_ranking': {},
            'vlm_horizon_count': {},
            'prompt_extension': {},
            'video_generation': {},
            'step_total': {}
        }

        if self.timing_stats['vlm_horizon_count']:
            times = [t[1] for t in self.timing_stats['vlm_horizon_count']]
            timing_analysis['vlm_horizon_count'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_run': [{'horizon': t[0], 'duration_sec': t[1]} for t in self.timing_stats['vlm_horizon_count']]
            }

        if self.timing_stats['vlm_propose_actions']:
            times = [t[2] for t in self.timing_stats['vlm_propose_actions']]
            timing_analysis['vlm_propose_actions'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_step': [{'step': t[0], 'beam': t[1], 'duration_sec': t[2]} for t in self.timing_stats['vlm_propose_actions']]
            }

        if self.timing_stats['vlm_score_rollout']:
            times = [t[4] for t in self.timing_stats['vlm_score_rollout']]
            timing_analysis['vlm_score_rollout'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_rollout': [{'step': t[0], 'beam': t[1], 'action_idx': t[2], 'sample_idx': t[3], 'duration_sec': t[4]}
                                for t in self.timing_stats['vlm_score_rollout']]
            }

        if self.timing_stats['vlm_batch_ranking']:
            times = [t[2] for t in self.timing_stats['vlm_batch_ranking']]
            timing_analysis['vlm_batch_ranking'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_step': [{'step': t[0], 'num_candidates': t[1], 'duration_sec': t[2]} 
                             for t in self.timing_stats['vlm_batch_ranking']]
            }

        if self.timing_stats['video_generation']:
            times = [t[4] for t in self.timing_stats['video_generation']]
            timing_analysis['video_generation'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_action': [{'step': t[0], 'beam': t[1], 'action_idx': t[2], 'num_samples': t[3], 'duration_sec': t[4]}
                               for t in self.timing_stats['video_generation']]
            }

        if self.timing_stats['prompt_extension']:
            times = [t[3] for t in self.timing_stats['prompt_extension']]
            timing_analysis['prompt_extension'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_action': [{'step': t[0], 'beam': t[1], 'action_idx': t[2], 'duration_sec': t[3]}
                               for t in self.timing_stats['prompt_extension']]
            }

        if self.timing_stats['step_total']:
            times = [t[1] for t in self.timing_stats['step_total']]
            timing_analysis['step_total'] = {
                'count': len(times), 'total_sec': sum(times),
                'mean_sec': statistics.mean(times), 'median_sec': statistics.median(times),
                'min_sec': min(times), 'max_sec': max(times),
                'stdev_sec': statistics.stdev(times) if len(times) > 1 else 0.0,
                'per_step': [{'step': t[0], 'duration_sec': t[1]} for t in self.timing_stats['step_total']]
            }

        with open(self.debug_dir / "timing_stats.json", 'w') as f:
            json.dump(timing_analysis, f, indent=2)

        with open(self.debug_dir / "timing_summary.txt", 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VISUAL LANGUAGE PLANNING - TIMING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Planning Time: {timing_analysis['planning_total_sec']:.2f}s\n\n")
            def section(name):
                f.write("=" * 80 + "\n"); f.write(name + "\n"); f.write("=" * 80 + "\n")
            if timing_analysis['vlm_horizon_count']:
                s = timing_analysis['vlm_horizon_count']; section("VLM HORIZON COUNT")
                f.write(f"Count: {s['count']}\nTotal: {s['total_sec']:.2f}s\nMean: {s['mean_sec']:.2f}s\n"
                        f"Median: {s['median_sec']:.2f}s\nRange: [{s['min_sec']:.2f}s, {s['max_sec']:.2f}s]\n\n")
            if timing_analysis['vlm_propose_actions']:
                s = timing_analysis['vlm_propose_actions']; section("VLM PROPOSE ACTIONS")
                f.write(f"Count: {s['count']}\nTotal: {s['total_sec']:.2f}s\nMean: {s['mean_sec']:.2f}s\n"
                        f"Median: {s['median_sec']:.2f}s\nRange: [{s['min_sec']:.2f}s, {s['max_sec']:.2f}s]\n\n")
            if timing_analysis['vlm_score_rollout']:
                s = timing_analysis['vlm_score_rollout']; section("VLM SCORE ROLLOUT")
                f.write(f"Count: {s['count']}\nTotal: {s['total_sec']:.2f}s\nMean: {s['mean_sec']:.2f}s\n"
                        f"Median: {s['median_sec']:.2f}s\nRange: [{s['min_sec']:.2f}s, {s['max_sec']:.2f}s]\n\n")
            if timing_analysis['video_generation']:
                s = timing_analysis['video_generation']; section("VIDEO GENERATION")
                f.write(f"Count: {s['count']}\nTotal: {s['total_sec']:.2f}s\nMean: {s['mean_sec']:.2f}s\n"
                        f"Median: {s['median_sec']:.2f}s\nRange: [{s['min_sec']:.2f}s, {s['max_sec']:.2f}s]\n\n")
            if timing_analysis['prompt_extension']:
                s = timing_analysis['prompt_extension']; section("PROMPT EXTENSION")
                f.write(f"Count: {s['count']}\nTotal: {s['total_sec']:.2f}s\nMean: {s['mean_sec']:.2f}s\n"
                        f"Median: {s['median_sec']:.2f}s\nRange: [{s['min_sec']:.2f}s, {s['max_sec']:.2f}s]\n\n")
            if timing_analysis['step_total']:
                s = timing_analysis['step_total']; section("STEP TOTALS")
                f.write(f"Count: {s['count']}\nTotal: {s['total_sec']:.2f}s\nMean: {s['mean_sec']:.2f}s\n"
                        f"Median: {s['median_sec']:.2f}s\nRange: [{s['min_sec']:.2f}s, {s['max_sec']:.2f}s]\n\n")

def main():
    """Run the command-line entry point."""
    p = argparse.ArgumentParser(description="NovaPlan planner with OpenAI VLM and WAN 2.2/Veo 3 adapters")
    reasoning_effort_choices = ("none", "low", "medium", "high", "xhigh", "max")
    p.add_argument("--goal", type=str, default="Put each block into the container of the matching color.")
    p.add_argument("--input_frame", type=Path, required=True)
    p.add_argument(
        "--output_video",
        type=Path,
        default=None,
        help="Selected plan video path. Defaults to RUN_DIR/selected_plan.mp4.",
    )
    p.add_argument("--debug_dir", type=Path, default=None,
                   help="Run artifact directory. Defaults to runs/planner/<timestamp>/.")
    p.add_argument("--beam_size", type=positive_int, default=2)
    p.add_argument("--num_action_per_beam", type=positive_int, default=2,
                   help="Strategic actions proposed per beam (paper setting: 2).")
    p.add_argument("--num_video_per_action", type=positive_int, default=4,
                   help="Strategic videos per action per enabled generation backend (paper setting: 4).")
    p.add_argument("--segment_T", type=positive_int, default=41)
    p.add_argument("--fps", type=positive_int, default=16)
    p.add_argument("--horizon", type=positive_int, default=3)
    p.add_argument("--disable_horizon_count", action="store_true",
                   help="Skip VLM horizon counting and use --horizon directly.")
    p.add_argument("--allow_horizon_fallback", action="store_true",
                   help="Allow fallback to --horizon if automatic task-structure/horizon counting fails.")
    p.add_argument("--horizon_count_min", type=positive_int, default=1,
                   help="Minimum nonzero horizon allowed for VLM horizon counting.")
    p.add_argument("--horizon_count_max", type=positive_int, default=int(os.getenv("NOVAPLAN_HORIZON_COUNT_MAX", "8")),
                   help="Maximum horizon allowed for VLM horizon counting.")
    p.add_argument(
        "--use_apis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use live video APIs; pass --no-use_apis for the local mock adapter.",
    )
    p.add_argument("--video_backend", type=str, default="both", choices=["wan22", "veo3", "both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vlm_model", type=str, default=os.getenv("NOVAPLAN_VLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_VLM_MODEL)),
                   help="Planner ChatGPT/OpenAI VLM model. Override with NOVAPLAN_VLM_MODEL or OPENAI_MODEL.")
    p.add_argument("--vlm_reasoning_effort", type=str, choices=reasoning_effort_choices,
                   default=os.getenv("NOVAPLAN_VLM_REASONING_EFFORT"),
                   help="Default reasoning effort for GPT-5-family VLM calls. Omit to use the model default.")
    p.add_argument("--horizon_vlm_model", type=str, default=os.getenv("NOVAPLAN_HORIZON_VLM_MODEL"),
                   help="Optional lighter model for coupling/horizon counting. Defaults to --vlm_model.")
    p.add_argument("--horizon_reasoning_effort", type=str, choices=reasoning_effort_choices,
                   default=os.getenv("NOVAPLAN_HORIZON_REASONING_EFFORT", "low"),
                   help="Reasoning effort for coupling/horizon counting. Default: low.")
    p.add_argument("--action_reasoning_effort", type=str, choices=reasoning_effort_choices,
                   default=os.getenv("NOVAPLAN_ACTION_REASONING_EFFORT"),
                   help="Reasoning effort for action proposal. Defaults to --vlm_reasoning_effort/model default.")
    p.add_argument("--prompt_extension_reasoning_effort", type=str, choices=reasoning_effort_choices,
                   default=os.getenv("NOVAPLAN_PROMPT_EXTENSION_REASONING_EFFORT", "low"),
                   help="Reasoning effort for WAN prompt extension. Default: low.")
    p.add_argument("--ranking_reasoning_effort", type=str, choices=reasoning_effort_choices,
                   default=os.getenv("NOVAPLAN_RANKING_REASONING_EFFORT"),
                   help="Reasoning effort for candidate ranking. Defaults to --vlm_reasoning_effort/model default.")
    p.add_argument("--scoring_reasoning_effort", type=str, choices=reasoning_effort_choices,
                   default=os.getenv("NOVAPLAN_SCORING_REASONING_EFFORT"),
                   help="Reasoning effort for per-rollout scoring fallback. Defaults to --vlm_reasoning_effort/model default.")
    p.add_argument("--disable_prompt_extension", action="store_true",
                   help="Skip OpenAI prompt rewriting before WAN video generation.")

    # Flow extraction options
    p.add_argument("--disable_flow", action="store_true",
                   help="Disable inline 2D flow extraction during video generation.")
    p.add_argument("--mask_prompt", type=str, default="object",
                   help="Fallback SAM3 mask prompt used only when a VLM action proposal omits track_object.")
    p.add_argument("--disable_flow_for_scoring", action="store_true",
                   help="Do not pass flow images to VLM candidate scoring.")
    p.add_argument("--flow_switch_theta_deg",
                   type=rotation_degrees, default=DEFAULT_FLOW_SWITCH_THETA_DEG,
                   help="Object-flow adjacent rotation threshold in degrees before trying hand flow.")
    
    # Video server and Wan options
    p.add_argument("--video_server_url", type=http_url,
                   default=_env_first("NOVAPLAN_VIDEO_SERVER_URL") or DEFAULT_VIDEO_SERVER_URL,
                   help="NovaPlan video-generation server URL.")
    p.add_argument("--wan_server", dest="video_server_url", type=http_url,
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--wan_size", type=image_size, default="1280*720",
                   help="Output WxH key supported by server (e.g. 1280*720, 720*1280, 832*480, 480*832)")
    p.add_argument("--wan_sample_steps", type=positive_int, default=None,
                   help="Sampler steps. None = server default")
    p.add_argument("--wan_sample_solver", type=str, default="euler", choices=["unipc", "dpm++","euler"],
                   help="Sampling solver")
    p.add_argument("--wan_guide_scale", type=positive_float, default=None,
                   help="Classifier-free guidance scale. None uses 1.0 in fast mode or 3.5 in slow mode.")
    p.add_argument("--save_debug_videos", action=argparse.BooleanOptionalAction, default=True,
                   help="Save each rollout video for debugging.")
    p.add_argument("--wan_fast_mode", action=argparse.BooleanOptionalAction, default=True,
                   help="Use WAN LightX2V 4-step LoRAs; pass --no-wan_fast_mode for 20-step inference.")
    p.add_argument("--serial_debug", action="store_true", default=False,
                   help="Run everything sequentially (1 worker) for debugging")
    args = p.parse_args()
    validate_horizon_bounds(
        p,
        minimum=args.horizon_count_min,
        maximum=args.horizon_count_max,
    )
    if args.video_server_url == "":
        args.video_server_url = (
            _env_first("NOVAPLAN_VIDEO_SERVER_URL") or DEFAULT_VIDEO_SERVER_URL
        )
    
    enable_flow = not args.disable_flow
    
    # Load input/start frame
    start = load_image_as_array(args.input_frame)
    start = np.array(start, dtype=np.uint8)

    task_models = {}
    if args.horizon_vlm_model:
        task_models["horizon"] = args.horizon_vlm_model
    task_reasoning_efforts = {
        "horizon": args.horizon_reasoning_effort,
        "prompt_extension": args.prompt_extension_reasoning_effort,
        "action": args.action_reasoning_effort or args.vlm_reasoning_effort,
        "ranking": args.ranking_reasoning_effort or args.vlm_reasoning_effort,
        "scoring": args.scoring_reasoning_effort or args.vlm_reasoning_effort,
    }
    vlm = VLMAdapter(
        model=args.vlm_model,
        reasoning_effort=args.vlm_reasoning_effort,
        task_models=task_models,
        task_reasoning_efforts=task_reasoning_efforts,
    )

    # Create NovaPlan client for flow extraction (used for Veo3 backend or separate flow calls)
    flow_extraction_client = None
    if enable_flow:
        # FlowExtractionClient talks to the video server for 2D flow-only processing.
        flow_extraction_client = FlowExtractionClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )
        print(
            f"2D flow extraction enabled on {args.video_server_url}; "
            f"fallback_mask_prompt={args.mask_prompt}"
        )

    # Prepare per-request WAN overrides passed on every rollout call.
    wan_overrides = {
        "size": args.wan_size,
        "num_frames": args.segment_T,
        "fps": args.fps,
        "seed": args.seed,
        "sampling_steps": args.wan_sample_steps,
        "guide_scale": args.wan_guide_scale,
        "sample_solver": args.wan_sample_solver,
        "fast_mode": args.wan_fast_mode,
    }

    if args.video_backend == "wan22":
        t2v = WanVideoGenerationClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )
    elif args.video_backend == "veo3":
        print("Using Veo3 backend")
        t2v = _make_veo_adapter(mock=(not args.use_apis), seed=args.seed)
        wan_overrides = {}
    else:
        print("Using hybrid WAN22 + Veo3 backend")
        wan_client = WanVideoGenerationClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )
        veo_client = _make_veo_adapter(mock=(not args.use_apis), seed=args.seed)
        t2v = HybridVideoGenerationClient(
            wan_client=wan_client,
            veo_client=veo_client,
            flow_client=flow_extraction_client,
        )

    # Planner
    planner = NovaPlanPlanner(
        vlm=vlm,
        t2v=t2v,
        beam_size=args.beam_size,
        num_action_per_beam=args.num_action_per_beam,
        num_video_per_action=args.num_video_per_action,
        segment_T=args.segment_T,
        fps=args.fps,
        seed=args.seed,
        save_debug_videos=args.save_debug_videos,
        wan_overrides=wan_overrides if args.video_backend in {"wan22", "both"} else None,
        serial_debug=args.serial_debug,
        horizon=args.horizon,
        use_prompt_extension=not args.disable_prompt_extension,
        use_horizon_count=not args.disable_horizon_count,
        allow_horizon_fallback=args.allow_horizon_fallback,
        horizon_count_min=args.horizon_count_min,
        horizon_count_max=args.horizon_count_max,
        enable_flow=enable_flow,
        mask_prompt=args.mask_prompt,
        use_flow_for_scoring=not args.disable_flow_for_scoring,
        flow_switch_theta_deg=args.flow_switch_theta_deg,
        debug_dir=args.debug_dir,
    )

    # Attach the flow client for later use inside the planner, especially for Veo.
    if flow_extraction_client is not None:
        planner.flow_extraction_client = flow_extraction_client

    # Run plan-only beam search.
    best_beam = planner.plan(start, args.goal)

    # Output best result (stitch all chosen segments)
    final_video = best_beam.stitched()
    if final_video.size > 0:
        output_video = args.output_video or (planner.debug_dir / "selected_plan.mp4")
        write_video(output_video, final_video, fps=args.fps)
        print(f"Final video saved to {output_video.resolve()}")
    else:
        print("Plan failed, no video generated.")


if __name__ == "__main__":
    main()
