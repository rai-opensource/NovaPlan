# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Segment tracked objects in video with SAM 3 ComfyUI nodes."""

import os
import sys
import torch
import numpy as np
import time
import contextlib
import io
import tempfile
import threading
import cv2
from fractions import Fraction

# Import ComfyUI video types if available
try:
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
except ImportError:
    VideoFromComponents = None
    VideoComponents = None

# Add SAM3 to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sam3_root = os.path.dirname(current_dir)
if sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)

# --- GLOBAL CACHE (PERSISTENT WEIGHTS) ---
GLOBAL_SAM3_PREDICTOR = None
VERBOSE_LOGS = os.environ.get("NOVAPLAN_VERBOSE_LOGS", "0").lower() in {"1", "true", "yes", "on"}


def _log_debug(*args, **kwargs):
    if VERBOSE_LOGS:
        print(*args, **kwargs)


def _log_timing(module, total_s, **fields):
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[TIMING] {module}: total={total_s:.2f}s{suffix}", flush=True)


def _quiet_external_logs():
    if VERBOSE_LOGS:
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


class Sam3VideoNode:
    """Expose SAM 3 video segmentation as a ComfyUI node."""
    DESCRIPTION = "NovaPlan SAM3 video flow; novaplan-model-ownership-v1"

    @classmethod
    def INPUT_TYPES(s):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "video": ("VIDEO",),
                "prompt": ("STRING", {"multiline": True, "default": "a person"}),
                "return_visualization": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO", "MASK", "BBOX")
    RETURN_NAMES = ("visualized_video", "mask", "bounding_box")
    FUNCTION = "process"
    CATEGORY = "NovaPlan/SAM3"

    def _save_frames_to_temp(self, video_frames, temp_dir):
        frames_np = (video_frames.cpu().numpy() * 255).astype(np.uint8)
        for i, frame in enumerate(frames_np):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(temp_dir, f"{i:05d}.jpg"), frame_bgr)
        return frames_np

    def process(self, video, prompt, return_visualization=False):
        """Execute this ComfyUI node."""
        node_start = time.perf_counter()
        global GLOBAL_SAM3_PREDICTOR

        # 1. Load Model ONCE (Fixes Reloading Lag)
        if GLOBAL_SAM3_PREDICTOR is None:
            _log_debug("[Sam3VideoNode] Loading SAM3 model into VRAM", flush=True)
            from sam3.model_builder import build_sam3_video_predictor
            with _quiet_external_logs():
                GLOBAL_SAM3_PREDICTOR = build_sam3_video_predictor()

            # SAFETY: Move to CUDA immediately to prevent CPU/GPU mismatches
            if hasattr(GLOBAL_SAM3_PREDICTOR, "model"):
                GLOBAL_SAM3_PREDICTOR.model.to("cuda")
        else:
            _log_debug("[Sam3VideoNode] Using cached SAM3 model", flush=True)

        video_predictor = GLOBAL_SAM3_PREDICTOR
        render_masklet_frame = None
        if return_visualization:
            try:
                from sam3.visualization_utils import render_masklet_frame as _render_masklet_frame
                render_masklet_frame = _render_masklet_frame
            except Exception as vis_error:
                _log_debug(f"[Sam3VideoNode] SAM3 renderer unavailable, using fallback overlay: {vis_error}", flush=True)

        # 2. Parse Video Input
        input_fps = 24.0
        if hasattr(video, 'get_components'):
            components = video.get_components()
            video_frames = components.images
            input_fps = float(components.frame_rate) if components.frame_rate else 24.0
        else:
            video_frames = video

        session_id = None

        # 3. Execution Context (Fixes BFloat16 Crash)
        # We wrap the ENTIRE inference block in autocast.
        # This acts as a "Universal Translator" for mixed precision types.
        inference_ms = 0.0

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Use /dev/shm for fastest I/O (RAM disk)
                # Create a unique temp dir in shared memory
                import shutil
                temp_dir_base = "/dev/shm/sam3_temp"
                os.makedirs(temp_dir_base, exist_ok=True)
                temp_dir = tempfile.mkdtemp(dir=temp_dir_base)

                try:
                    frames_np = self._save_frames_to_temp(video_frames, temp_dir)
                    H, W = frames_np.shape[1], frames_np.shape[2]

                    # Start Inference Session (Allocates ~4GB VRAM)
                    # Exclude start_session from timing as it's mostly disk I/O (reading images)
                    with _quiet_external_logs():
                        response = video_predictor.handle_request(
                            request=dict(type="start_session", resource_path=temp_dir)
                        )
                    session_id = response["session_id"]

                    # CUDA Event timing for accurate GPU inference measurement
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()

                    # Add Prompt (Runs backbone on frame 0) and propagate.
                    with _quiet_external_logs():
                        video_predictor.handle_request(
                            request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
                        )

                        _log_debug("[Sam3VideoNode] Starting propagation", flush=True)
                        propagated_outputs = {}
                        stream = video_predictor.handle_stream_request(
                            request=dict(type="propagate_in_video", session_id=session_id, propagation_direction="both", start_frame_index=0)
                        )

                        for item in stream:
                            propagated_outputs[item["frame_index"]] = item["outputs"]

                    _log_debug("[Sam3VideoNode] Propagation done", flush=True)
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_ms = start_event.elapsed_time(end_event)

                    # Print timing immediately after inference completes
                    _log_debug(f"[Sam3VideoNode] Inference: {inference_ms / 1000:.2f}s", flush=True)

                    # Build masks/bboxes. Visualization frames are opt-in because video tensors
                    # and SaveVideo encoding are pure debug overhead for production flow extraction.
                    visualized_frames = [] if return_visualization else None
                    mask_frames = []
                    bbox_frames = []

                    for i in range(len(frames_np)):
                        mask_combined = np.zeros((H, W), dtype=np.float32)
                        bbox = [0, 0, 0, 0]

                        if i in propagated_outputs:
                            outputs = propagated_outputs[i]
                            masks = outputs.get("out_binary_masks", [])
                            for m in masks:
                                m_np = m.astype(np.float32)
                                if m_np.shape != (H, W):
                                    m_np = cv2.resize(m_np, (W, H), interpolation=cv2.INTER_NEAREST)
                                mask_combined = np.maximum(mask_combined, m_np)

                            mask_combined = (mask_combined > 0.5).astype(np.float32)
                            rows = np.any(mask_combined, axis=0)
                            cols = np.any(mask_combined, axis=1)
                            y_indices, x_indices = np.where(cols)[0], np.where(rows)[0]

                            if len(x_indices) > 0 and len(y_indices) > 0:
                                bbox = [x_indices[0], y_indices[0], x_indices[-1] - x_indices[0] + 1, y_indices[-1] - y_indices[0] + 1]

                        if return_visualization and i in propagated_outputs and render_masklet_frame is not None:
                            visualized_frames.append(render_masklet_frame(frames_np[i], propagated_outputs[i], frame_idx=i))
                        elif return_visualization:
                            vis_frame = frames_np[i].copy()
                            mask_bool = mask_combined > 0.5
                            if mask_bool.any():
                                overlay_color = np.array([34, 197, 94], dtype=np.float32)
                                vis_frame[mask_bool] = (
                                    0.55 * vis_frame[mask_bool].astype(np.float32)
                                    + 0.45 * overlay_color
                                ).astype(np.uint8)
                                contours, _ = cv2.findContours(
                                    mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                                )
                                cv2.drawContours(vis_frame, contours, -1, (255, 255, 255), 2)
                                x, y, w, h = bbox
                                if w > 0 and h > 0:
                                    cv2.rectangle(vis_frame, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), (255, 255, 0), 2)
                            visualized_frames.append(vis_frame)

                        mask_frames.append(mask_combined)
                        bbox_frames.append(bbox)

                    if return_visualization:
                        vis_tensor = torch.from_numpy(np.stack(visualized_frames)).float() / 255.0
                    else:
                        vis_tensor = video_frames
                    mask_tensor = torch.from_numpy(np.stack(mask_frames))
                    bbox_tensor = torch.tensor(bbox_frames)

                finally:
                    # Cleanup temp dir
                    try:
                        shutil.rmtree(temp_dir)
                    except OSError:
                        pass

        except Exception as e:
            print(f"[Sam3VideoNode] ERROR: {e}", flush=True)
            raise e

        finally:
            _log_timing("Sam3VideoNode", time.perf_counter() - node_start, inference=f"{inference_ms / 1000:.2f}s")
            # 4. Cleanup Session State (Fixes 4GB Memory Leak)
            # Crucial: Destroy the session state so VRAM returns to the "Plateau"
            # The global predictor owns its cache; do not force-clear the CUDA
            # allocator from inside a ComfyUI node.
            if session_id is not None:
                def _close_session():
                    try:
                        with _quiet_external_logs():
                            video_predictor.handle_request(
                                request=dict(type="close_session", session_id=session_id)
                            )
                    except Exception as close_error:
                        print(f"[Sam3VideoNode] Session close failed: {close_error}", flush=True)

                if os.environ.get("SAM3_ASYNC_SESSION_CLOSE", "1") != "0":
                    threading.Thread(target=_close_session, daemon=True).start()
                else:
                    _close_session()

        # Output Formatting
        if VideoFromComponents and VideoComponents:
            video_out = VideoFromComponents(VideoComponents(images=vis_tensor, frame_rate=Fraction(input_fps)))
        else:
            video_out = vis_tensor

        return (video_out, mask_tensor, bbox_tensor)

NODE_CLASS_MAPPINGS = { "Sam3VideoNode": Sam3VideoNode }
NODE_DISPLAY_NAME_MAPPINGS = { "Sam3VideoNode": "SAM3 Video Predictor" }
