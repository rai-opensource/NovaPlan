# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Track persistent 3D points with the upstream TAPIP3D model."""

import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import contextlib
import io
import cv2
import threading
from fractions import Fraction

from comfy_api.input_impl import VideoFromComponents
from comfy_api.util import VideoComponents

from .upstream import (
    TAPIP3D_ROOT,
    get_grid_queries,
    inference,
    load_model,
    resize_depth_bilinear,
)

VERBOSE_LOGS = os.environ.get("NOVAPLAN_VERBOSE_LOGS", "0").lower() in {"1", "true", "yes", "on"}


def _log_debug(*args, **kwargs):
    if VERBOSE_LOGS:
        print(*args, **kwargs)


def _log_warning(*args, **kwargs):
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


def _video_to_uint8(video_frames):
    if isinstance(video_frames, torch.Tensor):
        arr = video_frames.detach().cpu().numpy()
    else:
        arr = np.asarray(video_frames)
    arr = np.nan_to_num(arr)
    if arr.dtype == np.uint8:
        return arr.copy()
    if arr.size and float(np.nanmax(arr)) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _project_tracks(coords_np, intrinsics):
    fx, fy, cx, cy = intrinsics
    z = coords_np[..., 2]
    valid = np.isfinite(coords_np).all(axis=-1) & (z > 1e-6)
    safe_z = np.where(valid, z, 1.0)
    x = coords_np[..., 0] * fx / safe_z + cx
    y = coords_np[..., 1] * fy / safe_z + cy
    points = np.stack([x, y], axis=-1)
    return points, valid


def _track_colors_bgr(num_tracks):
    if num_tracks <= 0:
        return []
    hues = np.linspace(0, 179, num_tracks, dtype=np.uint8)
    hsv = np.stack(
        [hues, np.full_like(hues, 255), np.full_like(hues, 255)],
        axis=1,
    ).reshape(num_tracks, 1, 3)
    colors_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(num_tracks, 3)
    return [tuple(int(v) for v in color.tolist()) for color in colors_bgr]


def _save_cv2_tracking_video(output_dir, save_prefix, video_frames, fps):
    video_uint8 = _video_to_uint8(video_frames)
    if video_uint8.ndim != 4 or video_uint8.shape[0] == 0:
        return False

    h, w = video_uint8.shape[1:3]
    tracking_video_path = os.path.join(output_dir, f"{save_prefix}_tracking_video.mp4")
    writer = cv2.VideoWriter(
        tracking_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(round(float(fps)))),
        (w, h),
    )
    try:
        for frame in video_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return os.path.exists(tracking_video_path)


def _build_visualizations(video_frames, coords_np, visibs_bool, intrinsics, return_debug_media):
    video_np = _video_to_uint8(video_frames)
    t_video, h, w, _ = video_np.shape
    flow_img_bgr = cv2.cvtColor(video_np[0].copy(), cv2.COLOR_RGB2BGR) if t_video else np.zeros((h, w, 3), dtype=np.uint8)

    if coords_np.ndim != 3 or coords_np.shape[1] == 0:
        flow_tensor = torch.from_numpy(cv2.cvtColor(flow_img_bgr, cv2.COLOR_BGR2RGB)).float().unsqueeze(0) / 255.0
        return video_frames, flow_tensor, 0

    coords_np = coords_np[:t_video]
    visibs_bool = visibs_bool[:coords_np.shape[0], :coords_np.shape[1]]
    points_2d, valid_geom = _project_tracks(coords_np, intrinsics)
    visible = valid_geom & visibs_bool
    in_bounds = (
        visible
        & (points_2d[..., 0] >= 0)
        & (points_2d[..., 0] < w)
        & (points_2d[..., 1] >= 0)
        & (points_2d[..., 1] < h)
    )
    colors_list = _track_colors_bgr(coords_np.shape[1])

    if return_debug_media:
        viz_frames = []
        for t in range(coords_np.shape[0]):
            img_bgr = cv2.cvtColor(video_np[t].copy(), cv2.COLOR_RGB2BGR)
            for i in range(coords_np.shape[1]):
                if in_bounds[t, i]:
                    px, py = int(points_2d[t, i, 0]), int(points_2d[t, i, 1])
                    cv2.circle(img_bgr, (px, py), 3, colors_list[i], -1)
            viz_frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        output_video = torch.from_numpy(np.stack(viz_frames)).float() / 255.0
    else:
        output_video = video_frames

    valid_points_count = 0
    drawn_tracks = 0
    for i in range(coords_np.shape[1]):
        valid_indices = np.where(in_bounds[:, i])[0]
        if len(valid_indices) < 2:
            continue
        pts = points_2d[valid_indices, i, :].astype(np.int32).reshape((-1, 1, 2))
        pts_in_bounds = (
            (pts[:, 0, 0] >= 0)
            & (pts[:, 0, 0] < w)
            & (pts[:, 0, 1] >= 0)
            & (pts[:, 0, 1] < h)
        )
        valid_points_count += int(np.sum(pts_in_bounds))
        if np.any(pts_in_bounds):
            cv2.polylines(flow_img_bgr, [pts], isClosed=False, color=colors_list[i], thickness=1)
            drawn_tracks += 1

    _log_debug(
        f"[TAPIP3DNode] Flow viz: {drawn_tracks}/{coords_np.shape[1]} tracks drawn, {valid_points_count} valid points in bounds",
        flush=True,
    )
    flow_tensor = torch.from_numpy(cv2.cvtColor(flow_img_bgr, cv2.COLOR_BGR2RGB)).float().unsqueeze(0) / 255.0
    return output_video, flow_tensor, drawn_tracks


class TAPIP3DNode:
    """Expose upstream TAPIP3D point tracking as a ComfyUI node."""
    _model_cache = {}
    _model_base_image_size = {}  # Store original base image size to prevent mutation issues
    _model_lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "video": ("VIDEO",),
                "depth": ("IMAGE",),  # Raw depth tensor [T, H, W, 1]
                "intrinsics": ("INTRINSICS",),
                "checkpoint": ("STRING", {"default": "checkpoints/tapip3d_final.pth"}),
                "resolution_factor": ("INT", {"default": 2, "min": 1, "max": 8}),
                "query_grid_size": ("INT", {"default": 32, "min": 0, "max": 128}),
                "vis_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "save_prefix": ("STRING", {"default": "flow_extraction"}),
                "return_debug_media": ("BOOLEAN", {"default": False}),
                "return_auxiliary_arrays": ("BOOLEAN", {"default": False}),

                # Safe mask query sampling parameters (like adaptive node)
                "mask_erosion": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Erode mask by this many pixels to avoid sampling near edges (0 = disabled)."}),
                "depth_grad_thresh": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Max depth gradient for safe sampling. Lower = stricter (0 = disabled)."}),
                "depth_laplacian_thresh": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Max depth Laplacian (2nd derivative) to filter boundary ramps. 0 = disabled. Try 0.02-0.05."}),
                "multiscale_grad": ("BOOLEAN", {"default": False,
                    "tooltip": "Use multi-scale gradient check (catches wider boundary artifacts)."}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("VIDEO", "TENSOR", "TENSOR", "IMAGE")
    RETURN_NAMES = ("output_video", "coords_3d", "visibilities", "flow_visualization")
    FUNCTION = "process"
    CATEGORY = "NovaPlan/TAPIP3D"

    def get_mask_queries(self, mask, depth, intrinsics, extrinsics, num_points,
                         mask_erosion=0, depth_grad_thresh=0.0, depth_laplacian_thresh=0.0,
                         multiscale_grad=False, query_frame=0):
        """
        Sample query points from mask with optional safety features.

        Args:
            mask: [H, W] mask tensor
            depth: [H, W] depth tensor
            intrinsics: [3, 3] camera intrinsics
            extrinsics: [4, 4] camera extrinsics
            num_points: Target number of points to sample
            mask_erosion: Pixels to erode mask by (avoids sampling near edges)
            depth_grad_thresh: Max depth gradient for safe sampling (avoids depth edges)
            depth_laplacian_thresh: Max depth Laplacian to filter boundary ramps (2nd derivative)
            multiscale_grad: Use multi-scale gradient check for wider boundary artifacts
        """
        _log_debug(f"[TAPIP3DNode] get_mask_queries() called with mask shape={mask.shape}, depth shape={depth.shape}", flush=True)
        _log_debug(f"[TAPIP3DNode] Safety params: mask_erosion={mask_erosion}, depth_grad_thresh={depth_grad_thresh}, laplacian={depth_laplacian_thresh}, multiscale={multiscale_grad}", flush=True)

        device = mask.device

        # 1. Start with basic mask
        valid_area = mask > 0.5

        # 2. Mask Erosion (optional) - sample from object interior, not edges
        if mask_erosion > 0:
            mask_in = mask.unsqueeze(0).unsqueeze(0).float()
            k = 2 * mask_erosion + 1
            # Erosion = negative max pooling of negated mask
            mask_eroded = -F.max_pool2d(-mask_in, kernel_size=k, stride=1, padding=mask_erosion)
            valid_area_eroded = mask_eroded.squeeze() > 0.5

            # Check if erosion left enough points
            eroded_count = valid_area_eroded.sum().item()
            original_count = valid_area.sum().item()

            if eroded_count >= num_points // 4:
                valid_area = valid_area_eroded
                _log_debug(f"[TAPIP3DNode] Mask erosion: {original_count} -> {eroded_count} valid pixels", flush=True)
            else:
                _log_debug(f"[TAPIP3DNode] Mask erosion too aggressive ({eroded_count} < {num_points // 4}), using original mask", flush=True)

        # 3. Depth Gradient Check (optional) - avoid depth discontinuities
        depth_uns = depth.unsqueeze(0).unsqueeze(0)

        if depth_grad_thresh > 0:
            if multiscale_grad:
                # Multi-scale gradient: catch wider boundary artifacts (the "lift up" ramps)
                grad_combined = torch.zeros_like(depth)
                for scale in [1, 2, 4]:
                    H_d, W_d = depth_uns.shape[2], depth_uns.shape[3]
                    if scale >= H_d or scale >= W_d:
                        continue
                    dy_s = torch.abs(depth_uns[:, :, scale:, :] - depth_uns[:, :, :-scale, :])
                    dx_s = torch.abs(depth_uns[:, :, :, scale:] - depth_uns[:, :, :, :-scale])
                    # Normalize by scale to make thresholds comparable
                    dy_s = dy_s / scale
                    dx_s = dx_s / scale
                    dy_s = F.pad(dy_s, (0, 0, 0, scale))
                    dx_s = F.pad(dx_s, (0, scale, 0, 0))
                    grad_combined = torch.maximum(grad_combined, (dx_s + dy_s).squeeze())
                grad_mag = grad_combined
                _log_debug("[TAPIP3DNode] Using multi-scale gradient (scales 1,2,4)", flush=True)
            else:
                # Original single-scale gradient
                dy = torch.abs(depth_uns[:, :, 1:, :] - depth_uns[:, :, :-1, :])
                dx = torch.abs(depth_uns[:, :, :, 1:] - depth_uns[:, :, :, :-1])
                dy = F.pad(dy, (0, 0, 0, 1))  # Pad bottom
                dx = F.pad(dx, (0, 1, 0, 0))  # Pad right
                grad_mag = (dx + dy).squeeze()

            low_grad = grad_mag < depth_grad_thresh
            valid_area_grad = valid_area & low_grad

            grad_count = valid_area_grad.sum().item()
            original_count = valid_area.sum().item()

            if grad_count >= num_points // 4:
                valid_area = valid_area_grad
                _log_debug(f"[TAPIP3DNode] Depth gradient filter: {original_count} -> {grad_count} valid pixels", flush=True)
            else:
                _log_debug(f"[TAPIP3DNode] Depth gradient too strict ({grad_count} < {num_points // 4}), skipping", flush=True)

        # 4. Depth Laplacian Check (optional) - detect boundary "ramps" that have low gradient but wrong curvature
        # These are the "lift up" surfaces you see at object boundaries where depth incorrectly transitions
        if depth_laplacian_thresh > 0:
            # Laplacian kernel: detects areas where depth curvature is non-zero
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                             device=device, dtype=depth.dtype).view(1, 1, 3, 3)
            depth_laplacian = F.conv2d(depth_uns, laplacian_kernel, padding=1).squeeze()

            # Areas with low Laplacian are "flat" in 3D - either actual surfaces or background
            # Areas with high Laplacian are curved/ramped - often boundary artifacts
            low_laplacian = torch.abs(depth_laplacian) < depth_laplacian_thresh
            valid_area_lap = valid_area & low_laplacian

            lap_count = valid_area_lap.sum().item()
            original_count = valid_area.sum().item()

            if lap_count >= num_points // 4:
                valid_area = valid_area_lap
                _log_debug(f"[TAPIP3DNode] Depth Laplacian filter: {original_count} -> {lap_count} valid pixels", flush=True)
            else:
                _log_debug(f"[TAPIP3DNode] Depth Laplacian too strict ({lap_count} < {num_points // 4}), skipping", flush=True)

        # 5. Require valid depth
        valid_area = valid_area & (depth > 0)

        y, x = torch.where(valid_area)

        if len(y) == 0:
            # Fallback to center if mask is empty
            _log_warning("[TAPIP3DNode] Warning: no valid mask/depth points, using center fallback", flush=True)
            y = torch.tensor([mask.shape[0]//2], device=device)
            x = torch.tensor([mask.shape[1]//2], device=device)

        if len(y) > num_points:
            idx = torch.randperm(len(y), device=device)[:num_points]
            y = y[idx]
            x = x[idx]

        _log_debug(f"[TAPIP3DNode] Sampled {len(y)} query points (target: {num_points})", flush=True)

        xy = torch.stack([x, y], dim=1).float()
        d = depth[y, x]

        inv_intrinsics = torch.linalg.inv(intrinsics)
        inv_extrinsics = torch.linalg.inv(extrinsics)

        xy_homo = torch.cat([xy, torch.ones_like(xy[:, :1])], dim=-1)
        local_coords = (xy_homo @ inv_intrinsics.T) * d[:, None]
        local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[:, :1])], dim=-1)
        world_coords = (local_coords_homo @ inv_extrinsics.T)[:, :3]

        t_zeros = torch.full((len(y), 1), float(query_frame), device=device)
        queries = torch.cat([t_zeros, world_coords], dim=-1)
        return queries.unsqueeze(0)

    def process(self, video, depth, intrinsics, checkpoint, resolution_factor, query_grid_size, vis_threshold, start_index=0, save_prefix="flow_extraction", return_debug_media=False, return_auxiliary_arrays=False, mask_erosion=3, depth_grad_thresh=0.05, depth_laplacian_thresh=0.0, multiscale_grad=False, mask=None):
        """Execute this ComfyUI node."""
        node_start = time.perf_counter()
        _log_debug(f"[TAPIP3DNode] process() called with start_index={start_index}, save_prefix={save_prefix}", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        # 1. Prepare Video
        if hasattr(video, 'get_components'):
            components = video.get_components()
            video_frames = components.images
            fps = float(components.frame_rate) if components.frame_rate else 24.0
        else:
            video_frames = video
            fps = 24.0

        T_video, H, W, C = video_frames.shape

        # 2. Prepare Depth
        if hasattr(depth, 'get_components'):
            depth_frames = depth.get_components().images
            _log_debug(f"[TAPIP3DNode] Depth from VIDEO components: shape={depth_frames.shape}", flush=True)
        else:
            depth_frames = depth
            _log_debug(f"[TAPIP3DNode] Depth from tensor: shape={depth_frames.shape}", flush=True)

        # Debug: Raw depth input statistics
        _log_debug(f"[TAPIP3DNode] RAW depth input: dtype={depth_frames.dtype}, min={depth_frames.min():.6f}, max={depth_frames.max():.6f}, mean={depth_frames.mean():.6f}", flush=True)

        if depth_frames.dim() == 4:
            if depth_frames.shape[-1] == 3:
                _log_debug("[TAPIP3DNode] ⚠️ WARNING: Depth has 3 channels (RGB) - might be COLORIZED depth! Taking channel 0", flush=True)
                # Check if this looks like colorized magma (values in 0-1 range, not actual metric depth)
                if depth_frames.max() <= 1.0 and depth_frames.min() >= 0.0:
                    _log_debug("[TAPIP3DNode] ⚠️ WARNING: Depth values in 0-1 range! This is likely COLORIZED depth, NOT raw metric depth!", flush=True)
                depth_frames = depth_frames[..., 0]
            elif depth_frames.shape[-1] == 1:
                depth_frames = depth_frames.squeeze(-1)

        # Debug: Processed depth statistics
        _log_debug(f"[TAPIP3DNode] PROCESSED depth: shape={depth_frames.shape}, min={depth_frames.min():.6f}, max={depth_frames.max():.6f}", flush=True)

        T_depth_full = depth_frames.shape[0]
        H_depth, W_depth = depth_frames.shape[1], depth_frames.shape[2]

        # CRITICAL: Resize depth to match video spatial dimensions FIRST
        # This is needed for cached depth mode where depth was computed at a different resolution
        # (e.g., depth estimation target_size=518) than the sub-video being processed
        if H_depth != H or W_depth != W:
            _log_debug(f"[TAPIP3DNode] Resizing depth from {H_depth}x{W_depth} to match video {H}x{W}", flush=True)
            # depth_frames is [T, H, W], need to resize each frame
            depth_np = depth_frames.cpu().numpy()
            depth_matched_list = []
            for d in depth_np:
                # resize_depth_bilinear expects (width, height) order
                d_resized = resize_depth_bilinear(d, (W, H))
                depth_matched_list.append(d_resized)
            depth_frames = torch.from_numpy(np.stack(depth_matched_list)).float()
            _log_debug(f"[TAPIP3DNode] Depth resized to: {depth_frames.shape}", flush=True)

        # Slicing Logic with Debug Prints
        end_index = start_index + T_video
        _log_debug(f"SHAPE CHECK: Start={start_index}, End={end_index}, DepthLen={T_depth_full}, VideoLen={T_video}", flush=True)

        if T_depth_full >= end_index:
            _log_debug(f"[TAPIP3DNode] Slicing Full Depth ({T_depth_full}) -> {start_index}:{end_index}", flush=True)
            depth_frames = depth_frames[start_index:end_index]
        else:
            if abs(T_depth_full - T_video) <= 1:
                 min_len = min(T_depth_full, T_video)
                 depth_frames = depth_frames[:min_len]
                 video_frames = video_frames[:min_len]
                 T_video = min_len
                 _log_debug(f"[TAPIP3DNode] Adjusted lengths to {T_video}", flush=True)
            else:
                raise ValueError(f"[TAPIP3DNode] Depth too short: needed {start_index}-{end_index}, got {T_depth_full}")

        # 3. Load Model
        checkpoint_path = checkpoint
        base_path = str(TAPIP3D_ROOT)
        if not os.path.isabs(checkpoint_path):
             checkpoint_path = os.path.join(base_path, checkpoint)

        if not os.path.exists(checkpoint_path) and os.path.basename(checkpoint_path) == "tapip3d_final.pth":
             os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
             torch.hub.download_url_to_file("https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth", checkpoint_path)

        cache_key = checkpoint_path
        with TAPIP3DNode._model_lock:
            if cache_key not in TAPIP3DNode._model_cache:
                with _quiet_external_logs():
                    TAPIP3DNode._model_cache[cache_key] = load_model(checkpoint_path)
                TAPIP3DNode._model_cache[cache_key].to(device).eval()
                for param in TAPIP3DNode._model_cache[cache_key].parameters():
                    param.requires_grad = False
                # CRITICAL: Store the original base image size before any set_image_size calls
                # This prevents the issue where model.image_size gets mutated and causes
                # inference_res to grow on each subsequent run
                TAPIP3DNode._model_base_image_size[cache_key] = tuple(TAPIP3DNode._model_cache[cache_key].image_size)
                _log_debug(f"[TAPIP3DNode] Model loaded, base image_size: {TAPIP3DNode._model_base_image_size[cache_key]}", flush=True)
            model = TAPIP3DNode._model_cache[cache_key]
            base_image_size = TAPIP3DNode._model_base_image_size[cache_key]
        model.to(device).eval()

        # 4. Resize & Preprocess
        # CRITICAL: Use the ORIGINAL base image size, not model.image_size which may have been mutated
        inference_res = (int(base_image_size[0] * np.sqrt(resolution_factor)), int(base_image_size[1] * np.sqrt(resolution_factor)))
        model.set_image_size(inference_res)
        _log_debug(f"[TAPIP3DNode] inference_res: {inference_res} (from base {base_image_size} * sqrt({resolution_factor}))", flush=True)

        video_tensor = video_frames.permute(0, 3, 1, 2).to(device)
        video_resized = torch.nn.functional.interpolate(video_tensor, size=inference_res, mode="bilinear", align_corners=False)

        depth_np = depth_frames.cpu().numpy()
        depth_resized_list = [resize_depth_bilinear(d, (inference_res[1], inference_res[0])) for d in depth_np]
        depth_tensor = torch.from_numpy(np.stack(depth_resized_list)).float().to(device)

        scale_x = inference_res[1] / W
        scale_y = inference_res[0] / H
        K = torch.tensor([[fx * scale_x, 0.0, cx * scale_x], [0.0, fy * scale_y, cy * scale_y], [0.0, 0.0, 1.0]], dtype=torch.float32)
        intrinsics_t = K.unsqueeze(0).unsqueeze(0).repeat(1, T_video, 1, 1).to(device)
        extrinsics_t = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(1, T_video, 1, 1).to(device)

        if mask is not None:
            # Find the first frame with non-empty mask (object might not appear in frame 0)
            mask_frame_idx = 0
            for i in range(mask.shape[0]):
                if mask[i].sum() > 0:
                    mask_frame_idx = i
                    break
            selected_mask_pixels = int(mask[mask_frame_idx].sum().item())
            if selected_mask_pixels == 0:
                _log_warning("[TAPIP3DNode] Warning: SAM3 mask is empty; using center fallback query", flush=True)
            _log_debug(f"[TAPIP3DNode] Using mask from frame {mask_frame_idx} (sum={selected_mask_pixels})", flush=True)

            mask_0 = mask[mask_frame_idx].cpu().numpy().astype(np.float32)
            mask_resized = cv2.resize(mask_0, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask_resized).to(device)

            _log_debug(f"[TAPIP3DNode] Mask after resize: shape={mask_tensor.shape}, min={mask_tensor.min():.4f}, max={mask_tensor.max():.4f}, sum={mask_tensor.sum():.0f}", flush=True)

            num_points = max(query_grid_size * query_grid_size, 1024)
            # Use depth from the same frame as the mask
            depth_frame_for_query = min(mask_frame_idx, depth_tensor.shape[0] - 1)
            query_point = self.get_mask_queries(
                mask_tensor, depth_tensor[depth_frame_for_query],
                intrinsics_t[0, 0], extrinsics_t[0, 0], num_points,
                mask_erosion=mask_erosion, depth_grad_thresh=depth_grad_thresh,
                depth_laplacian_thresh=depth_laplacian_thresh, multiscale_grad=multiscale_grad,
                query_frame=mask_frame_idx
            ).squeeze(0)
        else:
            query_point = get_grid_queries(grid_size=query_grid_size, depths=depth_tensor, intrinsics=intrinsics_t.squeeze(0), extrinsics=extrinsics_t.squeeze(0))

        mask_sum_value = int(mask[:T_video].sum().item()) if mask is not None else 0
        _log_debug(f"[TAPIP3DNode] query_point shape: {query_point.shape} and mask_sum={mask_sum_value}", flush=True)

        # 8. Inference
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16), _quiet_external_logs():
                coords, visibs = inference(
                    model=model,
                    video=video_resized,
                    depths=depth_tensor,
                    intrinsics=intrinsics_t.squeeze(0),
                    extrinsics=extrinsics_t.squeeze(0),
                    query_point=query_point,
                    num_iters=6,
                    grid_size=0,
                    vis_threshold=vis_threshold
                )
            end_event.record()
            torch.cuda.synchronize()
            inference_ms = start_event.elapsed_time(end_event)
        else:
            inference_start = time.perf_counter()
            with torch.no_grad(), _quiet_external_logs():
                coords, visibs = inference(
                    model=model,
                    video=video_resized,
                    depths=depth_tensor,
                    intrinsics=intrinsics_t.squeeze(0),
                    extrinsics=extrinsics_t.squeeze(0),
                    query_point=query_point,
                    num_iters=6,
                    grid_size=0,
                    vis_threshold=vis_threshold
                )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
        _log_debug(f"[TAPIP3DNode] Inference: {inference_ms:.2f}ms ({inference_ms/1000:.2f}s)", flush=True)

        # 9. Save Results - Output arrays match INPUT RGB video length (T_video)
        # Use RAM disk (/dev/shm) for fastest I/O - files are cleaned up by server after reading
        output_dir = "/dev/shm/comfyui_tapip3d"
        os.makedirs(output_dir, exist_ok=True)
        coords_np = coords.detach().cpu().numpy().astype(np.float32)
        visibs_np = visibs.detach().cpu().numpy().astype(np.float32)

        # TAPIP3D squeezes the track axis for a single query; keep server arrays stable.
        if coords_np.ndim == 2 and coords_np.shape[-1] == 3:
            coords_np = coords_np[:, None, :]
        elif coords_np.ndim == 1 and coords_np.size == 3:
            coords_np = coords_np.reshape(1, 1, 3)

        if visibs_np.ndim == 1:
            visibs_np = visibs_np[:, None]
        elif visibs_np.ndim == 0:
            visibs_np = visibs_np.reshape(1, 1)

        if coords_np.ndim == 3:
            if visibs_np.ndim != 2 or visibs_np.shape[:2] != coords_np.shape[:2]:
                visibs_np = np.ones(coords_np.shape[:2], dtype=np.float32)
            min_t = min(coords_np.shape[0], visibs_np.shape[0], T_video)
            coords_np = coords_np[:min_t]
            visibs_np = visibs_np[:min_t, :coords_np.shape[1]]

        visibs_bool = (visibs_np > vis_threshold).astype(np.bool_) if visibs_np.ndim == 2 else np.zeros((0, 0), dtype=np.bool_)
        coords_out = torch.from_numpy(coords_np).to(device) if coords_np.size else coords
        visibs_out = torch.from_numpy(visibs_np).to(device) if visibs_np.size else visibs
        query_points_np = query_point.detach().cpu().numpy()

        _log_debug(f"[TAPIP3DNode] coords_3d shape: {coords_np.shape} (matches input RGB video length)", flush=True)

        np.save(os.path.join(output_dir, f"{save_prefix}_coords_3d.npy"), coords_np)
        np.save(os.path.join(output_dir, f"{save_prefix}_visibilities.npy"), visibs_bool)

        if mask is not None and (return_debug_media or return_auxiliary_arrays):
            try:
                mask_orig_np = mask.detach().cpu().numpy().astype(np.float32)[:T_video]
                mask_orig_np = np.clip(mask_orig_np, 0.0, 1.0)
                mask_scaled_np = np.stack([
                    cv2.resize(frame, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_NEAREST)
                    for frame in mask_orig_np
                ]).astype(np.float32)
                np.save(os.path.join(output_dir, f"{save_prefix}_mask_orig.npy"), mask_orig_np)
                np.save(os.path.join(output_dir, f"{save_prefix}_mask_scaled.npy"), mask_scaled_np)
                _log_debug(f"[TAPIP3DNode] Saved masks: orig={mask_orig_np.shape}, scaled={mask_scaled_np.shape}", flush=True)
            except Exception as mask_error:
                _log_warning(f"[TAPIP3DNode] Warning: failed to save masks: {mask_error}", flush=True)

        if return_auxiliary_arrays:
            np.save(os.path.join(output_dir, f"{save_prefix}_query_points.npy"), query_points_np)
            video_scaled_np = video_resized.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
            video_scaled_np = np.clip(video_scaled_np, 0.0, 1.0)
            depth_scaled_np = depth_tensor.cpu().numpy().astype(np.float32)
            intrinsics_scaled_np = intrinsics_t.squeeze(0).cpu().numpy().astype(np.float32)
            extrinsics_scaled_np = extrinsics_t.squeeze(0).cpu().numpy().astype(np.float32)
            np.save(os.path.join(output_dir, f"{save_prefix}_video.npy"), video_scaled_np)
            np.save(os.path.join(output_dir, f"{save_prefix}_depths.npy"), depth_scaled_np)
            np.save(os.path.join(output_dir, f"{save_prefix}_intrinsics.npy"), intrinsics_scaled_np)
            np.save(os.path.join(output_dir, f"{save_prefix}_extrinsics.npy"), extrinsics_scaled_np)

        # 10. Visualization
        if coords_np.ndim != 3 or coords_np.shape[1] == 0:
            _log_warning("[TAPIP3DNode] Warning: no tracks found; flow image will contain the first video frame only", flush=True)
            output_video_tensor, flow_tensor, drawn_tracks = _build_visualizations(
                video_frames, np.zeros((0, 0, 3), dtype=np.float32), np.zeros((0, 0), dtype=np.bool_),
                (fx, fy, cx, cy), return_debug_media
            )
        else:
            output_video_tensor, flow_tensor, drawn_tracks = _build_visualizations(
                video_frames, coords_np, visibs_bool, (fx, fy, cx, cy), return_debug_media
            )
            if drawn_tracks == 0:
                _log_warning("[TAPIP3DNode] Warning: tracks were produced but none projected into the image bounds", flush=True)

        if return_debug_media:
            flow_debug_path = os.path.join(output_dir, f"{save_prefix}_flow_debug.png")
            try:
                flow_img_np = np.clip(flow_tensor[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(flow_debug_path, cv2.cvtColor(flow_img_np, cv2.COLOR_RGB2BGR))
            except Exception as flow_debug_error:
                _log_warning(f"[TAPIP3DNode] Warning: failed to save flow debug image: {flow_debug_error}", flush=True)

            _save_cv2_tracking_video(output_dir, save_prefix, output_video_tensor, fps)

        _log_timing(
            "TAPIP3DNode",
            time.perf_counter() - node_start,
            inference=f"{inference_ms / 1000:.2f}s",
            frames=T_video,
            tracks=coords_np.shape[1] if coords_np.ndim == 3 else 0,
            drawn=drawn_tracks,
            mask_pixels=mask_sum_value,
        )

        return (
            VideoFromComponents(VideoComponents(images=output_video_tensor, frame_rate=Fraction(fps))),
            coords_out,
            visibs_out,
            flow_tensor
        )

NODE_CLASS_MAPPINGS = {"TAPIP3DNode": TAPIP3DNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TAPIP3DNode": "TAPIP3D Tracker"}
