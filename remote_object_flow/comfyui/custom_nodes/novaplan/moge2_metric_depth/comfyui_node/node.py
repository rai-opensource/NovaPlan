# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Estimate and calibrate temporally consistent metric depth with MoGe2."""

import torch
import numpy as np
import os
import inspect
from fractions import Fraction
from PIL import Image
import torch.nn.functional as F
import folder_paths
import cv2

from comfy_api.input_impl import VideoFromComponents
from comfy_api.util import VideoComponents

MoGeModel = None
try:
    from moge.model.v2 import MoGeModel
except ImportError:
    try:
        from moge.model import MoGeModel
        print("MoGe2CVDMetricDepthNode: Using compatible MoGe model API")
    except ImportError as e2:
        print(f"MoGe2CVDMetricDepthNode: MoGe is unavailable: {e2}")

# -----------------------------
# Helpers
# -----------------------------

def normalize_depth_range(depth: np.ndarray, lower_percentile: float, upper_percentile: float) -> tuple[float, float]:
    """Normalize depth range."""
    valid_mask = np.isfinite(depth) & (depth > 0)
    if valid_mask.sum() == 0: return float(np.nanmin(depth)), float(np.nanmax(depth))
    valid_depth = depth[valid_mask]
    if 0.0 < lower_percentile < 100.0 and 0.0 < upper_percentile <= 100.0 and lower_percentile < upper_percentile:
        depth_min = float(np.percentile(valid_depth, lower_percentile))
        depth_max = float(np.percentile(valid_depth, upper_percentile))
    else:
        depth_min = float(valid_depth.min())
        depth_max = float(valid_depth.max())
    if depth_max - depth_min < 1e-6:
        depth_max = depth_min + 1.0
    return depth_min, depth_max


def _build_moge_infer_kwargs(infer_fn, intrinsics: np.ndarray | None, height: int, width: int, device) -> dict:
    """Build only the camera arguments supported by the installed MoGe API.

    MoGe2's released API accepts a known horizontal field of view, while some
    downstream variants accept normalized intrinsics directly.  Inspecting the
    bound method keeps this node compatible with both without swallowing an
    inference error and retrying with a different camera model.
    """
    try:
        parameters = inspect.signature(infer_fn).parameters
    except (TypeError, ValueError):
        parameters = {}

    kwargs = {}
    if "apply_mask" in parameters:
        # Dense finite depth is required by CVD and the downstream 3D tracker.
        kwargs["apply_mask"] = False

    if intrinsics is None:
        return kwargs

    K = np.asarray(intrinsics, dtype=np.float32)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("MoGe intrinsics must be a finite 3x3 matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("MoGe focal lengths must be positive")

    if "intrinsics" in parameters:
        # MoGe represents intrinsics in image-normalized coordinates.
        K_normalized = K.copy()
        K_normalized[0, :] /= float(width)
        K_normalized[1, :] /= float(height)
        kwargs["intrinsics"] = torch.as_tensor(K_normalized, dtype=torch.float32, device=device)
    elif "fov_x" in parameters:
        # The upstream MoGe2 API exposes known intrinsics through horizontal FoV.
        fov_x = np.degrees(2.0 * np.arctan(float(width) / (2.0 * float(K[0, 0]))))
        kwargs["fov_x"] = float(fov_x)

    return kwargs

def erode_mask(mask: np.ndarray, erosion_px: int) -> np.ndarray:
    """Erode mask."""
    if erosion_px <= 0: return mask
    k = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k, iterations=1).astype(bool)

def planar_mask(depth: np.ndarray, base_mask: np.ndarray, grad_thresh: float = 0.02) -> np.ndarray:
    """Compute a locally planar depth mask."""
    ys, xs = np.nonzero(base_mask)
    n_points = ys.size
    if n_points < 50: return base_mask
    zs = depth[ys, xs].astype(np.float64)
    w = 1.0 / np.clip(zs, 1e-6, None)
    cx, cy = np.mean(xs), np.mean(ys)
    u_centered = xs - cx
    v_centered = ys - cy
    pts = np.column_stack((u_centered, v_centered, np.ones_like(u_centered)))
    best_inliers = 0
    best_model = None
    candidate_idxs = np.arange(n_points)
    for _ in range(100):
        sample_idx = np.random.choice(candidate_idxs, size=3, replace=False)
        try: model = np.linalg.solve(pts[sample_idx], w[sample_idx])
        except np.linalg.LinAlgError: continue
        check_size = min(n_points, 500)
        check_idx = np.random.choice(candidate_idxs, size=check_size, replace=False)
        w_pred = pts[check_idx] @ model
        is_inlier = np.abs(w_pred - w[check_idx]) < (grad_thresh * w[check_idx])
        count = np.sum(is_inlier)
        estimated_count = count * (n_points / check_size)
        if estimated_count > best_inliers:
            best_inliers = estimated_count
            best_model = model
    if best_model is None: return base_mask
    all_w_pred = pts @ best_model
    all_z_pred = 1.0 / np.clip(all_w_pred, 1e-9, None)
    relative_error = np.abs(all_z_pred - zs) / np.clip(zs, 1e-6, None)
    inlier_mask_local = relative_error < grad_thresh
    final_mask = base_mask.copy()
    final_mask[:] = False
    final_mask[ys[inlier_mask_local], xs[inlier_mask_local]] = True
    return final_mask

def ratio_consistency_mask(pred_depth: np.ndarray, gt_depth: np.ndarray, base_mask: np.ndarray, ratio_thresh: float = 0.15, ksize: int = 5) -> np.ndarray:
    """Compute a depth-ratio consistency mask."""
    if base_mask.sum() < 10: return base_mask
    pred = pred_depth.astype(np.float32)
    gt = gt_depth.astype(np.float32)
    ratio = gt / np.clip(pred, 1e-6, None)
    k = max(3, int(ksize) | 1)
    ratio_med = cv2.medianBlur(ratio, k)
    rel = np.abs(ratio - ratio_med) / np.clip(np.abs(ratio_med), 1e-6, None)
    return base_mask & (rel < ratio_thresh)

def filter_pepper_noise_fast(depth: np.ndarray, valid_mask: np.ndarray, neighbor_threshold: float = 0.20, min_region_size: int = 50) -> np.ndarray:
    """Filter pepper noise fast."""
    depth_f32 = depth.astype(np.float32)
    depth_median = cv2.medianBlur(depth_f32, 5)
    rel_diff = np.abs(depth_f32 - depth_median) / np.clip(np.abs(depth_median), 1e-6, None)
    consistent = rel_diff < neighbor_threshold
    mask_to_filter = (valid_mask & consistent).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_to_filter = cv2.morphologyEx(mask_to_filter, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_to_filter, connectivity=8)
    output_mask = np.zeros_like(valid_mask, dtype=bool)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_region_size:
            output_mask[labels == label] = True
    return output_mask

def stratified_subsample_by_depth(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, valid_min: float, valid_max: float, num_bins: int = 10, max_total: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Subsample calibration pixels across depth strata."""
    idx = np.flatnonzero(mask)
    if idx.size == 0: return pred[mask], gt[mask]
    if idx.size <= max_total: return pred[mask], gt[mask]
    gt_vals = gt.flat[idx]
    denom = max(1e-6, (valid_max - valid_min))
    b = np.floor((gt_vals - valid_min) / denom * num_bins).astype(np.int32)
    b = np.clip(b, 0, num_bins - 1)
    per_bin = max(1, max_total // num_bins)
    keep = []
    rng = np.random.default_rng()
    for bi in range(num_bins):
        bi_idx = idx[b == bi]
        if bi_idx.size == 0: continue
        take = min(per_bin, bi_idx.size)
        keep.append(rng.choice(bi_idx, size=take, replace=False))
    if not keep: keep_idx = rng.choice(idx, size=max_total, replace=False)
    else:
        keep_idx = np.concatenate(keep)
        if keep_idx.size > max_total: keep_idx = rng.choice(keep_idx, size=max_total, replace=False)
    return pred.flat[keep_idx], gt.flat[keep_idx]

def estimate_scale_median_irls(pred_valid: np.ndarray, gt_valid: np.ndarray, iters: int = 3) -> float:
    """Estimate depth scale with median initialization and IRLS."""
    pred_valid = pred_valid.astype(np.float64)
    gt_valid = gt_valid.astype(np.float64)
    ratios = gt_valid / np.clip(pred_valid, 1e-9, None)
    s = float(np.median(ratios[np.isfinite(ratios)])) if ratios.size else 1.0
    if not np.isfinite(s): s = 1.0
    for _ in range(max(0, int(iters))):
        resid = (s * pred_valid - gt_valid) / np.clip(gt_valid, 1e-9, None)
        w = 1.0 / np.clip(np.abs(resid), 1e-3, None)
        num = np.sum(w * pred_valid * gt_valid)
        den = np.sum(w * pred_valid * pred_valid)
        if den <= 1e-12 or not np.isfinite(num) or not np.isfinite(den): break
        s_new = float(num / den)
        if not np.isfinite(s_new): break
        s = s_new
    return float(s)

def build_calibration_mask(pred_depth: np.ndarray, gt_depth: np.ndarray, valid_min: float, valid_max: float, filter_pepper: bool = True, use_planar: bool = True, planar_grad_thresh: float = 0.02, use_ratio_consistency: bool = False, ratio_thresh: float = 0.15, erosion_px: int = 2, min_region_size: int = 500) -> np.ndarray:
    """Build calibration mask."""
    base = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > valid_min) & (gt_depth < valid_max)
    if filter_pepper:
        pepper_filtered = filter_pepper_noise_fast(gt_depth, base, min_region_size=min_region_size)
    else:
        pepper_filtered = base.copy()
    
    # Start from finite, range-valid pixels after optional speckle filtering.
    m = pepper_filtered
    
    if use_planar: m = m & planar_mask(gt_depth, m, grad_thresh=planar_grad_thresh)
    if use_ratio_consistency: m = ratio_consistency_mask(pred_depth, gt_depth, m, ratio_thresh=ratio_thresh, ksize=5)
    m = erode_mask(m, erosion_px)
    return m

def estimate_depth_scale_robust(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_min: float = 0.9,
    valid_max: float = 6.0,
    filter_pepper: bool = True,
    use_planar: bool = True,
    planar_grad_thresh: float = 0.02,
    use_ratio_consistency: bool = True,
    ratio_thresh: float = 0.15,
    erosion_px: int = 4,
    min_pixels: int = 500,
    irls_iters: int = 3,
    subsample_bins: int = 10,
    max_points: int = 200_000,
    scale_bounds: tuple[float, float] = (0.05, 20.0),
    min_region_size: int = 50,
) -> tuple[float, bool]:
    """
    Robust scale initializer used by the affine RANSAC calibration:
      - robust masking (pepper + planar + ratio-consistency + erosion)
      - stratified subsampling
      - median-of-ratios + IRLS refinement

    This helper is not an exported scale-only depth mode. The public node
    always estimates and applies the full affine calibration.
    """
    # Build mask (strict)
    m_strict = build_calibration_mask(
        pred_depth, gt_depth,
        valid_min, valid_max,
        filter_pepper=filter_pepper,
        use_planar=use_planar,
        planar_grad_thresh=planar_grad_thresh,
        use_ratio_consistency=use_ratio_consistency,
        ratio_thresh=ratio_thresh,
        erosion_px=erosion_px,
        min_region_size=min_region_size,
    )
    # Relaxation ladder if too few pixels
    m = m_strict
    if m.sum() < min_pixels:
        # 1) drop ratio
        m = build_calibration_mask(
            pred_depth, gt_depth, valid_min, valid_max,
            filter_pepper=filter_pepper,
            use_planar=use_planar,
            planar_grad_thresh=planar_grad_thresh,
            use_ratio_consistency=False,
            ratio_thresh=ratio_thresh,
            erosion_px=erosion_px,
            min_region_size=min_region_size,
        )
    if m.sum() < min_pixels and erosion_px > 0:
        # 2) reduce erosion
        m = build_calibration_mask(
            pred_depth, gt_depth, valid_min, valid_max,
            filter_pepper=filter_pepper,
            use_planar=use_planar,
            planar_grad_thresh=planar_grad_thresh,
            use_ratio_consistency=False,
            ratio_thresh=ratio_thresh,
            erosion_px=max(0, erosion_px // 2),
            min_region_size=min_region_size,
        )
    if m.sum() < min_pixels and use_planar:
        # 3) drop planar too
        m = build_calibration_mask(
            pred_depth, gt_depth, valid_min, valid_max,
            filter_pepper=filter_pepper,
            use_planar=False,
            planar_grad_thresh=planar_grad_thresh,
            use_ratio_consistency=False,
            ratio_thresh=ratio_thresh,
            erosion_px=max(0, erosion_px // 2),
            min_region_size=min_region_size,
        )

    n = int(m.sum())
    if n < 10:
        print(f"[MoGe2Depth] WARNING: Too few calibration pixels ({n}). Using scale=1.0")
        return 1.0, False

    pred_s, gt_s = stratified_subsample_by_depth(
        pred_depth.astype(np.float32),
        gt_depth.astype(np.float32),
        m,
        valid_min, valid_max,
        num_bins=subsample_bins,
        max_total=max_points,
    )

    s = estimate_scale_median_irls(pred_s, gt_s, iters=irls_iters)
    if not np.isfinite(s):
        s = 1.0

    # clamp to sane range
    s = float(np.clip(s, scale_bounds[0], scale_bounds[1]))

    success = (n >= min_pixels)
    print(f"[MoGe2Depth] Scale (robust): scale={s:.4f}, pixels={n}, success={success}")
    return s, success

def _fit_affine_ransac(pred_valid: np.ndarray, gt_valid: np.ndarray, inlier_threshold: float = 0.10, num_iterations: int = 4000, scale_bounds: tuple[float, float] = (0.1, 100.0), shift_bounds: tuple[float, float] = (-10.0, 10.0), seed: int = 42) -> tuple[float, float, int]:
    """Fit ``sensor_depth = scale * generated_depth + shift``.

    ``inlier_threshold`` is an absolute metric-depth residual in meters, as in
    Appendix D.1 of the NovaPlan paper.
    """
    rng = np.random.default_rng(seed)
    n_points = pred_valid.size
    if n_points < 2: return 1.0, 0.0, 0
    p_min, p_max = np.min(pred_valid), np.max(pred_valid)
    min_separation = max(1e-5, (p_max - p_min) * 0.05)
    best_s, best_t, best_inliers, best_inlier_mask = 1.0, 0.0, 0, None
    num_iterations = max(1, int(num_iterations))
    idxs = rng.choice(n_points, size=(num_iterations, 2), replace=True)
    for i in range(num_iterations):
        idx1, idx2 = idxs[i]
        p1, p2 = pred_valid[idx1], pred_valid[idx2]
        denom = p1 - p2
        if abs(denom) < min_separation: continue
        g1, g2 = gt_valid[idx1], gt_valid[idx2]
        s = (g1 - g2) / denom
        t = g1 - s * p1
        if s <= 0 or s < scale_bounds[0] or s > scale_bounds[1]: continue
        if t < shift_bounds[0] or t > shift_bounds[1]: continue
        pred_calibrated = s * pred_valid + t
        diff = np.abs(pred_calibrated - gt_valid)
        inlier_mask = diff < inlier_threshold
        n_inliers = int(np.sum(inlier_mask))
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_s = s
            best_t = t
            best_inlier_mask = inlier_mask
    if best_inliers > 10 and best_inlier_mask is not None:
        try:
            p_in = pred_valid[best_inlier_mask]
            g_in = gt_valid[best_inlier_mask]
            A = np.vstack([p_in, np.ones(len(p_in))]).T
            s_ref, t_ref = np.linalg.lstsq(A, g_in, rcond=None)[0]
            if (
                np.isfinite(s_ref)
                and np.isfinite(t_ref)
                and s_ref > 0
                and scale_bounds[0] <= s_ref <= scale_bounds[1]
                and shift_bounds[0] <= t_ref <= shift_bounds[1]
            ):
                best_s = float(s_ref)
                best_t = float(t_ref)
        except np.linalg.LinAlgError:
            pass
    return float(best_s), float(best_t), best_inliers

def estimate_depth_affine_robust(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_min: float = 0.4,
    valid_max: float = 6.0,
    method: str = "irls",
    num_irls_iters: int = 10,
    num_ransac_iters: int = 1000,
    inlier_threshold: float = 0.15,
    filter_pepper: bool = True,
    use_planar: bool = False,
    planar_grad_thresh: float = 0.02,
    use_ratio_consistency: bool = False,
    ratio_thresh: float = 0.15,
    erosion_px: int = 2,
    min_region_size: int = 50,
    min_pixels: int = 500,
    scale_bounds: tuple[float, float] = (0.1, 10.0),
    shift_bounds: tuple[float, float] = (-3.0, 3.0),
    max_points: int = 100_000,
    seed: int = 42,
) -> tuple[float, float, bool]:
    """Estimate a robust affine depth calibration."""
    method = method.lower().strip()
    if method not in {"irls", "ransac"}:
        raise ValueError(f"Unsupported affine method: {method!r}")
    if valid_max <= valid_min:
        raise ValueError("valid_max must be greater than valid_min")
    if min_pixels < 2:
        raise ValueError("min_pixels must be at least 2")
    if inlier_threshold <= 0.0:
        raise ValueError("inlier_threshold must be positive")

    m = build_calibration_mask(
        pred_depth,
        gt_depth,
        valid_min,
        valid_max,
        filter_pepper=filter_pepper,
        use_planar=use_planar,
        planar_grad_thresh=planar_grad_thresh,
        use_ratio_consistency=use_ratio_consistency,
        ratio_thresh=ratio_thresh,
        erosion_px=erosion_px,
        min_region_size=min_region_size,
    )
    n = int(m.sum())
    if n < min_pixels:
        print(f"[MoGe2Depth] WARNING: Too few affine calibration pixels ({n} < {min_pixels})")
        return 1.0, 0.0, False
    idx = np.flatnonzero(m)
    pred_flat = pred_depth.flat[idx].astype(np.float64)
    gt_flat = gt_depth.flat[idx].astype(np.float64)
    rng = np.random.default_rng(seed)
    if idx.size > max_points:
        sampled_indices = rng.choice(len(idx), size=min(max_points, len(idx)), replace=False)
        pred_valid, gt_valid = pred_flat[sampled_indices], gt_flat[sampled_indices]
    else:
        pred_valid, gt_valid = pred_flat, gt_flat
    if method == "ransac":
        best_s, best_t, _ = _fit_affine_ransac(
            pred_valid,
            gt_valid,
            inlier_threshold,
            num_ransac_iters,
            scale_bounds,
            shift_bounds,
            seed,
        )
    else:
        ratios = gt_valid / np.clip(pred_valid, 1e-6, None)
        s_init = float(np.median(ratios))
        residuals = gt_valid - s_init * pred_valid
        t_init = float(np.median(residuals))
        best_s = s_init
        best_t = t_init
        huber_delta = inlier_threshold
        for _ in range(num_irls_iters):
            pred_scaled = pred_valid * best_s + best_t
            residuals = gt_valid - pred_scaled
            abs_residuals = np.abs(residuals)
            weights = np.ones_like(residuals)
            outlier_mask = abs_residuals > huber_delta
            weights[outlier_mask] = huber_delta / (abs_residuals[outlier_mask] + 1e-8)
            residual_weights = np.clip(1.0 - abs_residuals / inlier_threshold, 0.01, 1.0)
            weights = weights * residual_weights
            weights = weights / (weights.sum() + 1e-8) * pred_valid.size
            sqrt_w = np.sqrt(weights)
            try:
                sol, _, _, _ = np.linalg.lstsq(np.column_stack([pred_valid * sqrt_w, sqrt_w]), gt_valid * sqrt_w, rcond=None)
                best_s, best_t = float(np.clip(sol[0], scale_bounds[0], scale_bounds[1])), float(np.clip(sol[1], shift_bounds[0], shift_bounds[1]))
            except np.linalg.LinAlgError: break
    parameters_valid = (
        np.isfinite(best_s)
        and np.isfinite(best_t)
        and best_s > 0
        and scale_bounds[0] <= best_s <= scale_bounds[1]
        and shift_bounds[0] <= best_t <= shift_bounds[1]
    )
    if not parameters_valid:
        best_s, best_t = 1.0, 0.0

    pred_calibrated = pred_valid * best_s + best_t
    final_residuals = np.abs(gt_valid - pred_calibrated)
    inliers = final_residuals < inlier_threshold
    inlier_pct = 100.0 * int(inliers.sum()) / pred_valid.size
    success = parameters_valid and inlier_pct > 30.0
    if not success:
        # Keep the scale estimate for diagnostics, but do not let a failed
        # affine fit masquerade as a successful affine calibration.  The
        # caller fails closed and therefore never publishes this fallback.
        s0, _ = estimate_depth_scale_robust(
            pred_depth,
            gt_depth,
            valid_min,
            valid_max,
            filter_pepper,
            use_planar,
            planar_grad_thresh,
            use_ratio_consistency,
            ratio_thresh,
            erosion_px,
            min_pixels,
            3,
            10,
            max_points,
            scale_bounds,
            min_region_size,
        )
        return s0, 0.0, False
    return float(best_s), float(best_t), True

def estimate_depth_affine_ransac(
    pred_depth,
    gt_depth,
    valid_min,
    valid_max,
    method,
    num_irls_iters,
    num_ransac_iters,
    inlier_threshold,
    filter_pepper,
    use_planar,
    planar_grad_thresh,
    use_ratio_consistency,
    ratio_thresh,
    erosion_px,
    min_region_size,
    min_pixels,
):
    """Estimate a robust affine depth calibration with RANSAC."""
    return estimate_depth_affine_robust(
        pred_depth,
        gt_depth,
        valid_min=valid_min,
        valid_max=valid_max,
        method=method,
        num_irls_iters=num_irls_iters,
        num_ransac_iters=num_ransac_iters,
        inlier_threshold=inlier_threshold,
        filter_pepper=filter_pepper,
        use_planar=use_planar,
        planar_grad_thresh=planar_grad_thresh,
        use_ratio_consistency=use_ratio_consistency,
        ratio_thresh=ratio_thresh,
        erosion_px=erosion_px,
        min_region_size=min_region_size,
        min_pixels=min_pixels,
    )

class MoGe2CVDMetricDepthNode:
    # Cache for MoGe model
    """Expose MoGe2 metric-depth calibration as a ComfyUI node."""
    _moge_cache = {}
    
    @classmethod
    def INPUT_TYPES(s):
        """Declare the ComfyUI input specification."""
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "video": ("VIDEO",),
                "depth_factor": ("FLOAT", {
                    "default": 0.001, "min": 0.00001, "max": 1000.0, "step": 0.00001,
                    "tooltip": "Multiplier for input depth map units (e.g. 0.001 for mm to meters)."
                }),
                "calibration_mode": (["Affine (s * d + t)"], {"default": "Affine (s * d + t)"}),
                "depth_source": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "intrinsics": ("INTRINSICS",),
                # --- CVD Inputs ---
                "enable_cvd": (["disabled", "enabled"], {"default": "enabled", "tooltip": "Enable Consistent Video Depth (CVD) optimization."}),
                "cvd_w_grad": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cvd_w_normal": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cvd_resolution": ("INT", {"default": 196608, "min": 100000, "max": 500000, "step": 1024}),

                # Robust parameters
                "valid_depth_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 100.0, "step": 0.1}),
                "valid_depth_max": ("FLOAT", {"default": 6.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "use_planar_mask": ("BOOLEAN", {"default": False}),
                "planar_grad_thresh": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.2, "step": 0.001}),
                "use_ratio_consistency": ("BOOLEAN", {"default": False}),
                "ratio_thresh": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "mask_erosion_px": ("INT", {"default": 2, "min": 0, "max": 20, "step": 1}),
                "min_calib_pixels": ("INT", {"default": 500, "min": 10, "max": 1000000, "step": 10}),
                "irls_iters": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "affine_method": (["irls", "ransac"], {"default": "ransac"}),
                "ransac_iters": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "filter_pepper": ("BOOLEAN", {"default": True}),
                "min_region_size": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "inlier_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 10.0, "step": 0.01, "tooltip": "Absolute RANSAC residual threshold in meters."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "IMAGE")
    RETURN_NAMES = ("input_depth_visual", "depth_video", "calibrated_depth")
    FUNCTION = "process"
    CATEGORY = "NovaPlan/Metric Depth"
    
    def _load_moge_model(self, device):
        """Load MoGe2 model with caching."""
        cache_key = ("moge2", str(device))
        if cache_key in self._moge_cache:
            return self._moge_cache[cache_key]
        
        if MoGeModel is None:
            raise ImportError(
                "MoGe is unavailable. Run `pixi run -e object-flow-host setup-object-flow-host` "
                "to install the pinned NovaPlan dependency."
            )
        
        print("[MoGe2CVDMetricDepthNode] Loading MoGe2 model...")
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        model.eval()
        self._moge_cache[cache_key] = model
        return model
    
    def _infer_moge2(self, frames_list, intrinsics_np, device, orig_h, orig_w):
        """Run MoGe2 inference and return depth predictions + intrinsics."""
        model = self._load_moge_model(device)
        
        depth_list = []
        estimated_intrinsics = None
        
        for i, frame in enumerate(frames_list):
            image_tensor = torch.tensor(frame / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            
            frame_intrinsics = intrinsics_np[i] if intrinsics_np is not None else None
            infer_kwargs = _build_moge_infer_kwargs(
                model.infer,
                frame_intrinsics,
                orig_h,
                orig_w,
                device,
            )
            with torch.no_grad():
                output = model.infer(image_tensor, **infer_kwargs)
            
            depth = output['depth'].cpu()
            depth_list.append(depth)
            
            # Estimate intrinsics from first frame if not provided
            if intrinsics_np is None and estimated_intrinsics is None:
                moge_K = output['intrinsics'].cpu().numpy()
                K = moge_K.copy()
                K[0, 0] *= orig_w
                K[1, 1] *= orig_h
                K[0, 2] *= orig_w
                K[1, 2] *= orig_h
                estimated_intrinsics = K
        
        depth_pred = torch.stack(depth_list, dim=0)
        
        if intrinsics_np is None and estimated_intrinsics is not None:
            B = len(frames_list)
            intrinsics_np = np.tile(estimated_intrinsics[np.newaxis, :, :], (B, 1, 1)).astype(np.float32)
        
        return depth_pred, intrinsics_np

    def colorize_depth_magma(self, depth_tensor, vmin=None, vmax=None):
        """Colorize a depth map with the Magma colormap."""
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        B, H, W = depth_tensor.shape
        depth_np = depth_tensor.cpu().numpy()

        if vmin is None:
            vmin = float(np.nanmin(depth_np))
        if vmax is None:
            vmax = float(np.nanmax(depth_np))
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        norm_depth = (depth_np - vmin) / (vmax - vmin)
        norm_depth = np.clip(norm_depth, 0, 1)

        try:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("magma")
        except ImportError:
            raise ImportError("matplotlib is required for depth visualization.")

        colored_list = []
        for i in range(B):
            colored = cmap(norm_depth[i])
            colored_list.append(colored[:, :, :3])

        colored_np = np.stack(colored_list, axis=0)
        return torch.from_numpy(colored_np).float()

    def process(
        self,
        video,
        depth_factor,
        calibration_mode="Affine (s * d + t)",
        intrinsics=None,
        depth_source=None,
        enable_cvd="enabled", cvd_w_grad=2.0, cvd_w_normal=5.0, cvd_resolution=196608,
        valid_depth_min=0.6,
        valid_depth_max=6.0,
        use_planar_mask=False,
        planar_grad_thresh=0.02,
        use_ratio_consistency=False,
        ratio_thresh=0.15,
        mask_erosion_px=2,
        min_calib_pixels=500,
        irls_iters=3,
        affine_method="ransac",
        ransac_iters=1000,
        filter_pepper=True,
        min_region_size=50,
        inlier_threshold=0.15,
    ):
        """Execute this ComfyUI node."""
        print(f"\n{'='*60}", flush=True)
        print("[MoGe2CVDMetricDepthNode] Metric-depth node executing", flush=True)
        print(f"[MoGe2CVDMetricDepthNode] CVD: {enable_cvd}", flush=True)
        print(f"{'='*60}\n", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if depth_source is None:
            raise ValueError("depth_source is required: calibrated depth is the only numeric output")
        if int(min_calib_pixels) < 2:
            raise ValueError("min_calib_pixels must be at least 2")
        if int(min_region_size) < 1:
            raise ValueError("min_region_size must be at least 1")
        if float(valid_depth_max) <= float(valid_depth_min):
            raise ValueError("valid_depth_max must be greater than valid_depth_min")
        if calibration_mode != "Affine (s * d + t)":
            raise ValueError("Only robust affine depth calibration is supported")

        input_fps = 24.0
        if hasattr(video, 'get_components'):
            components = video.get_components()
            video_frames = components.images
            input_fps = float(components.frame_rate) if components.frame_rate else 24.0
        else:
            video_frames = video

        B, orig_h, orig_w, C = video_frames.shape
        video_frames_np = (video_frames.cpu().numpy() * 255).astype(np.uint8)
        frames_list = [video_frames_np[i] for i in range(B)]

        # MoGe2 estimates intrinsics only when the caller does not provide them.
        intrinsics_np = None
        if intrinsics is not None:
            K_in = intrinsics.cpu().numpy()
            intrinsics_np = np.tile(K_in[np.newaxis, :, :], (B, 1, 1))

        print("[MoGe2CVDMetricDepthNode] Running per-frame MoGe2 metric depth")
        depth_pred_raw, intrinsics_np = self._infer_moge2(
            frames_list, intrinsics_np, device, orig_h, orig_w
        )
        depth_pred = depth_pred_raw.unsqueeze(1)  # [N, 1, H, W]

        # Resize to original
        depth_resized = F.interpolate(depth_pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        depth_for_vis = depth_resized.squeeze(1)  # [N, H, W]
        
        # ====== CVD Optimization ======
        if enable_cvd == "enabled":
            print("[MoGe2Depth] CVD optimization enabled...")
            try:
                from .cvd_optimizer import CVDOptimizer
                
                if intrinsics is not None:
                     K_tensor = intrinsics.to(device)
                else:
                     # Use frame 0 default
                     K_tensor = torch.from_numpy(intrinsics_np[0]).float().to(device)
                
                cvd = CVDOptimizer(device=device, resolution=cvd_resolution)
                optimized_depth = cvd.optimize(
                    depth_prior=depth_for_vis.to(device),
                    images=video_frames_np,
                    intrinsics=K_tensor,
                    w_grad=cvd_w_grad,
                    w_normal=cvd_w_normal,
                    freeze_shift=True,
                )
                
                depth_for_vis = optimized_depth.cpu()
                depth_resized = optimized_depth.unsqueeze(1).cpu() # [N, 1, H, W]
                print("[MoGe2Depth] CVD optimization complete!")
            except FileNotFoundError as e:
                print(f"[MoGe2Depth] CVD error: {e}")
                raise e
            except Exception as e:
                print(f"[MoGe2Depth] CVD optimization failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"CVD optimization failed: {e}")
        
        # Calibration
        input_depth_visual = None
        vmin, vmax = None, None

        if depth_source is not None:
            depth_path = folder_paths.get_annotated_filepath(depth_source)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth calibration source does not exist: {depth_path}")
            if os.path.exists(depth_path):
                pil_depth_raw = Image.open(depth_path)
                depth_np = np.array(pil_depth_raw, dtype=np.float32)
                
                if depth_np.ndim == 3:
                    if depth_np.shape[2] == 4:
                        depth_np = depth_np[:, :, 0]
                    elif depth_np.shape[2] == 3:
                        depth_np = 0.299 * depth_np[:, :, 0] + 0.587 * depth_np[:, :, 1] + 0.114 * depth_np[:, :, 2]
                    else:
                        depth_np = depth_np[:, :, 0]
                
                depth_np = depth_np * float(depth_factor)

                pil_depth = Image.fromarray(depth_np, mode="F")
                pil_depth = pil_depth.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
                depth_ref = np.array(pil_depth, dtype=np.float32)

                vmin, vmax = normalize_depth_range(depth_ref, 2.0, 98.0)
                depth_ref_tensor = torch.from_numpy(depth_ref).unsqueeze(0)
                input_depth_visual = self.colorize_depth_magma(depth_ref_tensor, vmin, vmax)

                pred_depth_0 = depth_for_vis[0].cpu().numpy().astype(np.float32)

                scale, shift, success = estimate_depth_affine_ransac(
                    pred_depth_0, depth_ref,
                    valid_min=float(valid_depth_min), valid_max=float(valid_depth_max),
                    method=str(affine_method),
                    num_irls_iters=int(irls_iters),
                    num_ransac_iters=int(ransac_iters),
                    inlier_threshold=float(inlier_threshold),
                    filter_pepper=bool(filter_pepper),
                    use_planar=bool(use_planar_mask),
                    planar_grad_thresh=float(planar_grad_thresh),
                    use_ratio_consistency=bool(use_ratio_consistency),
                    ratio_thresh=float(ratio_thresh),
                    erosion_px=int(mask_erosion_px),
                    min_region_size=int(min_region_size),
                    min_pixels=int(min_calib_pixels),
                )

                if not success:
                    raise RuntimeError(
                        "Depth calibration failed: not enough valid first-frame correspondences "
                        "or no robust scale/affine model was found"
                    )

                depth_for_vis = depth_for_vis * scale + shift
                depth_resized = depth_resized * scale + shift
                depth_for_vis = torch.nan_to_num(
                    depth_for_vis, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp_min(0.0)
                depth_resized = torch.nan_to_num(
                    depth_resized, nan=0.0, posinf=0.0, neginf=0.0
                ).clamp_min(0.0)
                print(
                    f"[MoGe2Depth] First-frame calibration applied clip-wide: "
                    f"scale={scale:.6f}, shift={shift:.6f}",
                    flush=True,
                )

        if input_depth_visual is None:
            raise RuntimeError("Depth calibration prior was not processed")

        depth_visual = self.colorize_depth_magma(depth_for_vis, vmin, vmax)
        calibrated_depth_out = depth_resized.permute(0, 2, 3, 1).cpu()

        return (
            input_depth_visual,
            VideoFromComponents(VideoComponents(images=depth_visual, frame_rate=Fraction(input_fps))),
            calibrated_depth_out,
        )

NODE_CLASS_MAPPINGS = {
    "MoGe2CVDMetricDepthNode": MoGe2CVDMetricDepthNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MoGe2CVDMetricDepthNode": "MoGe2 + CVD Metric Depth",
}
