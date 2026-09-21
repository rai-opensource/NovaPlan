#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Rigid object motion recovered from tracked 3D points.

The reference object-flow path retains points that are valid in both adjacent
frames, estimates a robust Kabsch transform with RANSAC, and composes the
adjacent SE(3) transforms. No temporal smoothing is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_RANSAC_ITERATIONS = 1000
DEFAULT_RANSAC_INLIER_THRESHOLD_M = 0.02
DEFAULT_RANSAC_SEED = 0
FLOW_ARRAY_LAYOUTS = frozenset({"time_major", "point_major"})


@dataclass
class RigidObjectFlowFit:
    """Adjacent and accumulated rigid transforms for one object track set."""

    transforms: np.ndarray
    relative_transforms: np.ndarray
    rotation_degrees: List[Optional[float]] = field(default_factory=list)
    valid_point_counts: List[int] = field(default_factory=list)
    inlier_ratios: List[Optional[float]] = field(default_factory=list)
    rmse_by_step: List[Optional[float]] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Return whether the fitted object transform is valid."""
        return not self.failed_steps


def normalize_flow_arrays(
    coords: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    *,
    layout: str = "time_major",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Normalize TAPIP3D arrays to ``[T, N, 3]`` and ``[T, N]``.

    Layout cannot be inferred safely from array sizes: a valid time-major track
    may contain fewer points than frames. First-party TAPIP3D arrays are
    time-major; callers handling point-major arrays must say so explicitly.
    """

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [T,N,3] or [N,T,3], got {coords.shape}")

    layout = str(layout).strip().lower()
    if layout not in FLOW_ARRAY_LAYOUTS:
        expected = ", ".join(sorted(FLOW_ARRAY_LAYOUTS))
        raise ValueError(f"layout must be one of: {expected}; got {layout!r}")

    input_shape = coords.shape[:2]
    if layout == "point_major":
        coords = np.swapaxes(coords, 0, 1)

    vis = None if visibilities is None else np.asarray(visibilities)
    if vis is not None:
        if vis.ndim != 2:
            raise ValueError(f"visibilities must have shape [T,N] or [N,T], got {vis.shape}")
        if vis.shape == input_shape:
            if layout == "point_major":
                vis = vis.T
        elif vis.T.shape == input_shape:
            if layout == "time_major":
                vis = vis.T
        else:
            raise ValueError(f"visibilities shape {vis.shape} does not match coords shape {input_shape}")

    return coords, vis


def normalize_point_origins(
    point_origins: Optional[np.ndarray],
    num_points: int,
) -> Optional[np.ndarray]:
    """Normalize point origins."""
    if point_origins is None:
        return None
    origins = np.asarray(point_origins, dtype=np.int64).reshape(-1)
    if len(origins) != num_points:
        raise ValueError(f"point_origins length {len(origins)} does not match point count {num_points}")
    return origins


def valid_pair_mask(
    previous: np.ndarray,
    current: np.ndarray,
    previous_visibility: Optional[np.ndarray],
    current_visibility: Optional[np.ndarray],
    *,
    frame_idx: int,
    point_origins: Optional[np.ndarray],
    visibility_threshold: float,
) -> np.ndarray:
    """Return points that are finite, nonzero, born, and visible in both frames."""

    mask = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
    mask &= np.linalg.norm(previous, axis=1) > 1e-6
    mask &= np.linalg.norm(current, axis=1) > 1e-6
    if previous_visibility is not None and current_visibility is not None:
        mask &= previous_visibility > visibility_threshold
        mask &= current_visibility > visibility_threshold
    if point_origins is not None:
        mask &= point_origins <= frame_idx - 1
    return mask


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the rigid transform between two point sets."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be [N,3]")
    if source.shape[0] < 3:
        raise ValueError("at least 3 points are required")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    u, _, vh = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1
        rotation = vh.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def robust_kabsch_ransac(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rng: np.random.RandomState,
    iterations: int = DEFAULT_RANSAC_ITERATIONS,
    inlier_threshold_m: float = DEFAULT_RANSAC_INLIER_THRESHOLD_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-point RANSAC followed by an all-inlier Kabsch fit."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be [N,3]")
    if len(source) < 3:
        raise ValueError("at least 3 points are required")

    best_count = 0
    best_rotation = np.eye(3, dtype=np.float64)
    best_translation = np.zeros(3, dtype=np.float64)
    best_mask = np.zeros(len(source), dtype=bool)
    sample_sets = rng.randint(0, len(source), size=(int(iterations), 3))

    for sample_indices in sample_sets:
        try:
            rotation, translation = kabsch_transform(source[sample_indices], target[sample_indices])
        except np.linalg.LinAlgError:
            continue
        predicted = (rotation @ source.T).T + translation
        inlier_mask = np.linalg.norm(target - predicted, axis=1) < inlier_threshold_m
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count > best_count:
            best_count = inlier_count
            best_rotation = rotation
            best_translation = translation
            best_mask = inlier_mask

    if best_count >= 3:
        best_rotation, best_translation = kabsch_transform(source[best_mask], target[best_mask])
    return best_rotation, best_translation, best_mask


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cos_theta = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def estimate_rigid_object_flow(
    coords: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    *,
    layout: str = "time_major",
    point_origins: Optional[np.ndarray] = None,
    min_visible_points: int = 6,
    visibility_threshold: float = 0.5,
    ransac_iterations: int = DEFAULT_RANSAC_ITERATIONS,
    ransac_inlier_threshold_m: float = DEFAULT_RANSAC_INLIER_THRESHOLD_M,
    random_seed: int = DEFAULT_RANSAC_SEED,
) -> RigidObjectFlowFit:
    """Fit adjacent object transforms and compose them without smoothing."""

    coords_tn3, vis_tn = normalize_flow_arrays(
        coords,
        visibilities,
        layout=layout,
    )
    origins = normalize_point_origins(point_origins, coords_tn3.shape[1])
    num_frames = coords_tn3.shape[0]
    absolute = np.repeat(np.eye(4, dtype=np.float64)[None, ...], num_frames, axis=0)
    relative = np.repeat(np.eye(4, dtype=np.float64)[None, ...], num_frames, axis=0)
    rotations: List[Optional[float]] = []
    valid_counts: List[int] = []
    inlier_ratios: List[Optional[float]] = []
    rmse_by_step: List[Optional[float]] = []
    failed_steps: List[int] = []
    rng = np.random.RandomState(random_seed)

    for frame_idx in range(1, num_frames):
        previous = coords_tn3[frame_idx - 1]
        current = coords_tn3[frame_idx]
        previous_vis = vis_tn[frame_idx - 1] if vis_tn is not None else None
        current_vis = vis_tn[frame_idx] if vis_tn is not None else None
        mask = valid_pair_mask(
            previous,
            current,
            previous_vis,
            current_vis,
            frame_idx=frame_idx,
            point_origins=origins,
            visibility_threshold=visibility_threshold,
        )
        count = int(np.count_nonzero(mask))
        valid_counts.append(count)
        if count < max(3, int(min_visible_points)):
            rotations.append(None)
            inlier_ratios.append(None)
            rmse_by_step.append(None)
            failed_steps.append(frame_idx)
            absolute[frame_idx] = absolute[frame_idx - 1]
            continue

        source = previous[mask]
        target = current[mask]
        try:
            rotation, translation, inlier_mask = robust_kabsch_ransac(
                source,
                target,
                rng=rng,
                iterations=ransac_iterations,
                inlier_threshold_m=ransac_inlier_threshold_m,
            )
            if int(np.count_nonzero(inlier_mask)) < 3:
                raise ValueError("RANSAC found fewer than three inliers")
        except (ValueError, np.linalg.LinAlgError):
            rotations.append(None)
            inlier_ratios.append(None)
            rmse_by_step.append(None)
            failed_steps.append(frame_idx)
            absolute[frame_idx] = absolute[frame_idx - 1]
            continue

        delta = _make_transform(rotation, translation)
        relative[frame_idx] = delta
        absolute[frame_idx] = delta @ absolute[frame_idx - 1]
        predicted = (rotation @ source.T).T + translation
        residual = predicted - target
        rotations.append(_rotation_angle_deg(rotation))
        inlier_ratios.append(float(np.mean(inlier_mask)))
        rmse_by_step.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))

    return RigidObjectFlowFit(
        transforms=absolute,
        relative_transforms=relative,
        rotation_degrees=rotations,
        valid_point_counts=valid_counts,
        inlier_ratios=inlier_ratios,
        rmse_by_step=rmse_by_step,
        failed_steps=failed_steps,
    )
