#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Compile a selected NovaPlan rollout into execution pose trajectories.

The execution grounding follows the paper's split:
- object flow: fit object motion from TAPIP3D tracks,
- hand flow: when object flow is rejected, follow the calibrated hand pose.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .flow_switching import (
    DEFAULT_MIN_VISIBLE_POINTS,
    DEFAULT_FLOW_SWITCH_THETA_DEG,
    FlowSwitchDecision,
    HandFlowCandidate,
    ObjectFlowEvaluation,
    evaluate_object_flow,
    select_flow_reference,
)
from .object_flow import (
    DEFAULT_RANSAC_INLIER_THRESHOLD_M,
    FLOW_ARRAY_LAYOUTS,
    RigidObjectFlowFit,
    estimate_rigid_object_flow,
    normalize_flow_arrays,
)


VISIBILITY_THRESHOLD = 0.5


@dataclass
class FlowTrackData:
    """Store tracked 3D points, visibility, and confidence values."""
    coords: np.ndarray
    visibilities: Optional[np.ndarray] = None
    point_origins: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None
    source_path: Optional[Path] = None
    layout: str = "time_major"

    @property
    def num_frames(self) -> int:
        """Return the number of tracked frames."""
        if self.layout not in FLOW_ARRAY_LAYOUTS:
            raise ValueError(f"Unsupported flow layout: {self.layout!r}")
        axis = 1 if self.layout == "point_major" else 0
        return int(self.coords.shape[axis])

    @property
    def num_points(self) -> int:
        """Return the number of tracked points."""
        if self.layout not in FLOW_ARRAY_LAYOUTS:
            raise ValueError(f"Unsupported flow layout: {self.layout!r}")
        axis = 0 if self.layout == "point_major" else 1
        return int(self.coords.shape[axis])


@dataclass
class ObjectMotionResult:
    """Store an estimated rigid object transform and diagnostics."""
    transforms: np.ndarray
    valid: bool
    methods: List[str]
    valid_point_counts: List[int]
    failed_frames: List[int]
    relative_transforms: Optional[np.ndarray] = None
    inlier_ratios: List[Optional[float]] = field(default_factory=list)
    rmse_by_step: List[Optional[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        return {
            "valid": bool(self.valid),
            "num_frames": int(self.transforms.shape[0]),
            "has_relative_transforms": self.relative_transforms is not None,
            "methods": self.methods,
            "valid_point_counts": [int(x) for x in self.valid_point_counts],
            "inlier_ratios": [float(x) if x is not None else None for x in self.inlier_ratios],
            "rmse_by_step": [float(x) if x is not None else None for x in self.rmse_by_step],
            "failed_frames": [int(x) for x in self.failed_frames],
        }


@dataclass
class ExecutionStepResult:
    # Absolute poses are retained in memory as a geometric reference for
    # diagnostics only.  The public artifact written by save_execution_step is
    # the adjacent relative transform sequence below.
    """Store compiled end-effector poses and grounding metadata."""
    ee_poses: np.ndarray
    relative_ee_transforms: np.ndarray
    frame_indices: np.ndarray
    selected_flow: str
    flow_switch: FlowSwitchDecision
    object_motion: Optional[ObjectMotionResult] = None
    hand_poses: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        return {
            "selected_flow": self.selected_flow,
            "num_relative_transforms": int(self.relative_ee_transforms.shape[0]),
            "relative_transform_shape": list(self.relative_ee_transforms.shape),
            "frame_indices": self.frame_indices.astype(int).tolist(),
            "frame_gaps": np.diff(self.frame_indices).astype(int).tolist(),
            "transform_convention": (
                "left-multiplicative camera-frame delta: "
                "T_current_from_previous[t] = T_camera_motion[t] @ inv(T_camera_motion[t-1]); "
                "entry 0 is identity"
            ),
            "flow_switch": self.flow_switch.to_dict(),
            "object_motion": self.object_motion.to_dict() if self.object_motion else None,
            "has_hand_poses": self.hand_poses is not None,
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def estimate_object_motion_transforms(
    coords: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    *,
    layout: str = "time_major",
    point_origins: Optional[np.ndarray] = None,
    min_visible_points: int = DEFAULT_MIN_VISIBLE_POINTS,
    visibility_threshold: float = VISIBILITY_THRESHOLD,
    rigid_fit: Optional[RigidObjectFlowFit] = None,
) -> ObjectMotionResult:
    """Estimate robust adjacent transforms and compose them from frame 0."""

    coords_tn3, vis_tn = normalize_flow_arrays(
        coords,
        visibilities,
        layout=layout,
    )
    fit = rigid_fit or estimate_rigid_object_flow(
        coords_tn3,
        vis_tn,
        layout="time_major",
        point_origins=point_origins,
        min_visible_points=min_visible_points,
        visibility_threshold=visibility_threshold,
    )
    methods = ["identity"] + [
        "failed_reused_previous" if frame_idx in fit.failed_steps else "adjacent_ransac"
        for frame_idx in range(1, coords_tn3.shape[0])
    ]

    return ObjectMotionResult(
        transforms=fit.transforms,
        valid=fit.valid,
        methods=methods,
        valid_point_counts=[int(coords_tn3.shape[1]), *fit.valid_point_counts],
        failed_frames=fit.failed_steps,
        relative_transforms=fit.relative_transforms,
        inlier_ratios=[1.0, *fit.inlier_ratios],
        rmse_by_step=[0.0, *fit.rmse_by_step],
    )


def load_flow_tracks(
    path: Path,
    *,
    layout: Optional[str] = None,
) -> FlowTrackData:
    """Load tracked 3D coordinates and visibility metadata."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        coords = _first_key(data, ("coords", "coords_3d", "trajectories", "points_3d"))
        if coords is None:
            raise KeyError(f"No flow coordinate key found in {path}")
        vis = _first_key(data, ("visibilities", "visibs", "visibility"))
        point_origins = _first_key(data, ("point_origins", "query_frame_indices"))
        query_points = _first_key(data, ("query_points",))
        frame_indices = _first_key(data, ("frame_indices", "frame_ids", "frames"))
        stored_layout = _first_key(data, ("flow_layout",))
    resolved_layout = _resolve_flow_layout(
        coords,
        explicit_layout=layout,
        stored_layout=stored_layout,
        frame_indices=frame_indices,
        point_origins=point_origins,
        query_points=query_points,
    )
    coords_tn3, vis_tn = normalize_flow_arrays(
        coords,
        vis,
        layout=resolved_layout,
    )
    if point_origins is None and query_points is not None:
        query_points = np.asarray(query_points)
        if query_points.ndim == 3 and query_points.shape[0] == 1:
            query_points = query_points[0]
        if (
            query_points.ndim == 2
            and query_points.shape[0] == coords_tn3.shape[1]
            and query_points.shape[1] >= 1
        ):
            point_origins = query_points[:, 0]
    if point_origins is not None:
        point_origins = np.asarray(point_origins, dtype=np.int64).reshape(-1)
        if len(point_origins) != coords_tn3.shape[1]:
            raise ValueError(
                f"point_origins length {len(point_origins)} does not match flow point count {coords_tn3.shape[1]}"
            )
    if frame_indices is None:
        frame_indices = np.arange(coords_tn3.shape[0], dtype=np.int64)
    else:
        frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if len(frame_indices) != coords_tn3.shape[0]:
            raise ValueError(
                f"frame_indices length {len(frame_indices)} does not match flow frames {coords_tn3.shape[0]}"
            )
    return FlowTrackData(
        coords=coords_tn3,
        visibilities=vis_tn,
        point_origins=point_origins,
        frame_indices=frame_indices,
        source_path=path,
        layout="time_major",
    )


def _first_key(data: Any, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _layout_scalar(value: Any, *, source: str) -> Optional[str]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{source} must be a scalar flow layout marker")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    layout = str(scalar).strip().lower()
    if layout not in FLOW_ARRAY_LAYOUTS:
        expected = ", ".join(sorted(FLOW_ARRAY_LAYOUTS))
        raise ValueError(f"{source} must be one of: {expected}; got {layout!r}")
    return layout


def _resolve_flow_layout(
    coords: np.ndarray,
    *,
    explicit_layout: Optional[str],
    stored_layout: Any,
    frame_indices: Optional[np.ndarray],
    point_origins: Optional[np.ndarray],
    query_points: Optional[np.ndarray],
) -> str:
    """Resolve raw flow layout from declarations and unique axis metadata."""

    coords = np.asarray(coords)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [T,N,3] or [N,T,3], got {coords.shape}")
    axis_0, axis_1 = coords.shape[:2]
    evidence: List[Tuple[str, str]] = []

    declared = _layout_scalar(explicit_layout, source="layout")
    if declared is not None:
        evidence.append(("layout", declared))
    stored = _layout_scalar(stored_layout, source="flow_layout")
    if stored is not None:
        evidence.append(("flow_layout", stored))

    def add_length_hint(name: str, value: Any, *, represents: str) -> None:
        if value is None or axis_0 == axis_1:
            return
        length = int(np.asarray(value).reshape(-1).shape[0])
        if length == axis_0 and length != axis_1:
            inferred = "time_major" if represents == "frames" else "point_major"
            evidence.append((name, inferred))
        elif length == axis_1 and length != axis_0:
            inferred = "point_major" if represents == "frames" else "time_major"
            evidence.append((name, inferred))

    add_length_hint("frame_indices", frame_indices, represents="frames")
    add_length_hint("point_origins", point_origins, represents="points")
    if query_points is not None:
        query_array = np.asarray(query_points)
        if query_array.ndim == 3 and query_array.shape[0] == 1:
            query_array = query_array[0]
        if query_array.ndim == 2:
            add_length_hint("query_points", query_array[:, 0], represents="points")

    layouts = {inferred for _, inferred in evidence}
    if len(layouts) > 1:
        details = ", ".join(f"{name}={inferred}" for name, inferred in evidence)
        raise ValueError(f"Conflicting flow layout evidence for {coords.shape}: {details}")
    if layouts:
        return layouts.pop()

    # Main and release TAPIP3D artifacts are time-major.
    return "time_major"


def _load_object_file(path: Path) -> Any:
    path = Path(path)
    if path.suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    if path.suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
            return loaded.item()
        return loaded
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported file type: {path}")


def load_hand_poses(path: Path) -> np.ndarray:
    """Load calibrated hand poses from a HaMeR output directory."""
    raw = _load_object_file(path)
    if isinstance(raw, dict):
        for key in ("T_world_from_hand", "T_camera_from_hand", "hand_poses", "ee_poses", "poses"):
            if key in raw:
                poses = np.asarray(raw[key], dtype=np.float64)
                break
        else:
            if "position" not in raw or "orientation_R" not in raw:
                raise KeyError(f"No hand pose keys found in {path}")
            positions = np.asarray(raw["position"], dtype=np.float64)
            rotations = np.asarray(raw["orientation_R"], dtype=np.float64)
            poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(positions), axis=0)
            poses[:, :3, :3] = rotations
            poses[:, :3, 3] = positions
    else:
        poses = np.asarray(raw, dtype=np.float64)

    if poses.shape == (4, 4):
        poses = poses[None, ...]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"hand poses must have shape [T,4,4] or [4,4], got {poses.shape}")
    return poses


def load_hand_pose_track(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load hand poses together with their original generated-video indices."""

    raw = _load_object_file(path)
    poses = load_hand_poses(path)
    frame_indices = None
    if isinstance(raw, dict):
        for key in ("frame_indices", "frame_ids", "frames"):
            if key in raw:
                frame_indices = np.asarray(raw[key], dtype=np.int64).reshape(-1)
                break
    if frame_indices is None:
        frame_indices = np.arange(len(poses), dtype=np.int64)
    if len(frame_indices) != len(poses):
        raise ValueError(
            f"hand frame_indices length {len(frame_indices)} does not match pose count {len(poses)}"
        )
    if len(frame_indices) and np.any(np.diff(frame_indices) <= 0):
        raise ValueError("hand frame_indices must be strictly increasing")
    return poses, frame_indices


def relative_transforms_from_poses(poses: np.ndarray) -> np.ndarray:
    """Convert absolute camera-frame poses into adjacent left deltas."""

    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape [T,4,4], got {poses.shape}")
    relative = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(poses), axis=0)
    for idx in range(1, len(poses)):
        relative[idx] = poses[idx] @ np.linalg.inv(poses[idx - 1])
    return relative


def compile_flow_execution_step(
    flow: FlowTrackData,
    *,
    hand_poses: Optional[np.ndarray] = None,
    hand_frame_indices: Optional[np.ndarray] = None,
    selected_flow: str = "auto",
    flow_switch_theta_deg: float = DEFAULT_FLOW_SWITCH_THETA_DEG,
    min_visible_points: int = DEFAULT_MIN_VISIBLE_POINTS,
    visibility_threshold: float = VISIBILITY_THRESHOLD,
    rigid_fit: Optional[RigidObjectFlowFit] = None,
    object_evaluation: Optional[ObjectFlowEvaluation] = None,
) -> ExecutionStepResult:
    """Return flow-derived pose trajectories for the selected rollout.

    Object selection returns the fitted frame-0-to-frame-t object transforms.
    Hand selection returns the provided hand pose trajectory.
    """

    if selected_flow not in {"auto", "object", "hand"}:
        raise ValueError("selected_flow must be one of: auto, object, hand")

    coords, vis = normalize_flow_arrays(
        flow.coords,
        flow.visibilities,
        layout=flow.layout,
    )
    if rigid_fit is None:
        rigid_fit = estimate_rigid_object_flow(
            coords,
            vis,
            layout="time_major",
            point_origins=flow.point_origins,
            min_visible_points=min_visible_points,
            visibility_threshold=visibility_threshold,
        )
    object_eval = object_evaluation or evaluate_object_flow(
        coords,
        vis,
        layout="time_major",
        point_origins=flow.point_origins,
        flow_switch_theta_deg=flow_switch_theta_deg,
        min_visible_points=min_visible_points,
        visibility_threshold=visibility_threshold,
        rigid_fit=rigid_fit,
    )
    hand_candidate = None
    if hand_poses is not None:
        hand_poses = np.asarray(hand_poses, dtype=np.float64)
        hand_candidate = HandFlowCandidate(valid=True, trajectory=hand_poses)
    decision = select_flow_reference(object_eval, hand_candidate)
    effective_flow = decision.selected_flow if selected_flow == "auto" else selected_flow

    object_motion = estimate_object_motion_transforms(
        coords,
        vis,
        layout="time_major",
        point_origins=flow.point_origins,
        min_visible_points=min_visible_points,
        visibility_threshold=visibility_threshold,
        rigid_fit=rigid_fit,
    )

    if effective_flow == "object":
        if not object_eval.valid:
            raise ValueError(f"Object flow is invalid: {object_eval.reason}")
        if object_eval.should_switch_to_hand:
            raise ValueError(f"Object flow exceeded the flow-switch threshold: {object_eval.reason}")
        poses = object_motion.transforms
        frame_indices = (
            np.asarray(flow.frame_indices, dtype=np.int64).reshape(-1)
            if flow.frame_indices is not None
            else np.arange(len(poses), dtype=np.int64)
        )
        relative_ee = np.asarray(object_motion.relative_transforms, dtype=np.float64)
    elif effective_flow == "hand":
        if hand_poses is None or len(hand_poses) == 0:
            raise ValueError("Hand flow was selected but no hand poses were provided")
        poses = hand_poses
        frame_indices = (
            np.asarray(hand_frame_indices, dtype=np.int64).reshape(-1)
            if hand_frame_indices is not None
            else np.arange(len(poses), dtype=np.int64)
        )
        if len(frame_indices) != len(poses):
            raise ValueError("hand_frame_indices length must match hand_poses")
        relative_ee = relative_transforms_from_poses(poses)
    else:
        raise ValueError(f"Flow selection returned no executable flow: {effective_flow}")

    metadata = {
        "pipeline": "flow_only",
        "paper_rule": "object flow unless object tracking fails or rotation exceeds the flow-switch threshold; otherwise hand flow",
        "flow_switch_theta_deg": float(flow_switch_theta_deg),
        "min_visible_points": int(min_visible_points),
        "visibility_threshold": float(visibility_threshold),
        "object_motion_estimator": "adjacent_visibility_filtered_ransac_kabsch",
        "object_motion_ransac_inlier_threshold_m": DEFAULT_RANSAC_INLIER_THRESHOLD_M,
        "object_motion_temporal_smoothing": False,
        "flow_source": str(flow.source_path) if flow.source_path else None,
        "coordinate_frame": "metric_camera",
        "translation_units": "meters",
        "source_frame_indices": frame_indices.astype(int).tolist(),
        "source_frame_gaps": np.diff(frame_indices).astype(int).tolist(),
    }
    return ExecutionStepResult(
        ee_poses=np.asarray(poses, dtype=np.float64),
        relative_ee_transforms=relative_ee,
        frame_indices=frame_indices,
        selected_flow=effective_flow,
        flow_switch=decision if selected_flow == "auto" else _forced_decision(effective_flow, object_eval, hand_candidate),
        object_motion=object_motion,
        hand_poses=hand_poses,
        metadata=metadata,
    )


def _forced_decision(
    selected_flow: str,
    object_eval: ObjectFlowEvaluation,
    hand_flow: Optional[HandFlowCandidate],
) -> FlowSwitchDecision:
    return FlowSwitchDecision(
        selected_flow=selected_flow,
        object_flow=object_eval,
        hand_flow=hand_flow,
        reason=f"forced by execution CLI: {selected_flow}",
        used_fallback=(selected_flow == "hand"),
    )


def save_execution_step(result: ExecutionStepResult, out_dir: Path) -> Dict[str, Path]:
    """Save execution step."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    relative_path = out_dir / "relative_ee_transforms.npy"
    np.save(relative_path, result.relative_ee_transforms)
    saved["relative_ee_transforms"] = relative_path
    frame_indices_path = out_dir / "relative_ee_frame_indices.npy"
    np.save(frame_indices_path, result.frame_indices)
    saved["relative_ee_frame_indices"] = frame_indices_path
    npz_payload: Dict[str, Any] = {
        "relative_ee_transforms": result.relative_ee_transforms,
        "relative_ee_frame_indices": result.frame_indices,
        "selected_flow": np.array(result.selected_flow),
    }
    if result.object_motion is not None:
        npz_payload["object_motion_transforms"] = result.object_motion.transforms
        if result.object_motion.relative_transforms is not None:
            npz_payload["object_motion_relative_transforms"] = result.object_motion.relative_transforms

    bundle_path = out_dir / "execution_step.npz"
    np.savez(bundle_path, **npz_payload)
    saved["execution_step"] = bundle_path

    summary_path = out_dir / "execution_step.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2))
    saved["summary"] = summary_path
    return saved


def find_default_file(base: Path, patterns: Iterable[str]) -> Optional[Path]:
    """Find the first existing default file candidate."""
    for pattern in patterns:
        matches = sorted(Path(base).glob(pattern))
        if matches:
            return matches[0]
    return None


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Parse and validate a rotation angle in degrees."""
    trace = float(np.trace(rotation))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_theta)))
