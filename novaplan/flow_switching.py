#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Utilities for selecting between NovaPlan object flow and hand flow.

The NovaPlan paper switches from object flow to hand flow when the recovered
object trajectory becomes geometrically unreliable. In practice that means we
recover adjacent-frame rigid transforms from tracked 3D object keypoints and
check whether any frame-to-frame rotation magnitude exceeds the configured
flow-switch threshold.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .object_flow import RigidObjectFlowFit, estimate_rigid_object_flow


DEFAULT_FLOW_SWITCH_THETA_DEG = 45.0
DEFAULT_MIN_VISIBLE_POINTS = 6


@dataclass
class ObjectFlowEvaluation:
    """Reliability summary for a 3D object-flow trajectory."""

    valid: bool
    should_switch_to_hand: bool
    flow_switch_theta_deg: float
    max_rotation_deg: Optional[float]
    rotation_degrees: List[Optional[float]] = field(default_factory=list)
    valid_point_counts: List[int] = field(default_factory=list)
    inlier_ratios: List[Optional[float]] = field(default_factory=list)
    rmse_by_step: List[Optional[float]] = field(default_factory=list)
    failed_steps: List[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        return {
            "valid": self.valid,
            "should_switch_to_hand": self.should_switch_to_hand,
            "flow_switch_theta_deg": self.flow_switch_theta_deg,
            "max_rotation_deg": self.max_rotation_deg,
            "rotation_degrees": self.rotation_degrees,
            "valid_point_counts": self.valid_point_counts,
            "inlier_ratios": self.inlier_ratios,
            "rmse_by_step": self.rmse_by_step,
            "failed_steps": self.failed_steps,
            "reason": self.reason,
        }


@dataclass
class HandFlowCandidate:
    """Optional hand-flow data supplied by a reconstruction provider.

    This intentionally stores only generic data so the open-source planner can
    be wired to different hand-flow backends without depending on private robot or
    local_planning code.
    """

    valid: bool = True
    trajectory: Optional[np.ndarray] = None
    flow_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        trajectory_shape = None
        if self.trajectory is not None:
            trajectory_shape = tuple(int(x) for x in self.trajectory.shape)
        flow_image_shape = None
        if self.flow_image is not None:
            flow_image_shape = tuple(int(x) for x in self.flow_image.shape)
        return {
            "valid": self.valid,
            "has_trajectory": self.trajectory is not None,
            "trajectory_shape": trajectory_shape,
            "has_flow_image": self.flow_image is not None,
            "flow_image_shape": flow_image_shape,
            "metadata": self.metadata,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class FlowSwitchDecision:
    """Decision returned by the hand/object flow switch."""

    selected_flow: str
    object_flow: ObjectFlowEvaluation
    hand_flow: Optional[HandFlowCandidate] = None
    reason: str = ""
    used_fallback: bool = False

    @property
    def selected_is_hand(self) -> bool:
        """Return whether the decision selected hand-centric flow."""
        return self.selected_flow == "hand"

    @property
    def selected_is_object(self) -> bool:
        """Return whether the decision selected object-centric flow."""
        return self.selected_flow == "object"

    @property
    def effective_switch_to_hand(self) -> bool:
        """True when execution should use hand flow for the selected rollout."""

        return self.selected_flow == "hand"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        return {
            "selected_flow": self.selected_flow,
            "selected_is_hand": self.selected_is_hand,
            "selected_is_object": self.selected_is_object,
            "switch_to_hand": self.effective_switch_to_hand,
            "reason": self.reason,
            "used_fallback": self.used_fallback,
            "object_flow": self.object_flow.to_dict(),
            "hand_flow": self.hand_flow.to_dict() if self.hand_flow else None,
        }


def npy_bytes_to_array(data: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode a .npy payload returned by an object-flow server."""

    if not data:
        return None
    return np.load(io.BytesIO(data), allow_pickle=False)


def evaluate_object_flow(
    coords_3d: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    *,
    layout: str = "time_major",
    point_origins: Optional[np.ndarray] = None,
    flow_switch_theta_deg: float = DEFAULT_FLOW_SWITCH_THETA_DEG,
    min_visible_points: int = DEFAULT_MIN_VISIBLE_POINTS,
    visibility_threshold: float = 0.5,
    rigid_fit: Optional[RigidObjectFlowFit] = None,
) -> ObjectFlowEvaluation:
    """Evaluate adjacent-frame rotations using the shared rigid object-flow fit."""

    coords = np.asarray(coords_3d)
    if coords.ndim != 3 or min(coords.shape[:2]) < 2:
        return ObjectFlowEvaluation(
            valid=False,
            should_switch_to_hand=False,
            flow_switch_theta_deg=float(flow_switch_theta_deg),
            max_rotation_deg=None,
            reason="object flow has fewer than two frames",
        )

    fit = rigid_fit or estimate_rigid_object_flow(
        coords_3d,
        visibilities,
        layout=layout,
        point_origins=point_origins,
        min_visible_points=min_visible_points,
        visibility_threshold=visibility_threshold,
    )
    rotation_degrees = fit.rotation_degrees
    valid_point_counts = fit.valid_point_counts
    inlier_ratios = fit.inlier_ratios
    rmse_by_step = fit.rmse_by_step
    failed_steps = fit.failed_steps

    finite_rotations = [float(r) for r in rotation_degrees if r is not None and np.isfinite(r)]
    max_rotation = max(finite_rotations) if finite_rotations else None
    invalid = bool(failed_steps) or not finite_rotations
    should_switch = bool(max_rotation is not None and max_rotation > flow_switch_theta_deg)

    if invalid:
        reason = "object flow missing reliable adjacent-frame transforms"
    elif should_switch:
        reason = (
            "object rotation exceeded flow-switch threshold "
            f"({max_rotation:.2f} > {flow_switch_theta_deg:.2f} deg)"
        )
    else:
        reason = (
            "object rotation within flow-switch threshold "
            f"({max_rotation:.2f} <= {flow_switch_theta_deg:.2f} deg)"
        )

    return ObjectFlowEvaluation(
        valid=not invalid,
        should_switch_to_hand=should_switch,
        flow_switch_theta_deg=float(flow_switch_theta_deg),
        max_rotation_deg=max_rotation,
        rotation_degrees=rotation_degrees,
        valid_point_counts=valid_point_counts,
        inlier_ratios=inlier_ratios,
        rmse_by_step=rmse_by_step,
        failed_steps=failed_steps,
        reason=reason,
    )


def select_flow_reference(
    object_flow: ObjectFlowEvaluation,
    hand_flow: Optional[HandFlowCandidate] = None,
) -> FlowSwitchDecision:
    """Select object or hand flow using the NovaPlan switching rule."""

    hand_valid = bool(hand_flow is not None and hand_flow.valid)

    if object_flow.should_switch_to_hand:
        if hand_valid:
            return FlowSwitchDecision(
                selected_flow="hand",
                object_flow=object_flow,
                hand_flow=hand_flow,
                reason=f"{object_flow.reason}; using hand flow",
            )
        return FlowSwitchDecision(
            selected_flow="object" if object_flow.valid else "none",
            object_flow=object_flow,
            hand_flow=hand_flow,
            reason=f"{object_flow.reason}; hand flow unavailable or rejected",
            used_fallback=True,
        )

    if not object_flow.valid:
        if hand_valid:
            return FlowSwitchDecision(
                selected_flow="hand",
                object_flow=object_flow,
                hand_flow=hand_flow,
                reason=f"{object_flow.reason}; using hand flow fallback",
                used_fallback=True,
            )
        return FlowSwitchDecision(
            selected_flow="none",
            object_flow=object_flow,
            hand_flow=hand_flow,
            reason=f"{object_flow.reason}; no valid hand flow fallback",
            used_fallback=True,
        )

    return FlowSwitchDecision(
        selected_flow="object",
        object_flow=object_flow,
        hand_flow=hand_flow,
        reason=object_flow.reason,
    )


def select_flow_reference_from_arrays(
    coords_3d: np.ndarray,
    visibilities: Optional[np.ndarray] = None,
    hand_flow: Optional[HandFlowCandidate] = None,
    *,
    layout: str = "time_major",
    point_origins: Optional[np.ndarray] = None,
    flow_switch_theta_deg: float = DEFAULT_FLOW_SWITCH_THETA_DEG,
    min_visible_points: int = DEFAULT_MIN_VISIBLE_POINTS,
    visibility_threshold: float = 0.5,
) -> FlowSwitchDecision:
    """Select object- or hand-centric grounding from flow arrays."""
    object_eval = evaluate_object_flow(
        coords_3d,
        visibilities,
        layout=layout,
        point_origins=point_origins,
        flow_switch_theta_deg=flow_switch_theta_deg,
        min_visible_points=min_visible_points,
        visibility_threshold=visibility_threshold,
    )
    return select_flow_reference(object_eval, hand_flow)


def decisions_to_jsonable(decisions: Sequence[Optional[FlowSwitchDecision]]) -> List[Optional[Dict[str, Any]]]:
    """Serialize flow-switch decisions for JSON output."""
    return [decision.to_dict() if decision is not None else None for decision in decisions]
