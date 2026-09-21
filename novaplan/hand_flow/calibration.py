# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Calibration helpers for HaMeR hand-flow meshes.

HaMeR meshes are useful kinematic priors, but the raw reconstruction can be in a
weak-camera coordinate frame. These helpers keep the hand-flow execution path in
the same metric camera frame as the TAPIP3D/depth outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


FINGER_JOINTS = {
    "thumb": {"mcp": 1, "pip": 2, "dip": 3, "tip": 4},
    "index": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
    "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
    "ring": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    "pinky": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
}

_FINGER_ALIASES = {
    "thumb": "thumb",
    "thumb_tip": "thumb",
    "index": "index",
    "index_finger": "index",
    "index_tip": "index",
    "middle": "middle",
    "middle_finger": "middle",
    "middle_tip": "middle",
    "ring": "ring",
    "ring_finger": "ring",
    "ring_tip": "ring",
    "pinky": "pinky",
    "little": "pinky",
    "little_finger": "pinky",
    "pinky_tip": "pinky",
}


def canonical_contact_finger(value: Optional[str]) -> str:
    """Return a single canonical HaMeR finger name or reject ambiguity."""

    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    finger = _FINGER_ALIASES.get(normalized)
    if finger is None:
        raise ValueError(
            "contact_finger must identify exactly one of thumb, index, middle, ring, or pinky; "
            f"got {value!r}"
        )
    return finger


def scale_pixel_between_resolutions(
    point_xy: tuple[float, float] | list[float] | np.ndarray,
    source_shape_hw: tuple[int, int] | list[int] | np.ndarray,
    target_shape_hw: tuple[int, int] | list[int] | np.ndarray,
) -> np.ndarray:
    """Map one ``[x,y]`` pixel between aligned image resolutions."""

    point = np.asarray(point_xy, dtype=np.float64).reshape(-1)
    source = np.asarray(source_shape_hw, dtype=np.int64).reshape(-1)
    target = np.asarray(target_shape_hw, dtype=np.int64).reshape(-1)
    if point.size != 2 or not np.isfinite(point).all():
        raise ValueError("point_xy must contain one finite [x,y] pixel")
    if source.size != 2 or target.size != 2 or np.any(source <= 0) or np.any(target <= 0):
        raise ValueError("source_shape_hw and target_shape_hw must be positive [height,width]")
    source_h, source_w = int(source[0]), int(source[1])
    target_h, target_w = int(target[0]), int(target[1])
    if not (0.0 <= point[0] < source_w and 0.0 <= point[1] < source_h):
        raise ValueError(
            f"contact pixel {point.tolist()} is outside source image {source_h}x{source_w}"
        )
    return np.asarray(
        [point[0] * target_w / source_w, point[1] * target_h / source_h],
        dtype=np.float64,
    )


def interaction_interval_from_object_masks(
    object_masks: np.ndarray,
    *,
    epsilon: float = 0.9,
) -> tuple[int, int]:
    """Return the paper's coarse object-motion interval from mask differences.

    Equation (7)'s final active frame is an upper bound on hand interaction:
    after placement, the object can remain outside its initial footprint while
    the hand releases and withdraws.
    """

    masks = np.asarray(object_masks)
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError(f"object_masks must have shape [T,H,W], got {masks.shape}")
    masks = masks > 0.5
    initial = masks[0]
    denom = int(np.count_nonzero(initial))
    if denom <= 0:
        raise ValueError("initial object mask is empty")
    # Paper Eq. (7): |M_t \ M_t0| / |M_t0|. This measures target-object
    # support newly exposed outside its initial footprint. It avoids treating
    # pure occlusion or segmentation shrinkage as the start of interaction.
    changed = np.count_nonzero(masks & ~initial[None, ...], axis=(1, 2)) / float(denom)
    active = np.flatnonzero(changed >= float(epsilon))
    if active.size == 0:
        max_changed = float(np.max(changed)) if changed.size else 0.0
        raise ValueError(
            "object-mask motion never reached the interaction threshold: "
            f"max_new_support_ratio={max_changed:.3f} < epsilon={float(epsilon):.3f}"
        )
    return int(active[0]), int(active[-1])


@dataclass
class HandMeshDepthCalibration:
    """Store hand-mesh metric calibration results and diagnostics."""
    vertices: np.ndarray
    scales: np.ndarray
    offsets: np.ndarray
    frame_indices: np.ndarray
    status: list[str]
    projection_bbox_px: np.ndarray
    valid_depth_points: np.ndarray
    method: str = "depth_projection_similarity"
    extra_summary: dict[str, Any] | None = None

    @property
    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible calibration diagnostics."""
        valid = [s for s in self.status if s == "ok"]
        summary = {
            "method": self.method,
            "num_frames": int(len(self.frame_indices)),
            "num_ok_frames": int(len(valid)),
            "scale_median": float(np.median(self.scales)) if len(self.scales) else 1.0,
            "scale_min": float(np.min(self.scales)) if len(self.scales) else 1.0,
            "scale_max": float(np.max(self.scales)) if len(self.scales) else 1.0,
            "offset_median": np.median(self.offsets, axis=0).tolist() if len(self.offsets) else [0.0, 0.0, 0.0],
            "status_counts": {name: int(self.status.count(name)) for name in sorted(set(self.status))},
            "projection_bbox_px_median": (
                np.median(self.projection_bbox_px, axis=0).tolist() if len(self.projection_bbox_px) else [0.0, 0.0]
            ),
        }
        if self.extra_summary:
            summary.update(self.extra_summary)
        return summary


def decode_semantic_xyz(value: Any) -> dict[str, Any]:
    """Decode semantic xyz."""
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return {}
        value = value.item()
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        return json.loads(value)
    return {}


def _as_intrinsic(intrinsics: np.ndarray, frame_idx: int) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 3:
        idx = max(0, min(int(frame_idx), intrinsics.shape[0] - 1))
        intrinsics = intrinsics[idx]
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3,3] or [T,3,3], got {intrinsics.shape}")
    return intrinsics


def project_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project points."""
    points = np.asarray(points, dtype=np.float64)
    intr = np.asarray(intrinsics, dtype=np.float64)
    z = points[:, 2]
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (z > 1e-9)
    uv[valid, 0] = intr[0, 0] * points[valid, 0] / z[valid] + intr[0, 2]
    uv[valid, 1] = intr[1, 1] * points[valid, 1] / z[valid] + intr[1, 2]
    return uv


def unproject_pixels(uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Unproject pixels."""
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    intr = np.asarray(intrinsics, dtype=np.float64)
    x = (uv[:, 0] - intr[0, 2]) * depth / intr[0, 0]
    y = (uv[:, 1] - intr[1, 2]) * depth / intr[1, 1]
    return np.stack([x, y, depth], axis=1)


def _robust_median(values: np.ndarray, *, trim: float = 0.1) -> Optional[np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if values.ndim == 1:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        if values.size >= 10 and trim > 0:
            lo, hi = np.quantile(values, [trim, 1.0 - trim])
            values = values[(values >= lo) & (values <= hi)]
        return np.asarray(np.median(values), dtype=np.float64)

    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if len(values) == 0:
        return None
    if len(values) >= 10 and trim > 0:
        norms = np.linalg.norm(values - np.median(values, axis=0), axis=1)
        keep = norms <= np.quantile(norms, 1.0 - trim)
        values = values[keep]
    return np.median(values, axis=0)


def _calibrate_one_frame(
    vertices: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    min_points: int,
    min_projection_bbox_px: float,
    scale_bounds: tuple[float, float],
) -> tuple[float, np.ndarray, str, np.ndarray, int]:
    vertices = np.asarray(vertices, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    intr = np.asarray(intrinsics, dtype=np.float64)
    h, w = depth.shape[:2]

    uv = project_points(vertices, intr)
    finite_uv = np.isfinite(uv).all(axis=1)
    if not np.any(finite_uv):
        return 1.0, np.zeros(3, dtype=np.float64), "no_projection", np.zeros(2), 0

    bbox = np.array([np.ptp(uv[finite_uv, 0]), np.ptp(uv[finite_uv, 1])], dtype=np.float64)
    if float(max(bbox)) < min_projection_bbox_px:
        return 1.0, np.zeros(3, dtype=np.float64), "projection_too_small", bbox, 0

    ui = np.rint(uv[:, 0]).astype(np.int64)
    vi = np.rint(uv[:, 1]).astype(np.int64)
    z = vertices[:, 2]
    in_bounds = finite_uv & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & np.isfinite(z) & (z > 1e-6)
    if not np.any(in_bounds):
        return 1.0, np.zeros(3, dtype=np.float64), "projection_out_of_bounds", bbox, 0

    sampled_depth = depth[vi[in_bounds], ui[in_bounds]]
    mesh_z = z[in_bounds]
    valid_depth = np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
    sampled_depth = sampled_depth[valid_depth]
    mesh_z = mesh_z[valid_depth]
    sampled_uv = uv[in_bounds][valid_depth]
    sampled_vertices = vertices[in_bounds][valid_depth]
    if len(sampled_depth) < min_points:
        return 1.0, np.zeros(3, dtype=np.float64), "not_enough_depth_points", bbox, int(len(sampled_depth))

    ratios = sampled_depth / mesh_z
    ratio_mask = np.isfinite(ratios) & (ratios >= scale_bounds[0]) & (ratios <= scale_bounds[1])
    if int(np.count_nonzero(ratio_mask)) < min_points:
        return 1.0, np.zeros(3, dtype=np.float64), "scale_out_of_bounds", bbox, int(np.count_nonzero(ratio_mask))

    scale_value = _robust_median(ratios[ratio_mask])
    if scale_value is None:
        return 1.0, np.zeros(3, dtype=np.float64), "scale_failed", bbox, int(np.count_nonzero(ratio_mask))
    scale = float(scale_value)

    target_points = unproject_pixels(sampled_uv[ratio_mask], sampled_depth[ratio_mask], intr)
    scaled_vertices = sampled_vertices[ratio_mask] * scale
    offsets = target_points - scaled_vertices
    offset_value = _robust_median(offsets, trim=0.2)
    if offset_value is None:
        return scale, np.zeros(3, dtype=np.float64), "offset_failed", bbox, int(np.count_nonzero(ratio_mask))
    return scale, np.asarray(offset_value, dtype=np.float64).reshape(3), "ok", bbox, int(np.count_nonzero(ratio_mask))


def calibrate_mesh_sequence_to_depth(
    vertices: np.ndarray,
    frame_indices: np.ndarray,
    depths: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    *,
    min_points: int = 30,
    min_projection_bbox_px: float = 18.0,
    scale_bounds: tuple[float, float] = (0.2, 5.0),
) -> HandMeshDepthCalibration:
    """Calibrate mesh sequence to depth."""
    vertices = np.asarray(vertices, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape [T,V,3], got {vertices.shape}")
    if len(frame_indices) != vertices.shape[0]:
        raise ValueError("frame_indices length must match vertices.shape[0]")

    calibrated = vertices.copy()
    scales = np.ones(vertices.shape[0], dtype=np.float64)
    offsets = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    status = ["no_depth"] * vertices.shape[0]
    bbox = np.zeros((vertices.shape[0], 2), dtype=np.float64)
    valid_depth_points = np.zeros(vertices.shape[0], dtype=np.int64)

    if depths is None or intrinsics is None:
        return HandMeshDepthCalibration(calibrated, scales, offsets, frame_indices, status, bbox, valid_depth_points)

    depths_arr = np.asarray(depths, dtype=np.float64)
    if depths_arr.ndim != 3:
        raise ValueError(f"depths must have shape [T,H,W], got {depths_arr.shape}")

    for i, frame_idx in enumerate(frame_indices):
        depth_idx = max(0, min(int(frame_idx), depths_arr.shape[0] - 1))
        intr = _as_intrinsic(np.asarray(intrinsics), depth_idx)
        scale, offset, frame_status, frame_bbox, frame_valid = _calibrate_one_frame(
            vertices[i],
            depths_arr[depth_idx],
            intr,
            min_points=min_points,
            min_projection_bbox_px=min_projection_bbox_px,
            scale_bounds=scale_bounds,
        )
        scales[i] = scale
        offsets[i] = offset
        status[i] = frame_status
        bbox[i] = frame_bbox
        valid_depth_points[i] = frame_valid
        calibrated[i] = vertices[i] * scale + offset.reshape(1, 3)

    return HandMeshDepthCalibration(calibrated, scales, offsets, frame_indices, status, bbox, valid_depth_points)


def _resize_mask_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape[:2] == shape:
        return mask.astype(bool)
    src_h, src_w = mask.shape[:2]
    dst_h, dst_w = shape
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return np.zeros(shape, dtype=bool)
    ys = np.floor(np.arange(dst_h, dtype=np.float64) * (src_h / dst_h)).astype(np.int64).clip(0, src_h - 1)
    xs = np.floor(np.arange(dst_w, dtype=np.float64) * (src_w / dst_w)).astype(np.int64).clip(0, src_w - 1)
    return np.asarray(mask)[ys[:, None], xs[None, :]].astype(bool)


def _select_mask_frame(object_masks: np.ndarray, frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(object_masks)
    if masks.ndim == 2:
        mask = masks
    elif masks.ndim == 3:
        idx = max(0, min(int(frame_idx), masks.shape[0] - 1))
        mask = masks[idx]
    else:
        raise ValueError(f"object_masks must have shape [H,W] or [T,H,W], got {masks.shape}")
    return _resize_mask_nearest(mask, shape)


def _nearest_mask_pixel(mask: np.ndarray, u: int, v: int, *, radius: int) -> Optional[tuple[int, int, float]]:
    h, w = mask.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None
    if mask[v, u]:
        return u, v, 0.0
    u0 = max(0, u - radius)
    u1 = min(w, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(h, v + radius + 1)
    window = mask[v0:v1, u0:u1]
    ys, xs = np.nonzero(window)
    if len(xs) == 0:
        return None
    xs = xs + u0
    ys = ys + v0
    d2 = (xs - u) ** 2 + (ys - v) ** 2
    best = int(np.argmin(d2))
    distance = float(np.sqrt(d2[best]))
    if distance > float(radius):
        return None
    return int(xs[best]), int(ys[best]), distance


def _nearest_metric_mask_pixel(
    mask: np.ndarray,
    depth: np.ndarray,
    u: int,
    v: int,
    *,
    radius: int,
) -> Optional[tuple[int, int, float]]:
    """Find the nearest masked pixel with finite positive metric depth."""

    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float64)
    valid = mask & np.isfinite(depth) & (depth > 1e-6)
    return _nearest_mask_pixel(valid, int(u), int(v), radius=int(radius))


def _outside_frame_fractions(
    vertices: np.ndarray,
    frame_indices: np.ndarray,
    intrinsics: np.ndarray,
    image_shapes: list[tuple[int, int]],
) -> np.ndarray:
    fractions = np.ones(len(vertices), dtype=np.float64)
    for seq_i, frame_idx in enumerate(frame_indices):
        intr = _as_intrinsic(np.asarray(intrinsics), int(frame_idx))
        uv = project_points(vertices[seq_i], intr)
        h, w = image_shapes[seq_i]
        inside = (
            np.isfinite(uv).all(axis=1)
            & np.isfinite(vertices[seq_i]).all(axis=1)
            & (vertices[seq_i, :, 2] > 1e-6)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        fractions[seq_i] = 1.0 - float(np.count_nonzero(inside)) / float(max(1, len(inside)))
    return fractions


def _mesh_vertex_adjacency(faces: np.ndarray, n_vertices: int) -> list[set[int]]:
    adjacency = [set() for _ in range(int(n_vertices))]
    faces = np.asarray(faces, dtype=np.int64)
    for face in faces:
        if face.shape[0] < 3:
            continue
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        if not (0 <= a < n_vertices and 0 <= b < n_vertices and 0 <= c < n_vertices):
            continue
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return adjacency


def _fingertip_vertex_indices(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    tip: np.ndarray,
    mcp: Optional[np.ndarray],
    *,
    max_vertices: int = 200,
    fallback_radius: float = 0.015,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    if len(vertices) == 0:
        return np.zeros(0, dtype=np.int64)

    if faces is not None and mcp is not None and len(faces) > 0:
        mcp = np.asarray(mcp, dtype=np.float64).reshape(3)
        adjacency = _mesh_vertex_adjacency(faces, len(vertices))
        seed = int(np.argmin(np.linalg.norm(vertices - tip.reshape(1, 3), axis=1)))
        queue = [seed]
        visited: set[int] = set()
        selected: list[int] = []
        while queue and len(selected) < max_vertices:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)
            v_pos = vertices[idx]
            if float(np.linalg.norm(v_pos - mcp)) < float(np.linalg.norm(v_pos - tip)):
                continue
            selected.append(idx)
            queue.extend(sorted(adjacency[idx] - visited))
        if selected:
            return np.asarray(selected, dtype=np.int64)

    distances = np.linalg.norm(vertices - tip.reshape(1, 3), axis=1)
    selected = np.flatnonzero(distances <= fallback_radius)
    if len(selected) > 0:
        return selected.astype(np.int64)
    return np.asarray([int(np.argmin(distances))], dtype=np.int64)


def _calibrate_contact_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    semantic: dict[str, Any],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    object_mask: np.ndarray,
    *,
    contact_finger: Optional[str],
    projection_radius: int,
    scale_bounds: tuple[float, float],
) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    intr = np.asarray(intrinsics, dtype=np.float64)
    h, w = depth.shape[:2]
    mask = _resize_mask_nearest(object_mask, (h, w))
    if not np.any(mask):
        return {"ok": False, "status": "empty_object_mask"}

    candidates_by_finger: dict[str, list[dict[str, Any]]] = {}
    finger_names = list(FINGER_JOINTS.keys())
    if contact_finger in FINGER_JOINTS:
        finger_names = [str(contact_finger)]

    for finger in finger_names:
        tip = semantic_finger_point(semantic, finger, "tip")
        if tip is None:
            continue
        mcp = semantic_finger_point(semantic, finger, "mcp")
        vertex_indices = _fingertip_vertex_indices(vertices, faces, tip, mcp)
        if len(vertex_indices) == 0:
            continue
        fingertip_vertices = vertices[vertex_indices]
        uv = project_points(fingertip_vertices, intr)
        finger_candidates = []
        for local_idx, (u_float, v_float) in enumerate(uv):
            if not np.isfinite(u_float) or not np.isfinite(v_float):
                continue
            u = int(round(float(u_float)))
            v = int(round(float(v_float)))
            nearest = _nearest_mask_pixel(mask, u, v, radius=projection_radius)
            if nearest is None:
                continue
            mask_u, mask_v, mask_dist = nearest
            d = float(depth[mask_v, mask_u])
            vertex = fingertip_vertices[local_idx]
            z = float(vertex[2])
            if not (np.isfinite(d) and np.isfinite(z) and d > 1e-6 and z > 1e-6):
                continue
            scale = d / z
            if not (scale_bounds[0] <= scale <= scale_bounds[1]):
                continue
            contact_xyz = unproject_pixels(np.asarray([[mask_u, mask_v]], dtype=np.float64), np.asarray([d]), intr)[0]
            offset = contact_xyz - scale * vertex
            finger_candidates.append(
                {
                    "scale": float(scale),
                    "offset": np.asarray(offset, dtype=np.float64),
                    "finger": finger,
                    "vertex_index": int(vertex_indices[local_idx]),
                    "vertex_xyz": vertex.astype(float).tolist(),
                    "contact_xyz": contact_xyz.astype(float).tolist(),
                    "mask_uv": [int(mask_u), int(mask_v)],
                    "projection_uv": [float(u_float), float(v_float)],
                    "mask_distance_px": float(mask_dist),
                }
            )
        if finger_candidates:
            candidates_by_finger[finger] = finger_candidates

    if not candidates_by_finger:
        return {"ok": False, "status": "no_fingertip_object_contact"}

    per_finger: dict[str, dict[str, Any]] = {}
    for finger, candidates in candidates_by_finger.items():
        # Same convention as the main branch: each finger contributes its
        # closest contact (minimum scale), then the selected contact is the
        # maximum of those per-finger minima unless a finger was requested.
        best = min(candidates, key=lambda item: item["scale"])
        per_finger[finger] = best

    if contact_finger in per_finger:
        selected = per_finger[str(contact_finger)]
    else:
        selected = max(per_finger.values(), key=lambda item: item["scale"])

    return {
        "ok": True,
        "status": "ok",
        "scale": float(selected["scale"]),
        "offset": np.asarray(selected["offset"], dtype=np.float64),
        "contact_finger": str(selected["finger"]),
        "num_valid_contacts": int(sum(len(v) for v in candidates_by_finger.values())),
        "num_contact_fingers": int(len(candidates_by_finger)),
        "per_finger_scale": {finger: float(item["scale"]) for finger, item in per_finger.items()},
        "selected_contact": {
            key: value
            for key, value in selected.items()
            if key not in {"offset"}
        },
    }


def _sequence_index_at_or_after(frame_indices: np.ndarray, target_frame: int) -> int:
    """Map an onset bound inward to the first available sequence frame."""

    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    candidates = np.flatnonzero(frame_indices >= int(target_frame))
    if not candidates.size:
        raise ValueError(
            "hand trajectory rejected because HaMeR has no frame at or after "
            f"interaction onset {int(target_frame)}"
        )
    return int(candidates[0])


def _sequence_index_at_or_before(frame_indices: np.ndarray, target_frame: int) -> int:
    """Map an end bound inward to the last available sequence frame."""

    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    candidates = np.flatnonzero(frame_indices <= int(target_frame))
    if not candidates.size:
        raise ValueError(
            "hand trajectory rejected because HaMeR has no frame at or before "
            f"interaction end {int(target_frame)}"
        )
    return int(candidates[-1])


def calibrate_mesh_sequence_to_object_contact(
    vertices: np.ndarray,
    faces: np.ndarray,
    semantics: list[dict[str, Any]],
    frame_indices: np.ndarray,
    depths: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    object_masks: Optional[np.ndarray],
    *,
    movement_start_frame: Optional[int] = None,
    movement_end_frame: Optional[int] = None,
    contact_finger: Optional[str] = None,
    projection_radius: int = 15,
    scale_bounds: tuple[float, float] = (0.2, 5.0),
    release_delta_m: float = 0.05,
    max_outside_fraction: float = 0.30,
    strict_contact: bool = False,
) -> HandMeshDepthCalibration:
    """Calibrate HaMeR meshes using object-contact scale/translation anchors.

    This mirrors the main-branch HaMeR calibration strategy: estimate a start
    contact scale from fingertip/object-mask overlap, re-estimate the same
    contact finger at release, hold the start scale/translation globally, and
    apply only the release-local translation drift correction. The returned
    vertices are the only meshes that the renderer should consume.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape [T,V,3], got {vertices.shape}")
    if len(frame_indices) != vertices.shape[0]:
        raise ValueError("frame_indices length must match vertices.shape[0]")
    if len(semantics) != vertices.shape[0]:
        raise ValueError("semantics length must match vertices.shape[0]")
    if not len(frame_indices):
        raise ValueError("contact calibration received no HaMeR frames")
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError("HaMeR frame_indices must be strictly increasing")

    if depths is None or intrinsics is None or object_masks is None:
        if strict_contact:
            raise ValueError("contact calibration requires depth, intrinsics, and object masks")
        base = calibrate_mesh_sequence_to_depth(vertices, frame_indices, depths, intrinsics)
        base.method = "fallback_depth_projection_no_object_mask"
        return base

    depths_arr = np.asarray(depths, dtype=np.float64)
    if depths_arr.ndim != 3:
        raise ValueError(f"depths must have shape [T,H,W], got {depths_arr.shape}")

    first_frame = int(frame_indices[0])
    last_frame = int(frame_indices[-1])
    start_frame = first_frame if movement_start_frame is None else int(movement_start_frame)
    end_frame = last_frame if movement_end_frame is None else int(movement_end_frame)
    if end_frame < start_frame:
        end_frame = last_frame

    start_i = _sequence_index_at_or_after(frame_indices, start_frame)
    requested_end_i = _sequence_index_at_or_before(frame_indices, end_frame)
    if requested_end_i < start_i:
        raise ValueError(
            "hand trajectory rejected because HaMeR frames do not overlap the "
            f"interaction interval [{start_frame}, {end_frame}]"
        )
    end_i = requested_end_i

    def calibrate_at(seq_i: int, requested_finger: Optional[str]) -> dict[str, Any]:
        frame_idx = int(frame_indices[seq_i])
        depth_idx = max(0, min(frame_idx, depths_arr.shape[0] - 1))
        intr = _as_intrinsic(np.asarray(intrinsics), depth_idx)
        mask = _select_mask_frame(np.asarray(object_masks), depth_idx, depths_arr[depth_idx].shape[:2])
        result = _calibrate_contact_frame(
            vertices[seq_i],
            faces,
            semantics[seq_i],
            depths_arr[depth_idx],
            intr,
            mask,
            contact_finger=requested_finger,
            projection_radius=projection_radius,
            scale_bounds=scale_bounds,
        )
        result["frame_idx"] = frame_idx
        result["sequence_index"] = int(seq_i)
        return result

    start = calibrate_at(start_i, contact_finger)
    selected_finger = start.get("contact_finger") if start.get("ok") else contact_finger
    requested_end = calibrate_at(
        requested_end_i,
        str(selected_finger) if selected_finger else contact_finger,
    )
    end = requested_end
    release_contact_found = bool(end.get("ok"))
    release_contact_search_used = bool(start.get("ok") and not end.get("ok"))
    release_contact_search_attempts = 0

    # Equation (7)'s last active object-mask frame can occur after the hand has
    # released and withdrawn because the placed object remains outside its
    # initial footprint.  Resolve the paper's release anchor as the last
    # designated-fingertip contact before that mask-derived endpoint.
    if release_contact_search_used:
        for candidate_i in range(requested_end_i - 1, start_i, -1):
            release_contact_search_attempts += 1
            candidate = calibrate_at(candidate_i, str(selected_finger))
            if candidate.get("ok"):
                end_i = candidate_i
                end = candidate
                release_contact_found = True
                break

    if not start.get("ok") or not end.get("ok"):
        if strict_contact:
            raise ValueError(
                "hand contact calibration failed: "
                f"start={start.get('status')} end={end.get('status')}; "
                f"requested_end_frame={int(frame_indices[requested_end_i])} "
                f"release_search_attempts={release_contact_search_attempts}"
            )
        if not start.get("ok") and end.get("ok"):
            start = dict(end)
            start["frame_idx"] = int(frame_indices[start_i])
            start["sequence_index"] = int(start_i)
        if start.get("ok") and not end.get("ok"):
            end = dict(start)
            end["frame_idx"] = int(frame_indices[requested_end_i])
            end["sequence_index"] = int(requested_end_i)
            end_i = requested_end_i
    if not start.get("ok") or not end.get("ok"):
        base = calibrate_mesh_sequence_to_depth(vertices, frame_indices, depths, intrinsics)
        base.method = "fallback_depth_projection_no_contact"
        base.extra_summary = {
            "contact_calibration_start": start,
            "contact_calibration_end": end,
        }
        return base

    start_scale = float(start["scale"])
    end_scale = float(end["scale"])
    start_offset = np.asarray(start["offset"], dtype=np.float64).reshape(3)
    end_offset = np.asarray(end["offset"], dtype=np.float64).reshape(3)
    selected_finger = str(start.get("contact_finger") or end.get("contact_finger") or contact_finger or "")
    if selected_finger not in FINGER_JOINTS:
        raise ValueError("contact calibration did not identify an unambiguous contact finger")
    raw_tips = []
    for semantic in semantics:
        tip = semantic_finger_point(semantic, selected_finger, "tip")
        if tip is None:
            raise ValueError(f"HaMeR semantic output is missing the {selected_finger} fingertip")
        raw_tips.append(np.asarray(tip, dtype=np.float64).reshape(3))
    raw_tips_arr = np.stack(raw_tips, axis=0)
    baseline_tips = raw_tips_arr * start_scale + start_offset.reshape(1, 3)
    end_target = raw_tips_arr[end_i] * end_scale + end_offset
    drift_error = end_target - baseline_tips[end_i]
    distances_to_release = np.linalg.norm(
        baseline_tips - baseline_tips[end_i].reshape(1, 3), axis=1
    )
    correction_candidates = np.flatnonzero(
        (np.arange(len(frame_indices)) >= start_i)
        & (np.arange(len(frame_indices)) <= end_i)
        & (distances_to_release < float(release_delta_m))
    )
    correction_i = int(correction_candidates[0]) if correction_candidates.size else int(end_i)
    alphas = np.zeros(len(frame_indices), dtype=np.float64)
    if end_i <= correction_i:
        alphas[end_i:] = 1.0
    else:
        alphas[correction_i : end_i + 1] = np.linspace(0.0, 1.0, end_i - correction_i + 1)
        alphas[end_i + 1 :] = 1.0
    scales = np.full(vertices.shape[0], start_scale, dtype=np.float64)
    offsets = start_offset.reshape(1, 3) + alphas[:, None] * drift_error.reshape(1, 3)
    calibrated = vertices * scales[:, None, None] + offsets[:, None, :]
    status = ["ok"] * vertices.shape[0]
    bbox = np.zeros((vertices.shape[0], 2), dtype=np.float64)
    valid_depth_points = np.zeros(vertices.shape[0], dtype=np.int64)

    image_shapes = [
        tuple(int(x) for x in depths_arr[max(0, min(int(frame_idx), depths_arr.shape[0] - 1))].shape[:2])
        for frame_idx in frame_indices
    ]
    outside_fractions = _outside_frame_fractions(
        calibrated,
        frame_indices,
        np.asarray(intrinsics),
        image_shapes,
    )
    critical_outside = outside_fractions[start_i : end_i + 1]
    max_critical_outside = float(np.max(critical_outside)) if critical_outside.size else 1.0
    if max_critical_outside > float(max_outside_fraction):
        raise ValueError(
            "hand trajectory rejected because the projected hand leaves the image: "
            f"{max_critical_outside:.3f} > {float(max_outside_fraction):.3f}"
        )

    return HandMeshDepthCalibration(
        calibrated,
        scales,
        offsets,
        frame_indices,
        status,
        bbox,
        valid_depth_points,
        method="object_contact_scale_translation",
        extra_summary={
            "movement_start_frame": int(frame_indices[start_i]),
            "movement_end_frame": int(frame_indices[end_i]),
            "requested_movement_end_frame": int(end_frame),
            "requested_movement_end_hand_frame": int(frame_indices[requested_end_i]),
            "release_contact_frame": (
                int(frame_indices[end_i]) if release_contact_found else None
            ),
            "release_contact_search_used": release_contact_search_used,
            "release_contact_search_attempts": release_contact_search_attempts,
            "hand_mesh_scale_start": start_scale,
            "hand_mesh_scale_end": end_scale,
            "translation_offset_start": start_offset.astype(float).tolist(),
            "translation_offset_end": end_offset.astype(float).tolist(),
            "contact_finger": selected_finger,
            "release_correction_frame": int(frame_indices[correction_i]),
            "release_delta_m": float(release_delta_m),
            "release_drift_offset": drift_error.astype(float).tolist(),
            "outside_frame_fraction": outside_fractions.astype(float).tolist(),
            "max_critical_outside_fraction": max_critical_outside,
            "contact_calibration_start": {
                key: (value.astype(float).tolist() if isinstance(value, np.ndarray) else value)
                for key, value in start.items()
            },
            "contact_calibration_end": {
                key: (value.astype(float).tolist() if isinstance(value, np.ndarray) else value)
                for key, value in end.items()
            },
            "contact_calibration_end_requested": {
                key: (value.astype(float).tolist() if isinstance(value, np.ndarray) else value)
                for key, value in requested_end.items()
            },
        },
    )


def calibrate_mesh_sequence_to_prompted_contact(
    vertices: np.ndarray,
    faces: np.ndarray,
    semantics: list[dict[str, Any]],
    frame_indices: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    object_masks: np.ndarray,
    *,
    contact_finger: str,
    contact_point_2d: tuple[float, float] | list[float] | np.ndarray,
    contact_point_source_shape: Optional[tuple[int, int] | list[int] | np.ndarray] = None,
    movement_start_frame: Optional[int] = None,
    movement_end_frame: Optional[int] = None,
    projection_radius: int = 15,
    release_delta_m: float = 0.02,
    scale_bounds: tuple[float, float] = (0.2, 5.0),
    max_outside_fraction: float = 0.30,
) -> HandMeshDepthCalibration:
    """Paper D.4 calibration for a prompted non-prehensile contact.

    The annotated pixel anchors the prompted fingertip to a metric object
    surface point at contact onset.  That start scale/translation is applied to
    the full sequence; only a release-local translation correction is ramped in.
    """

    finger = canonical_contact_finger(contact_finger)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    depths_arr = np.asarray(depths, dtype=np.float64)
    masks_arr = np.asarray(object_masks)
    source_point = np.asarray(contact_point_2d, dtype=np.float64).reshape(-1)
    if source_point.size != 2 or not np.isfinite(source_point).all():
        raise ValueError("contact_point_2d must contain one finite [x,y] pixel")
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"vertices must have shape [T,V,3], got {vertices.shape}")
    if len(frame_indices) != len(vertices) or len(semantics) != len(vertices):
        raise ValueError("vertices, semantics, and frame_indices must have matching lengths")
    if depths_arr.ndim != 3:
        raise ValueError(f"depths must have shape [T,H,W], got {depths_arr.shape}")
    if not len(vertices):
        raise ValueError("prompted contact calibration received no hand frames")
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError("HaMeR frame_indices must be strictly increasing")

    depth_shape = tuple(int(value) for value in depths_arr.shape[1:3])
    source_shape = (
        tuple(int(value) for value in np.asarray(contact_point_source_shape).reshape(-1))
        if contact_point_source_shape is not None
        else depth_shape
    )
    point = scale_pixel_between_resolutions(source_point, source_shape, depth_shape)

    start_frame = int(frame_indices[0] if movement_start_frame is None else movement_start_frame)
    end_frame = int(frame_indices[-1] if movement_end_frame is None else movement_end_frame)
    start_i = _sequence_index_at_or_after(frame_indices, start_frame)
    requested_end_i = _sequence_index_at_or_before(frame_indices, end_frame)
    if requested_end_i < start_i:
        raise ValueError(
            "prompted hand trajectory rejected because HaMeR frames do not "
            f"overlap the interaction interval [{start_frame}, {end_frame}]"
        )
    end_i = requested_end_i

    start_depth_i = max(0, min(int(frame_indices[start_i]), depths_arr.shape[0] - 1))
    start_depth = depths_arr[start_depth_i]
    start_intr = _as_intrinsic(np.asarray(intrinsics), start_depth_i)
    start_mask = _select_mask_frame(masks_arr, start_depth_i, start_depth.shape[:2])
    start_pixel = _nearest_metric_mask_pixel(
        start_mask,
        start_depth,
        int(round(float(point[0]))),
        int(round(float(point[1]))),
        radius=projection_radius,
    )
    if start_pixel is None:
        raise ValueError("annotated recovery contact does not land on a metric target-object surface")
    start_u, start_v, start_pixel_distance = start_pixel
    start_contact_depth = float(start_depth[start_v, start_u])
    start_contact_xyz = unproject_pixels(
        np.asarray([[start_u, start_v]], dtype=np.float64),
        np.asarray([start_contact_depth], dtype=np.float64),
        start_intr,
    )[0]
    start_tip = semantic_finger_point(semantics[start_i], finger, "tip")
    if start_tip is None:
        raise ValueError(f"HaMeR semantic output is missing the prompted {finger} fingertip")
    start_tip = np.asarray(start_tip, dtype=np.float64).reshape(3)
    if not np.isfinite(start_tip).all() or start_tip[2] <= 1e-6:
        raise ValueError(f"prompted {finger} fingertip has invalid contact-onset geometry")
    start_scale = start_contact_depth / float(start_tip[2])
    if not (scale_bounds[0] <= start_scale <= scale_bounds[1]):
        raise ValueError(
            f"prompted fingertip scale {start_scale:.4f} is outside {scale_bounds}"
        )
    start_offset = start_contact_xyz - start_scale * start_tip

    raw_tips = []
    for semantic in semantics:
        tip = semantic_finger_point(semantic, finger, "tip")
        if tip is None:
            raise ValueError(f"HaMeR semantic output is missing the prompted {finger} fingertip")
        raw_tips.append(np.asarray(tip, dtype=np.float64).reshape(3))
    raw_tips_arr = np.stack(raw_tips, axis=0)
    baseline_tips = raw_tips_arr * start_scale + start_offset.reshape(1, 3)

    def release_contact_at(seq_i: int) -> dict[str, Any]:
        depth_i = max(0, min(int(frame_indices[seq_i]), depths_arr.shape[0] - 1))
        depth = depths_arr[depth_i]
        intr = _as_intrinsic(np.asarray(intrinsics), depth_i)
        mask = _select_mask_frame(masks_arr, depth_i, depth.shape[:2])
        uv = project_points(baseline_tips[seq_i : seq_i + 1], intr)[0]
        result: dict[str, Any] = {
            "ok": False,
            "status": "unprojectable_fingertip",
            "frame_idx": int(frame_indices[seq_i]),
            "sequence_index": int(seq_i),
        }
        if not np.isfinite(uv).all():
            return result
        pixel = _nearest_metric_mask_pixel(
            mask,
            depth,
            int(round(float(uv[0]))),
            int(round(float(uv[1]))),
            radius=projection_radius,
        )
        if pixel is None:
            result["status"] = "no_metric_object_surface_contact"
            return result
        u, v, pixel_distance = pixel
        contact_depth = float(depth[v, u])
        contact_xyz = unproject_pixels(
            np.asarray([[u, v]], dtype=np.float64),
            np.asarray([contact_depth], dtype=np.float64),
            intr,
        )[0]
        result.update(
            {
                "ok": True,
                "status": "ok",
                "u": int(u),
                "v": int(v),
                "pixel_distance": float(pixel_distance),
                "contact_xyz": contact_xyz,
            }
        )
        return result

    requested_release = release_contact_at(requested_end_i)
    release = requested_release
    release_contact_search_used = not bool(release.get("ok"))
    release_contact_search_attempts = 0
    if release_contact_search_used:
        for candidate_i in range(requested_end_i - 1, start_i, -1):
            release_contact_search_attempts += 1
            candidate = release_contact_at(candidate_i)
            if candidate.get("ok"):
                end_i = candidate_i
                release = candidate
                break
    if not release.get("ok"):
        raise ValueError(
            "prompted fingertip has no metric object-surface contact at release; "
            f"requested_end_frame={int(frame_indices[requested_end_i])} "
            f"release_search_attempts={release_contact_search_attempts}"
        )

    end_u = int(release["u"])
    end_v = int(release["v"])
    end_pixel_distance = float(release["pixel_distance"])
    end_contact_xyz = np.asarray(release["contact_xyz"], dtype=np.float64)
    drift_error = end_contact_xyz - baseline_tips[end_i]

    distances_to_release = np.linalg.norm(
        baseline_tips - baseline_tips[end_i].reshape(1, 3), axis=1
    )
    candidates = np.flatnonzero(
        (np.arange(len(frame_indices)) >= start_i)
        & (np.arange(len(frame_indices)) <= end_i)
        & (distances_to_release < float(release_delta_m))
    )
    correction_i = int(candidates[0]) if candidates.size else int(end_i)
    alphas = np.zeros(len(frame_indices), dtype=np.float64)
    if end_i <= correction_i:
        alphas[end_i:] = 1.0
    else:
        alphas[correction_i : end_i + 1] = np.linspace(0.0, 1.0, end_i - correction_i + 1)
        alphas[end_i + 1 :] = 1.0
    scales = np.full(len(vertices), start_scale, dtype=np.float64)
    offsets = start_offset.reshape(1, 3) + alphas[:, None] * drift_error.reshape(1, 3)
    calibrated = vertices * scales[:, None, None] + offsets[:, None, :]

    image_shapes = [
        tuple(int(x) for x in depths_arr[max(0, min(int(frame_idx), depths_arr.shape[0] - 1))].shape[:2])
        for frame_idx in frame_indices
    ]
    outside_fractions = _outside_frame_fractions(
        calibrated,
        frame_indices,
        np.asarray(intrinsics),
        image_shapes,
    )
    critical_outside = outside_fractions[start_i : end_i + 1]
    max_critical_outside = float(np.max(critical_outside)) if critical_outside.size else 1.0
    if max_critical_outside > float(max_outside_fraction):
        raise ValueError(
            "prompted hand trajectory rejected because the hand leaves the image: "
            f"{max_critical_outside:.3f} > {float(max_outside_fraction):.3f}"
        )

    return HandMeshDepthCalibration(
        calibrated,
        scales,
        offsets,
        frame_indices,
        ["ok"] * len(vertices),
        np.zeros((len(vertices), 2), dtype=np.float64),
        np.zeros(len(vertices), dtype=np.int64),
        method="prompted_fingertip_contact",
        extra_summary={
            "movement_start_frame": int(frame_indices[start_i]),
            "movement_end_frame": int(frame_indices[end_i]),
            "requested_movement_end_frame": int(end_frame),
            "requested_movement_end_hand_frame": int(frame_indices[requested_end_i]),
            "release_contact_frame": int(frame_indices[end_i]),
            "release_contact_search_used": release_contact_search_used,
            "release_contact_search_attempts": release_contact_search_attempts,
            "release_contact_requested_status": str(requested_release.get("status")),
            "contact_finger": finger,
            "contact_point_2d_requested": source_point.astype(float).tolist(),
            "contact_point_source_shape": list(source_shape),
            "contact_point_2d_depth_resolution": point.astype(float).tolist(),
            "contact_point_depth_shape": list(depth_shape),
            "contact_point_2d_metric": [int(start_u), int(start_v)],
            "contact_point_distance_px": float(start_pixel_distance),
            "contact_point_xyz": start_contact_xyz.astype(float).tolist(),
            "release_contact_point_2d": [int(end_u), int(end_v)],
            "release_contact_distance_px": float(end_pixel_distance),
            "release_contact_point_xyz": end_contact_xyz.astype(float).tolist(),
            "hand_mesh_scale_start": float(start_scale),
            "hand_mesh_scale_end": float(start_scale),
            "translation_offset_start": start_offset.astype(float).tolist(),
            "translation_offset_end": (start_offset + drift_error).astype(float).tolist(),
            "release_correction_frame": int(frame_indices[correction_i]),
            "release_delta_m": float(release_delta_m),
            "release_drift_offset": drift_error.astype(float).tolist(),
            "outside_frame_fraction": outside_fractions.astype(float).tolist(),
            "max_critical_outside_fraction": max_critical_outside,
        },
    )


def transform_semantic_xyz(semantic: dict[str, Any], scale: float, offset: np.ndarray) -> dict[str, Any]:
    """Transform semantic xyz."""
    offset = np.asarray(offset, dtype=np.float64).reshape(3)

    def transform_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: transform_value(child) for key, child in value.items()}
        if isinstance(value, list):
            arr = np.asarray(value, dtype=object)
            try:
                numeric = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return [transform_value(child) for child in value]
            if numeric.shape == (3,):
                return (numeric * float(scale) + offset).tolist()
            if numeric.ndim == 2 and numeric.shape[1] >= 3:
                out = numeric.copy()
                out[:, :3] = out[:, :3] * float(scale) + offset.reshape(1, 3)
                return out.tolist()
            if arr.ndim == 1:
                return [transform_value(child) for child in value]
        return value

    transformed = transform_value(semantic)
    return transformed if isinstance(transformed, dict) else {}


def _array_semantic_keypoints(semantic: dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("keypoints_3d", "joints", "keypoints3d", "xyz", "points"):
        value = semantic.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 21 and arr.shape[1] >= 3:
            return arr[:, :3]
    return None


def semantic_keypoints(semantic: dict[str, Any]) -> Optional[np.ndarray]:
    """Extract keypoints."""
    keypoints = _array_semantic_keypoints(semantic)
    if keypoints is not None:
        return keypoints

    fingers = semantic.get("fingers")
    if not isinstance(fingers, dict):
        return None

    def point_xyz(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            point = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        return point[:3] if point.size >= 3 else None

    wrist = point_xyz(semantic.get("wrist"))
    if wrist is None:
        return None

    # Main-branch HaMeR artifacts store semantic joints by name.
    # Reconstruct the canonical MANO-21 ordering while retaining NaN markers
    # for joints that were not emitted.
    keypoints = np.full((21, 3), np.nan, dtype=np.float64)
    keypoints[0] = wrist
    for finger, joints in FINGER_JOINTS.items():
        finger_data = fingers.get(finger)
        if not isinstance(finger_data, dict):
            continue
        for joint, index in joints.items():
            point = point_xyz(finger_data.get(joint))
            if point is not None:
                keypoints[index] = point
    return keypoints


def semantic_finger_point(semantic: dict[str, Any], finger: str, joint: str) -> Optional[np.ndarray]:
    """Extract finger point."""
    fingers = semantic.get("fingers")
    if isinstance(fingers, dict):
        finger_data = fingers.get(finger)
        if isinstance(finger_data, dict) and finger_data.get(joint) is not None:
            try:
                arr = np.asarray(finger_data[joint], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                arr = np.empty(0, dtype=np.float64)
            if arr.size >= 3 and np.isfinite(arr[:3]).all():
                return arr[:3]
    # Keep the array-schema lookup separate from synthesized nested keypoints
    # so an absent nested fingertip cannot become a NaN translation.
    keypoints = _array_semantic_keypoints(semantic)
    idx = FINGER_JOINTS.get(finger, {}).get(joint)
    if keypoints is not None and idx is not None and idx < len(keypoints):
        point = np.asarray(keypoints[idx], dtype=np.float64).reshape(-1)
        if point.size >= 3 and np.isfinite(point[:3]).all():
            return point[:3]
    return None


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return v / norm


def palm_frame_from_semantic(semantic: dict[str, Any]) -> Optional[np.ndarray]:
    """Construct a palm frame from semantic hand keypoints."""
    keypoints = semantic_keypoints(semantic)
    if keypoints is None:
        return None

    wrist = np.asarray(keypoints[0], dtype=np.float64)
    mcp_by_finger = {
        finger: np.asarray(keypoints[joints["mcp"]], dtype=np.float64)
        for finger, joints in FINGER_JOINTS.items()
    }
    palm_points = [wrist, *mcp_by_finger.values()]
    palm_points = [point for point in palm_points if np.isfinite(point).all()]
    if len(palm_points) < 3 or not np.isfinite(wrist).all():
        return None

    # Paper D.2 and the main-branch implementation estimate the palm normal by
    # fitting a plane to the wrist and every MCP joint, rather than relying on
    # a single three-landmark cross product.
    centered = np.stack(palm_points, axis=0)
    centered -= np.mean(centered, axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = _normalize(vt[-1])
    if normal is None:
        return None

    # SVD leaves the normal sign ambiguous. Preserve the release's anatomical
    # convention whenever the index/pinky landmarks provide one.
    index_mcp = mcp_by_finger["index"]
    pinky_mcp = mcp_by_finger["pinky"]
    if np.isfinite(index_mcp).all() and np.isfinite(pinky_mcp).all():
        anatomical_normal = _normalize(
            np.cross(index_mcp - wrist, pinky_mcp - wrist)
        )
        if anatomical_normal is not None and float(np.dot(normal, anatomical_normal)) < 0.0:
            normal = -normal

    # Use the wrist-to-middle-MCP direction projected into the fitted plane,
    # with the same fallbacks as the main implementation.
    middle_mcp = mcp_by_finger["middle"]
    if np.isfinite(middle_mcp).all():
        x_direction = middle_mcp - wrist
    elif np.isfinite(index_mcp).all():
        x_direction = index_mcp - wrist
    else:
        finite_mcps = [
            point for point in mcp_by_finger.values() if np.isfinite(point).all()
        ]
        if not finite_mcps:
            return None
        x_direction = np.mean(finite_mcps, axis=0) - wrist

    x_direction = x_direction - float(np.dot(x_direction, normal)) * normal
    x_axis = _normalize(x_direction)
    if x_axis is None:
        arbitrary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(normal, arbitrary))) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = _normalize(arbitrary - float(np.dot(arbitrary, normal)) * normal)
    if x_axis is None:
        return None

    y_axis = _normalize(np.cross(normal, x_axis))
    if y_axis is None:
        return None
    x_axis = _normalize(np.cross(y_axis, normal))
    if x_axis is None:
        return None
    rotation = np.stack([x_axis, y_axis, normal], axis=1)
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0
    return rotation


def hand_pose_from_semantic(
    semantic: dict[str, Any],
    vertices: np.ndarray,
    *,
    contact_finger: str = "index",
) -> np.ndarray:
    """Construct a hand pose from semantic hand keypoints."""
    pose = np.eye(4, dtype=np.float64)
    tip = semantic_finger_point(semantic, contact_finger, "tip")
    if tip is None:
        keypoints = _array_semantic_keypoints(semantic)
        if keypoints is not None and keypoints.shape[0] > 8:
            fallback_tip = np.asarray(keypoints[8], dtype=np.float64).reshape(-1)
            if fallback_tip.size >= 3 and np.isfinite(fallback_tip[:3]).all():
                tip = fallback_tip[:3]
    if tip is None:
        verts = np.asarray(vertices, dtype=np.float64)
        finite = np.isfinite(verts).all(axis=1)
        tip = np.median(verts[finite], axis=0) if np.any(finite) else np.zeros(3, dtype=np.float64)
    pose[:3, 3] = np.asarray(tip, dtype=np.float64).reshape(3)

    rotation = palm_frame_from_semantic(semantic)
    if rotation is not None and np.isfinite(rotation).all():
        pose[:3, :3] = rotation
    return pose
