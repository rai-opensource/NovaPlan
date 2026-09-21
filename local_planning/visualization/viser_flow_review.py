#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Interactive Viser review for one NovaPlan closed-loop execution step."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from novaplan.cli_args import positive_int, tcp_port
from novaplan.execution_step import load_flow_tracks
from novaplan.object_flow import estimate_rigid_object_flow


def _import_viser():
    try:
        import viser
    except ImportError as exc:
        raise SystemExit(
            "viser is required for --debug_flow_review. Run with the viz/full environment, e.g. "
            "`pixi run -e local-planning-full closed-loop-execution ...`."
        ) from exc
    return viser


def _load_summary(execution_dir: Path) -> dict:
    summary_path = execution_dir / "execution_step.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _normalize_video(video: np.ndarray) -> np.ndarray:
    video = np.asarray(video)
    if video.ndim != 4:
        raise ValueError(f"video must have shape [T,C,H,W] or [T,H,W,C], got {video.shape}")
    if video.shape[1] in {1, 3, 4}:
        video = np.moveaxis(video, 1, -1)
    if video.dtype != np.uint8:
        if np.nanmax(video) <= 1.5:
            video = video * 255.0
        video = np.clip(video, 0, 255).astype(np.uint8)
    if video.shape[-1] > 3:
        video = video[..., :3]
    if video.shape[-1] == 1:
        video = np.repeat(video, 3, axis=-1)
    return video


def _camera_point_cloud(
    video: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    *,
    frame_idx: int = 0,
    stride: int = 6,
    max_points: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = _normalize_video(video)[frame_idx]
    depth = np.asarray(depths, dtype=np.float64)[frame_idx]
    if intrinsics.ndim == 3:
        intr = intrinsics[frame_idx]
    else:
        intr = intrinsics
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])

    h, w = depth.shape[:2]
    if rgb.shape[:2] != (h, w):
        ys_rgb = (np.arange(0, h, stride) * (rgb.shape[0] / h)).astype(int).clip(0, rgb.shape[0] - 1)
        xs_rgb = (np.arange(0, w, stride) * (rgb.shape[1] / w)).astype(int).clip(0, rgb.shape[1] - 1)
    else:
        ys_rgb = np.arange(0, h, stride)
        xs_rgb = np.arange(0, w, stride)

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 1e-6)
    x = (grid_x - cx) * z / fx
    y = (grid_y - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)[valid]

    rgb_x, rgb_y = np.meshgrid(xs_rgb, ys_rgb)
    colors = rgb[rgb_y, rgb_x][valid]
    if len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[keep]
        colors = colors[keep]
    return points, colors


def _load_flow_tracks(
    flow_npz: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    flow = load_flow_tracks(flow_npz)
    return flow.coords, flow.visibilities, flow.point_origins


def _load_hand_motion_tracks(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
        raw = loaded.item()
    else:
        raw = loaded

    if isinstance(raw, dict):
        ordered_keys = [
            "wrist",
            "thumb_tip",
            "thumb_mcp",
            "index_tip",
            "index_mcp",
            "middle_tip",
            "middle_mcp",
            "ring_tip",
            "ring_mcp",
            "pinky_tip",
            "pinky_mcp",
        ]
        tracks = []
        for key in ordered_keys:
            value = raw.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 3:
                tracks.append(arr)
        if tracks:
            min_t = min(track.shape[0] for track in tracks)
            return np.stack([track[:min_t] for track in tracks], axis=1)
        for key in ("position",):
            value = raw.get(key)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr[:, None, :]
        for key in ("T_world_from_hand", "T_camera_from_hand", "hand_poses", "ee_poses", "poses"):
            value = raw.get(key)
            if value is not None:
                arr = np.asarray(value, dtype=np.float64)
                if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                    return arr[:, None, :3, 3]
        return None

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr[:, None, :3, 3]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr[:, None, :]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    return None


def _load_hand_motion_frame_indices(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    loaded = np.load(path, allow_pickle=True)
    if not (isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object):
        return None
    raw = loaded.item()
    if not isinstance(raw, dict) or "pose_frame_indices" not in raw:
        return None
    indices = np.asarray(raw["pose_frame_indices"], dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        return None
    return indices


def _frame_idx_from_mesh_name(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("frame"):
        return 0
    digits = []
    for char in stem[len("frame") :]:
        if not char.isdigit():
            break
        digits.append(char)
    return int("".join(digits)) if digits else 0


@dataclass
class HandMeshSequence:
    """Store a time-indexed hand mesh sequence for visualization."""
    vertices: np.ndarray
    faces: np.ndarray
    frame_indices: np.ndarray


def _load_hamer_mesh_sequence(mesh_dir: Path) -> Optional[HandMeshSequence]:
    if not mesh_dir.exists():
        return None

    records = []
    for path in sorted(mesh_dir.glob("frame*_hand*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                vertices = np.asarray(data["vertices"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int32)
                frame_idx = int(data["frame_idx"]) if "frame_idx" in data else _frame_idx_from_mesh_name(path)
                hand_id = int(data["hand_id"]) if "hand_id" in data else 0
                is_right = bool(int(data["is_right"])) if "is_right" in data else True
        except Exception:
            continue
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) == 0
            or faces.ndim != 2
            or faces.shape[1] != 3
            or len(faces) == 0
        ):
            continue
        records.append(
            {
                "frame_idx": frame_idx,
                "hand_id": hand_id,
                "is_right": is_right,
                "vertices": vertices,
                "faces": faces,
            }
        )
    if not records:
        return None

    grouped_by_side: dict[bool, list[dict]] = {}
    for record in records:
        grouped_by_side.setdefault(bool(record["is_right"]), []).append(record)
    _selected_side, selected_records = max(
        grouped_by_side.items(),
        key=lambda item: (len(item[1]), int(item[0])),
    )

    by_frame: dict[int, dict] = {}
    for record in selected_records:
        frame_idx = int(record["frame_idx"])
        current = by_frame.get(frame_idx)
        if current is None or int(record["hand_id"]) < int(current["hand_id"]):
            by_frame[frame_idx] = record
    ordered = [by_frame[idx] for idx in sorted(by_frame)]
    min_vertices = min(int(record["vertices"].shape[0]) for record in ordered)
    if min_vertices <= 0:
        return None

    vertices = np.stack([record["vertices"][:min_vertices] for record in ordered], axis=0)
    faces = np.asarray(ordered[0]["faces"], dtype=np.int32)
    faces = faces[np.all((faces >= 0) & (faces < min_vertices), axis=1)]
    if len(faces) == 0:
        return None
    frame_indices = np.asarray([int(record["frame_idx"]) for record in ordered], dtype=np.int64)
    return HandMeshSequence(vertices=vertices, faces=faces, frame_indices=frame_indices)


def _find_hamer_mesh_dir(hand_path: Path) -> Optional[Path]:
    candidates = [
        hand_path.parent / "calibrated_mesh_dump",
        hand_path.parent / "real_mesh_dump",
        hand_path.parent / "hamer_outputs" / "calibrated_mesh_dump",
        hand_path.parent / "hamer_outputs" / "real_mesh_dump",
        hand_path.parent.parent / "hamer_outputs" / "calibrated_mesh_dump",
        hand_path.parent.parent / "hamer_outputs" / "real_mesh_dump",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _create_rainbow_colors(normalized_values: np.ndarray) -> np.ndarray:
    """Same custom red-orange-yellow-green-blue-purple-magenta map as mola_viz."""
    colors_list = np.array(
        [
            (1.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.5),
            (1.0, 0.0, 1.0),
        ],
        dtype=np.float64,
    )
    normalized_values = np.asarray(normalized_values, dtype=np.float64)
    xp = np.linspace(0.0, 1.0, len(colors_list))
    clipped = np.clip(normalized_values, 0.0, 1.0)
    return np.stack(
        [
            np.interp(clipped, xp, colors_list[:, 0]),
            np.interp(clipped, xp, colors_list[:, 1]),
            np.interp(clipped, xp, colors_list[:, 2]),
        ],
        axis=1,
    )


def _valid_points_mask(
    frame_idx: int,
    coords: np.ndarray,
    vis: Optional[np.ndarray],
    point_origins: Optional[np.ndarray] = None,
) -> np.ndarray:
    points = coords[frame_idx]
    mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
    if vis is not None and vis.shape == coords.shape[:2]:
        mask &= vis[frame_idx].astype(bool)
        if point_origins is not None:
            mask &= point_origins <= frame_idx
    return mask


def _safe_remove(handle: Optional[object]) -> None:
    if handle is None:
        return
    try:
        handle.remove()
    except Exception:
        pass


def _safe_float(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if np.isfinite(parsed) else fallback


def _needs_numeric_reset(value: object, expected: float) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return True
    return (not np.isfinite(parsed)) or parsed != expected


def _safe_int(value: object, fallback: int, *, min_value: int, max_value: int) -> int:
    parsed = int(round(_safe_float(value, float(fallback))))
    return max(min_value, min(max_value, parsed))


def _build_current_trails(
    coords: np.ndarray,
    vis: Optional[np.ndarray],
    point_origins: Optional[np.ndarray],
    colors: np.ndarray,
    frame_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    trail_segments = []
    trail_colors = []
    frame_pairs = [(idx - 1, idx) for idx in range(1, frame_idx + 1)]

    for prev_idx, curr_idx in frame_pairs:
        curr_points = coords[curr_idx]
        prev_points = coords[prev_idx]
        if vis is not None and vis.shape == coords.shape[:2]:
            valid_mask = vis[curr_idx].astype(bool) & vis[prev_idx].astype(bool)
            if point_origins is not None:
                valid_mask &= point_origins <= prev_idx
        else:
            curr_nonzero = np.linalg.norm(curr_points, axis=1) > 1e-6
            prev_nonzero = np.linalg.norm(prev_points, axis=1) > 1e-6
            valid_mask = curr_nonzero & prev_nonzero
        valid_mask &= np.isfinite(curr_points).all(axis=1) & np.isfinite(prev_points).all(axis=1)
        if not np.any(valid_mask):
            continue

        segments = np.stack([prev_points[valid_mask], curr_points[valid_mask]], axis=1)
        segment_colors = np.stack([colors[valid_mask], colors[valid_mask]], axis=1)
        trail_segments.append(segments)
        trail_colors.append(segment_colors)

    if not trail_segments:
        return np.zeros((0, 2, 3), dtype=np.float64), np.zeros((0, 2, 3), dtype=np.float64)
    return np.concatenate(trail_segments, axis=0), np.concatenate(trail_colors, axis=0)


def _rigid_tracks_from_transforms(coords: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Move frame-zero object points with the fitted camera-frame transforms."""
    coords = np.asarray(coords, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [T,N,3], got {coords.shape}")
    if transforms.shape != (coords.shape[0], 4, 4):
        raise ValueError(
            f"transforms must have shape [{coords.shape[0]},4,4], got {transforms.shape}"
        )
    base_points = coords[0]
    return np.einsum("tij,nj->tni", transforms[:, :3, :3], base_points) + transforms[:, None, :3, 3]


def _load_saved_object_transforms(execution_dir: Path, num_frames: int) -> Optional[np.ndarray]:
    bundle_path = execution_dir / "execution_step.npz"
    if not bundle_path.exists():
        return None
    with np.load(bundle_path, allow_pickle=False) as data:
        if "object_motion_transforms" not in data:
            return None
        transforms = np.asarray(data["object_motion_transforms"], dtype=np.float64)
    if transforms.shape != (num_frames, 4, 4):
        return None
    return transforms


def _spatial_track_order(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a deterministic farthest-point order for an evenly spread display subset."""
    points = np.asarray(points, dtype=np.float64)
    candidates = np.flatnonzero(np.asarray(valid, dtype=bool))
    if len(candidates) <= 1:
        return candidates

    candidate_points = points[candidates]
    centroid = candidate_points.mean(axis=0)
    first = int(np.argmax(np.sum((candidate_points - centroid) ** 2, axis=1)))
    selected_local = np.empty(len(candidates), dtype=np.int64)
    selected_local[0] = first
    min_distances = np.sum((candidate_points - candidate_points[first]) ** 2, axis=1)
    min_distances[first] = -1.0
    for slot in range(1, len(candidates)):
        next_local = int(np.argmax(min_distances))
        selected_local[slot] = next_local
        distances = np.sum((candidate_points - candidate_points[next_local]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected_local[: slot + 1]] = -1.0
    return candidates[selected_local]


@dataclass
class FlowLayer:
    """Store one visualized object- or hand-flow layer."""
    name: str
    coords: np.ndarray
    vis: Optional[np.ndarray]
    point_origins: Optional[np.ndarray]
    colors: np.ndarray
    frame_indices: Optional[np.ndarray]
    display_order: np.ndarray
    raw_coords: Optional[np.ndarray] = None
    mesh_faces: Optional[np.ndarray] = None
    point_size_scale: float = 1.0
    line_width_scale: float = 1.0


class FlowReviewServer:
    """Persistent flow-only reviewer with frame playback controls."""

    def __init__(self, *, port: int = 8097, point_stride: int = 6, max_tracks: int = 20000) -> None:
        viser = _import_viser()
        self.server = viser.ViserServer(port=port)
        self.port = int(port)
        self.point_stride = int(point_stride)
        self.max_tracks = int(max_tracks)

        self.server.scene.add_grid("/ground", width=2, height=2)
        self.server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.005)

        self.layers: dict[str, FlowLayer] = {}
        self.video: Optional[np.ndarray] = None
        self.depths: Optional[np.ndarray] = None
        self.intrinsics: Optional[np.ndarray] = None
        self.num_frames = 1
        self.current_frame = 0
        self.is_playing = False
        self._playback_armed = False
        self._playback_generation = 0
        self._playback_timer: Optional[threading.Timer] = None
        self._done: Optional[threading.Event] = None
        self._decision: Optional[dict] = None

        self._step_handles: list[object] = []
        self._scene_handle: Optional[object] = None
        self._point_handles: dict[str, object] = {}
        self._trail_handles: dict[str, object] = {}
        self._mesh_handles: dict[str, list[object]] = {}
        self._flow_layer_options: tuple[str, ...] = ("object",)

        with self.server.gui.add_folder("Flow Review"):
            self.step_text = self.server.gui.add_text("Step", "")
            self.selected_flow_text = self.server.gui.add_text("Selected Flow", "")
            self.object_valid_text = self.server.gui.add_text("Object Valid", "")
            self.switch_text = self.server.gui.add_text("Using Hand Flow", "")
            self.reason_text = self.server.gui.add_text("Reason", "")
            self.continue_button = self.server.gui.add_button("Continue to next step")
            self.stop_button = self.server.gui.add_button("Stop execution")

        with self.server.gui.add_folder("Visualization Controls"):
            self.flow_layer_dropdown = self.server.gui.add_dropdown(
                "Flow Layer",
                ("object",),
                initial_value="object",
                hint="Choose which flow layer to render.",
            )
            self.frame_slider = self.server.gui.add_slider("Frame", min=0.0, max=1.0, step=1.0, initial_value=0.0)
            self.scene_point_size_slider = self.server.gui.add_slider(
                "RGB Point Size",
                min=0.001,
                max=0.06,
                step=0.001,
                initial_value=0.008,
            )
            self.trajectory_point_size_slider = self.server.gui.add_slider(
                "Trajectory Point Size",
                min=0.002,
                max=0.06,
                step=0.001,
                initial_value=0.01,
            )
            self.object_trajectory_dropdown = self.server.gui.add_dropdown(
                "Object Trajectory",
                ("Rigid fit", "Raw TAPIP3D"),
                initial_value="Rigid fit",
                hint="Rigid fit is the object motion used for execution; raw tracks are diagnostic.",
            )
            self.trail_line_width_slider = self.server.gui.add_slider(
                "Trail Line Width",
                min=0.05,
                max=5.0,
                step=0.05,
                initial_value=1.5,
            )
            self.displayed_tracks_slider = self.server.gui.add_slider(
                "Displayed Tracks",
                min=1.0,
                max=1024.0,
                step=1.0,
                initial_value=256.0,
            )
            self.hand_mesh_ghost_slider = self.server.gui.add_slider(
                "Hand Mesh Ghosts",
                min=1.0,
                max=24.0,
                step=1.0,
                initial_value=12.0,
            )
            self.hand_mesh_opacity_slider = self.server.gui.add_slider(
                "Hand Mesh Opacity",
                min=0.05,
                max=1.0,
                step=0.05,
                initial_value=0.75,
            )
            self.play_button = self.server.gui.add_button("Play")
            self.pause_button = self.server.gui.add_button("Pause")
            self.reset_button = self.server.gui.add_button("Reset")

        self.flow_layer_dropdown.on_update(lambda _: self._on_flow_layer_update())
        self.frame_slider.on_update(lambda _: self._on_frame_slider())
        self.scene_point_size_slider.on_update(lambda _: self.update_visuals())
        self.trajectory_point_size_slider.on_update(lambda _: self.update_visuals())
        self.object_trajectory_dropdown.on_update(lambda _: self.update_visuals())
        self.trail_line_width_slider.on_update(lambda _: self.update_visuals())
        self.displayed_tracks_slider.on_update(lambda _: self.update_visuals())
        self.hand_mesh_ghost_slider.on_update(lambda _: self.update_visuals())
        self.hand_mesh_opacity_slider.on_update(lambda _: self.update_visuals())
        self.play_button.on_click(lambda _: self._on_play())
        self.pause_button.on_click(lambda _: self._on_pause())
        self.reset_button.on_click(lambda _: self._on_reset())
        self.continue_button.on_click(lambda _: self._set_decision(True, "continue"))
        self.stop_button.on_click(lambda _: self._set_decision(False, "stop"))
        self.server.on_client_connect(lambda _: self._schedule_armed_playback())

        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        print(f"[flow-review] Persistent Viser server running at http://127.0.0.1:{self.port}")

    def review_step(
        self,
        *,
        sample_dir: Path,
        execution_dir: Path,
        flow_npz: Path,
        step: int,
        decision_path: Path,
        label: Optional[str] = None,
    ) -> dict:
        """Launch interactive review for one execution step."""
        del sample_dir
        summary = _load_summary(execution_dir)
        flow_data = _load_npz(flow_npz)
        video = flow_data.get("video")
        depths = flow_data.get("depths")
        intrinsics = flow_data.get("intrinsics")
        if video is None or depths is None or intrinsics is None:
            raise RuntimeError(f"{flow_npz} must include video, depths, and intrinsics for scene pointcloud review.")

        # Reset playback before replacing layers. GUI updates triggered while the
        # new step is configured must never reuse the previous step's final frame.
        self._cancel_playback_timer()
        self._playback_armed = False
        self.is_playing = False
        self.current_frame = 0
        self._clear_step_scene()
        self.video = video
        self.depths = depths
        self.intrinsics = intrinsics

        selected_flow = summary.get("selected_flow")
        artifact_paths = summary.get("artifact_paths") or {}
        layers: dict[str, FlowLayer] = {}
        object_coords, object_vis, object_origins = _load_flow_tracks(flow_npz)
        object_transforms = _load_saved_object_transforms(execution_dir, len(object_coords))
        if object_transforms is None:
            fit = estimate_rigid_object_flow(
                object_coords,
                object_vis,
                layout="time_major",
                point_origins=object_origins,
                min_visible_points=int((summary.get("metadata") or {}).get("min_visible_points", 6)),
                visibility_threshold=float((summary.get("metadata") or {}).get("visibility_threshold", 0.5)),
            )
            object_transforms = fit.transforms
        rigid_object_coords = _rigid_tracks_from_transforms(object_coords, object_transforms)
        layers["object"] = self._build_layer(
            name="object",
            coords=rigid_object_coords,
            raw_coords=object_coords,
            vis=object_vis,
            point_origins=object_origins,
            frame_indices=None,
            point_size_scale=1.0,
            line_width_scale=1.0,
        )
        if selected_flow == "hand":
            hand_path = artifact_paths.get("hand_poses_input") or artifact_paths.get("hand_poses")
            hand_coords = None
            hand_frame_indices = None
            hand_mesh_faces = None
            if hand_path:
                resolved_hand_path = Path(hand_path)
                mesh_dir = _find_hamer_mesh_dir(resolved_hand_path)
                mesh_sequence = _load_hamer_mesh_sequence(mesh_dir) if mesh_dir is not None else None
                if mesh_sequence is not None:
                    hand_coords = mesh_sequence.vertices
                    hand_frame_indices = mesh_sequence.frame_indices
                    hand_mesh_faces = mesh_sequence.faces
                    if mesh_dir is not None and mesh_dir.name != "calibrated_mesh_dump":
                        print(
                            "[flow-review] WARNING: displaying raw HaMeR meshes because no calibrated_mesh_dump "
                            "was found. No render-time calibration is applied."
                        )
                else:
                    hand_coords = _load_hand_motion_tracks(resolved_hand_path)
                    hand_frame_indices = _load_hand_motion_frame_indices(resolved_hand_path)
            if hand_coords is not None:
                layers["hand"] = self._build_layer(
                    name="hand",
                    coords=hand_coords,
                    vis=None,
                    point_origins=None,
                    frame_indices=hand_frame_indices,
                    mesh_faces=hand_mesh_faces,
                    point_size_scale=1.35,
                    line_width_scale=1.2,
                )
        self._load_layers(layers)
        self._load_summary_text(summary, label or str(step))
        self._configure_layer_dropdown(selected_flow)

        self.current_frame = 0
        self.frame_slider.max = float(max(1, self.num_frames - 1))
        self.frame_slider.value = 0.0
        self.is_playing = False
        self._cancel_playback_timer()
        self._playback_generation += 1
        self._playback_armed = True
        self.update_visuals()
        if self.server.get_clients():
            self._schedule_armed_playback()

        self._done = threading.Event()
        self._decision = {"continue": False, "step": int(step), "label": label or str(step)}
        try:
            while self._done is not None and not self._done.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            self._set_decision(False, "keyboard_interrupt")

        decision = dict(self._decision or {"continue": False, "step": int(step), "clicked": "missing_decision"})
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, indent=2))
        print(f"[flow-review] decision: {decision}")
        return {
            "status": "completed",
            "decision": decision,
            "decision_path": str(decision_path),
            "url": f"http://127.0.0.1:{self.port}",
        }

    def _build_layer(
        self,
        *,
        name: str,
        coords: np.ndarray,
        raw_coords: Optional[np.ndarray] = None,
        vis: Optional[np.ndarray],
        point_origins: Optional[np.ndarray],
        frame_indices: Optional[np.ndarray],
        point_size_scale: float,
        line_width_scale: float,
        mesh_faces: Optional[np.ndarray] = None,
    ) -> FlowLayer:
        num_frames, num_tracks = coords.shape[:2]
        track_indices = np.arange(num_tracks)
        if mesh_faces is None and num_tracks > self.max_tracks:
            track_indices = track_indices[np.linspace(0, num_tracks - 1, self.max_tracks).astype(int)]
        layer_coords = coords[:, track_indices, :]
        layer_raw_coords = raw_coords[:, track_indices, :] if raw_coords is not None else None
        layer_vis = vis[:, track_indices] if vis is not None and vis.shape[1] == num_tracks else None
        layer_origins = (
            np.asarray(point_origins, dtype=np.int64).reshape(-1)[track_indices]
            if point_origins is not None
            else None
        )
        layer_frame_indices = None
        if frame_indices is not None:
            layer_frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
            keep_t = min(len(layer_frame_indices), layer_coords.shape[0])
            layer_coords = layer_coords[:keep_t]
            layer_raw_coords = layer_raw_coords[:keep_t] if layer_raw_coords is not None else None
            layer_vis = layer_vis[:keep_t] if layer_vis is not None else None
            layer_frame_indices = layer_frame_indices[:keep_t]
        denom = max(1, num_tracks - 1)
        color_phase = 0.5 if name == "hand" else 0.0
        layer_colors = _create_rainbow_colors((track_indices / denom + color_phase) % 1.0)
        display_valid = np.isfinite(layer_coords[0]).all(axis=1)
        display_valid &= np.linalg.norm(layer_coords[0], axis=1) > 1e-6
        if layer_vis is not None:
            display_valid &= layer_vis[0].astype(bool)
        if layer_origins is not None:
            display_valid &= layer_origins <= 0
        display_order = _spatial_track_order(layer_coords[0], display_valid)
        return FlowLayer(
            name=name,
            coords=layer_coords,
            vis=layer_vis,
            point_origins=layer_origins,
            colors=layer_colors,
            frame_indices=layer_frame_indices,
            display_order=display_order,
            raw_coords=layer_raw_coords,
            mesh_faces=np.asarray(mesh_faces, dtype=np.int32) if mesh_faces is not None else None,
            point_size_scale=point_size_scale,
            line_width_scale=line_width_scale,
        )

    def _load_layers(self, layers: dict[str, FlowLayer]) -> None:
        self.layers = layers
        frame_counts = []
        if self.depths is not None:
            frame_counts.append(int(np.asarray(self.depths).shape[0]))
        for layer in layers.values():
            if layer.frame_indices is not None and len(layer.frame_indices) > 0:
                frame_counts.append(int(np.max(layer.frame_indices)) + 1)
            else:
                frame_counts.append(int(layer.coords.shape[0]))
        self.num_frames = max(1, max(frame_counts, default=1))

    def _configure_layer_dropdown(self, selected_flow: object) -> None:
        layer_names = tuple(self.layers.keys())
        options = list(layer_names)
        if not options:
            options = ["object"]

        previous_value = str(getattr(self.flow_layer_dropdown, "value", "object"))
        if selected_flow in self.layers:
            default_value = str(selected_flow)
        elif previous_value in options:
            default_value = previous_value
        elif "object" in options:
            default_value = "object"
        else:
            default_value = options[0]

        self._flow_layer_options = tuple(options)
        self.flow_layer_dropdown.options = self._flow_layer_options
        self.flow_layer_dropdown.value = default_value
        self.flow_layer_dropdown.disabled = len(options) <= 1
        self.object_trajectory_dropdown.disabled = default_value != "object"

    def _load_summary_text(self, summary: dict, step: object) -> None:
        flow_switch = summary.get("flow_switch") or {}
        object_eval = flow_switch.get("object_flow") or {}
        using_hand = flow_switch.get("switch_to_hand")
        if using_hand is None:
            using_hand = flow_switch.get("selected_is_hand")
        if using_hand is None:
            using_hand = summary.get("selected_flow") == "hand"
        self.step_text.value = str(step)
        self.selected_flow_text.value = str(summary.get("selected_flow"))
        self.object_valid_text.value = str(object_eval.get("valid"))
        self.switch_text.value = str(bool(using_hand))
        self.reason_text.value = str(flow_switch.get("reason") or object_eval.get("reason") or "")

    def _clear_step_scene(self) -> None:
        self._clear_dynamic_handles()
        for handle in self._step_handles:
            _safe_remove(handle)
        self._step_handles.clear()

    def _clear_dynamic_handles(self) -> None:
        _safe_remove(self._scene_handle)
        self._scene_handle = None
        for handle in self._point_handles.values():
            _safe_remove(handle)
        for handle in self._trail_handles.values():
            _safe_remove(handle)
        for handles in self._mesh_handles.values():
            for handle in handles:
                _safe_remove(handle)
        self._point_handles.clear()
        self._trail_handles.clear()
        self._mesh_handles.clear()

    @staticmethod
    def _handle_is_live(handle: Optional[object]) -> bool:
        impl = getattr(handle, "_impl", None)
        return handle is not None and not bool(getattr(impl, "removed", False))

    def _upsert_point_cloud_handle(
        self,
        handle: Optional[object],
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        *,
        point_size: float,
    ) -> object:
        points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        colors = np.asarray(colors)
        if colors.size == 0:
            colors = np.zeros((0, 3), dtype=np.uint8)
        colors = colors.reshape((-1, 3))

        if not self._handle_is_live(handle):
            return self.server.scene.add_point_cloud(
                name,
                points=points,
                colors=colors,
                point_size=point_size,
                point_shape="circle",
                precision="float32",
            )

        handle.points = points
        handle.colors = colors
        handle.point_size = point_size
        handle.visible = len(points) > 0
        return handle

    def _upsert_line_segments_handle(
        self,
        handle: Optional[object],
        name: str,
        points: np.ndarray,
        colors: np.ndarray,
        *,
        line_width: float,
    ) -> object:
        points = np.asarray(points, dtype=np.float32).reshape((-1, 2, 3))
        colors = np.asarray(colors)
        if colors.size == 0:
            colors = np.zeros((0, 2, 3), dtype=np.uint8)
        colors = colors.reshape((-1, 2, 3))

        if not self._handle_is_live(handle):
            return self.server.scene.add_line_segments(
                name,
                points=points,
                colors=colors,
                line_width=line_width,
                visible=len(points) > 0,
            )

        handle.points = points
        handle.colors = colors
        handle.line_width = line_width
        handle.visible = len(points) > 0
        return handle

    def _clear_mesh_handles(self, layer_name: str) -> None:
        for handle in self._mesh_handles.get(layer_name, []):
            _safe_remove(handle)
        self._mesh_handles[layer_name] = []

    def _hide_non_mesh_flow_handles(self, layer: FlowLayer) -> None:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_point_colors = np.zeros((0, 3), dtype=np.uint8)
        empty_segments = np.zeros((0, 2, 3), dtype=np.float32)
        empty_segment_colors = np.zeros((0, 2, 3), dtype=np.uint8)
        self._point_handles[layer.name] = self._upsert_point_cloud_handle(
            self._point_handles.get(layer.name),
            f"/flow/{layer.name}/trajectory_points",
            points=empty_points,
            colors=empty_point_colors,
            point_size=_safe_float(self.trajectory_point_size_slider.value, 0.01) * layer.point_size_scale,
        )
        self._trail_handles[layer.name] = self._upsert_line_segments_handle(
            self._trail_handles.get(layer.name),
            f"/flow/{layer.name}/trajectory_trails",
            points=empty_segments,
            colors=empty_segment_colors,
            line_width=_safe_float(self.trail_line_width_slider.value, 1.5) * layer.line_width_scale,
        )

    def _update_mesh_ghosts(self, layer: FlowLayer, layer_idx: int, *, visible: bool) -> None:
        self._clear_mesh_handles(layer.name)
        if not visible or layer.mesh_faces is None or layer_idx < 0:
            return

        ghost_count = _safe_int(
            self.hand_mesh_ghost_slider.value,
            12,
            min_value=1,
            max_value=max(1, min(24, layer.coords.shape[0])),
        )
        max_opacity = _safe_float(self.hand_mesh_opacity_slider.value, 0.75)
        start_idx = max(0, layer_idx - ghost_count + 1)
        ghost_indices = list(range(start_idx, layer_idx + 1))
        if not ghost_indices:
            return

        old_color = np.array([226, 226, 226], dtype=np.float64)
        current_color = np.array([255, 194, 166], dtype=np.float64)
        handles = []
        denom = max(1, len(ghost_indices) - 1)
        for slot, mesh_idx in enumerate(ghost_indices):
            age_phase = 1.0 if len(ghost_indices) == 1 else slot / denom
            opacity = max(0.08, max_opacity * (0.18 + 0.82 * (age_phase ** 1.5)))
            color = np.round(old_color * (1.0 - age_phase) + current_color * age_phase).astype(np.uint8)
            vertices = np.asarray(layer.coords[mesh_idx], dtype=np.float32)
            finite = np.isfinite(vertices).all(axis=1)
            if not np.all(finite):
                continue
            handles.append(
                self.server.scene.add_mesh_simple(
                    f"/flow/{layer.name}/mano_mesh_{slot:02d}",
                    vertices=vertices,
                    faces=np.asarray(layer.mesh_faces, dtype=np.int32),
                    color=tuple(int(x) for x in color),
                    opacity=float(opacity),
                    material="standard",
                    flat_shading=False,
                    side="double",
                    cast_shadow=False,
                    receive_shadow=False,
                    visible=True,
                )
            )
        self._mesh_handles[layer.name] = handles

    def _on_frame_slider(self) -> None:
        self.current_frame = _safe_int(
            self.frame_slider.value,
            self.current_frame,
            min_value=0,
            max_value=max(0, self.num_frames - 1),
        )
        if _needs_numeric_reset(self.frame_slider.value, float(self.current_frame)):
            self.frame_slider.value = float(self.current_frame)
        self.update_visuals()

    def _on_flow_layer_update(self) -> None:
        self.object_trajectory_dropdown.disabled = str(self.flow_layer_dropdown.value) != "object"
        self.update_visuals()

    def _cancel_playback_timer(self) -> None:
        timer = self._playback_timer
        self._playback_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_armed_playback(self, delay_s: float = 0.75) -> None:
        if not self._playback_armed:
            return
        timer = self._playback_timer
        if timer is not None and timer.is_alive():
            return
        generation = self._playback_generation
        timer = threading.Timer(delay_s, self._begin_armed_playback, args=(generation,))
        timer.daemon = True
        self._playback_timer = timer
        timer.start()

    def _begin_armed_playback(self, generation: Optional[int] = None) -> None:
        if not self._playback_armed:
            return
        if generation is not None and generation != self._playback_generation:
            return
        self._playback_timer = None
        self._playback_armed = False
        self.current_frame = 0
        self.frame_slider.value = 0.0
        self.update_visuals()
        self.is_playing = self.num_frames > 1

    def _render_playback_frame(self, frame_idx: int) -> None:
        self.current_frame = max(0, min(int(frame_idx), self.num_frames - 1))
        self.frame_slider.value = float(self.current_frame)
        self.update_visuals()

    def _advance_playback_frame(self) -> None:
        next_frame = 0 if self.current_frame >= self.num_frames - 1 else self.current_frame + 1
        self._render_playback_frame(next_frame)

    def _on_play(self) -> None:
        self._cancel_playback_timer()
        self._playback_armed = False
        if self.current_frame >= self.num_frames - 1:
            self._render_playback_frame(0)
        self.is_playing = True

    def _on_pause(self) -> None:
        self._cancel_playback_timer()
        self._playback_armed = False
        self.is_playing = False

    def _on_reset(self) -> None:
        self._cancel_playback_timer()
        self._playback_armed = False
        self.is_playing = False
        self.current_frame = 0
        self.frame_slider.value = 0.0
        self.update_visuals()

    def _set_decision(self, should_continue: bool, clicked: str) -> None:
        self._cancel_playback_timer()
        self._playback_armed = False
        self.is_playing = False
        if self._decision is not None:
            self._decision["continue"] = bool(should_continue)
            self._decision["clicked"] = clicked
        if self._done is not None:
            self._done.set()

    def _playback_loop(self) -> None:
        while True:
            if self.is_playing and self.layers and self.num_frames > 0:
                if self.current_frame >= self.num_frames - 1:
                    time.sleep(0.75)
                    if self.is_playing:
                        self._advance_playback_frame()
                else:
                    self._advance_playback_frame()
                    time.sleep(1.0 / 10.0)
            else:
                time.sleep(0.1)

    def update_visuals(self) -> None:
        """Refresh all visible flow-review layers."""
        if not self.layers:
            return
        frame_idx = _safe_int(
            self.current_frame,
            0,
            min_value=0,
            max_value=max(0, self.num_frames - 1),
        )
        self.current_frame = frame_idx
        if _needs_numeric_reset(self.frame_slider.value, float(frame_idx)):
            self.frame_slider.value = float(frame_idx)
        with self.server.atomic():
            self._update_visual_handles(frame_idx)

    def _update_visual_handles(self, frame_idx: int) -> None:
        if self.video is not None and self.depths is not None and self.intrinsics is not None:
            scene_frame_idx = frame_idx
            scene_frame_idx = max(0, min(scene_frame_idx, int(np.asarray(self.depths).shape[0]) - 1))
            points, colors = _camera_point_cloud(
                self.video,
                self.depths,
                self.intrinsics,
                frame_idx=scene_frame_idx,
                stride=self.point_stride,
            )
            self._scene_handle = self._upsert_point_cloud_handle(
                self._scene_handle,
                # Singleton current-frame RGB-D cloud. Flow history is accumulated
                # only in /flow/trajectory_trails below.
                "/scene/rgb_pointcloud",
                points=points,
                colors=colors,
                point_size=_safe_float(self.scene_point_size_slider.value, 0.008),
            )

        max_display_tracks = max((len(layer.display_order) for layer in self.layers.values()), default=16)
        slider_max = float(max(16, max_display_tracks))
        if self.displayed_tracks_slider.max != slider_max:
            self.displayed_tracks_slider.max = slider_max
        displayed_tracks = _safe_int(
            self.displayed_tracks_slider.value,
            256,
            min_value=1,
            max_value=max(1, max_display_tracks),
        )
        if _needs_numeric_reset(self.displayed_tracks_slider.value, float(displayed_tracks)):
            self.displayed_tracks_slider.value = float(displayed_tracks)
        visible_layer_names = self._visible_layer_names()
        for layer in self.layers.values():
            self._update_flow_layer(
                frame_idx,
                layer,
                displayed_tracks,
                visible=layer.name in visible_layer_names,
            )

    def _visible_layer_names(self) -> set[str]:
        choice = str(self.flow_layer_dropdown.value)
        if choice in self.layers:
            return {choice}
        first_layer = next(iter(self.layers), None)
        return {first_layer} if first_layer is not None else set()

    @staticmethod
    def _layer_index_for_frame(layer: FlowLayer, frame_idx: int) -> Optional[int]:
        if layer.frame_indices is None:
            return frame_idx if 0 <= frame_idx < layer.coords.shape[0] else None
        if len(layer.frame_indices) == 0:
            return None
        if layer.mesh_faces is not None:
            matches = np.flatnonzero(layer.frame_indices == frame_idx)
            return int(matches[-1]) if len(matches) else None
        matches = np.flatnonzero(layer.frame_indices <= frame_idx)
        if len(matches) == 0:
            return None
        return int(matches[-1])

    def _update_flow_layer(self, frame_idx: int, layer: FlowLayer, displayed_tracks: int, *, visible: bool) -> None:
        layer_idx = self._layer_index_for_frame(layer, frame_idx)
        if layer_idx is None or not visible:
            self._hide_non_mesh_flow_handles(layer)
            self._clear_mesh_handles(layer.name)
            return

        if layer.mesh_faces is not None:
            self._hide_non_mesh_flow_handles(layer)
            self._update_mesh_ghosts(layer, layer_idx, visible=visible)
            return

        self._clear_mesh_handles(layer.name)

        track_slice = layer.display_order[:displayed_tracks]
        source_coords = layer.coords
        if layer.name == "object" and str(self.object_trajectory_dropdown.value) == "Raw TAPIP3D":
            source_coords = layer.raw_coords if layer.raw_coords is not None else layer.coords
        coords = source_coords[:, track_slice, :]
        vis = layer.vis[:, track_slice] if layer.vis is not None and layer.vis.shape[1] == layer.coords.shape[1] else None
        colors = layer.colors[track_slice]

        point_origins = layer.point_origins[track_slice] if layer.point_origins is not None else None
        valid_mask = _valid_points_mask(layer_idx, coords, vis, point_origins)
        self._point_handles[layer.name] = self._upsert_point_cloud_handle(
            self._point_handles.get(layer.name),
            f"/flow/{layer.name}/trajectory_points",
            points=coords[layer_idx][valid_mask],
            colors=colors[valid_mask],
            point_size=_safe_float(self.trajectory_point_size_slider.value, 0.01) * layer.point_size_scale,
        )

        segments, segment_colors = _build_current_trails(
            coords,
            vis,
            point_origins,
            colors,
            layer_idx,
        )
        self._trail_handles[layer.name] = self._upsert_line_segments_handle(
            self._trail_handles.get(layer.name),
            f"/flow/{layer.name}/trajectory_trails",
            points=segments,
            colors=segment_colors,
            line_width=_safe_float(self.trail_line_width_slider.value, 1.5) * layer.line_width_scale,
        )


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Review one NovaPlan flow grounding step in Viser.")
    parser.add_argument("--sample_dir", type=Path, required=True)
    parser.add_argument("--execution_dir", type=Path, required=True)
    parser.add_argument("--flow_npz", type=Path, required=True)
    parser.add_argument("--step", type=positive_int, required=True)
    parser.add_argument("--port", type=tcp_port, default=8097)
    parser.add_argument("--decision_path", type=Path, required=True)
    parser.add_argument("--point_stride", type=positive_int, default=6)
    parser.add_argument("--max_tracks", type=positive_int, default=20000)
    args = parser.parse_args()

    reviewer = FlowReviewServer(port=args.port, point_stride=args.point_stride, max_tracks=args.max_tracks)
    reviewer.review_step(
        sample_dir=args.sample_dir,
        execution_dir=args.execution_dir,
        flow_npz=args.flow_npz,
        step=args.step,
        decision_path=args.decision_path,
    )


if __name__ == "__main__":
    main()
