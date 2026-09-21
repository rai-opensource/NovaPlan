#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Visualize compiled NovaPlan execution poses in Viser."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_planning.visualization.pointcloud_io import load_scene_point_cloud  # noqa: E402
from novaplan.cli_args import positive_float, positive_int, tcp_port  # noqa: E402
from novaplan.execution_step import load_flow_tracks  # noqa: E402


SCENE_COLOR = np.array([170, 170, 170], dtype=np.uint8)
OBJECT_COLOR = np.array([80, 220, 120], dtype=np.uint8)
TRAJECTORY_COLOR = np.array([0, 220, 255], dtype=np.uint8)
AXIS_COLORS = np.array([[255, 60, 60], [60, 220, 90], [70, 130, 255]], dtype=np.uint8)


def _safe_remove(handles: Iterable[Any]) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


def _sample(points: Optional[np.ndarray], colors: Optional[np.ndarray], max_points: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if points is None or len(points) == 0:
        return None, None
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    sampled_colors = colors[idx] if colors is not None and len(colors) == len(points) else None
    return points[idx], sampled_colors


def _line_segments_from_polyline(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1).astype(np.float32)


def _pose_axes(pose: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    origin = pose[:3, 3]
    rot = pose[:3, :3]
    segments = []
    colors = []
    for axis in range(3):
        segments.append(np.stack([origin, origin + rot[:, axis] * scale], axis=0))
        colors.append(np.stack([AXIS_COLORS[axis], AXIS_COLORS[axis]], axis=0))
    return np.stack(segments, axis=0).astype(np.float32), np.stack(colors, axis=0)


def _load_execution_dir(execution_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    npz_path = execution_dir / "execution_step.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"execution_step.npz not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        if "relative_ee_transforms" in data:
            relative = data["relative_ee_transforms"].astype(np.float64)
            ee_poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(relative), axis=0)
            for idx in range(1, len(relative)):
                ee_poses[idx] = relative[idx] @ ee_poses[idx - 1]
            object_motion = None
        else:
            # Compatibility with execution bundles produced before the public
            # artifact contract became relative-transform-only.
            ee_poses = data["ee_poses"].astype(np.float64)
            object_motion = data["object_motion"].astype(np.float64) if "object_motion" in data else None

    summary_path = execution_dir / "execution_step.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return ee_poses, object_motion, summary


def _scene_path_from_manifest(execution_dir: Path) -> Optional[Path]:
    manifest_path = execution_dir / "visualization_inputs.json"
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text())
    raw = data.get("scene_pc")
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def run_viewer(args: argparse.Namespace) -> None:
    """Run the interactive Viser execution-step viewer."""
    try:
        import viser
    except ImportError as exc:
        raise SystemExit("viser is required. Install with: pixi install -e local-planning-viz") from exc

    execution_dir = args.execution_dir.resolve()
    ee_poses, object_motion, summary = _load_execution_dir(execution_dir)

    scene_path = args.scene_pc or _scene_path_from_manifest(execution_dir)
    scene_points = scene_colors = object_points = object_colors = None
    if scene_path is not None:
        scene_points, scene_colors, object_points, object_colors = load_scene_point_cloud(scene_path)

    object_flow = None
    flow_path = args.object_flow
    if flow_path is None:
        flow_source = summary.get("metadata", {}).get("flow_source") if summary else None
        if flow_source:
            candidate = Path(flow_source)
            if candidate.exists():
                flow_path = candidate
    if flow_path is not None and flow_path.exists():
        object_flow = load_flow_tracks(flow_path)
        if object_points is None:
            object_points = object_flow.coords[0]

    scene_points, scene_colors = _sample(scene_points, scene_colors, args.max_scene_points)
    object_points, object_colors = _sample(object_points, object_colors, args.max_object_points)
    if scene_colors is None and scene_points is not None:
        scene_colors = np.repeat(SCENE_COLOR[None, :], len(scene_points), axis=0)
    if object_colors is None and object_points is not None:
        object_colors = np.repeat(OBJECT_COLOR[None, :], len(object_points), axis=0)

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"[viser] serving NovaPlan execution debugger at http://{args.host}:{args.port}")
    print(f"[viser] execution: {execution_dir} ({len(ee_poses)} EE poses, flow={summary.get('selected_flow')})")
    if scene_path is not None:
        print(f"[viser] scene: {scene_path}")

    frame_slider = server.gui.add_slider("Frame", min=0, max=len(ee_poses) - 1, step=1, initial_value=0)
    play_button = server.gui.add_button("Play")
    pause_button = server.gui.add_button("Pause")
    show_scene = server.gui.add_checkbox("Scene point cloud", initial_value=scene_points is not None)
    show_object = server.gui.add_checkbox("Object points", initial_value=object_points is not None)
    show_object_motion = server.gui.add_checkbox("Object motion", initial_value=object_motion is not None)
    show_full_traj = server.gui.add_checkbox("Full EE trajectory", initial_value=True)
    show_all_axes = server.gui.add_checkbox("All EE axes", initial_value=False)
    point_size = server.gui.add_slider("Point size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size)
    axis_scale = server.gui.add_slider("Axis scale", min=0.01, max=0.20, step=0.01, initial_value=args.axis_scale)
    line_width = server.gui.add_slider("Line width", min=1.0, max=10.0, step=0.5, initial_value=args.line_width)

    handles: List[Any] = []
    state = {"playing": False, "last": time.time()}

    def redraw(_: Any = None) -> None:
        nonlocal handles
        _safe_remove(handles)
        handles = []
        frame = int(frame_slider.value)

        if scene_points is not None and show_scene.value:
            handles.append(
                server.scene.add_point_cloud(
                    "/novaplan/scene",
                    points=scene_points.astype(np.float32),
                    colors=scene_colors,
                    point_size=point_size.value,
                    point_shape="circle",
                )
            )

        if object_points is not None and show_object.value:
            if object_flow is not None and frame < object_flow.num_frames:
                points = object_flow.coords[frame]
                colors = np.repeat(OBJECT_COLOR[None, :], len(points), axis=0)
                points, colors = _sample(points, colors, args.max_object_points)
            else:
                points, colors = object_points, object_colors
            if points is not None:
                handles.append(
                    server.scene.add_point_cloud(
                        "/novaplan/object",
                        points=points.astype(np.float32),
                        colors=colors,
                        point_size=point_size.value * 1.4,
                        point_shape="circle",
                    )
                )

        origins = ee_poses[:, :3, 3]
        if show_full_traj.value:
            segments = _line_segments_from_polyline(origins)
            if len(segments):
                colors = np.repeat(TRAJECTORY_COLOR[None, None, :], len(segments), axis=0)
                colors = np.repeat(colors, 2, axis=1)
                handles.append(
                    server.scene.add_line_segments(
                        "/novaplan/ee/trajectory",
                        points=segments,
                        colors=colors,
                        line_width=line_width.value,
                    )
                )

        axes_to_draw = range(len(ee_poses)) if show_all_axes.value else (frame,)
        for idx in axes_to_draw:
            segments, colors = _pose_axes(ee_poses[idx], axis_scale.value)
            handles.append(
                server.scene.add_line_segments(
                    f"/novaplan/ee/axes_{idx:04d}",
                    points=segments,
                    colors=colors,
                    line_width=line_width.value,
                )
            )

        if show_object_motion.value and object_motion is not None:
            object_origins = object_motion[:, :3, 3]
            segments = _line_segments_from_polyline(object_origins)
            if len(segments):
                colors = np.repeat(OBJECT_COLOR[None, None, :], len(segments), axis=0)
                colors = np.repeat(colors, 2, axis=1)
                handles.append(
                    server.scene.add_line_segments(
                        "/novaplan/object_motion",
                        points=segments,
                        colors=colors,
                        line_width=max(1.0, line_width.value - 1.0),
                    )
                )

    play_button.on_click(lambda _: state.update({"playing": True}))
    pause_button.on_click(lambda _: state.update({"playing": False}))
    for control in (
        frame_slider,
        show_scene,
        show_object,
        show_object_motion,
        show_full_traj,
        show_all_axes,
        point_size,
        axis_scale,
        line_width,
    ):
        control.on_update(redraw)

    redraw()
    while True:
        if state["playing"] and time.time() - state["last"] > 1.0 / max(1.0, args.fps):
            state["last"] = time.time()
            frame_slider.value = (int(frame_slider.value) + 1) % len(ee_poses)
            redraw()
        time.sleep(0.01)


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Visualize NovaPlan EE poses over scene/object flow in Viser.")
    parser.add_argument("--execution_dir", type=Path, required=True)
    parser.add_argument("--scene_pc", type=Path, default=None)
    parser.add_argument("--object_flow", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=tcp_port, default=8082)
    parser.add_argument("--fps", type=positive_float, default=8.0)
    parser.add_argument("--max_scene_points", type=positive_int, default=20000)
    parser.add_argument("--max_object_points", type=positive_int, default=2048)
    parser.add_argument("--point_size", type=positive_float, default=0.004)
    parser.add_argument("--axis_scale", type=positive_float, default=0.06)
    parser.add_argument("--line_width", type=positive_float, default=3.0)
    run_viewer(parser.parse_args())


if __name__ == "__main__":
    main()
