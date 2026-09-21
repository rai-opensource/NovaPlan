#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Compile NovaPlan flow outputs into execution pose trajectories."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novaplan.execution_step import (  # noqa: E402
    DEFAULT_FLOW_SWITCH_THETA_DEG,
    compile_flow_execution_step,
    find_default_file,
    load_hand_pose_track,
    load_flow_tracks,
    normalize_flow_arrays,
    save_execution_step,
)
from novaplan.cli_args import (  # noqa: E402
    http_url,
    nonnegative_float,
    positive_int,
    rotation_degrees,
    unit_interval,
)
from novaplan.flow_switching import evaluate_object_flow  # noqa: E402
from novaplan.object_flow import estimate_rigid_object_flow  # noqa: E402
from novaplan.hand_flow.calibration import (  # noqa: E402
    calibrate_mesh_sequence_to_object_contact,
    calibrate_mesh_sequence_to_prompted_contact,
    canonical_contact_finger,
    decode_semantic_xyz,
    hand_pose_from_semantic,
    interaction_interval_from_object_masks,
    palm_frame_from_semantic,
    transform_semantic_xyz,
)


DEFAULT_OBJECT_FLOW_SERVER_URL = "http://127.0.0.1:7001"
DEFAULT_HAND_FLOW_SERVER_URL = "http://127.0.0.1:8080/predict"


def _write_grounding_progress(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grounding_progress.json").write_text(json.dumps(payload, indent=2))


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _canonical_url(value: Optional[str], *env_names: str) -> Optional[str]:
    if value:
        return value
    return _env_first(*env_names)


def _resolve_results_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.results_dir:
        return args.results_dir.resolve()
    if args.sample_dir:
        sample_dir = args.sample_dir.resolve()
        for rel in ("test_results", "test_results_human_hand"):
            candidate = sample_dir / rel
            if candidate.exists():
                return candidate
    return None


def _default_flow_path(results_dir: Optional[Path]) -> Optional[Path]:
    if results_dir is None:
        return None
    return find_default_file(results_dir, ("tapip3d_output.npz", "tapip3d_output_*.npz", "*flow_arrays.npz"))


def _load_optional_hand(path: Optional[Path]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None:
        return None, None
    return load_hand_pose_track(path)


def _trim_hand_pose_track(
    poses: np.ndarray,
    frame_indices: np.ndarray,
    calibration_summary: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the paper's executable contact-onset through release interval."""

    poses = np.asarray(poses, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if len(poses) != len(frame_indices):
        raise ValueError("hand poses and frame_indices must have matching lengths")
    start_frame = calibration_summary.get("movement_start_frame")
    end_frame = calibration_summary.get("movement_end_frame")
    if start_frame is None or end_frame is None:
        raise ValueError("hand calibration did not report an executable interaction interval")
    keep = (frame_indices >= int(start_frame)) & (frame_indices <= int(end_frame))
    trimmed_poses = poses[keep]
    trimmed_indices = frame_indices[keep]
    if len(trimmed_indices) < 2:
        raise ValueError(
            "hand trajectory rejected because the calibrated interaction interval "
            f"[{int(start_frame)}, {int(end_frame)}] contains fewer than two HaMeR poses"
        )
    return trimmed_poses, trimmed_indices


def _hand_pose_with_reference_palm_orientation(
    semantic: dict,
    vertices: np.ndarray,
    *,
    contact_finger: str,
    reference_rotation: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Match the palm-normal sign to the first valid frame, as in main."""

    pose = hand_pose_from_semantic(
        semantic,
        vertices,
        contact_finger=contact_finger,
    )
    rotation = palm_frame_from_semantic(semantic)
    if rotation is None or not np.isfinite(rotation).all():
        return pose, reference_rotation

    rotation = np.asarray(rotation, dtype=np.float64).copy()
    if reference_rotation is None:
        reference_rotation = rotation.copy()
    elif float(np.dot(rotation[:, 2], reference_rotation[:, 2])) < 0.0:
        # A plane normal has two equivalent signs. Flip its normal and
        # accompanying in-plane y-axis together to preserve a right-handed
        # frame, matching main's sequence-level sign convention.
        rotation[:, 1] *= -1.0
        rotation[:, 2] *= -1.0

    pose[:3, :3] = rotation
    return pose, reference_rotation


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


def _load_hamer_mesh_records(mesh_dir: Path) -> list[dict]:
    mesh_paths = sorted(mesh_dir.glob("frame*_hand*.npz"))
    if not mesh_paths:
        raise SystemExit(f"HaMeR completed but no mesh npz files were found under {mesh_dir}.")

    records = []
    for mesh_path in mesh_paths:
        with np.load(mesh_path, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            faces = np.asarray(data["faces"], dtype=np.int32)
            frame_idx = int(data["frame_idx"]) if "frame_idx" in data else _frame_idx_from_mesh_name(mesh_path)
            hand_id = int(data["hand_id"]) if "hand_id" in data else 0
            is_right = bool(int(data["is_right"])) if "is_right" in data else True
            semantic = decode_semantic_xyz(data["semantic_xyz_json"]) if "semantic_xyz_json" in data else {}
            ok_pnp = bool(data["ok_pnp"]) if "ok_pnp" in data else False
            reprojection_error = float(data["pnp_reprojection_error"]) if "pnp_reprojection_error" in data else None
            k_real = np.asarray(data["K_REAL"], dtype=np.float64) if "K_REAL" in data else None
            k_weak = np.asarray(data["K_WEAK"], dtype=np.float64) if "K_WEAK" in data else None
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            continue
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            continue
        records.append(
            {
                "path": mesh_path,
                "frame_idx": frame_idx,
                "hand_id": hand_id,
                "is_right": is_right,
                "vertices": vertices,
                "faces": faces,
                "semantic": semantic,
                "ok_pnp": ok_pnp,
                "pnp_reprojection_error": reprojection_error,
                "K_REAL": k_real,
                "K_WEAK": k_weak,
            }
        )
    if not records:
        raise SystemExit(f"HaMeR mesh dump did not contain usable vertices: {mesh_dir}")

    grouped_by_side: dict[bool, list[dict]] = {}
    for record in records:
        grouped_by_side.setdefault(bool(record["is_right"]), []).append(record)
    _selected_side, selected_records = max(grouped_by_side.items(), key=lambda item: (len(item[1]), int(item[0])))

    by_frame: dict[int, dict] = {}
    for record in selected_records:
        frame_idx = int(record["frame_idx"])
        current = by_frame.get(frame_idx)
        if current is None or int(record["hand_id"]) < int(current["hand_id"]):
            by_frame[frame_idx] = record
    return [by_frame[idx] for idx in sorted(by_frame)]


def _load_flow_depth_calibration_inputs(flow_path: Optional[Path]):
    if flow_path is None or not flow_path.exists():
        return None, None, None
    with np.load(flow_path, allow_pickle=False) as data:
        depths = np.asarray(data["depths"], dtype=np.float64) if "depths" in data else None
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float64) if "intrinsics" in data else None
        video_shape = None
        if "depths_video_res" in data and np.asarray(data["depths_video_res"]).ndim >= 3:
            video_shape = tuple(int(value) for value in np.asarray(data["depths_video_res"]).shape[-2:])
        elif "video" in data:
            video = np.asarray(data["video"])
            if video.ndim == 4:
                video_shape = (
                    (int(video.shape[2]), int(video.shape[3]))
                    if video.shape[1] in {1, 3, 4}
                    else (int(video.shape[1]), int(video.shape[2]))
                )
    return depths, intrinsics, video_shape


def _flow_suffix(flow_path: Path) -> str:
    stem = flow_path.stem
    for prefix in ("tapip3d_output_", "flow_arrays_", "object_flow_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _find_flow_mask_path(flow_path: Optional[Path]) -> Optional[Path]:
    if flow_path is None or not flow_path.exists():
        return None
    suffix = _flow_suffix(flow_path)
    parent = flow_path.parent
    candidates = [
        parent / f"segmentation_masks_scaled_{suffix}.npz",
        parent / f"segmentation_masks_{suffix}.npz",
        parent / f"segmentation_masks_full_{suffix}.npz",
    ]
    candidates.extend(sorted(parent.glob("segmentation_masks_scaled*.npz")))
    candidates.extend(sorted(parent.glob("segmentation_masks_*.npz")))
    candidates.extend(sorted(parent.glob("segmentation_masks_full*.npz")))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _load_flow_object_masks(flow_path: Optional[Path]) -> Optional[np.ndarray]:
    mask_path = _find_flow_mask_path(flow_path)
    if mask_path is None:
        return None
    with np.load(mask_path, allow_pickle=False) as data:
        for key in ("masks", "mask", "object_mask", "segmentation"):
            if key in data:
                return np.asarray(data[key])
        if data.files:
            return np.asarray(data[data.files[0]])
    return None


def _load_hamer_meta(mesh_dir: Path) -> dict:
    meta_path = mesh_dir.parent / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _save_calibrated_mesh_dump(
    records: list[dict],
    calibrated_vertices: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
    output_dir: Path,
    *,
    contact_finger: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame*_hand*.npz"):
        stale.unlink()
    for i, record in enumerate(records):
        semantic = transform_semantic_xyz(record["semantic"], float(scales[i]), offsets[i])
        stem = f"frame{int(record['frame_idx']):06d}_hand{int(record['hand_id']):02d}"
        faces = np.asarray(record["faces"], dtype=np.int32)
        faces = faces[np.all((faces >= 0) & (faces < calibrated_vertices.shape[1]), axis=1)]
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            frame_idx=np.int64(record["frame_idx"]),
            hand_id=np.int64(record["hand_id"]),
            is_right=np.int64(1 if record["is_right"] else 0),
            vertices=np.asarray(calibrated_vertices[i], dtype=np.float32),
            faces=faces,
            semantic_xyz_json=np.bytes_(json.dumps(semantic)),
            calibration_scale=np.float32(scales[i]),
            calibration_offset=np.asarray(offsets[i], dtype=np.float32),
            contact_finger=np.bytes_(contact_finger),
        )


def _update_hamer_meta(
    meta_path: Path,
    *,
    calibration_summary: dict,
    scales: np.ndarray,
    offsets: np.ndarray,
    contact_finger: str,
    calibrated_mesh_dir: Path,
) -> None:
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    start_offset = calibration_summary.get("translation_offset_start")
    if start_offset is None:
        start_offset = offsets[0].astype(float).tolist() if len(offsets) else [0.0, 0.0, 0.0]
    end_offset = calibration_summary.get("translation_offset_end")
    if end_offset is None:
        end_offset = offsets[-1].astype(float).tolist() if len(offsets) else [0.0, 0.0, 0.0]
    meta.update(
        {
            "hand_mesh_scale": float(np.median(scales)) if len(scales) else 1.0,
            "hand_mesh_scale_start": float(calibration_summary.get("hand_mesh_scale_start", scales[0] if len(scales) else 1.0)),
            "hand_mesh_scale_end": float(calibration_summary.get("hand_mesh_scale_end", scales[-1] if len(scales) else 1.0)),
            "translation_offset_start": start_offset,
            "translation_offset_end": end_offset,
            "contact_finger": contact_finger,
            "calibrated_mesh_dump": str(calibrated_mesh_dir),
            "hand_mesh_calibration": calibration_summary,
        }
    )
    if "movement_start_frame" in calibration_summary:
        meta["movement_start_frame"] = int(calibration_summary["movement_start_frame"])
        meta["grasp_frame_idx"] = int(calibration_summary["movement_start_frame"])
    if "movement_end_frame" in calibration_summary:
        meta["movement_end_frame"] = int(calibration_summary["movement_end_frame"])
    meta_path.write_text(json.dumps(meta, indent=2))


def _hand_poses_from_hamer_mesh_dump(
    mesh_dir: Path,
    output_path: Path,
    *,
    flow_path: Optional[Path] = None,
    grounding_mode: str = "grasp",
    contact_finger: Optional[str] = None,
    contact_point_2d: Optional[tuple[float, float]] = None,
    release_delta_m: Optional[float] = None,
    max_outside_fraction: float = 0.30,
    hand_flow_interaction_epsilon: float = 0.9,
) -> Path:
    records = _load_hamer_mesh_records(mesh_dir)
    min_vertices = min(int(record["vertices"].shape[0]) for record in records)
    vertices = np.stack([record["vertices"][:min_vertices] for record in records], axis=0)
    frame_indices = np.asarray([int(record["frame_idx"]) for record in records], dtype=np.int64)
    faces = np.asarray(records[0]["faces"], dtype=np.int32)
    faces = faces[np.all((faces >= 0) & (faces < min_vertices), axis=1)]
    semantics = [record["semantic"] for record in records]
    hamer_meta = _load_hamer_meta(mesh_dir)
    movement_start_frame = hamer_meta.get("movement_start_frame")
    if movement_start_frame is None:
        movement_start_frame = hamer_meta.get("grasp_frame_idx")
    movement_end_frame = hamer_meta.get("movement_end_frame")
    depths, intrinsics, flow_video_shape = _load_flow_depth_calibration_inputs(flow_path)
    object_masks = _load_flow_object_masks(flow_path)
    if object_masks is not None:
        mask_start, mask_end = interaction_interval_from_object_masks(
            object_masks,
            epsilon=hand_flow_interaction_epsilon,
        )
        movement_start_frame = mask_start
        movement_end_frame = mask_end

    if grounding_mode == "non_prehensile":
        if contact_finger is None:
            raise ValueError("non-prehensile grounding requires a prompted contact_finger")
        if contact_point_2d is None:
            raise ValueError("non-prehensile grounding requires one annotated contact_point_2d")
        if depths is None or intrinsics is None or object_masks is None:
            raise ValueError(
                "non-prehensile grounding requires aligned depth, intrinsics, and object masks"
            )
        if flow_video_shape is None:
            raise ValueError(
                "non-prehensile grounding requires the generated-video resolution "
                "(depths_video_res or video in the TAPIP3D output) to map its contact pixel"
            )
        contact_finger = canonical_contact_finger(contact_finger)
        calibration = calibrate_mesh_sequence_to_prompted_contact(
            vertices,
            faces,
            semantics,
            frame_indices,
            depths,
            intrinsics,
            object_masks,
            movement_start_frame=int(movement_start_frame) if movement_start_frame is not None else None,
            movement_end_frame=int(movement_end_frame) if movement_end_frame is not None else None,
            contact_finger=contact_finger,
            contact_point_2d=contact_point_2d,
            contact_point_source_shape=flow_video_shape,
            release_delta_m=float(release_delta_m if release_delta_m is not None else 0.02),
            max_outside_fraction=float(max_outside_fraction),
        )
    elif object_masks is not None:
        calibration = calibrate_mesh_sequence_to_object_contact(
            vertices,
            faces,
            semantics,
            frame_indices,
            depths,
            intrinsics,
            object_masks,
            movement_start_frame=int(movement_start_frame) if movement_start_frame is not None else None,
            movement_end_frame=int(movement_end_frame) if movement_end_frame is not None else None,
            contact_finger=contact_finger,
            release_delta_m=float(release_delta_m if release_delta_m is not None else 0.05),
            max_outside_fraction=float(max_outside_fraction),
            strict_contact=True,
        )
    else:
        raise ValueError("hand grounding requires target-object masks for contact calibration")

    calibration_summary = dict(calibration.summary)
    calibration_summary["hand_flow_interaction_epsilon"] = float(hand_flow_interaction_epsilon)
    contact_finger = canonical_contact_finger(
        str(calibration_summary.get("contact_finger") or contact_finger or "")
    )

    poses = []
    reference_palm_rotation: Optional[np.ndarray] = None
    for i, record in enumerate(records):
        semantic = transform_semantic_xyz(record["semantic"], float(calibration.scales[i]), calibration.offsets[i])
        pose, reference_palm_rotation = _hand_pose_with_reference_palm_orientation(
            semantic,
            calibration.vertices[i],
            contact_finger=contact_finger,
            reference_rotation=reference_palm_rotation,
        )
        poses.append(pose)

    pose_array = np.stack(poses, axis=0)
    execution_poses, execution_frame_indices = _trim_hand_pose_track(
        pose_array,
        frame_indices,
        calibration_summary,
    )
    calibration_summary.update(
        {
            "execution_num_frames": int(len(execution_frame_indices)),
            "execution_start_frame": int(execution_frame_indices[0]),
            "execution_end_frame": int(execution_frame_indices[-1]),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        hand_poses=execution_poses,
        frame_indices=execution_frame_indices,
        contact_finger=np.bytes_(contact_finger),
        grounding_mode=np.bytes_(grounding_mode),
        hand_flow_interaction_epsilon=np.float32(hand_flow_interaction_epsilon),
    )
    calibrated_mesh_dir = output_path.parent / "calibrated_mesh_dump"
    _save_calibrated_mesh_dump(
        records,
        calibration.vertices,
        calibration.scales,
        calibration.offsets,
        calibrated_mesh_dir,
        contact_finger=contact_finger,
    )
    _update_hamer_meta(
        output_path.parent / "meta.json",
        calibration_summary=calibration_summary,
        scales=calibration.scales,
        offsets=calibration.offsets,
        contact_finger=contact_finger,
        calibrated_mesh_dir=calibrated_mesh_dir,
    )
    print(
        "[execution-step] HaMeR calibration: "
        f"method={calibration_summary['method']} "
        f"{calibration_summary['num_ok_frames']}/{calibration_summary['num_frames']} frames, "
        f"scale_median={calibration_summary['scale_median']:.3f}; "
        f"execution_frames={len(execution_frame_indices)} "
        f"[{int(execution_frame_indices[0])}, {int(execution_frame_indices[-1])}]"
    )
    return output_path


def _run_hand_flow_if_needed(
    *,
    args: argparse.Namespace,
    sample_dir: Optional[Path],
    flow_video: Optional[Path],
    flow_path: Path,
    out_dir: Path,
    grounding_mode: str,
    contact_finger: Optional[str],
    contact_point_2d: Optional[tuple[float, float]],
    release_delta_m: Optional[float],
    max_outside_fraction: float,
    hand_flow_interaction_epsilon: float,
) -> Optional[Path]:
    if not args.hand_flow_server_url:
        return None
    if sample_dir is None:
        raise SystemExit("--hand_flow_server_url requires --sample_dir so hand-flow inputs can be resolved.")

    hamer_out = (
        args.debug_artifact_dir.resolve() / "hand_flow"
        if args.debug_artifact_dir is not None
        else out_dir / "hamer_outputs"
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "novaplan" / "hand_flow" / "hamer_client.py"),
        "--sample_dir",
        str(sample_dir),
        "--url",
        args.hand_flow_server_url,
        "--flow_file",
        str(flow_path),
        "--out_folder",
        str(hamer_out),
        "--all_frames",
    ]
    if flow_video is not None:
        cmd.extend(["--rgb_path", str(flow_video.resolve())])

    print("[execution-step] object flow requests hand fallback; querying the hand-flow service")
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"Hand-flow service request failed with exit code {proc.returncode}.")

    hand_path = _hand_poses_from_hamer_mesh_dump(
        hamer_out / "real_mesh_dump",
        hamer_out / "hand_pose_track.npz",
        flow_path=flow_path,
        grounding_mode=grounding_mode,
        contact_finger=contact_finger,
        contact_point_2d=contact_point_2d,
        release_delta_m=release_delta_m,
        max_outside_fraction=max_outside_fraction,
        hand_flow_interaction_epsilon=hand_flow_interaction_epsilon,
    )
    print(f"[execution-step] wrote HaMeR hand poses: {hand_path}")
    return hand_path


def _run_object_flow_if_needed(
    *,
    args: argparse.Namespace,
    sample_dir: Optional[Path],
    results_dir: Optional[Path],
    flow_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    if flow_path is not None:
        return flow_path, results_dir
    if not args.object_flow_server_url:
        if args.flow_video:
            raise SystemExit(
                "Selected rollout video was provided with --flow_video, but no remote 3D object-flow server URL "
                "is configured. Set --object_flow_server_url or NOVAPLAN_OBJECT_FLOW_SERVER_URL."
            )
        return flow_path, results_dir
    if sample_dir is None:
        raise SystemExit("--object_flow_server_url requires --sample_dir so the flow inputs can be resolved.")

    if args.flow_video and args.out_dir:
        output_dir = args.out_dir.resolve() / "flow_outputs"
    else:
        output_dir = results_dir or (sample_dir / "test_results")
    mask_prompt = args.flow_mask_prompt or args.track_object or "object"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "remote_object_flow" / "tests" / "test_object_flow_server.py"),
        "--server",
        args.object_flow_server_url,
        "--example_dir",
        str(sample_dir.parent),
        "--step",
        sample_dir.name,
        "--mask_prompt",
        mask_prompt,
        "--flow_switch_theta_deg",
        str(args.flow_switch_theta_deg),
        "--output_dir",
        str(output_dir),
        "--auxiliary_arrays",
    ]
    if args.debug_artifact_dir is not None:
        cmd.extend([
            "--debug_output_dir",
            str(args.debug_artifact_dir.resolve() / "object_flow"),
        ])
    if args.flow_video:
        cmd.extend(["--video", str(args.flow_video.resolve())])
    if args.flow_depth_source_job_id:
        cmd.extend(["--depth_source_job_id", args.flow_depth_source_job_id])
    if args.flow_sam3_debug_video:
        cmd.append("--sam3_debug_video")
    if args.flow_debug_artifacts:
        cmd.append("--debug_artifacts")
    if args.disable_cvd:
        cmd.append("--disable_cvd")

    print(f"[execution-step] object flow missing; requesting remote object flow with mask_prompt='{mask_prompt}'")
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"Remote object-flow extraction failed with exit code {proc.returncode}.")

    flow_path = _default_flow_path(output_dir)
    if flow_path is None:
        raise SystemExit(f"Remote object flow completed but no tapip3d_output*.npz was found in {output_dir}.")
    return flow_path, output_dir


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Compile a selected NovaPlan rollout into flow-derived poses.")
    parser.add_argument("--sample_dir", type=Path, default=None, help="Sample step dir containing test_results/.")
    parser.add_argument("--results_dir", type=Path, default=None, help="Directory with TAPIP3D and optional hand outputs.")
    parser.add_argument("--action", default=None, help="Action being grounded; recorded for logs/debug.")
    parser.add_argument("--track_object", default=None, help="Object prompt from the planner for execution-time flow.")
    parser.add_argument("--object_flow", type=Path, default=None, help="TAPIP3D/object-flow NPZ or planner *_flow_arrays.npz.")
    parser.add_argument(
        "--hand_poses",
        type=Path,
        default=None,
        help="Explicit manual hand pose override. Normal execution recomputes HaMeR hand flow when needed.",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for execution_step.*.")
    parser.add_argument(
        "--debug_artifact_dir",
        type=Path,
        default=None,
        help="Per-step directory for flow media, HaMeR meshes, and other diagnostics.",
    )
    parser.add_argument(
        "--flow_video",
        type=Path,
        default=None,
        help="Selected rollout MP4 to send to the remote 3D object-flow server when object-flow arrays are missing.",
    )
    parser.add_argument("--selected_flow", choices=("auto", "object", "hand"), default="auto")
    parser.add_argument(
        "--grounding_mode",
        choices=("standard", "grasp", "non_prehensile"),
        default="standard",
        help=(
            "Recovery-aware grounding route. Grasp/standard use object-first hand fallback; "
            "non_prehensile forces prompted fingertip hand flow."
        ),
    )
    parser.add_argument(
        "--contact_finger",
        type=canonical_contact_finger,
        default=None,
        help="Prompted contact finger for non-prehensile grounding (thumb/index/middle/ring/pinky).",
    )
    parser.add_argument(
        "--contact_point_2d",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="Annotated recovery contact pixel in the selected video's first-frame coordinates.",
    )
    parser.add_argument("--release_delta_m", type=nonnegative_float, default=None)
    parser.add_argument("--max_hand_outside_fraction", type=unit_interval, default=0.30)
    parser.add_argument(
        "--flow_switch_theta_deg",
        type=rotation_degrees,
        default=DEFAULT_FLOW_SWITCH_THETA_DEG,
    )
    parser.add_argument(
        "--hand_flow_interaction_epsilon",
        type=unit_interval,
        default=0.9,
        help=(
            "Object-mask new-support ratio for the hand interaction interval "
            "(paper Appendix D.2, epsilon; default: 0.9)."
        ),
    )
    parser.add_argument("--min_visible_points", type=positive_int, default=6)
    parser.add_argument(
        "--object_flow_server_url",
        type=http_url,
        default=_env_first("NOVAPLAN_OBJECT_FLOW_SERVER_URL")
        or DEFAULT_OBJECT_FLOW_SERVER_URL,
        help="Optional remote 3D object-flow server URL used when object-flow outputs are missing.",
    )
    parser.add_argument(
        "--hand_flow_server_url",
        type=http_url,
        default=_env_first("NOVAPLAN_HAND_FLOW_SERVER_URL") or DEFAULT_HAND_FLOW_SERVER_URL,
        help=(
            "Hand-flow service /predict URL used for object-flow fallback and "
            "forced non-prehensile grounding."
        ),
    )
    parser.add_argument("--flow_mask_prompt", default=None, help="Override SAM3 prompt for execution-time object flow.")
    parser.add_argument("--flow_depth_source_job_id", default=None, help="Reuse live cached depth from the remote object-flow server.")
    parser.add_argument("--flow_sam3_debug_video", action="store_true", help="Also request the SAM3 segmentation debug video.")
    parser.add_argument("--flow_debug_artifacts", action="store_true", help="Request full remote object-flow debug media/artifacts.")
    parser.add_argument(
        "--disable_cvd",
        action="store_true",
        help="Use calibrated per-frame MoGe2 depth without optional CVD refinement.",
    )
    args = parser.parse_args()
    args.object_flow_server_url = _canonical_url(
        args.object_flow_server_url,
        "NOVAPLAN_OBJECT_FLOW_SERVER_URL",
    ) or DEFAULT_OBJECT_FLOW_SERVER_URL
    args.hand_flow_server_url = _canonical_url(
        args.hand_flow_server_url,
        "NOVAPLAN_HAND_FLOW_SERVER_URL",
    ) or DEFAULT_HAND_FLOW_SERVER_URL

    results_dir = _resolve_results_dir(args)
    sample_dir = args.sample_dir.resolve() if args.sample_dir else None
    flow_path = args.object_flow or (None if args.flow_video else _default_flow_path(results_dir))
    flow_path, results_dir = _run_object_flow_if_needed(
        args=args,
        sample_dir=sample_dir,
        results_dir=results_dir,
        flow_path=flow_path,
    )
    out_dir = args.out_dir or ((results_dir or Path.cwd()) / "execution_step")

    if flow_path is None:
        raise SystemExit("No object-flow NPZ found. Pass --object_flow or --results_dir/--sample_dir.")

    print(f"[execution-step] object flow: {flow_path}")
    flow = load_flow_tracks(flow_path)

    hand_path = args.hand_poses
    coords, vis = normalize_flow_arrays(
        flow.coords,
        flow.visibilities,
        layout="time_major",
    )
    rigid_fit = estimate_rigid_object_flow(
        coords,
        vis,
        layout="time_major",
        point_origins=flow.point_origins,
        min_visible_points=args.min_visible_points,
    )
    object_eval = evaluate_object_flow(
        coords,
        vis,
        layout="time_major",
        point_origins=flow.point_origins,
        flow_switch_theta_deg=args.flow_switch_theta_deg,
        min_visible_points=args.min_visible_points,
        rigid_fit=rigid_fit,
    )
    object_flow_unusable = (not object_eval.valid) or object_eval.should_switch_to_hand
    selected_flow = "hand" if args.grounding_mode == "non_prehensile" else args.selected_flow
    needs_hand = selected_flow == "hand" or (selected_flow == "auto" and object_flow_unusable)
    if selected_flow == "hand":
        selection_reason = (
            "non-prehensile recovery requires hand-centric grounding"
            if args.grounding_mode == "non_prehensile"
            else "hand-centric grounding was explicitly requested"
        )
    else:
        selection_reason = object_eval.reason
    progress = {
        "stage": "hand_flow_computation_started" if needs_hand and hand_path is None else "flow_selected",
        "object_flow_path": str(flow_path),
        "object_flow": object_eval.to_dict(),
        "requested_flow": selected_flow,
        "effective_switch_to_hand": bool(needs_hand),
        "selected_grounding": "hand" if needs_hand else "object",
        "selection_reason": selection_reason,
    }
    _write_grounding_progress(out_dir, progress)
    print(
        "[execution-step] object-flow evaluation: "
        f"valid={object_eval.valid} "
        f"rotation_threshold_exceeded={object_eval.should_switch_to_hand} "
        f"effective_switch_to_hand={needs_hand} "
        f"reason={object_eval.reason}"
    )
    if hand_path is None and needs_hand:
        print("[execution-step] computing fresh HaMeR hand flow; automatic hand-pose reuse is disabled")
        hand_path = _run_hand_flow_if_needed(
            args=args,
            sample_dir=sample_dir,
            flow_video=args.flow_video,
            flow_path=flow_path,
            out_dir=out_dir,
            grounding_mode=args.grounding_mode,
            contact_finger=args.contact_finger,
            contact_point_2d=(tuple(args.contact_point_2d) if args.contact_point_2d is not None else None),
            release_delta_m=args.release_delta_m,
            max_outside_fraction=args.max_hand_outside_fraction,
            hand_flow_interaction_epsilon=args.hand_flow_interaction_epsilon,
        )
        progress["stage"] = "hand_flow_computed"
        progress["hand_poses_path"] = str(hand_path) if hand_path is not None else None
        _write_grounding_progress(out_dir, progress)
    if needs_hand and hand_path is None:
        raise SystemExit(
            "Object flow requested hand fallback, but no hand poses were available and no hand-flow service URL "
            "was configured. Set --hand_flow_server_url or NOVAPLAN_HAND_FLOW_SERVER_URL."
        )

    hand_poses, hand_frame_indices = _load_optional_hand(hand_path)
    if hand_path is not None:
        progress["stage"] = "hand_flow_computed"
        progress["hand_poses_path"] = str(hand_path)
        _write_grounding_progress(out_dir, progress)
        print(f"[execution-step] hand poses: {hand_path} ({len(hand_poses) if hand_poses is not None else 0} poses)")

    result = compile_flow_execution_step(
        flow,
        hand_poses=hand_poses,
        hand_frame_indices=hand_frame_indices,
        selected_flow=selected_flow,
        flow_switch_theta_deg=args.flow_switch_theta_deg,
        min_visible_points=args.min_visible_points,
        rigid_fit=rigid_fit,
        object_evaluation=object_eval,
    )
    result.metadata.update(
        {
            "action": args.action,
            "track_object": args.track_object,
            "grounding_mode": args.grounding_mode,
            "requested_contact_finger": args.contact_finger,
            "requested_contact_point_2d": args.contact_point_2d,
        }
    )
    saved = save_execution_step(result, out_dir)
    summary_path = saved["summary"]
    summary = json.loads(summary_path.read_text())
    summary["artifact_paths"] = {
        "object_flow": str(flow_path),
        "hand_poses_input": str(hand_path) if hand_path is not None else None,
        **{name: str(path) for name, path in saved.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    progress["stage"] = "geometric_grounding_complete"
    progress["selected_grounding"] = result.selected_flow
    progress["execution_summary"] = str(summary_path)
    _write_grounding_progress(out_dir, progress)

    print(f"[execution-step] selected flow: {result.selected_flow}")
    print(f"[execution-step] flow switch reason: {result.flow_switch.reason}")
    print(f"[execution-step] relative transforms: {result.relative_ee_transforms.shape}")
    print(f"[execution-step] source frame indices: {result.frame_indices.tolist()}")
    print(f"[execution-step] wrote: {saved['execution_step']}")
    print(f"[execution-step] summary: {saved['summary']}")
    print(
        "[execution-step] visualize with: "
        f"pixi run -e local-planning-viz python local_planning/visualization/viser_execution_step.py "
        f"--execution_dir {out_dir}"
    )


if __name__ == "__main__":
    main()
