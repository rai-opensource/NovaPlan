# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""In-process closed-loop NovaPlan execution orchestration.

The CLI wrapper in ``local_planning/run_closed_loop_execution.py`` uses this
module, and ``novaplan.planner`` can call it directly after horizon counting.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .llm_client import VLMAdapter
from .observations import (
    FilesystemObservationProvider,
    ObservationBundle,
    ObservationProvider,
    RecordedTraceObservationProvider,
    observation_depth_scale,
)
from .recovery_policy import RecoveryContext, choose_recovery_strategy
from .vlm_prompts import build_direct_video_prompt, build_video_negative_prompt

ROOT = Path(__file__).resolve().parents[1]
RECORDED_OBSERVATION_CONTEXTS = {"illustration", "recorded_observations"}


@dataclass
class ClosedLoopExecutionConfig:
    """Configure one closed-loop NovaPlan execution run."""
    goal: str
    input_frame: Path
    output_dir: Optional[Path] = None
    sample_root: Optional[Path] = None
    step_dirs: Sequence[Path] = field(default_factory=list)
    step_glob: str = "step_*"
    plan_actions: Sequence[Dict[str, Any] | str] = field(default_factory=list)
    illustration_actions: Sequence[Dict[str, Any] | str] = field(default_factory=list)
    execution_context: str = "online"
    horizon_result: Optional[Dict[str, Any]] = None
    fallback_horizon: int = 3
    execution_mode: Optional[str] = None
    fallback_action: Optional[str] = None
    fallback_track_object: str = "object"
    object_flow_server_url: Optional[str] = None
    enable_cvd: bool = True
    hand_flow_server_url: Optional[str] = None
    flow_sam3_debug_video: bool = False
    flow_debug_artifacts: bool = False
    selected_flow: str = "auto"
    flow_switch_theta_deg: float = 45.0
    hand_flow_interaction_epsilon: float = 0.9
    post_image: Optional[Path] = None
    max_recovery_attempts: int = 1
    disable_verification: bool = False
    recovery_video_backend: str = "both"
    recovery_num_videos: int = 1
    recovery_num_frames: int = 41
    recovery_fps: int = 16
    rollout_fps: int = 16
    recovery_size: str = "1280*720"
    recovery_seed: int = 0
    max_hand_flow_regenerations: int = 2
    debug_flow_review: bool = False
    debug_flow_review_port: int = 8097
    show_execution_stdout: bool = False
    observation_provider: Optional[ObservationProvider] = None


def _default_run_dir() -> Path:
    return ROOT / "runs" / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")


def _call_with_verbose_capture(
    config: ClosedLoopExecutionConfig,
    label: str,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Keep component detail in the run directory while the console stays readable."""

    if config.show_execution_stdout:
        return function(*args, **kwargs)
    output_dir = Path(config.output_dir or _default_run_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runtime_verbose.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {label} ===\n")
        log.flush()
        with redirect_stdout(log), redirect_stderr(log):
            return function(*args, **kwargs)


def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _decode_png_base64(data: Optional[str]) -> Optional[np.ndarray]:
    if not data:
        return None
    try:
        return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _write_video(path: Path, video: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        bitrate="8M",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    with writer:
        for frame in video:
            writer.append_data(frame)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _object_flow_console_summary(
    object_flow: Dict[str, Any],
    fallback_threshold: float,
) -> str:
    max_rotation = object_flow.get("max_rotation_deg")
    rotation_text = (
        f"{float(max_rotation):.2f}"
        if isinstance(max_rotation, (int, float))
        else "unavailable"
    )
    threshold = object_flow.get("flow_switch_theta_deg", fallback_threshold)
    threshold_text = (
        f"{float(threshold):.2f}"
        if isinstance(threshold, (int, float))
        else "unavailable"
    )
    details = [
        f"object_valid={object_flow.get('valid')}",
        f"rotation_threshold_exceeded={bool(object_flow.get('should_switch_to_hand'))}",
        f"max_rotation_deg={rotation_text}",
        f"threshold_deg={threshold_text}",
    ]

    failed_steps = [int(step) for step in object_flow.get("failed_steps") or []]
    if failed_steps:
        valid_point_counts = list(object_flow.get("valid_point_counts") or [])
        failed_counts = [
            valid_point_counts[step - 1]
            if 0 < step <= len(valid_point_counts)
            else None
            for step in failed_steps
        ]
        details.extend(
            [
                f"failed_adjacent_transforms={len(failed_steps)}",
                f"failed_steps={failed_steps}",
                f"visible_points_at_failed_steps={failed_counts}",
            ]
        )
    return " ".join(details)


def _action_from_beam(beam: Any, *, source: str) -> Dict[str, Any]:
    actions = list(getattr(beam, "actions", []) or [])
    track_objects = list(getattr(beam, "track_objects", []) or [])
    if not actions:
        raise RuntimeError(f"{source} did not return an executable action.")
    return {
        "action": str(actions[-1]),
        "track_object": str(track_objects[-1] if track_objects else "object"),
        "source": source,
        "score": float(getattr(beam, "score", 0.0)),
    }


def _actions_from_beam(beam: Any) -> List[Dict[str, Any]]:
    actions = list(getattr(beam, "actions", []) or [])
    track_objects = list(getattr(beam, "track_objects", []) or [])
    return [
        {
            "action": str(action),
            "track_object": str(track_objects[idx] if idx < len(track_objects) else "object"),
            "source": "strategic_video_language_plan",
            "score": float(getattr(beam, "score", 0.0)),
        }
        for idx, action in enumerate(actions)
    ]


def _save_selected_rollout(
    *,
    output_dir: Path,
    step_index: int,
    rollout: Optional[np.ndarray],
    fps: int,
    source: str,
    generation_attempt: int = 0,
) -> Dict[str, Any]:
    if rollout is None:
        return {}
    step_root = output_dir / f"step_{step_index + 1:03d}"
    if generation_attempt > 0:
        step_dir = (
            step_root
            / "grounding_regenerations"
            / f"attempt_{generation_attempt + 1:03d}"
            / "selected_rollout"
        )
    else:
        step_dir = step_root / "selected_rollout"
    video_path = step_dir / "selected_rollout.mp4"
    target_path = step_dir / "target.png"
    _write_video(video_path, rollout, fps)
    Image.fromarray(np.asarray(rollout[-1], dtype=np.uint8)).save(target_path)
    return {
        "source": source,
        "video": str(video_path),
        "target_image": str(target_path),
        "num_frames": int(len(rollout)),
        "fps": int(fps),
        "generation_attempt": int(generation_attempt),
    }


def _is_retryable_hand_flow_rejection(exc: BaseException) -> bool:
    """Identify paper D.2 calibration rejections, excluding service failures."""

    text = str(exc).lower()
    markers = (
        "hand contact calibration failed",
        "hand trajectory rejected",
        "contact calibration did not identify",
        "semantic output is missing",
        "prompted fingertip",
        "fingertip has invalid contact-onset geometry",
        "annotated recovery contact does not land",
        "fingertip cannot be projected at release",
        "no metric object-surface contact",
        "candidate scale set is empty",
        "object-mask motion never reached the interaction threshold",
        "initial object mask is empty",
        "hamer returned no hand meshes",
        "hamer completed but no mesh npz files",
        "hamer mesh dump did not contain usable vertices",
    )
    return any(marker in text for marker in markers)


def _discover_step_dirs(config: ClosedLoopExecutionConfig, horizon: int) -> List[Path]:
    if config.step_dirs:
        return [Path(p).resolve() for p in config.step_dirs]
    if config.sample_root is None:
        return []
    candidates = sorted(Path(config.sample_root).resolve().glob(config.step_glob))
    # A recorded observation after step N is stored as step N+1's start state.
    # Keep that boundary state available even when only one action is executed.
    extra_observation = 1 if config.execution_context in RECORDED_OBSERVATION_CONTEXTS else 0
    return candidates[: max(1, horizon + extra_observation)]


def _normalize_actions(actions: Sequence[Dict[str, Any] | str]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for item in actions:
        if isinstance(item, str):
            if item:
                result.append({"action": item, "track_object": "object"})
        elif isinstance(item, dict):
            action = str(item.get("action") or item.get("text") or "")
            if action:
                result.append({
                    "action": action,
                    "track_object": str(item.get("track_object") or item.get("mask_prompt") or "object"),
                    **{k: v for k, v in item.items() if k not in {"action", "text", "track_object", "mask_prompt"}},
                })
    return result


def load_plan_actions(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load normalized action records from a saved plan."""
    if path is None:
        return []
    data = _read_json(path)
    if isinstance(data, dict) and "actions" in data:
        return _normalize_actions(data["actions"])
    if isinstance(data, dict) and "best_beam" in data:
        beam = data.get("best_beam", {})
        actions = beam.get("actions", [])
        tracks = beam.get("track_objects", [])
        return _normalize_actions([
            {"action": action, "track_object": tracks[idx] if idx < len(tracks) else "object"}
            for idx, action in enumerate(actions)
        ])
    if not isinstance(data, list):
        raise ValueError(f"Plan actions must be a list or object with actions: {path}")
    return _normalize_actions(data)


def load_illustration_actions(
    sample_root: Optional[Path],
    explicit_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load a labeled example-data action sequence without treating it as a plan."""

    candidates: List[Path] = []
    if explicit_path is not None:
        explicit = Path(explicit_path)
        if not explicit.is_file():
            raise FileNotFoundError(f"Illustration action file does not exist: {explicit}")
        candidates.append(explicit)
    if sample_root is not None:
        root = Path(sample_root)
        candidates.append(root / "illustration_actions.json")
    for path in candidates:
        if path.is_file():
            actions = load_plan_actions(path)
            for action in actions:
                action["source"] = "recorded_example_data_action"
                action["illustration_only"] = True
                action["not_evidence_of_execution"] = True
                action["example_data_path"] = str(path.resolve())
            return actions
    return []


def _step_summary_action(step_dir: Path) -> Optional[Dict[str, str]]:
    summary_path = step_dir / "step_summary.json"
    if not summary_path.exists():
        return None
    data = _read_json(summary_path)
    action = data.get("action") or data.get("prompt")
    if not action:
        return None
    return {
        "action": str(action),
        "track_object": str(data.get("track_object") or data.get("mask_prompt") or "object"),
    }


def _target_image_for_step(step_dir: Path) -> Optional[Path]:
    for candidate in (
        step_dir / "target.png",
        step_dir / "end_frame_rgb.png",
        step_dir / "test_results" / "end_frame_rgb.png",
    ):
        if candidate.exists():
            return candidate
    video = step_dir / "rgb_video_16fps.mp4"
    if not video.exists():
        return None
    target = step_dir / "execution_target_last_frame.png"
    if not target.exists():
        import imageio.v3 as iio

        frames = iio.imread(video)
        Image.fromarray(np.asarray(frames[-1], dtype=np.uint8)).save(target)
    return target


def _resolve_horizon_and_mode(config: ClosedLoopExecutionConfig) -> tuple[Dict[str, Any], str, int]:
    result = dict(config.horizon_result or {})
    if not result:
        result = {
            "horizon": int(config.fallback_horizon),
            "planner_horizon": int(config.fallback_horizon),
            "reactive_execution_horizon": None,
            "plan_in_advance_allowed": config.execution_mode == "strategic",
            "is_coupled": config.execution_mode == "strategic",
            "horizon_source": "manual_or_standalone",
        }
    if config.execution_mode:
        mode = config.execution_mode
    else:
        mode = "strategic" if result.get("plan_in_advance_allowed") else "reactive"
    if mode == "reactive" and result.get("reactive_execution_horizon") is not None:
        horizon = int(result.get("reactive_execution_horizon") or 0)
    else:
        if result.get("horizon") is not None:
            horizon = int(result["horizon"])
        elif result.get("planner_horizon") is not None:
            horizon = int(result["planner_horizon"])
        else:
            horizon = int(config.fallback_horizon)
    return result, mode, max(0, horizon)


def _select_action(
    *,
    step_idx: int,
    step_dir: Optional[Path],
    plan_actions: List[Dict[str, Any]],
    prefer_reactive_vlm: bool,
    vlm: Optional[VLMAdapter],
    current_image: np.ndarray,
    goal: str,
    history: List[str],
    horizon_remaining: int,
    constraint_context: Optional[str] = None,
    fallback_action: Optional[str],
    fallback_track_object: str,
) -> Dict[str, Any]:
    if step_idx < len(plan_actions):
        action = dict(plan_actions[step_idx])
        action.setdefault("source", "planner_in_memory")
        return action
    if prefer_reactive_vlm and vlm is not None:
        proposals = vlm.propose_actions(
            image=current_image,
            goal=goal,
            num_actions=1,
            previous_action=history[-1] if history else None,
            action_history=list(history),
            steps_remaining=horizon_remaining,
            constraint_context=constraint_context,
        )
        action = dict(proposals[0])
        action["source"] = "vlm_reactive"
        return action
    if step_dir is not None:
        summary_action = _step_summary_action(step_dir)
        if summary_action:
            summary_action["source"] = "step_summary"
            return summary_action
    if vlm is not None:
        proposals = vlm.propose_actions(
            image=current_image,
            goal=goal,
            num_actions=1,
            previous_action=history[-1] if history else None,
            action_history=list(history),
            steps_remaining=horizon_remaining,
            constraint_context=constraint_context,
        )
        action = dict(proposals[0])
        action["source"] = "vlm_reactive"
        return action
    if fallback_action:
        return {
            "action": fallback_action,
            "track_object": fallback_track_object,
            "source": "fallback_cli",
        }
    raise RuntimeError("No action source available.")


def _run_execution_step(
    step_dir: Path,
    action: Dict[str, Any],
    config: ClosedLoopExecutionConfig,
    step_index: int,
    *,
    flow_video: Optional[Path] = None,
    out_dir_override: Optional[Path] = None,
    selected_flow_override: Optional[str] = None,
    grounding_mode: str = "standard",
    contact_finger: Optional[str] = None,
    contact_point_2d: Optional[Sequence[float]] = None,
    console_label: Optional[str] = None,
    debug_artifact_dir_override: Optional[Path] = None,
) -> Dict[str, Any]:
    if flow_video is not None and not config.object_flow_server_url:
        raise RuntimeError(
            "Selected rollout video is available, but no remote 3D object-flow server URL is configured. "
            "Set --object_flow_server_url or NOVAPLAN_OBJECT_FLOW_SERVER_URL so the selected "
            "video can be converted into object flow before pose compilation."
        )
    out_dir = (
        Path(out_dir_override)
        if out_dir_override is not None
        else Path(config.output_dir) / f"step_{step_index + 1:03d}" / "execution_step"
    ).resolve()
    debug_artifact_dir = (
        Path(debug_artifact_dir_override)
        if debug_artifact_dir_override is not None
        else Path(config.output_dir)
        / f"step_{step_index + 1:03d}"
        / "debug_artifacts"
        / "geometric_grounding"
        / "selected_rollout"
    ).resolve()
    cmd = [
        sys.executable,
        str(ROOT / "local_planning" / "run_execution_step.py"),
        "--sample_dir",
        str(step_dir),
        "--out_dir",
        str(out_dir),
        "--debug_artifact_dir",
        str(debug_artifact_dir),
        "--action",
        str(action.get("action", "")),
        "--track_object",
        str(action.get("track_object", "object")),
        "--selected_flow",
        selected_flow_override or config.selected_flow,
        "--flow_switch_theta_deg",
        str(config.flow_switch_theta_deg),
        "--hand_flow_interaction_epsilon",
        str(config.hand_flow_interaction_epsilon),
        "--grounding_mode",
        grounding_mode,
    ]
    if contact_finger:
        cmd.extend(["--contact_finger", str(contact_finger)])
    if contact_point_2d is not None:
        point = list(contact_point_2d)
        if len(point) != 2:
            raise ValueError("contact_point_2d must contain exactly [x,y]")
        cmd.extend(["--contact_point_2d", str(float(point[0])), str(float(point[1]))])
    if config.object_flow_server_url:
        cmd.extend(["--object_flow_server_url", config.object_flow_server_url])
        cmd.extend(["--flow_mask_prompt", str(action.get("track_object", "object"))])
        if not config.enable_cvd:
            cmd.append("--disable_cvd")
    if config.hand_flow_server_url:
        cmd.extend(["--hand_flow_server_url", config.hand_flow_server_url])
    if flow_video is not None:
        cmd.extend(["--flow_video", str(flow_video)])
    if config.flow_sam3_debug_video:
        cmd.append("--flow_sam3_debug_video")
    if config.flow_debug_artifacts:
        cmd.append("--flow_debug_artifacts")
    label = console_label or f"Step {step_index + 1}"
    print(
        f"[Object-Centric Grounding] {label}: computing metric object flow for "
        f"'{action.get('track_object', 'object')}'"
    )
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    subprocess_log = out_dir / "execution_step_subprocess.log"
    subprocess_log.parent.mkdir(parents=True, exist_ok=True)
    subprocess_log.write_text(
        "COMMAND\n"
        + " ".join(cmd)
        + "\n\nSTDOUT\n"
        + (proc.stdout or "")
        + "\n\nSTDERR\n"
        + (proc.stderr or "")
    )
    if config.show_execution_stdout:
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "out_dir": str(out_dir),
        "subprocess_log": str(subprocess_log),
    }
    if proc.returncode != 0:
        progress_path = out_dir / "grounding_progress.json"
        if progress_path.exists():
            progress = _read_json(progress_path)
            object_flow = progress.get("object_flow") or {}
            print(
                f"[Object-Centric Grounding] {label}: metric object flow computed "
                f"(valid={object_flow.get('valid')})"
            )
            selection_reason = str(
                progress.get("selection_reason")
                or object_flow.get("reason")
                or "no flow-switch reason reported"
            )
            selected_grounding = str(progress.get("selected_grounding") or "unknown")
            print(
                f"[Flow Switching] {label}: selected {selected_grounding}-centric grounding; "
                f"{_object_flow_console_summary(object_flow, config.flow_switch_theta_deg)}; "
                f"reason={selection_reason}"
            )
        combined_lines = [
            line.strip()
            for line in ((proc.stderr or "") + "\n" + (proc.stdout or "")).splitlines()
            if line.strip()
        ]
        preferred = next(
            (
                line
                for line in reversed(combined_lines)
                if line.startswith(("ValueError:", "RuntimeError:", "ERROR:", "SystemExit:"))
            ),
            combined_lines[-1] if combined_lines else f"exit code {proc.returncode}",
        )
        raise RuntimeError(
            f"run_execution_step.py failed for {step_dir}: {preferred}. "
            f"Full subprocess log: {subprocess_log}"
        )
    summary_path = out_dir / "execution_step.json"
    if summary_path.exists():
        result["execution_summary"] = _read_json(summary_path)
        execution_summary = result["execution_summary"]
        flow_switch = execution_summary.get("flow_switch") or {}
        object_flow = flow_switch.get("object_flow") or {}
        selected_flow = str(execution_summary.get("selected_flow") or "unknown")
        switch_reason = str(
            flow_switch.get("reason")
            or object_flow.get("reason")
            or "no flow-switch reason reported"
        )
        object_flow_summary = _object_flow_console_summary(
            object_flow,
            config.flow_switch_theta_deg,
        )
        print(
            f"[Object-Centric Grounding] {label}: metric object flow computed "
            f"(valid={object_flow.get('valid')})"
        )
        if selected_flow == "hand":
            print(
                f"[Flow Switching] {label}: object flow rejected; "
                f"selected hand-centric grounding; "
                f"{object_flow_summary}; reason={switch_reason}"
            )
            print(f"[Hand-Centric Grounding] {label}: calibrated hand flow computed")
        elif selected_flow == "object":
            print(
                f"[Flow Switching] {label}: selected object-centric grounding; "
                f"{object_flow_summary}; reason={switch_reason}"
            )
        else:
            print(
                f"[Flow Switching] {label}: selected_flow={selected_flow}; "
                f"{object_flow_summary}; reason={switch_reason}"
            )
    return result


def _run_flow_review(
    *,
    step_index: int,
    step_dir: Path,
    grounding: Dict[str, Any],
    config: ClosedLoopExecutionConfig,
    reviewer: Any,
    review_label: Optional[str] = None,
) -> Dict[str, Any]:
    execution_summary = grounding.get("execution_summary") or {}
    artifact_paths = execution_summary.get("artifact_paths") or {}
    flow_source = (
        execution_summary.get("metadata", {}).get("flow_source")
        or artifact_paths.get("object_flow")
    )
    if not flow_source:
        return {
            "status": "skipped",
            "reason": "execution summary did not include object-flow source path",
        }

    out_dir = Path(grounding.get("out_dir", config.output_dir)).resolve()
    decision_path = out_dir / "flow_review_decision.json"
    label = review_label or f"Step {step_index + 1}"
    print(
        f"[Geometric Grounding Review] {label}: Viser ready at "
        f"http://127.0.0.1:{config.debug_flow_review_port}"
    )
    return reviewer.review_step(
        sample_dir=step_dir,
        execution_dir=out_dir,
        flow_npz=Path(flow_source),
        step=step_index + 1,
        label=label,
        decision_path=decision_path,
    )


def _verify_step(
    *,
    vlm: Optional[VLMAdapter],
    start_path: Optional[Path],
    post_path: Optional[Path],
    target_path: Optional[Path],
    action: str,
    goal: str,
    disabled: bool,
) -> Dict[str, Any]:
    if disabled:
        return {"status": "disabled", "success": None, "reason": "verification disabled"}
    if vlm is None:
        return {"status": "skipped", "success": None, "reason": "VLM disabled/unavailable"}
    if start_path is None or post_path is None or target_path is None:
        return {
            "status": "skipped",
            "success": None,
            "reason": "missing start, post, or target image",
            "start_image": str(start_path) if start_path else None,
            "post_image": str(post_path) if post_path else None,
            "target_image": str(target_path) if target_path else None,
        }
    result = vlm.verify_transition(
        start_image=_load_image(start_path),
        current_image=_load_image(post_path),
        target_image=_load_image(target_path),
        action=action,
        goal=goal,
    )
    result["status"] = "completed"
    result["start_image"] = str(start_path)
    result["post_image"] = str(post_path)
    result["target_image"] = str(target_path)
    return result


def _recovery_contact_finger(recovery: Any) -> Optional[str]:
    """Read one explicit contact finger without guessing an ambiguous hand part."""

    candidates: List[str] = []
    direct = getattr(recovery, "contact_finger", None)
    if direct:
        candidates.append(str(direct))
    metadata = getattr(recovery, "metadata", {}) or {}
    raw = metadata.get("raw") if isinstance(metadata, dict) else None
    if isinstance(raw, dict):
        for container in (raw, raw.get("annotation"), raw.get("prompt_P")):
            if not isinstance(container, dict):
                continue
            value = container.get("contact_finger") or container.get("finger")
            if value:
                candidates.append(str(value))
    aliases = {
        "thumb": "thumb", "thumb_tip": "thumb",
        "index": "index", "index_finger": "index", "index_tip": "index",
        "middle": "middle", "middle_finger": "middle", "middle_tip": "middle",
        "ring": "ring", "ring_finger": "ring", "ring_tip": "ring",
        "pinky": "pinky", "little": "pinky", "little_finger": "pinky", "pinky_tip": "pinky",
    }
    normalized = {
        aliases.get(value.strip().lower().replace("-", "_").replace(" ", "_"))
        for value in candidates
    }
    normalized.discard(None)
    return next(iter(normalized)) if len(normalized) == 1 else None


def _relative_transform_path(grounding: Dict[str, Any]) -> Optional[Path]:
    execution_summary = grounding.get("execution_summary") or {}
    value = (execution_summary.get("artifact_paths") or {}).get("relative_ee_transforms")
    return Path(value) if value else None


def _generate_recovery_rollouts(
    client: Any,
    *,
    start_frame: np.ndarray,
    action_text: str,
    num_samples: int,
    num_frames: int,
    fps: int,
    size: str,
    seed: int,
    last_frame: np.ndarray,
    mask_prompt: str,
    wan_action_text: Optional[str] = None,
    veo_action_text: Optional[str] = None,
) -> Any:
    """Call Wan, Veo, or the combined client with the matching negative prompt."""

    kwargs: Dict[str, Any] = {
        "start_frame": start_frame,
        "action_text": action_text,
        "num_samples": num_samples,
        "num_frames": num_frames,
        "fps": fps,
        "size": size,
        "seed": seed,
        "last_frame": last_frame,
        "enable_flow": True,
        "mask_prompt": mask_prompt,
    }
    if hasattr(client, "wan_client") and hasattr(client, "veo_client"):
        # The hybrid adapter dispatches these to their corresponding backend.
        kwargs["wan_action_text"] = wan_action_text or action_text
        kwargs["veo_action_text"] = veo_action_text or build_direct_video_prompt(
            backend="veo3", action=action_text
        )
        kwargs["wan_negative_prompt"] = build_video_negative_prompt("wan22")
        kwargs["veo_negative_prompt"] = build_video_negative_prompt("veo3")
    elif hasattr(client, "base"):
        # The ComfyUI client exposes ``base`` and drives Wan directly.
        kwargs["action_text"] = wan_action_text or action_text
        kwargs["negative_prompt"] = build_video_negative_prompt("wan22")
    else:
        # Veo's adapter (and compatible single-backend adapters) accepts one
        # negative prompt. Unknown test/dummy adapters follow this contract too.
        kwargs["action_text"] = veo_action_text or build_direct_video_prompt(
            backend="veo3", action=action_text
        )
        kwargs["negative_prompt"] = build_video_negative_prompt("veo3")
    return client.generate_rollouts(**kwargs)


def _materialize_grounding_observation(bundle: ObservationBundle, sample_dir: Path) -> Path:
    """Stage an aligned RGB-D observation in the flow client's example-data format."""

    sample_dir = Path(sample_dir).resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(bundle.rgb).save(sample_dir / "start.png")
    if bundle.depth is None:
        raise ValueError("recovery grounding requires an aligned metric depth observation")
    config = dict(bundle.config or {})
    nested = config.get("camera_intrinsics_and_extrinsics") or {}
    depth_scale = observation_depth_scale(config)
    raw_depth = np.rint(np.asarray(bundle.depth, dtype=np.float64) / depth_scale)
    raw_depth = np.clip(raw_depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(raw_depth).save(sample_dir / "start_depth.png")
    if bundle.intrinsics and "intrinsics" not in config:
        config["intrinsics"] = dict(bundle.intrinsics)
    config["depth_scale"] = depth_scale
    if isinstance(nested, dict) and "depth_scale" in nested:
        normalized_nested = dict(nested)
        normalized_nested["depth_scale"] = depth_scale
        config["camera_intrinsics_and_extrinsics"] = normalized_nested
    (sample_dir / "config.json").write_text(json.dumps(config, indent=2))
    return sample_dir


def _run_one_recovery_attempt(
    *,
    attempt_index: int,
    step_index: int,
    step_dir: Path,
    failed_observation: ObservationBundle,
    target_path: Path,
    previous_action: Dict[str, Any],
    recovery: Any,
    config: ClosedLoopExecutionConfig,
    planner: Optional[Any],
    vlm: Optional[VLMAdapter],
    recovery_video_client: Optional[Any],
    history: List[str],
    constraint_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate, rank, and ground one recovery attempt without robot control."""

    attempt_dir = (
        Path(config.output_dir)
        / f"step_{step_index + 1:03d}"
        / "recovery"
        / f"attempt_{attempt_index + 1:03d}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempt_debug_dir = (
        Path(config.output_dir)
        / f"step_{step_index + 1:03d}"
        / "debug_artifacts"
        / "recovery"
        / f"attempt_{attempt_index + 1:03d}"
    )
    attempt_debug_dir.mkdir(parents=True, exist_ok=True)
    decision_payload = recovery.to_dict()
    _write_json(attempt_dir / "recovery_decision.json", decision_payload)
    object_prompt = str(
        getattr(recovery, "object_prompt", "")
        or previous_action.get("track_object", "object")
    )
    is_non_prehensile = bool(recovery.use_non_prehensile_pipeline)
    videos: List[np.ndarray] = []
    flow_images: List[Any] = []
    video_sources: List[str] = []
    annotated_start_shape: Optional[tuple[int, int]] = None

    if is_non_prehensile:
        contact_finger = _recovery_contact_finger(recovery)
        contact_point_2d = getattr(recovery, "contact_point_2d", None)
        if contact_finger is None:
            raise RuntimeError("non-prehensile recovery requires one explicit canonical contact finger")
        if contact_point_2d is None or len(contact_point_2d) != 2:
            raise RuntimeError("non-prehensile recovery requires one annotated contact_point_2d")
        start_frame = _decode_png_base64(getattr(recovery, "annotated_image_png_base64", None))
        if start_frame is None:
            raise RuntimeError("non-prehensile recovery returned no valid annotated start image")
        annotated_start_shape = tuple(int(value) for value in start_frame.shape[:2])
    else:
        start_frame = failed_observation.rgb

    proposed_recovery_action = str(getattr(recovery, "recovery_action", "") or "").strip()
    if not proposed_recovery_action:
        raise RuntimeError("recovery decision returned no recovery_action")
    generation_prompt = str(
        getattr(recovery, "prompt_text", "")
        or getattr(recovery, "recovery_prompt", "")
        or proposed_recovery_action
    ).strip()
    if not generation_prompt:
        raise RuntimeError("recovery decision returned no action or generation prompt")
    if recovery_video_client is None or not hasattr(recovery_video_client, "generate_rollouts"):
        raise RuntimeError("recovery generation requires a Wan/Veo generate_rollouts client")

    recovery_action = {
        "action": proposed_recovery_action,
        "track_object": object_prompt,
        "source": "non_prehensile_recovery" if is_non_prehensile else "grasp_recovery",
        "constraint_context": constraint_context,
    }
    target_frame = _load_image(target_path)
    wan_generation_prompt = generation_prompt
    veo_generation_prompt = generation_prompt
    uses_wan = hasattr(recovery_video_client, "base") or (
        hasattr(recovery_video_client, "wan_client")
        and hasattr(recovery_video_client, "veo_client")
    )
    uses_veo = not hasattr(recovery_video_client, "base")
    if (
        uses_wan
        and not is_non_prehensile
        and vlm is not None
        and bool(getattr(planner, "use_prompt_extension", False))
    ):
        wan_generation_prompt = _call_with_verbose_capture(
            config,
            f"step {step_index + 1} recovery {attempt_index + 1} WAN prompt extension",
            vlm.extend_video_prompt,
            image=start_frame,
            action=generation_prompt,
            goal=config.goal,
            track_object=object_prompt,
            backend="wan22",
            last_image=target_frame,
        )
    if (
        uses_veo
        and not is_non_prehensile
        and vlm is not None
        and bool(getattr(planner, "use_prompt_extension", False))
    ):
        veo_generation_prompt = _call_with_verbose_capture(
            config,
            f"step {step_index + 1} recovery {attempt_index + 1} Veo prompt extension",
            vlm.extend_video_prompt,
            image=start_frame,
            action=generation_prompt,
            goal=config.goal,
            track_object=object_prompt,
            backend="veo3",
            last_image=target_frame,
        )
    backend = str(config.recovery_video_backend)
    print(
        f"[Recovery Video Generation] Step {step_index + 1} attempt {attempt_index + 1}: "
        f"generating {config.recovery_num_videos} candidate(s) per backend with {backend}"
    )
    result = _call_with_verbose_capture(
        config,
        f"step {step_index + 1} recovery {attempt_index + 1} video generation",
        _generate_recovery_rollouts,
        recovery_video_client,
        start_frame=start_frame,
        action_text=generation_prompt,
        num_samples=config.recovery_num_videos,
        num_frames=config.recovery_num_frames,
        fps=config.recovery_fps,
        size=config.recovery_size,
        seed=config.recovery_seed + attempt_index * 1009,
        last_frame=target_frame,
        mask_prompt=object_prompt,
        wan_action_text=wan_generation_prompt,
        veo_action_text=veo_generation_prompt,
    )
    videos = [np.asarray(video, dtype=np.uint8) for video in list(getattr(result, "videos", result))]
    flow_images = list(getattr(result, "flow_images", []))
    video_sources = [str(value) for value in list(getattr(result, "video_sources", []))]
    requested_backends = list(getattr(result, "requested_backends", []) or [])
    if not requested_backends:
        requested_backends = ["wan22", "veo3"] if backend == "both" else [backend]
    requested_samples = int(
        getattr(result, "requested_samples_per_backend", None)
        or config.recovery_num_videos
    )
    backend_errors = dict(getattr(result, "backend_errors", {}) or {})
    backend_counts: Dict[str, int] = {}
    for source in video_sources:
        backend_counts[source] = backend_counts.get(source, 0) + 1
    for requested_backend in requested_backends:
        backend_counts.setdefault(requested_backend, 0)
    pool = ", ".join(
        f"{requested_backend}={backend_counts[requested_backend]}/{requested_samples}"
        for requested_backend in requested_backends
    )
    pool_message = (
        f"[Recovery Video Generation] Step {step_index + 1} attempt {attempt_index + 1}: "
        f"candidate pool ready ({pool})"
    )
    if backend_errors:
        failures = []
        for failed_backend, error in backend_errors.items():
            compact_error = " ".join(str(error).split())
            if len(compact_error) > 300:
                compact_error = compact_error[:297] + "..."
            failures.append(f"{failed_backend} failed: {compact_error}")
        pool_message += "; " + "; ".join(failures)
    print(pool_message)
    if not videos:
        raise RuntimeError("Wan/Veo recovery generation returned no videos")
    if len(flow_images) < len(videos):
        flow_images.extend([None] * (len(videos) - len(flow_images)))
    missing_flow = (
        [idx for idx, flow in enumerate(flow_images[: len(videos)]) if flow is None]
        if len(videos) > 1
        else []
    )
    recovery_flow_client = getattr(planner, "flow_extraction_client", None)
    if missing_flow and recovery_flow_client is not None:
        print(
            f"[Recovery Video Rollout Selection] Step {step_index + 1} "
            f"attempt {attempt_index + 1}: computing 2D selection flow for "
            f"{len(missing_flow)} candidate(s)"
        )
        extracted = _call_with_verbose_capture(
            config,
            f"step {step_index + 1} recovery {attempt_index + 1} selection flow",
            recovery_flow_client.extract_flow,
            videos=[videos[idx] for idx in missing_flow],
            mask_prompt=object_prompt,
            fps=config.recovery_fps,
        )
        for local_idx, candidate_idx in enumerate(missing_flow):
            if local_idx < len(extracted.flow_images):
                flow_images[candidate_idx] = extracted.flow_images[local_idx]
    if not video_sources:
        video_sources = [backend] * len(videos)
    elif len(video_sources) < len(videos):
        video_sources.extend([backend] * (len(videos) - len(video_sources)))

    ranking_candidates = [
        {
            "candidate_id": idx,
            "action": recovery_action["action"],
            "track_object": object_prompt,
            "generation_prompt": (
                wan_generation_prompt if video_sources[idx] == "wan22" else generation_prompt
            ),
            "rollout": video,
            "flow_image": flow_images[idx] if idx < len(flow_images) else None,
            "backend": video_sources[idx],
        }
        for idx, video in enumerate(videos)
    ]
    if len(ranking_candidates) == 1:
        ranked = [dict(ranking_candidates[0], score=1.0, success=True)]
    elif vlm is not None and hasattr(vlm, "rank_rollouts_batch"):
        ranked = _call_with_verbose_capture(
            config,
            f"step {step_index + 1} recovery {attempt_index + 1} candidate ranking",
            vlm.rank_rollouts_batch,
            goal=recovery_action["action"],
            candidates=ranking_candidates,
            top_n=len(ranking_candidates),
            debug_dir=attempt_debug_dir / "video_rollout_selection",
            step=step_index + 1,
        )
    else:
        raise RuntimeError(
            "multiple recovery candidates require the VLM batch ranker; "
            "NovaPlan will not silently execute candidate 0"
        )
    if not ranked:
        raise RuntimeError(
            "recovery ranking returned no candidate with both motion-flow and final-frame evidence"
        )
    selection: Dict[str, Any] = {
        "selection_method": (
            "single_candidate" if len(ranking_candidates) == 1 else "standard_flow_batch_ranking"
        ),
        "recovery_action": recovery_action["action"],
        "selected_index": int(ranked[0].get("candidate_id", 0)),
        "rankings": [
            {
                "candidate_id": int(candidate.get("candidate_id", idx)),
                "backend": video_sources[int(candidate.get("candidate_id", idx))],
                "success": bool(candidate.get("success", False)),
                "score": float(candidate.get("score", 0.0)),
                "reason": str(candidate.get("rank_reason", "")),
            }
            for idx, candidate in enumerate(ranked)
        ],
    }
    print(
        f"[Recovery Video Rollout Selection] Step {step_index + 1} "
        f"attempt {attempt_index + 1}: selected candidate "
        f"{selection['selected_index'] + 1}/{len(videos)} "
        f"(backend={video_sources[selection['selected_index']]}, "
        f"score={selection['rankings'][0]['score']:.3f})"
    )

    saved_videos: List[str] = []
    saved_flows: List[Optional[str]] = []
    for idx, video in enumerate(videos):
        source_backend = video_sources[idx]
        video_path = attempt_dir / f"candidate_{idx:03d}_{source_backend}.mp4"
        _write_video(video_path, video, config.recovery_fps)
        saved_videos.append(str(video_path))
        flow_image = flow_images[idx] if idx < len(flow_images) else None
        if flow_image is None:
            saved_flows.append(None)
        else:
            flow_path = (
                attempt_debug_dir
                / "video_rollout_selection"
                / f"candidate_{idx:03d}_{source_backend}_flow.png"
            )
            flow_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(flow_image, dtype=np.uint8)).save(flow_path)
            saved_flows.append(str(flow_path))

    grounding_sample = _materialize_grounding_observation(
        failed_observation,
        attempt_dir / "grounding_input",
    )
    selected_idx = int(selection["selected_index"])
    if not 0 <= selected_idx < len(saved_videos):
        raise RuntimeError(f"recovery ranker selected invalid candidate {selected_idx}")
    selection.update(
        {
            "selected_video": saved_videos[selected_idx],
            "selected_backend": video_sources[selected_idx],
            "candidates": [
                {
                    "candidate_id": idx,
                    "backend": video_sources[idx],
                    "video": saved_videos[idx],
                    "flow_image": saved_flows[idx],
                }
                for idx in range(len(videos))
            ],
            "grounding": {"candidate_id": selected_idx, "status": "pending"},
        }
    )
    _write_json(attempt_dir / "recovery_video_selection.json", selection)
    contact_finger = _recovery_contact_finger(recovery) if is_non_prehensile else None
    contact_point_2d = getattr(recovery, "contact_point_2d", None) if is_non_prehensile else None
    candidate_contact_point = contact_point_2d
    if is_non_prehensile:
        if annotated_start_shape is None:
            raise RuntimeError("annotated recovery image shape is unavailable")
        source_h, source_w = annotated_start_shape
        candidate_h, candidate_w = videos[selected_idx].shape[1:3]
        candidate_contact_point = [
            float(contact_point_2d[0]) * float(candidate_w) / float(source_w),
            float(contact_point_2d[1]) * float(candidate_h) / float(source_h),
        ]
    grounded = _run_execution_step(
        grounding_sample,
        recovery_action,
        config,
        step_index,
        flow_video=Path(saved_videos[selected_idx]),
        out_dir_override=attempt_dir / f"candidate_{selected_idx:03d}_grounding",
        selected_flow_override="hand" if is_non_prehensile else "auto",
        grounding_mode="non_prehensile" if is_non_prehensile else "grasp",
        contact_finger=contact_finger,
        contact_point_2d=candidate_contact_point,
        console_label=f"Step {step_index + 1} recovery {attempt_index + 1}",
        debug_artifact_dir_override=(
            Path(config.output_dir)
            / f"step_{step_index + 1:03d}"
            / "debug_artifacts"
            / "recovery"
            / f"attempt_{attempt_index + 1:03d}"
            / f"candidate_{selected_idx:03d}"
        ),
    )
    relative_path = _relative_transform_path(grounded)
    if relative_path is None or not relative_path.exists():
        raise RuntimeError("selected recovery video grounding returned no relative_ee_transforms artifact")
    selection.update(
        {
            "selected_index": selected_idx,
            "selected_video": saved_videos[selected_idx],
            "grounding": {
                "candidate_id": selected_idx,
                "status": "grounded",
                "recovery_action": recovery_action["action"],
                "relative_ee_transforms": str(relative_path),
                "contact_point_2d_annotated": contact_point_2d,
                "contact_point_2d_candidate": candidate_contact_point,
            },
        }
    )
    _write_json(attempt_dir / "recovery_video_selection.json", selection)
    print(
        f"[Geometric Grounding] Step {step_index + 1} recovery attempt "
        f"{attempt_index + 1}: selected recovery video {selected_idx + 1} grounded"
    )
    return {
        "attempt": attempt_index + 1,
        "status": "grounded",
        "recovery_mode": recovery.recovery_mode,
        "decision": decision_payload,
        "action": recovery_action,
        "prompt": generation_prompt,
        "wan_generation_prompt": wan_generation_prompt if uses_wan else None,
        "videos": saved_videos,
        "flow_images": saved_flows,
        "video_sources": video_sources,
        "requested_backends": requested_backends,
        "requested_samples_per_backend": requested_samples,
        "backend_errors": backend_errors,
        "selection": selection,
        "grounding": grounded,
        "relative_ee_transforms": str(relative_path),
    }


def _run_recovery_loop(
    *,
    step_index: int,
    step_dir: Path,
    next_step_dir: Optional[Path],
    failed_observation: ObservationBundle,
    target_path: Path,
    previous_action: Dict[str, Any],
    initial_failure_reason: str,
    config: ClosedLoopExecutionConfig,
    planner: Optional[Any],
    vlm: Optional[VLMAdapter],
    recovery_video_client: Optional[Any],
    observation_provider: ObservationProvider,
    history: List[str],
    constraint_context: Optional[str] = None,
    flow_reviewer: Optional[Any] = None,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    current_observation = failed_observation
    failure_reason = initial_failure_reason
    max_attempts = max(0, int(config.max_recovery_attempts))
    current_constraint_context = constraint_context
    if max_attempts == 0:
        return {"status": "failed_no_recovery_attempts", "attempts": attempts}

    batch_attempt_index = 0
    max_hand_regenerations = max(0, int(config.max_hand_flow_regenerations))
    for decision_index in range(max_attempts):
        context = RecoveryContext(
            goal=config.goal,
            current_state="latest post-execution/recovery observation did not verify",
            target_state="selected rollout target frame",
            previous_action=str(previous_action.get("action", "")),
            failure_reason=failure_reason,
            object_prompt=str(previous_action.get("track_object", "object")),
            current_image_path=str(current_observation.rgb_path),
            target_image_path=str(target_path),
            planner_summary={"history": history, "attempt": decision_index + 1},
        )
        recovery = None
        try:
            if vlm is not None and hasattr(vlm, "decide_recovery"):
                recovery = _call_with_verbose_capture(
                    config,
                    f"step {step_index + 1} recovery decision {decision_index + 1}",
                    vlm.decide_recovery,
                    context,
                )
            else:
                recovery = choose_recovery_strategy(context)
        except Exception as exc:
            failure = {
                "attempt": batch_attempt_index + 1,
                "status": "recovery_decision_failed",
                "recovery_mode": None,
                "decision": None,
                "reason": str(exc),
            }
            attempts.append(failure)
            failure_reason = str(exc)
            batch_attempt_index += 1
            continue

        print(
            f"[Recovery Mode Selection] Step {step_index + 1}: "
            f"mode={getattr(recovery, 'recovery_mode', 'unknown')}"
        )
        recovery_action_text = str(getattr(recovery, "recovery_action", "") or "").strip()
        print(f"[Recovery Action Proposal] Step {step_index + 1}: {recovery_action_text}")

        for generation_retry in range(max_hand_regenerations + 1):
            attempt_index = batch_attempt_index
            batch_attempt_index += 1
            try:
                attempt = _run_one_recovery_attempt(
                    attempt_index=attempt_index,
                    step_index=step_index,
                    step_dir=step_dir,
                    failed_observation=current_observation,
                    target_path=target_path,
                    previous_action=previous_action,
                    recovery=recovery,
                    config=config,
                    planner=planner,
                    vlm=vlm,
                    recovery_video_client=recovery_video_client,
                    history=history,
                    constraint_context=current_constraint_context,
                )
            except Exception as exc:
                retryable = _is_retryable_hand_flow_rejection(exc)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "status": "generation_or_grounding_failed",
                        "recovery_mode": getattr(recovery, "recovery_mode", None),
                        "decision": recovery.to_dict(),
                        "generation_retry": generation_retry,
                        "retryable_hand_flow_rejection": retryable,
                        "reason": str(exc),
                    }
                )
                failure_reason = str(exc)
                if retryable and generation_retry < max_hand_regenerations:
                    print(
                        f"[Hand-Flow Calibration] Recovery hand flow rejected: {exc}; "
                        "generating and ranking a fresh recovery batch "
                        f"({generation_retry + 1}/{max_hand_regenerations})"
                    )
                    continue
                break

            attempts.append(attempt)
            current_constraint_context = (
                attempt.get("action", {}).get("constraint_context")
                or current_constraint_context
            )
            relative_path = Path(attempt["relative_ee_transforms"])
            if config.debug_flow_review and flow_reviewer is not None:
                flow_review = _run_flow_review(
                    step_index=step_index,
                    step_dir=step_dir,
                    grounding=attempt["grounding"],
                    config=config,
                    reviewer=flow_reviewer,
                    review_label=f"Step {step_index + 1} recovery {attempt_index + 1}",
                )
                attempt["flow_review"] = flow_review
                if flow_review.get("decision", {}).get("continue") is False:
                    return {
                        "status": "stopped_by_flow_review",
                        "attempts": attempts,
                        "relative_ee_transforms": str(relative_path),
                    }
            observation_id = f"step_{step_index + 1:03d}_recovery_{attempt_index + 1:03d}"
            next_observation = observation_provider.acquire(
                step_index=step_index,
                step_dir=step_dir,
                next_step_dir=next_step_dir,
                output_dir=Path(config.output_dir),
                relative_transforms_path=relative_path,
                observation_id=observation_id,
            )
            # Online providers must return a distinct capture. Illustration
            # replay may explicitly label reuse as a non-evidentiary stand-in.
            if (
                next_observation is not None
                and next_observation.rgb_path.resolve() == current_observation.rgb_path.resolve()
                and not (
                    config.execution_context == "illustration"
                    and bool(next_observation.metadata.get("allow_recovery_state_reuse"))
                )
            ):
                next_observation = None
            if next_observation is None:
                attempt["observation"] = {
                    "status": "awaiting_external_recovery_execution",
                    "observation_id": observation_id,
                    "relative_ee_transforms": str(relative_path),
                    "robot_commanded_by_novaplan": False,
                }
                print(
                    f"[Closed-Loop Execution] Step {step_index + 1} recovery grounded, but no distinct "
                    "post-recovery observation is available; pausing before verification"
                )
                return {
                    "status": "awaiting_external_recovery_execution",
                    "attempts": attempts,
                    "relative_ee_transforms": str(relative_path),
                    "observation_id": observation_id,
                }

            attempt["observation"] = next_observation.to_dict()
            verification = _call_with_verbose_capture(
                config,
                f"step {step_index + 1} recovery {attempt_index + 1} verification",
                _verify_step,
                vlm=vlm,
                start_path=current_observation.rgb_path,
                post_path=next_observation.rgb_path,
                target_path=target_path,
                action=str(attempt["action"].get("action", "")),
                goal=config.goal,
                disabled=config.disable_verification,
            )
            attempt["verification"] = verification
            print(
                f"[State Verification] Step {step_index + 1} recovery: "
                f"success={verification.get('success')} "
                f"reason={verification.get('reason') or verification.get('summary') or ''}"
            )
            if verification.get("success") is True:
                return {
                    "status": "recovered",
                    "attempts": attempts,
                    "observation": next_observation,
                    "constraint_context": current_constraint_context,
                }
            failure_reason = str(verification.get("reason") or "recovery transition did not verify")
            current_observation = next_observation
            break

    return {
        "status": "recovery_attempts_exhausted",
        "attempts": attempts,
        "reason": failure_reason,
    }


def run_closed_loop_execution(
    config: ClosedLoopExecutionConfig,
    *,
    planner: Optional[Any] = None,
    vlm: Optional[VLMAdapter] = None,
    recovery_video_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run closed-loop planning, grounding, and verification."""
    valid_execution_contexts = {"online", *RECORDED_OBSERVATION_CONTEXTS}
    if config.execution_context not in valid_execution_contexts:
        raise ValueError(
            "execution_context must be 'online', 'recorded_observations', or 'illustration'"
        )
    configured_output = config.output_dir or getattr(planner, "debug_dir", None) or _default_run_dir()
    output_dir = Path(configured_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = output_dir
    print(f"[Closed-Loop Execution] Artifacts: {output_dir}")

    planner_status_context = {"label": "Planner"}
    console_stream = sys.stdout

    def planner_status(event: str, payload: Dict[str, Any]) -> None:
        label = planner_status_context["label"]
        if event == "action_proposed":
            message = (
                f"[Action Proposal] {label}: {payload.get('action', '')} "
                f"| track_object='{payload.get('track_object') or 'object'}'"
            )
        elif event == "video_generation_started":
            backends = "+".join(payload.get("backends") or ["configured backend"])
            message = (
                f"[Video Rollout Generation] {label}: generating {payload.get('num_samples')} "
                f"video(s) per backend with {backends}"
            )
        elif event == "video_generation_completed":
            requested = payload.get("requested_backends") or []
            expected = int(payload.get("requested_samples_per_backend") or 0)
            counts = payload.get("backend_counts") or {}
            if requested:
                pool = ", ".join(
                    f"{backend}={int(counts.get(backend, 0))}/{expected}"
                    for backend in requested
                )
            else:
                pool = f"total={int(payload.get('generated_total') or 0)}"
            message = f"[Video Rollout Generation] {label}: candidate pool ready ({pool})"
            errors = payload.get("backend_errors") or {}
            if errors:
                failures = []
                for backend, error in errors.items():
                    compact_error = " ".join(str(error).split())
                    if len(compact_error) > 300:
                        compact_error = compact_error[:297] + "..."
                    failures.append(f"{backend} failed: {compact_error}")
                message += "; " + "; ".join(failures)
        elif event == "video_selected":
            candidate_id = payload.get("candidate_id")
            candidate_number = int(candidate_id) + 1 if candidate_id is not None else "?"
            candidate_count = payload.get("candidate_count", "?")
            message = (
                f"[Physics-Based Video Rollout Selection] {label}: selected candidate "
                f"{candidate_number}/{candidate_count} "
                f"(backend={payload.get('backend') or 'unknown'}, "
                f"score={payload.get('score', 0.0):.3f})"
            )
        else:
            return
        console_stream.write(message + "\n")
        console_stream.flush()

    if planner is not None:
        planner.status_callback = planner_status

    current_image_path = Path(config.input_frame).resolve()
    current_image = _load_image(current_image_path)
    latest_observation: Optional[ObservationBundle] = None
    if planner is not None and config.horizon_result is None:
        config.horizon_result = _call_with_verbose_capture(
            config,
            "task structure and horizon",
            planner.assess_task_structure,
            current_image,
            config.goal,
        )

    horizon_result, execution_mode, execution_horizon = _resolve_horizon_and_mode(config)
    step_dirs = _discover_step_dirs(config, execution_horizon)
    strategic_actions = _normalize_actions(config.plan_actions) if execution_mode == "strategic" else []
    illustration_actions = (
        _normalize_actions(config.illustration_actions)
        if config.execution_context == "illustration"
        else []
    )
    for action in illustration_actions:
        action["source"] = "recorded_example_data_action"
        action["illustration_only"] = True
        action["not_evidence_of_execution"] = True
    if (
        config.execution_context == "illustration"
        and execution_mode == "reactive"
        and len(illustration_actions) < execution_horizon
    ):
        raise RuntimeError(
            "Illustration mode requires one recorded example-data action per reactive horizon step; "
            f"found {len(illustration_actions)} for horizon {execution_horizon}."
        )
    print(
        f"[Task Structure Assessment] mode={execution_mode}; horizon={execution_horizon}; "
        f"coupled={bool(horizon_result.get('is_coupled', False))}"
    )
    if config.execution_context == "illustration":
        print(
            "[Closed-Loop Execution] Context: illustration; example-data actions/observations "
            "are recorded stand-ins and do not represent live robot execution"
        )
    elif config.execution_context == "recorded_observations":
        print(
            "[Closed-Loop Execution] Context: recorded observations; action proposal, video "
            "generation, selection, and grounding are live, while verification observations "
            "are recorded stand-ins"
        )
    else:
        print(
            "[Closed-Loop Execution] Context: online; every verification requires a newly "
            "captured external observation"
        )

    if execution_mode == "strategic" and planner is not None and not strategic_actions and execution_horizon > 0:
        planner_status_context["label"] = "Strategic planning"
        best_beam = _call_with_verbose_capture(
            config,
            "strategic beam search",
            planner.run_strategic_beam_search,
            current_image,
            config.goal,
            horizon_result,
        )
        strategic_actions = _actions_from_beam(best_beam)

    if config.observation_provider is not None:
        observation_provider = config.observation_provider
    elif config.execution_context in RECORDED_OBSERVATION_CONTEXTS:
        observation_provider = RecordedTraceObservationProvider(explicit_post_image=config.post_image)
    else:
        observation_provider = FilesystemObservationProvider(
            Path(config.output_dir or _default_run_dir()) / "external_observations"
        )

    flow_reviewer = None
    if config.debug_flow_review:
        from local_planning.visualization.viser_flow_review import FlowReviewServer

        flow_reviewer = FlowReviewServer(port=config.debug_flow_review_port)

    summary: Dict[str, Any] = {
        "goal": config.goal,
        "input_frame": str(Path(config.input_frame).resolve()),
        "horizon": horizon_result,
        "execution_mode": execution_mode,
        "execution_context": config.execution_context,
        "provenance_notice": (
            "Recorded example data actions and next-step observations are illustration-only stand-ins; "
            "they are not evidence of online robot execution."
            if config.execution_context == "illustration"
            else (
                "Actions and generated rollouts are produced live; recorded next-step observations "
                "are used only as execution stand-ins for verification and recovery."
                if config.execution_context == "recorded_observations"
                else "Actions are grounded online and verification consumes newly captured external observations."
            )
        ),
        "orchestrator": "closed_loop_video_language_planning",
        "planner_debug_dir": str(getattr(planner, "debug_dir", "")) if planner is not None else None,
        "steps": [],
    }
    if execution_horizon == 0:
        summary["termination"] = {
            "status": "task_complete",
            "step": 0,
            "reason": "initial task-structure assessment found no remaining macro-actions",
        }
    history: List[str] = []
    reactive_constraint_context: Optional[str] = None
    for step_idx in range(execution_horizon):
        if execution_mode == "strategic" and step_idx >= len(strategic_actions):
            summary["termination"] = {
                "status": "planned_sequence_completed",
                "step": step_idx + 1,
                "reason": "strategic action queue is empty",
            }
            break
        pre_execution_image_path = current_image_path
        step_dir = step_dirs[step_idx] if step_idx < len(step_dirs) else None
        next_step_dir = step_dirs[step_idx + 1] if (step_idx + 1) < len(step_dirs) else None
        selected_rollout: Optional[np.ndarray] = None
        illustration_action = (
            dict(illustration_actions[step_idx])
            if execution_mode == "reactive" and step_idx < len(illustration_actions)
            else None
        )
        if execution_mode == "reactive" and planner is not None and illustration_action is None:
            planner_status_context["label"] = f"Step {step_idx + 1}"
            reactive_beam = _call_with_verbose_capture(
                config,
                f"step {step_idx + 1} reactive planning and rollout selection",
                planner.plan_next_reactive_step,
                current_image,
                config.goal,
                history=history,
                steps_remaining=execution_horizon - step_idx,
                constraint_context=reactive_constraint_context,
                debug_step_index=step_idx + 1,
            )
            if bool(getattr(reactive_beam, "task_complete", False)):
                summary["steps"].append(
                    {
                        "step": step_idx + 1,
                        "status": "task_complete",
                        "source": "reactive_action_proposal",
                        "reason": "the current real observation visibly satisfies the goal",
                    }
                )
                summary["termination"] = {
                    "status": "task_complete",
                    "step": step_idx + 1,
                    "reason": "reactive VLM returned FINISH before proposing another action",
                }
                break
            action = _action_from_beam(reactive_beam, source="reactive_video_language_planner")
            action["constraint_context"] = getattr(reactive_beam, "constraint_context", None)
            if getattr(reactive_beam, "video", None):
                selected_rollout = reactive_beam.video[-1]
        else:
            if illustration_action is not None:
                action = illustration_action
                print(
                    f"[Action Proposal] Step {step_idx + 1} illustration action: "
                    f"{action['action']} | track_object='{action.get('track_object', 'object')}'"
                )
            else:
                action = _select_action(
                    step_idx=step_idx,
                    step_dir=step_dir,
                    plan_actions=strategic_actions,
                    prefer_reactive_vlm=execution_mode == "reactive",
                    vlm=vlm,
                    current_image=current_image,
                    goal=config.goal,
                    history=history,
                    horizon_remaining=execution_horizon - step_idx,
                    constraint_context=reactive_constraint_context,
                    fallback_action=config.fallback_action,
                    fallback_track_object=config.fallback_track_object,
                )
                print(f"[Action Proposal] Step {step_idx + 1}: {action['action']}")
            if planner is not None and hasattr(planner, "generate_execution_rollout"):
                planner_status_context["label"] = f"Step {step_idx + 1}"
                execution_beam = _call_with_verbose_capture(
                    config,
                    f"step {step_idx + 1} execution rollout selection",
                    planner.generate_execution_rollout,
                    current_image,
                    config.goal,
                    action=action["action"],
                    track_object=action.get("track_object", "object"),
                    history=history,
                    debug_step_index=step_idx + 1,
                )
                if getattr(execution_beam, "video", None):
                    selected_rollout = execution_beam.video[-1]

        if bool(action.get("is_finish")):
            summary["steps"].append(
                {
                    "step": step_idx + 1,
                    "status": "task_complete",
                    "source": action.get("source", "vlm_action_proposal"),
                    "reason": "the current real observation visibly satisfies the goal",
                }
            )
            summary["termination"] = {
                "status": "task_complete",
                "step": step_idx + 1,
                "reason": "VLM returned FINISH before proposing another action",
            }
            break

        rollout_artifacts = _save_selected_rollout(
            output_dir=output_dir,
            step_index=step_idx,
            rollout=selected_rollout,
            fps=config.rollout_fps,
            source=str(action.get("source", execution_mode)),
        )
        flow_video_path = Path(rollout_artifacts["video"]) if rollout_artifacts.get("video") else None

        step_record: Dict[str, Any] = {
            "step": step_idx + 1,
            "step_dir": str(step_dir) if step_dir else None,
            "action": action,
            "selected_rollout": rollout_artifacts or None,
        }
        grounding_step_dir = step_dir
        if latest_observation is not None:
            grounding_step_dir = _materialize_grounding_observation(
                latest_observation,
                output_dir / f"step_{step_idx + 1:03d}" / "grounding_input",
            )
            step_record["grounding_input"] = {
                "source": (
                    "latest_live_observation"
                    if latest_observation.is_live
                    else "latest_recorded_observation"
                ),
                "provenance": latest_observation.provenance,
                "rgb_path": str(latest_observation.rgb_path),
                "materialized_sample_dir": str(grounding_step_dir),
            }
        elif grounding_step_dir is not None:
            step_record["grounding_input"] = {
                "source": "recorded_example_data",
                "sample_dir": str(grounding_step_dir),
            }
        if grounding_step_dir is None:
            step_record["status"] = "needs_rollout_and_flow"
            step_record["reason"] = (
                "No recorded step_dir or prior live RGB-D observation is available "
                "for geometric grounding."
            )
            summary["steps"].append(step_record)
            break

        grounding_attempts: List[Dict[str, Any]] = []
        max_regenerations = max(0, int(config.max_hand_flow_regenerations))
        generation_attempt = 0
        while True:
            attempt_out_dir = None
            if generation_attempt > 0:
                if planner is None or not hasattr(planner, "generate_execution_rollout"):
                    raise RuntimeError(
                        "hand-flow calibration requested rollout regeneration, but no planner/video stack is available"
                    )
                planner_status_context["label"] = f"Step {step_idx + 1} regeneration {generation_attempt}"
                execution_beam = _call_with_verbose_capture(
                    config,
                    f"step {step_idx + 1} rollout regeneration {generation_attempt}",
                    planner.generate_execution_rollout,
                    current_image,
                    config.goal,
                    action=action["action"],
                    track_object=action.get("track_object", "object"),
                    history=history,
                    generation_attempt=generation_attempt,
                    debug_step_index=step_idx + 1,
                )
                if not getattr(execution_beam, "video", None):
                    raise RuntimeError("regenerated execution rollout selection returned no video")
                selected_rollout = execution_beam.video[-1]
                rollout_artifacts = _save_selected_rollout(
                    output_dir=output_dir,
                    step_index=step_idx,
                    rollout=selected_rollout,
                    fps=config.rollout_fps,
                    source=str(action.get("source", execution_mode)),
                    generation_attempt=generation_attempt,
                )
                step_record["selected_rollout"] = rollout_artifacts
                flow_video_path = Path(rollout_artifacts["video"])
                attempt_out_dir = (
                    output_dir
                    / f"step_{step_idx + 1:03d}"
                    / "grounding_regenerations"
                    / f"attempt_{generation_attempt + 1:03d}"
                    / "execution_step"
                )
            try:
                grounding = _run_execution_step(
                    grounding_step_dir,
                    action,
                    config,
                    step_idx,
                    flow_video=flow_video_path,
                    out_dir_override=attempt_out_dir,
                    debug_artifact_dir_override=(
                        output_dir
                        / f"step_{step_idx + 1:03d}"
                        / "debug_artifacts"
                        / "geometric_grounding"
                        / (
                            "selected_rollout"
                            if generation_attempt == 0
                            else f"regeneration_{generation_attempt:03d}"
                        )
                    ),
                )
                grounding_attempts.append(
                    {
                        "generation_attempt": generation_attempt,
                        "status": "grounded",
                        "selected_rollout": rollout_artifacts or None,
                    }
                )
                break
            except Exception as exc:
                retryable = _is_retryable_hand_flow_rejection(exc)
                grounding_attempts.append(
                    {
                        "generation_attempt": generation_attempt,
                        "status": "rejected",
                        "retryable_hand_flow_rejection": retryable,
                        "reason": str(exc),
                        "selected_rollout": rollout_artifacts or None,
                    }
                )
                if not retryable or generation_attempt >= max_regenerations:
                    raise
                generation_attempt += 1
                print(
                    f"[Hand-Flow Calibration] Selected rollout rejected: {exc}; "
                    f"generating and ranking a fresh candidate batch "
                    f"({generation_attempt}/{max_regenerations})"
                )
        step_record["grounding_attempts"] = grounding_attempts
        step_record["grounding"] = grounding
        execution_summary = grounding.get("execution_summary") or {}

        if config.debug_flow_review:
            step_record["flow_review"] = _run_flow_review(
                step_index=step_idx,
                step_dir=grounding_step_dir,
                grounding=grounding,
                config=config,
                reviewer=flow_reviewer,
            )
            if step_record["flow_review"].get("decision", {}).get("continue") is False:
                step_record["status"] = "stopped_by_flow_review"
                summary["steps"].append(step_record)
                break

        target_path = (
            Path(rollout_artifacts["target_image"])
            if rollout_artifacts.get("target_image")
            else _target_image_for_step(step_dir or grounding_step_dir)
        )
        artifact_paths = execution_summary.get("artifact_paths") or {}
        relative_path_value = artifact_paths.get("relative_ee_transforms")
        observation = observation_provider.acquire(
            step_index=step_idx,
            step_dir=step_dir,
            next_step_dir=next_step_dir,
            output_dir=output_dir,
            relative_transforms_path=Path(relative_path_value) if relative_path_value else None,
        )
        step_record["observation"] = observation.to_dict() if observation is not None else {
            "status": "awaiting_external_execution",
            "robot_commanded_by_novaplan": False,
        }
        if observation is None:
            step_record["status"] = "awaiting_external_execution"
            summary["steps"].append(step_record)
            request_path = output_dir / f"step_{step_idx + 1:03d}" / "observation_request.json"
            print(
                f"[Observation Handoff] Step {step_idx + 1}: awaiting external execution "
                f"and a fresh observation; request={request_path}"
            )
            break
        post_path = observation.rgb_path
        verification = _call_with_verbose_capture(
            config,
            f"step {step_idx + 1} verification",
            _verify_step,
            vlm=vlm,
            start_path=pre_execution_image_path,
            post_path=post_path,
            target_path=target_path,
            action=action["action"],
            goal=config.goal,
            disabled=config.disable_verification,
        )
        step_record["verification"] = verification
        print(
            f"[State Verification] Step {step_idx + 1}: "
            f"success={verification.get('success')} "
            f"reason={verification.get('reason') or verification.get('summary') or verification.get('decision')}"
        )

        if verification.get("success") is False:
            recovery_outcome = _run_recovery_loop(
                step_index=step_idx,
                step_dir=step_dir,
                next_step_dir=next_step_dir,
                failed_observation=observation,
                target_path=target_path,
                previous_action=action,
                initial_failure_reason=str(verification.get("reason") or ""),
                config=config,
                planner=planner,
                vlm=vlm,
                recovery_video_client=recovery_video_client,
                observation_provider=observation_provider,
                history=history,
                constraint_context=reactive_constraint_context,
                flow_reviewer=flow_reviewer,
            )
            recovered_observation = recovery_outcome.pop("observation", None)
            step_record["recovery"] = recovery_outcome
            step_record["status"] = recovery_outcome.get("status", "recovery_failed")
            summary["steps"].append(step_record)
            if recovered_observation is not None and step_record["status"] == "recovered":
                history.append(action["action"])
                current_image = recovered_observation.rgb
                current_image_path = recovered_observation.rgb_path.resolve()
                latest_observation = recovered_observation
                reactive_constraint_context = (
                    recovery_outcome.get("constraint_context")
                    or action.get("constraint_context")
                    or reactive_constraint_context
                )
                continue
            summary["termination"] = {
                "status": step_record["status"],
                "step": step_idx + 1,
                "completed_horizon_steps": step_idx,
                "remaining_horizon_steps": execution_horizon - step_idx,
                "reason": recovery_outcome.get("reason") or (
                    "Recovery transforms were grounded, but a distinct post-recovery observation "
                    "is required before verification and the next task step."
                ),
            }
            print(
                f"[Closed-Loop Execution] Paused at step {step_idx + 1}: "
                f"{step_record['status']}; "
                f"{execution_horizon - step_idx} horizon step(s) remain"
            )
            break

        step_record["status"] = "verified" if verification.get("success") else "grounded_unverified"
        summary["steps"].append(step_record)
        history.append(action["action"])
        reactive_constraint_context = action.get("constraint_context") or reactive_constraint_context
        current_image = observation.rgb
        current_image_path = observation.rgb_path.resolve()
        latest_observation = observation

    _write_json(output_dir / "closed_loop_summary.json", summary)
    print(f"[Closed-Loop Execution] Summary: {output_dir / 'closed_loop_summary.json'}")
    return summary
