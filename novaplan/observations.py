# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Observation hand-off contracts for controller-agnostic closed-loop runs.

NovaPlan produces relative end-effector transforms but intentionally does not
command a robot. An observation provider is the narrow boundary between that
output and the next planning iteration: showcase runs can replay only the next
recorded RGB-D start state while keeping model decisions live, and an online
experiment can atomically deposit a fresh RGB-D bundle after an external
controller has consumed the transforms.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

import numpy as np
from PIL import Image

DEFAULT_OBSERVATION_DEPTH_SCALE_M = 0.001


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}


def _file_sha256(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_intrinsics(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    candidates = [
        config.get("intrinsics"),
        config.get("depth_intrinsics"),
        (config.get("camera_intrinsics_and_extrinsics") or {}).get("depth_intrinsics"),
    ]
    for value in candidates:
        if not isinstance(value, dict):
            continue
        fx = value.get("fx")
        fy = value.get("fy")
        cx = value.get("cx", value.get("ppx"))
        cy = value.get("cy", value.get("ppy"))
        if all(item is not None for item in (fx, fy, cx, cy)):
            return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}
    return None


def observation_depth_scale(config: Dict[str, Any]) -> float:
    """Return the meters-per-unit scale for an encoded observation depth image."""

    nested = config.get("camera_intrinsics_and_extrinsics") or {}
    value = config.get("depth_scale")
    if value is None and isinstance(nested, dict):
        value = nested.get("depth_scale")
    if value is None:
        return DEFAULT_OBSERVATION_DEPTH_SCALE_M
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation depth_scale must be a finite positive number") from exc
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("observation depth_scale must be a finite positive number")
    return scale


def _load_depth(path: Optional[Path], config: Dict[str, Any]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        depth = np.asarray(np.load(path), dtype=np.float32)
    else:
        depth = np.asarray(Image.open(path), dtype=np.float32) * observation_depth_scale(config)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Observation depth must be HxW, got {depth.shape} from {path}")
    return depth


@dataclass
class ObservationBundle:
    """One post-execution RGB-D observation and its provenance."""

    rgb: np.ndarray
    rgb_path: Path
    provenance: str
    is_live: bool
    step_index: int
    depth: Optional[np.ndarray] = None
    depth_path: Optional[Path] = None
    intrinsics: Optional[Dict[str, float]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None
    captured_at: str = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rgb = np.asarray(self.rgb, dtype=np.uint8)
        if self.rgb.ndim != 3 or self.rgb.shape[-1] != 3:
            raise ValueError(f"Observation RGB must be HxWx3, got {self.rgb.shape}")
        if self.depth is not None and self.depth.shape != self.rgb.shape[:2]:
            raise ValueError(
                "Observation RGB/depth must be pixel-aligned: "
                f"rgb={self.rgb.shape[:2]} depth={self.depth.shape}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this value to a JSON-compatible dictionary."""
        return {
            "step": self.step_index + 1,
            "rgb_path": str(self.rgb_path),
            "depth_path": str(self.depth_path) if self.depth_path else None,
            "config_path": str(self.config_path) if self.config_path else None,
            "rgb_shape": list(self.rgb.shape),
            "depth_shape": list(self.depth.shape) if self.depth is not None else None,
            "intrinsics": self.intrinsics,
            "provenance": self.provenance,
            "is_live": self.is_live,
            "captured_at": self.captured_at,
            "metadata": self.metadata,
        }


class ObservationProvider(Protocol):
    """Define the interface for acquiring post-execution observations."""
    def acquire(
        self,
        *,
        step_index: int,
        step_dir: Optional[Path],
        next_step_dir: Optional[Path],
        output_dir: Path,
        relative_transforms_path: Optional[Path],
        observation_id: Optional[str] = None,
    ) -> Optional[ObservationBundle]:
        """Acquire the next observation bundle."""
        ...


def load_observation_bundle(
    *,
    rgb_path: Path,
    depth_path: Optional[Path],
    config_path: Optional[Path],
    provenance: str,
    is_live: bool,
    step_index: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> ObservationBundle:
    """Load and validate one RGB-D observation bundle."""
    rgb_path = Path(rgb_path).resolve()
    config_path = Path(config_path).resolve() if config_path and Path(config_path).exists() else None
    depth_path = Path(depth_path).resolve() if depth_path and Path(depth_path).exists() else None
    config = _read_json(config_path)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = _load_depth(depth_path, config)
    return ObservationBundle(
        rgb=rgb,
        rgb_path=rgb_path,
        depth=depth,
        depth_path=depth_path,
        intrinsics=_nested_intrinsics(config),
        config=config,
        config_path=config_path,
        provenance=provenance,
        is_live=is_live,
        step_index=step_index,
        metadata=dict(metadata or {}),
    )


class RecordedTraceObservationProvider:
    """Replay post-execution observation stand-ins from immutable example data."""

    def __init__(self, explicit_post_image: Optional[Path] = None):
        self.explicit_post_image = Path(explicit_post_image) if explicit_post_image else None

    @staticmethod
    def _companions(rgb_path: Path, base_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
        stem = rgb_path.stem
        if stem == "start":
            names = [base_dir / "start_depth.npy", base_dir / "start_depth.png"]
        elif stem == "end_frame_rgb":
            names = [base_dir / "end_frame_depth.npy", base_dir / "end_frame_depth.png"]
        else:
            names = [
                rgb_path.with_name(f"{stem}_depth.npy"),
                rgb_path.with_name(f"{stem}_depth.png"),
                base_dir / "depth.npy",
                base_dir / "depth.png",
            ]
        depth = next((path for path in names if path.exists()), None)
        configs = [base_dir / "config.json", base_dir / "intrinsics.json"]
        config = next((path for path in configs if path.exists()), None)
        return depth, config

    def acquire(
        self,
        *,
        step_index: int,
        step_dir: Optional[Path],
        next_step_dir: Optional[Path],
        output_dir: Path,
        relative_transforms_path: Optional[Path],
        observation_id: Optional[str] = None,
    ) -> Optional[ObservationBundle]:
        """Acquire the next observation bundle."""
        del output_dir
        is_recovery = bool(observation_id and "_recovery_" in observation_id)
        candidate: Optional[Path] = None
        provenance = ""
        base_dir: Optional[Path] = None
        # An explicit post image represents the normal action outcome only.
        # Recovery must consume a distinct recorded state, just as an online
        # run requires a new camera capture after executing recovery.
        if (
            not is_recovery
            and step_index == 0
            and self.explicit_post_image
            and self.explicit_post_image.exists()
        ):
            candidate = self.explicit_post_image
            provenance = "recorded_explicit_post_image"
            base_dir = candidate.parent
        elif step_dir is not None and (Path(step_dir) / "post.png").exists():
            candidate = Path(step_dir) / "post.png"
            provenance = "recorded_current_step_post"
            base_dir = Path(step_dir)
        elif next_step_dir is not None and (Path(next_step_dir) / "start.png").exists():
            candidate = Path(next_step_dir) / "start.png"
            provenance = (
                "recorded_next_step_start_recovery_standin"
                if is_recovery
                else "recorded_next_step_start_standin"
            )
            base_dir = Path(next_step_dir)
        elif step_dir is not None and (Path(step_dir) / "end_frame_rgb.png").exists():
            candidate = Path(step_dir) / "end_frame_rgb.png"
            provenance = (
                "recorded_current_step_end_frame_recovery_standin"
                if is_recovery
                else "recorded_current_step_end_frame_standin"
            )
            base_dir = Path(step_dir)
        if candidate is None or base_dir is None:
            return None
        depth, config = self._companions(candidate, base_dir)
        return load_observation_bundle(
            rgb_path=candidate,
            depth_path=depth,
            config_path=config,
            provenance=provenance,
            is_live=False,
            step_index=step_index,
            metadata={
                "example_data_replay": True,
                "illustration_only": True,
                "not_evidence_of_execution": True,
                "observation_role": (
                    "post_recovery_standin" if is_recovery else "post_action_standin"
                ),
                "allow_recovery_state_reuse": is_recovery,
                "robot_commanded_by_novaplan": False,
                "relative_transforms_path": (
                    str(relative_transforms_path) if relative_transforms_path else None
                ),
            },
        )


class FilesystemObservationProvider:
    """Read observations atomically deposited by an external experiment.

    For step N, the external process writes ``rgb.png`` plus optional
    ``depth.npy``/``depth.png`` and ``config.json`` beneath ``step_NNN/``.  It
    atomically writes ``READY`` with the matching request token only after every
    file is complete. NovaPlan writes an
    ``observation_request.json`` beside its transform output so no controller
    integration is required in this repository.
    """

    def __init__(self, root: Path, *, wait_seconds: float = 0.0, poll_seconds: float = 0.25):
        self.root = Path(root).resolve()
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.poll_seconds = max(0.05, float(poll_seconds))

    def acquire(
        self,
        *,
        step_index: int,
        step_dir: Optional[Path],
        next_step_dir: Optional[Path],
        output_dir: Path,
        relative_transforms_path: Optional[Path],
        observation_id: Optional[str] = None,
    ) -> Optional[ObservationBundle]:
        """Acquire the next observation bundle."""
        del step_dir, next_step_dir
        bundle_id = str(observation_id or f"step_{step_index + 1:03d}")
        if not bundle_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in bundle_id):
            raise ValueError("observation_id may contain only letters, numbers, '_' and '-'")
        inbox = self.root / bundle_id
        request_name = "observation_request.json" if observation_id is None else f"observation_request_{bundle_id}.json"
        request_path = Path(output_dir) / f"step_{step_index + 1:03d}" / request_name
        request_path.parent.mkdir(parents=True, exist_ok=True)
        relative_path_text = str(relative_transforms_path) if relative_transforms_path else None
        transform_sha256 = _file_sha256(relative_transforms_path)
        existing_request = _read_json(request_path) if request_path.exists() else {}
        resume_pending = bool(
            existing_request
            and existing_request.get("status") == "awaiting_external_execution"
            and existing_request.get("observation_id") == bundle_id
            and existing_request.get("relative_ee_transforms") == relative_path_text
            and existing_request.get("relative_ee_transforms_sha256") == transform_sha256
            and existing_request.get("request_id")
        )
        if resume_pending:
            request_payload = existing_request
            request_id = str(existing_request["request_id"])
        else:
            request_id = uuid.uuid4().hex
            request_payload = {
                "status": "awaiting_external_execution",
                "step": step_index + 1,
                "observation_id": bundle_id,
                "request_id": request_id,
                "relative_ee_transforms": relative_path_text,
                "relative_ee_transforms_sha256": transform_sha256,
                "write_bundle_to": str(inbox),
                "required": ["rgb.png", "READY (JSON containing the matching request_id)"],
                "optional": ["depth.npy", "depth.png", "config.json", "intrinsics.json"],
                "depth_requirement": (
                    "Aligned metric depth and camera intrinsics are required before "
                    "NovaPlan can geometrically ground another normal or recovery step; "
                    "RGB alone is sufficient only for terminal verification."
                ),
                "atomicity": (
                    "After all files are complete, atomically write READY as JSON: "
                    '{"request_id": "<value from this request>"}.'
                ),
                "requested_at": _utc_now(),
            }
            request_path.write_text(json.dumps(request_payload, indent=2))

        def ready_matches_request() -> bool:
            ready_path = inbox / "READY"
            if not ready_path.exists():
                return False
            try:
                ready_payload = json.loads(ready_path.read_text())
            except (OSError, json.JSONDecodeError):
                return False
            return (
                isinstance(ready_payload, dict)
                and str(ready_payload.get("request_id") or "") == request_id
            )

        deadline = time.monotonic() + self.wait_seconds
        while not ready_matches_request():
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))

        rgb_path = inbox / "rgb.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"{inbox / 'READY'} exists but required {rgb_path} is missing")
        depth_path = next(
            (path for path in (inbox / "depth.npy", inbox / "depth.png") if path.exists()),
            None,
        )
        config_path = next(
            (path for path in (inbox / "config.json", inbox / "intrinsics.json") if path.exists()),
            None,
        )
        bundle = load_observation_bundle(
            rgb_path=rgb_path,
            depth_path=depth_path,
            config_path=config_path,
            provenance="external_filesystem_observation",
            is_live=True,
            step_index=step_index,
            metadata={
                "ready_marker": str(inbox / "READY"),
                "request_path": str(request_path),
                "request_id": request_id,
                "resumed_pending_request": resume_pending,
                "robot_commanded_by_novaplan": False,
            },
        )
        request_payload["status"] = "observation_consumed"
        request_payload["consumed_at"] = _utc_now()
        request_path.write_text(json.dumps(request_payload, indent=2))
        return bundle


class CallbackObservationProvider:
    """In-process adapter for experiments that already expose an observation callback."""

    def __init__(self, callback: Callable[..., Optional[ObservationBundle]]):
        self.callback = callback

    def acquire(self, **kwargs: Any) -> Optional[ObservationBundle]:
        """Acquire the next observation bundle."""
        bundle = self.callback(**kwargs)
        if bundle is not None and not isinstance(bundle, ObservationBundle):
            raise TypeError("Observation callback must return ObservationBundle or None")
        return bundle
