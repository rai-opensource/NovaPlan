#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Closed-loop video-language planning entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from novaplan.closed_loop_execution import (  # noqa: E402
    ClosedLoopExecutionConfig,
    load_illustration_actions,
    load_plan_actions,
    run_closed_loop_execution,
)
from novaplan.cli_args import (  # noqa: E402
    http_url,
    image_size,
    nonnegative_float,
    nonnegative_int,
    positive_int,
    rotation_degrees,
    tcp_port,
    unit_interval,
    validate_horizon_bounds,
)
from novaplan.flow_extraction_client import FlowExtractionClient  # noqa: E402
from novaplan.llm_client import DEFAULT_OPENAI_VLM_MODEL, VLMAdapter  # noqa: E402
from novaplan.observations import FilesystemObservationProvider  # noqa: E402
from novaplan.planner import (  # noqa: E402
    NovaPlanPlanner,
    _make_veo_adapter,
)
from novaplan.video_generation import (  # noqa: E402
    HybridVideoGenerationClient,
    WanVideoGenerationClient,
)


DEFAULT_VIDEO_SERVER_URL = "http://127.0.0.1:7000"
DEFAULT_OBJECT_FLOW_SERVER_URL = "http://127.0.0.1:7001"
DEFAULT_HAND_FLOW_SERVER_URL = "http://127.0.0.1:8080/predict"


class _TeeStream:
    def __init__(self, console: TextIO, transcript: TextIO, lock: threading.Lock):
        self.console = console
        self.transcript = transcript
        self.lock = lock
        self.encoding = getattr(console, "encoding", "utf-8")

    def write(self, text: str) -> int:
        with self.lock:
            self.console.write(text)
            self.transcript.write(text)
            self.console.flush()
            self.transcript.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.transcript.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.console.fileno()


@contextmanager
def _terminal_transcript(output_dir: Path) -> Iterator[Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "terminal.log"
    lock = threading.Lock()
    with transcript_path.open("a", encoding="utf-8", buffering=1) as transcript:
        stdout_tee = _TeeStream(sys.stdout, transcript, lock)
        stderr_tee = _TeeStream(sys.stderr, transcript, lock)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield transcript_path


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _canonical_url(value: str | None, *env_names: str) -> str | None:
    if value:
        return value
    return _env_first(*env_names)


def _make_vlm(args: argparse.Namespace) -> VLMAdapter | None:
    if args.disable_vlm:
        return None
    task_models = {}
    if args.horizon_vlm_model:
        task_models["horizon"] = args.horizon_vlm_model
    if args.verification_vlm_model:
        task_models["verification"] = args.verification_vlm_model
    if args.recovery_vlm_model:
        task_models["recovery"] = args.recovery_vlm_model
    efforts = {
        "horizon": args.horizon_reasoning_effort,
        "prompt_extension": args.prompt_extension_reasoning_effort,
        "action": args.action_reasoning_effort or args.vlm_reasoning_effort,
        "ranking": args.ranking_reasoning_effort or args.vlm_reasoning_effort,
        "scoring": args.scoring_reasoning_effort or args.vlm_reasoning_effort,
        "verification": args.verification_reasoning_effort or args.vlm_reasoning_effort,
        "recovery": args.recovery_reasoning_effort or args.vlm_reasoning_effort,
    }
    return VLMAdapter(
        model=args.vlm_model,
        reasoning_effort=args.vlm_reasoning_effort,
        task_models=task_models,
        task_reasoning_efforts=efforts,
    )


def _make_video_stack(args: argparse.Namespace) -> tuple[Any, FlowExtractionClient | None, dict[str, Any]]:
    enable_flow = not args.disable_flow
    flow_client = None
    if enable_flow:
        flow_client = FlowExtractionClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )

    wan_overrides = {
        "size": args.wan_size,
        "num_frames": args.segment_T,
        "fps": args.fps,
        "seed": args.seed,
        "fast_mode": args.wan_fast_mode,
    }

    if args.video_backend == "wan22":
        t2v = WanVideoGenerationClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )
    elif args.video_backend == "veo3":
        t2v = _make_veo_adapter(mock=(not args.use_apis), seed=args.seed)
        wan_overrides = {}
    else:
        wan_client = WanVideoGenerationClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )
        veo_client = _make_veo_adapter(mock=(not args.use_apis), seed=args.seed)
        t2v = HybridVideoGenerationClient(
            wan_client=wan_client,
            veo_client=veo_client,
            flow_client=flow_client,
        )
    return t2v, flow_client, wan_overrides


def _make_planner(
    args: argparse.Namespace,
    vlm: VLMAdapter | None,
) -> tuple[NovaPlanPlanner | None, Any, FlowExtractionClient | None]:
    if vlm is None:
        return None, None, None
    t2v, flow_client, wan_overrides = _make_video_stack(args)
    planner = NovaPlanPlanner(
        vlm=vlm,
        t2v=t2v,
        beam_size=args.beam_size,
        num_action_per_beam=args.num_action_per_beam,
        num_video_per_action=args.num_video_per_action,
        execution_num_action_per_step=args.execution_num_action_per_step,
        execution_num_video_per_action=args.execution_num_video_per_action,
        segment_T=args.segment_T,
        fps=args.fps,
        seed=args.seed,
        save_debug_videos=args.save_debug_videos,
        wan_overrides=wan_overrides if args.video_backend in {"wan22", "both"} else None,
        serial_debug=args.serial_debug,
        horizon=args.horizon,
        use_prompt_extension=not args.disable_prompt_extension,
        use_horizon_count=not args.disable_horizon_count,
        allow_horizon_fallback=args.allow_horizon_fallback,
        horizon_count_min=args.horizon_count_min,
        horizon_count_max=args.horizon_count_max,
        enable_flow=not args.disable_flow,
        mask_prompt=args.mask_prompt,
        use_flow_for_scoring=not args.disable_flow_for_scoring,
        flow_switch_theta_deg=args.flow_switch_theta_deg,
        debug_dir=args.debug_dir,
        flow_extraction_client=flow_client,
    )
    return planner, t2v, flow_client


def _make_recovery_video_client(
    args: argparse.Namespace,
    t2v: Any,
    flow_client: FlowExtractionClient | None,
) -> Any:
    """Select recovery generation independently from normal planning/execution."""

    backend = args.recovery_video_backend
    if backend == "wan22":
        if args.video_backend == "wan22":
            return t2v
        if args.video_backend == "both":
            return t2v.wan_client
        return WanVideoGenerationClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )

    if backend == "veo3":
        if args.video_backend == "veo3":
            return t2v
        if args.video_backend == "both":
            return t2v.veo_client
        return _make_veo_adapter(mock=(not args.use_apis), seed=args.seed)

    if args.video_backend == "both":
        return t2v
    wan_client = (
        t2v
        if args.video_backend == "wan22"
        else WanVideoGenerationClient(
            server_base=args.video_server_url,
            flow_switch_theta_deg=args.flow_switch_theta_deg,
        )
    )
    veo_client = (
        t2v
        if args.video_backend == "veo3"
        else _make_veo_adapter(mock=(not args.use_apis), seed=args.seed)
    )
    return HybridVideoGenerationClient(
        wan_client=wan_client,
        veo_client=veo_client,
        flow_client=flow_client,
    )


def _run_configured(args: argparse.Namespace, run_output_dir: Path) -> None:
    vlm = _make_vlm(args)
    planner, t2v, flow_client = _make_planner(args, vlm)
    recovery_video_client = (
        _make_recovery_video_client(args, t2v, flow_client)
        if t2v is not None
        else None
    )
    observation_provider = None
    if args.execution_context == "online":
        observation_root = args.observation_dir or (run_output_dir / "external_observations")
        observation_provider = FilesystemObservationProvider(
            observation_root,
            wait_seconds=args.observation_wait_seconds,
        )

    config = ClosedLoopExecutionConfig(
        goal=args.goal,
        input_frame=args.input_frame,
        output_dir=run_output_dir,
        sample_root=args.sample_root,
        step_dirs=args.step_dir or [],
        step_glob=args.step_glob,
        plan_actions=load_plan_actions(args.plan_actions_json),
        illustration_actions=(
            load_illustration_actions(args.sample_root, args.illustration_actions_json)
            if args.execution_context == "illustration"
            else []
        ),
        execution_context=args.execution_context,
        fallback_horizon=args.horizon,
        execution_mode=args.execution_mode_override,
        fallback_action=args.fallback_action,
        fallback_track_object=args.fallback_track_object,
        object_flow_server_url=args.object_flow_server_url,
        enable_cvd=not args.disable_cvd,
        hand_flow_server_url=args.hand_flow_server_url,
        flow_sam3_debug_video=args.flow_sam3_debug_video,
        flow_debug_artifacts=args.flow_debug_artifacts,
        selected_flow=args.selected_flow,
        flow_switch_theta_deg=args.flow_switch_theta_deg,
        hand_flow_interaction_epsilon=args.hand_flow_interaction_epsilon,
        post_image=args.post_image,
        max_recovery_attempts=args.max_recovery_attempts,
        recovery_video_backend=args.recovery_video_backend,
        disable_verification=args.disable_verification,
        recovery_num_videos=max(1, int(args.recovery_num_videos)),
        max_hand_flow_regenerations=max(0, int(args.max_hand_flow_regenerations)),
        rollout_fps=args.fps,
        recovery_fps=args.fps,
        recovery_num_frames=args.segment_T,
        recovery_size=args.wan_size,
        recovery_seed=args.seed,
        debug_flow_review=args.debug_flow_review,
        debug_flow_review_port=args.debug_flow_review_port,
        show_execution_stdout=args.show_execution_stdout,
        observation_provider=observation_provider,
    )
    run_closed_loop_execution(
        config,
        planner=planner,
        vlm=vlm,
        recovery_video_client=recovery_video_client,
    )


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Run NovaPlan closed-loop video-language planning.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--input_frame", type=Path, required=True)
    parser.add_argument("--sample_root", type=Path, default=None, help="Directory containing step_XXX folders.")
    parser.add_argument("--step_dir", type=Path, nargs="*", default=None, help="Explicit per-step sample dirs.")
    parser.add_argument("--step_glob", default="step_*")
    parser.add_argument("--plan_actions_json", type=Path, default=None, help="Optional fixed plan for smoke tests.")
    parser.add_argument(
        "--illustration_actions_json",
        type=Path,
        default=None,
        help=(
            "Optional recorded action sequence for --execution_context illustration. "
            "Defaults to SAMPLE_ROOT/illustration_actions.json."
        ),
    )
    parser.add_argument("--fallback_action", default=None, help="Manual action used only when planner/VLM are disabled.")
    parser.add_argument("--fallback_track_object", default="object")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Run artifact directory. Defaults to runs/closed_loop/<timestamp>/.",
    )
    parser.add_argument("--debug_dir", type=Path, default=None)

    parser.add_argument("--horizon", type=positive_int, default=3, help="Fallback horizon if task-structure assessment is disabled.")
    parser.add_argument("--horizon_count_min", type=positive_int, default=1)
    parser.add_argument("--horizon_count_max", type=positive_int, default=int(os.getenv("NOVAPLAN_HORIZON_COUNT_MAX", "8")))
    parser.add_argument("--disable_horizon_count", action="store_true")
    parser.add_argument(
        "--allow_horizon_fallback",
        action="store_true",
        help="Allow fallback to --horizon if automatic task-structure/horizon counting fails.",
    )
    parser.add_argument("--execution_mode_override", choices=("strategic", "reactive"), default=None, help="Debug override; normally let the planner decide.")
    parser.add_argument("--disable_vlm", action="store_true")
    reasoning_effort_choices = ("none", "low", "medium", "high", "xhigh", "max")
    parser.add_argument("--vlm_model", default=os.getenv("NOVAPLAN_VLM_MODEL", os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_VLM_MODEL)))
    parser.add_argument("--vlm_reasoning_effort", choices=reasoning_effort_choices, default=os.getenv("NOVAPLAN_VLM_REASONING_EFFORT"))
    parser.add_argument("--horizon_vlm_model", default=os.getenv("NOVAPLAN_HORIZON_VLM_MODEL"))
    parser.add_argument("--horizon_reasoning_effort", choices=reasoning_effort_choices, default=os.getenv("NOVAPLAN_HORIZON_REASONING_EFFORT", "low"))
    parser.add_argument("--action_reasoning_effort", choices=reasoning_effort_choices, default=os.getenv("NOVAPLAN_ACTION_REASONING_EFFORT"))
    parser.add_argument("--prompt_extension_reasoning_effort", choices=reasoning_effort_choices, default=os.getenv("NOVAPLAN_PROMPT_EXTENSION_REASONING_EFFORT", "low"))
    parser.add_argument("--ranking_reasoning_effort", choices=reasoning_effort_choices, default=os.getenv("NOVAPLAN_RANKING_REASONING_EFFORT"))
    parser.add_argument("--scoring_reasoning_effort", choices=reasoning_effort_choices, default=os.getenv("NOVAPLAN_SCORING_REASONING_EFFORT"))
    parser.add_argument("--verification_vlm_model", default=os.getenv("NOVAPLAN_VERIFICATION_VLM_MODEL"))
    parser.add_argument(
        "--verification_reasoning_effort",
        choices=reasoning_effort_choices,
        default=os.getenv("NOVAPLAN_VERIFICATION_REASONING_EFFORT", "low"),
    )
    parser.add_argument("--recovery_vlm_model", default=os.getenv("NOVAPLAN_RECOVERY_VLM_MODEL"))
    parser.add_argument(
        "--recovery_reasoning_effort",
        choices=reasoning_effort_choices,
        default=os.getenv("NOVAPLAN_RECOVERY_REASONING_EFFORT", "low"),
    )

    parser.add_argument("--video_backend", choices=("wan22", "veo3", "both"), default="both")
    parser.add_argument(
        "--use_apis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use live video APIs; pass --no-use_apis for the local Veo mock adapter.",
    )
    parser.add_argument("--beam_size", type=positive_int, default=2)
    parser.add_argument("--num_action_per_beam", type=positive_int, default=2, help="Strategic actions per beam (paper setting: 2).")
    parser.add_argument("--num_video_per_action", type=positive_int, default=4, help="Strategic videos per action per enabled backend (paper setting: 4).")
    parser.add_argument("--execution_num_action_per_step", type=positive_int, default=1, help="Reactive execution actions proposed per online step.")
    parser.add_argument("--execution_num_video_per_action", type=positive_int, default=8, help="Reactive execution videos per action per enabled backend.")
    parser.add_argument("--segment_T", type=positive_int, default=41)
    parser.add_argument("--fps", type=positive_int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_prompt_extension", action="store_true")
    parser.add_argument("--save_debug_videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serial_debug", action="store_true")

    parser.add_argument(
        "--video_server_url",
        type=http_url,
        default=_env_first("NOVAPLAN_VIDEO_SERVER_URL") or DEFAULT_VIDEO_SERVER_URL,
        help="NovaPlan video-generation server URL.",
    )
    parser.add_argument(
        "--wan_server",
        dest="video_server_url",
        type=http_url,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--wan_size", type=image_size, default="1280*720")
    parser.add_argument("--wan_fast_mode", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--disable_flow", action="store_true", help="Disable 2D flow extraction during video ranking.")
    parser.add_argument("--disable_flow_for_scoring", action="store_true")
    parser.add_argument("--mask_prompt", default="object", help="Fallback flow prompt if the VLM omits track_object.")
    parser.add_argument(
        "--flow_switch_theta_deg",
        type=rotation_degrees,
        default=45.0,
        help="Maximum adjacent-frame rotation in degrees for object-flow validity and hand-flow switching.",
    )
    parser.add_argument(
        "--hand_flow_interaction_epsilon",
        type=unit_interval,
        default=0.9,
        help=(
            "Object-mask new-support ratio used to detect the hand interaction interval "
            "(paper Appendix D.2, epsilon; default: 0.9)."
        ),
    )
    parser.add_argument(
        "--object_flow_server_url",
        type=http_url,
        default=_env_first("NOVAPLAN_OBJECT_FLOW_SERVER_URL")
        or DEFAULT_OBJECT_FLOW_SERVER_URL,
        help="Remote 3D object-flow server used by per-step geometric grounding.",
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
    parser.add_argument("--flow_sam3_debug_video", action="store_true")
    parser.add_argument("--flow_debug_artifacts", action="store_true")
    parser.add_argument(
        "--disable_cvd",
        action="store_true",
        help="Use MoGe2 plus metric calibration without optional CVD for 3D flow grounding.",
    )
    parser.add_argument(
        "--debug_flow_review",
        action="store_true",
        help="After each execution grounding, open a Viser flow/pointcloud review and wait for Continue/Stop.",
    )
    parser.add_argument("--debug_flow_review_port", type=tcp_port, default=8097)
    parser.add_argument(
        "--show_execution_stdout",
        action="store_true",
        help="Print run_execution_step.py stdout/stderr even when grounding succeeds.",
    )
    parser.add_argument("--selected_flow", choices=("auto", "object", "hand"), default="auto")

    parser.add_argument(
        "--post_image",
        type=Path,
        default=None,
        help="Optional recorded first-step post image for a recorded-observation context.",
    )
    parser.add_argument(
        "--execution_context",
        choices=("online", "recorded_observations", "illustration"),
        default="online",
        help=(
            "online waits for newly captured external observations; recorded_observations keeps "
            "planning/generation live and replays only next-step observations; illustration "
            "replays both labeled example-data actions and observations."
        ),
    )
    parser.add_argument(
        "--observation_mode",
        choices=("recorded", "filesystem"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--observation_dir",
        type=Path,
        default=None,
        help="Filesystem observation inbox root; defaults to OUTPUT_DIR/external_observations.",
    )
    parser.add_argument(
        "--observation_wait_seconds",
        type=nonnegative_float,
        default=0.0,
        help="Optional bounded wait for an external READY marker; zero returns awaiting_external_execution immediately.",
    )
    parser.add_argument("--disable_verification", action="store_true")
    parser.add_argument("--max_recovery_attempts", type=nonnegative_int, default=1)
    parser.add_argument(
        "--recovery_video_backend",
        choices=("wan22", "veo3", "both"),
        default=os.getenv("NOVAPLAN_RECOVERY_VIDEO_BACKEND", "both"),
        help=(
            "Recovery video backend. The release ranks WAN and Veo by default; use veo3 "
            "to reproduce the paper's non-prehensile recovery backend exactly."
        ),
    )
    parser.add_argument(
        "--recovery_num_videos",
        type=positive_int,
        default=8,
        help=(
            "Recovery candidates per enabled backend. Values above one trigger flow-based ranking."
        ),
    )
    parser.add_argument(
        "--max_hand_flow_regenerations",
        type=nonnegative_int,
        default=2,
        help=(
            "Fresh rollout batches to generate and re-rank when execution or recovery "
            "hand-flow calibration is rejected."
        ),
    )
    args = parser.parse_args()
    if hasattr(args, "observation_mode"):
        args.execution_context = {
            "recorded": "recorded_observations",
            "filesystem": "online",
        }[args.observation_mode]
    if args.execution_context != "illustration" and args.illustration_actions_json is not None:
        parser.error("--illustration_actions_json requires --execution_context illustration")
    if args.execution_context == "online" and args.post_image is not None:
        parser.error("--post_image requires a recorded-observation execution context")
    if args.execution_context != "online" and args.observation_dir is not None:
        parser.error("--observation_dir is only used with --execution_context online")
    if args.execution_context != "online" and args.observation_wait_seconds > 0:
        parser.error("--observation_wait_seconds is only used with --execution_context online")
    validate_horizon_bounds(
        parser,
        minimum=args.horizon_count_min,
        maximum=args.horizon_count_max,
    )
    args.video_server_url = (
        _canonical_url(args.video_server_url, "NOVAPLAN_VIDEO_SERVER_URL")
        or DEFAULT_VIDEO_SERVER_URL
    )
    args.object_flow_server_url = _canonical_url(
        args.object_flow_server_url,
        "NOVAPLAN_OBJECT_FLOW_SERVER_URL",
    ) or DEFAULT_OBJECT_FLOW_SERVER_URL
    args.hand_flow_server_url = _canonical_url(
        args.hand_flow_server_url,
        "NOVAPLAN_HAND_FLOW_SERVER_URL",
    ) or DEFAULT_HAND_FLOW_SERVER_URL
    if args.debug_dir is None:
        args.debug_dir = args.output_dir or Path(
            ROOT / "runs" / "closed_loop" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    run_output_dir = args.output_dir or args.debug_dir
    with _terminal_transcript(run_output_dir) as transcript_path:
        print(f"[Closed-Loop Execution] Terminal transcript: {transcript_path}")
        _run_configured(args, Path(run_output_dir))


if __name__ == "__main__":
    main()
