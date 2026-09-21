#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.closed_loop_execution import (  # noqa: E402
    ClosedLoopExecutionConfig,
    _generate_recovery_rollouts,
    _run_one_recovery_attempt,
    _run_recovery_loop,
)
from novaplan.observations import (  # noqa: E402
    ObservationBundle,
    RecordedTraceObservationProvider,
)


def _image(path: Path, value: int) -> np.ndarray:
    arr = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return arr


def _bundle(path: Path, value: int, *, step: int = 0) -> ObservationBundle:
    rgb = _image(path, value)
    return ObservationBundle(
        rgb=rgb,
        rgb_path=path,
        provenance="unit_test",
        is_live=True,
        step_index=step,
        depth=np.ones((16, 16), dtype=np.float32),
        intrinsics={"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0},
        config={"intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0}},
    )


class _Recovery:
    recovery_mode = "grasp"
    use_non_prehensile_pipeline = False
    object_prompt = "block"
    recovery_action = "Regrasp the block and place it at the target."
    recovery_prompt = ""
    prompt_text = None

    def to_dict(self):
        return {
            "recovery_mode": self.recovery_mode,
            "recovery_action": self.recovery_action,
        }


class RecoveryGroundingTest(unittest.TestCase):
    def test_illustration_recovery_can_verify_against_labeled_next_step_standin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = root / "step_001"
            step2 = root / "step_002"
            step1.mkdir()
            step2.mkdir()
            _image(step2 / "start.png", 60)
            target = step1 / "end_frame_rgb.png"
            _image(target, 80)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            provider = RecordedTraceObservationProvider()
            failed = provider.acquire(
                step_index=0,
                step_dir=step1,
                next_step_dir=step2,
                output_dir=root / "out",
                relative_transforms_path=relative,
            )
            vlm = SimpleNamespace(
                decide_recovery=Mock(return_value=_Recovery()),
                verify_transition=Mock(return_value={"success": True, "reason": "illustrated"}),
            )
            config = ClosedLoopExecutionConfig(
                goal="place block",
                input_frame=step2 / "start.png",
                output_dir=root / "out",
                execution_context="illustration",
                max_recovery_attempts=1,
            )
            attempt = {
                "status": "grounded",
                "action": {"action": _Recovery.recovery_action},
                "relative_ee_transforms": str(relative),
            }

            with patch(
                "novaplan.closed_loop_execution._run_one_recovery_attempt",
                return_value=attempt,
            ):
                result = _run_recovery_loop(
                    step_index=0,
                    step_dir=step1,
                    next_step_dir=step2,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "place block", "track_object": "block"},
                    initial_failure_reason="failed",
                    config=config,
                    planner=SimpleNamespace(),
                    vlm=vlm,
                    recovery_video_client=SimpleNamespace(),
                    observation_provider=provider,
                    history=[],
                )

            self.assertEqual(result["status"], "recovered")
            self.assertEqual(result["observation"].rgb_path, failed.rgb_path)
            self.assertTrue(result["observation"].metadata["allow_recovery_state_reuse"])
            vlm.verify_transition.assert_called_once()
            self.assertEqual(
                vlm.verify_transition.call_args.kwargs["action"],
                _Recovery.recovery_action,
            )

    def test_real_verification_contexts_reject_reused_recovery_observation(self):
        for execution_context in ("online", "recorded_observations"):
            with self.subTest(execution_context=execution_context), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                failed = _bundle(root / "failed.png", 20)
                failed.metadata["allow_recovery_state_reuse"] = True
                target = root / "target.png"
                _image(target, 80)
                relative = root / "relative.npy"
                np.save(relative, np.eye(4)[None])
                provider = SimpleNamespace(acquire=Mock(return_value=failed))
                vlm = SimpleNamespace(
                    decide_recovery=Mock(return_value=_Recovery()),
                    verify_transition=Mock(),
                )
                config = ClosedLoopExecutionConfig(
                    goal="place block",
                    input_frame=failed.rgb_path,
                    output_dir=root / "out",
                    execution_context=execution_context,
                    max_recovery_attempts=1,
                )
                attempt = {
                    "status": "grounded",
                    "action": {"action": "correct block"},
                    "relative_ee_transforms": str(relative),
                }

                with patch(
                    "novaplan.closed_loop_execution._run_one_recovery_attempt",
                    return_value=attempt,
                ):
                    result = _run_recovery_loop(
                        step_index=0,
                        step_dir=root,
                        next_step_dir=None,
                        failed_observation=failed,
                        target_path=target,
                        previous_action={"action": "place block", "track_object": "block"},
                        initial_failure_reason="failed",
                        config=config,
                        planner=SimpleNamespace(),
                        vlm=vlm,
                        recovery_video_client=SimpleNamespace(),
                        observation_provider=provider,
                        history=[],
                    )

                self.assertEqual(result["status"], "awaiting_external_recovery_execution")
                vlm.verify_transition.assert_not_called()

    def test_recovery_generation_routes_negative_prompts_by_backend(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        class _Client:
            def __init__(self, backend):
                if backend == "wan":
                    self.base = "http://wan"
                elif backend == "hybrid":
                    self.wan_client = object()
                    self.veo_client = object()
                self.kwargs = None

            def generate_rollouts(self, **kwargs):
                self.kwargs = kwargs
                return []

        common = {
            "start_frame": frame,
            "action_text": "CONTACT FINGER: index\nPoke once with continuous contact.",
            "veo_action_text": "extended English Veo recovery prompt",
            "num_samples": 1,
            "num_frames": 2,
            "fps": 1,
            "size": "4*4",
            "seed": 0,
            "last_frame": frame,
            "mask_prompt": "block",
        }
        wan = _Client("wan")
        veo = _Client("veo")
        hybrid = _Client("hybrid")
        for client in (wan, veo, hybrid):
            _generate_recovery_rollouts(client, **common)

        self.assertIn("negative_prompt", wan.kwargs)
        self.assertIn("negative_prompt", veo.kwargs)
        self.assertEqual(
            veo.kwargs["action_text"],
            common["veo_action_text"],
        )
        self.assertNotEqual(wan.kwargs["negative_prompt"], veo.kwargs["negative_prompt"])
        self.assertIn("wan_negative_prompt", hybrid.kwargs)
        self.assertIn("veo_negative_prompt", hybrid.kwargs)
        self.assertEqual(hybrid.kwargs["action_text"], common["action_text"])
        self.assertEqual(hybrid.kwargs["wan_action_text"], common["action_text"])
        self.assertEqual(
            hybrid.kwargs["veo_action_text"],
            common["veo_action_text"],
        )

    def test_grasp_recovery_uses_flf_generation_and_auto_grounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 20)
            target = root / "target.png"
            _image(target, 80)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            client = SimpleNamespace(
                base="http://wan",
                generate_rollouts=Mock(
                    return_value=SimpleNamespace(
                        videos=[np.zeros((2, 16, 16, 3), dtype=np.uint8)],
                        flow_images=[np.zeros((16, 16, 3), dtype=np.uint8)],
                        video_sources=["wan22"],
                    )
                )
            )
            planner = SimpleNamespace(flow_extraction_client=None, use_prompt_extension=True)
            vlm = SimpleNamespace(
                extend_video_prompt=Mock(return_value="wan extended recovery prompt")
            )
            grounding = {
                "execution_summary": {
                    "artifact_paths": {"relative_ee_transforms": str(relative)}
                }
            }
            with (
                patch("novaplan.closed_loop_execution._write_video"),
                patch("novaplan.closed_loop_execution._run_execution_step", return_value=grounding) as run_grounding,
            ):
                result = _run_one_recovery_attempt(
                    attempt_index=0,
                    step_index=0,
                    step_dir=root,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "place block", "track_object": "block"},
                    recovery=_Recovery(),
                    config=ClosedLoopExecutionConfig(
                        goal="place the block",
                        input_frame=failed.rgb_path,
                        output_dir=root / "out",
                    ),
                    planner=planner,
                    vlm=vlm,
                    recovery_video_client=client,
                    history=[],
                )

            generation = client.generate_rollouts.call_args.kwargs
            vlm.extend_video_prompt.assert_called_once()
            self.assertEqual(
                vlm.extend_video_prompt.call_args.kwargs["action"],
                _Recovery.recovery_action,
            )
            self.assertEqual(generation["action_text"], "wan extended recovery prompt")
            np.testing.assert_array_equal(generation["start_frame"], failed.rgb)
            np.testing.assert_array_equal(
                generation["last_frame"],
                np.asarray(Image.open(target).convert("RGB"), dtype=np.uint8),
            )
            kwargs = run_grounding.call_args.kwargs
            self.assertEqual(
                run_grounding.call_args.args[1]["action"],
                _Recovery.recovery_action,
            )
            self.assertEqual(kwargs["selected_flow_override"], "auto")
            self.assertEqual(kwargs["grounding_mode"], "grasp")
            self.assertEqual(result["action"]["action"], _Recovery.recovery_action)
            self.assertEqual(
                result["selection"]["grounding"]["recovery_action"],
                _Recovery.recovery_action,
            )
            self.assertEqual(result["relative_ee_transforms"], str(relative))
            self.assertEqual(result["selection"]["selected_backend"], "wan22")
            self.assertIn("candidate_000_wan22.mp4", result["videos"][0])

    def test_single_non_prehensile_candidate_skips_selection_flow_and_vlm_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 20)
            target = root / "target.png"
            _image(target, 80)
            buf = io.BytesIO()
            Image.fromarray(failed.rgb).save(buf, format="PNG")
            recovery = SimpleNamespace(
                recovery_mode="non_prehensile",
                use_non_prehensile_pipeline=True,
                object_prompt="block",
                contact_finger="index",
                contact_point_2d=[8, 8],
                prompt_text="Poke once with the index finger.",
                recovery_prompt="Poke once with the index finger.",
                recovery_action="Poke the block once.",
                annotated_image_png_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
                metadata={},
                to_dict=lambda: {"recovery_mode": "non_prehensile"},
            )
            client = SimpleNamespace(
                generate_rollouts=Mock(
                    return_value=SimpleNamespace(
                        videos=[np.zeros((2, 16, 16, 3), dtype=np.uint8)],
                        flow_images=[],
                        video_sources=["veo3"],
                    )
                )
            )
            flow_client = SimpleNamespace(extract_flow=Mock())
            planner = SimpleNamespace(flow_extraction_client=flow_client)
            vlm = SimpleNamespace(rank_rollouts_batch=Mock())
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            grounding = {
                "execution_summary": {
                    "artifact_paths": {"relative_ee_transforms": str(relative)}
                }
            }
            with (
                patch("novaplan.closed_loop_execution._write_video"),
                patch("novaplan.closed_loop_execution._run_execution_step", return_value=grounding),
            ):
                result = _run_one_recovery_attempt(
                    attempt_index=0,
                    step_index=0,
                    step_dir=root,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "align block", "track_object": "block"},
                    recovery=recovery,
                    config=ClosedLoopExecutionConfig(
                        goal="align block",
                        input_frame=failed.rgb_path,
                        output_dir=root / "out",
                        recovery_num_videos=1,
                    ),
                    planner=planner,
                    vlm=vlm,
                    recovery_video_client=client,
                    history=[],
                )

            flow_client.extract_flow.assert_not_called()
            vlm.rank_rollouts_batch.assert_not_called()
            self.assertEqual(result["selection"]["selection_method"], "single_candidate")
            self.assertEqual(result["selection"]["selected_backend"], "veo3")

    def test_non_prehensile_grounds_only_flow_ranked_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 20)
            target = root / "target.png"
            _image(target, 80)
            buf = io.BytesIO()
            Image.fromarray(failed.rgb).save(buf, format="PNG")
            recovery = SimpleNamespace(
                recovery_mode="non_prehensile",
                use_non_prehensile_pipeline=True,
                object_prompt="block",
                contact_finger="index_tip",
                contact_point_2d=[8, 8],
                prompt_text="Use one index-finger poke.",
                recovery_prompt="Use one index-finger poke.",
                recovery_action="Poke the block once.",
                annotated_image_png_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
                metadata={"raw": {"contact_finger": "index_tip"}},
                to_dict=lambda: {"recovery_mode": "non_prehensile"},
            )
            videos = [
                np.zeros((2, 16, 16, 3), dtype=np.uint8),
                np.ones((2, 8, 32, 3), dtype=np.uint8),
            ]
            client = SimpleNamespace(
                generate_rollouts=Mock(
                    return_value=SimpleNamespace(
                        videos=videos,
                        flow_images=[None, None],
                        video_sources=["wan22", "veo3"],
                    )
                )
            )
            vlm = SimpleNamespace(
                rank_rollouts_batch=Mock(
                    return_value=[
                        {"candidate_id": 0, "score": 0.9, "success": True},
                        {"candidate_id": 1, "score": 0.8, "success": True},
                    ]
                )
            )
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            grounded = {
                "execution_summary": {
                    "artifact_paths": {"relative_ee_transforms": str(relative)}
                }
            }
            with (
                patch("novaplan.closed_loop_execution._write_video"),
                patch(
                    "novaplan.closed_loop_execution._run_execution_step",
                    return_value=grounded,
                ) as run_grounding,
            ):
                result = _run_one_recovery_attempt(
                    attempt_index=0,
                    step_index=0,
                    step_dir=root,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "align block", "track_object": "block"},
                    recovery=recovery,
                    config=ClosedLoopExecutionConfig(
                        goal="align the block",
                        input_frame=failed.rgb_path,
                        output_dir=root / "out",
                        recovery_num_videos=1,
                    ),
                    planner=None,
                    vlm=vlm,
                    recovery_video_client=client,
                    history=[],
                )

            run_grounding.assert_called_once()
            selected = run_grounding.call_args.kwargs
            self.assertEqual(
                run_grounding.call_args.args[1]["action"],
                "Poke the block once.",
            )
            self.assertEqual(selected["selected_flow_override"], "hand")
            self.assertEqual(selected["grounding_mode"], "non_prehensile")
            self.assertEqual(selected["contact_finger"], "index")
            self.assertEqual(selected["contact_point_2d"], [8.0, 8.0])
            self.assertEqual(result["selection"]["selected_index"], 0)
            self.assertEqual(
                vlm.rank_rollouts_batch.call_args.kwargs["goal"],
                "Poke the block once.",
            )
            self.assertTrue(
                all(
                    candidate["action"] == "Poke the block once."
                    for candidate in vlm.rank_rollouts_batch.call_args.kwargs["candidates"]
                )
            )
            self.assertEqual(
                result["selection"]["grounding"]["recovery_action"],
                "Poke the block once.",
            )
            self.assertEqual(
                result["selection"]["grounding"]["contact_point_2d_candidate"],
                [8.0, 8.0],
            )

    def test_recovery_loop_redecides_and_retries_after_failed_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 10)
            obs1 = _bundle(root / "recovery1.png", 30)
            obs2 = _bundle(root / "recovery2.png", 60)
            target = root / "target.png"
            _image(target, 80)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            provider = SimpleNamespace(acquire=Mock(side_effect=[obs1, obs2]))
            vlm = SimpleNamespace(
                decide_recovery=Mock(side_effect=[_Recovery(), _Recovery()]),
                verify_transition=Mock(
                    side_effect=[
                        {"success": False, "reason": "still misaligned"},
                        {"success": True, "reason": "recovered"},
                    ]
                ),
            )
            attempt = {
                "status": "grounded",
                "action": {"action": "correct block"},
                "relative_ee_transforms": str(relative),
            }
            config = ClosedLoopExecutionConfig(
                goal="align block",
                input_frame=failed.rgb_path,
                output_dir=root / "out",
                max_recovery_attempts=2,
            )
            with patch(
                "novaplan.closed_loop_execution._run_one_recovery_attempt",
                side_effect=[dict(attempt), dict(attempt)],
            ) as run_attempt:
                result = _run_recovery_loop(
                    step_index=0,
                    step_dir=root,
                    next_step_dir=None,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "place block", "track_object": "block"},
                    initial_failure_reason="failed",
                    config=config,
                    planner=SimpleNamespace(),
                    vlm=vlm,
                    recovery_video_client=SimpleNamespace(),
                    observation_provider=provider,
                    history=[],
                )

            self.assertEqual(result["status"], "recovered")
            self.assertEqual(run_attempt.call_count, 2)
            self.assertEqual(vlm.decide_recovery.call_count, 2)
            ids = [call.kwargs["observation_id"] for call in provider.acquire.call_args_list]
            self.assertEqual(ids, ["step_001_recovery_001", "step_001_recovery_002"])

    def test_recovery_loop_regenerates_with_same_decision_after_hand_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 10)
            recovered = _bundle(root / "recovered.png", 60)
            target = root / "target.png"
            _image(target, 80)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            recovery = _Recovery()
            provider = SimpleNamespace(acquire=Mock(return_value=recovered))
            vlm = SimpleNamespace(
                decide_recovery=Mock(return_value=recovery),
                verify_transition=Mock(return_value={"success": True, "reason": "recovered"}),
            )
            attempt = {
                "status": "grounded",
                "action": {"action": "correct block"},
                "relative_ee_transforms": str(relative),
            }
            config = ClosedLoopExecutionConfig(
                goal="align block",
                input_frame=failed.rgb_path,
                output_dir=root / "out",
                max_recovery_attempts=1,
                max_hand_flow_regenerations=2,
            )
            with patch(
                "novaplan.closed_loop_execution._run_one_recovery_attempt",
                side_effect=[
                    RuntimeError("initial object mask is empty"),
                    attempt,
                ],
            ) as run_attempt:
                result = _run_recovery_loop(
                    step_index=0,
                    step_dir=root,
                    next_step_dir=None,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "place block", "track_object": "block"},
                    initial_failure_reason="failed",
                    config=config,
                    planner=SimpleNamespace(),
                    vlm=vlm,
                    recovery_video_client=SimpleNamespace(),
                    observation_provider=provider,
                    history=[],
                )

            self.assertEqual(result["status"], "recovered")
            self.assertEqual(run_attempt.call_count, 2)
            self.assertEqual(vlm.decide_recovery.call_count, 1)
            self.assertIs(run_attempt.call_args_list[0].kwargs["recovery"], recovery)
            self.assertIs(run_attempt.call_args_list[1].kwargs["recovery"], recovery)
            self.assertEqual(
                [call.kwargs["attempt_index"] for call in run_attempt.call_args_list],
                [0, 1],
            )
            provider.acquire.assert_called_once()
            self.assertEqual(
                provider.acquire.call_args.kwargs["observation_id"],
                "step_001_recovery_002",
            )

    def test_recovery_grounding_uses_persistent_flow_reviewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 10)
            recovered = _bundle(root / "recovered.png", 60)
            target = root / "target.png"
            _image(target, 80)
            relative = root / "relative.npy"
            flow = root / "flow.npz"
            np.save(relative, np.eye(4)[None])
            np.savez(flow, coords_3d=np.zeros((2, 1, 3)))
            grounding = {
                "out_dir": str(root / "grounding"),
                "execution_summary": {
                    "metadata": {"flow_source": str(flow)},
                    "artifact_paths": {"relative_ee_transforms": str(relative)},
                },
            }
            attempt = {
                "status": "grounded",
                "action": {"action": "correct block"},
                "grounding": grounding,
                "relative_ee_transforms": str(relative),
            }
            provider = SimpleNamespace(acquire=Mock(return_value=recovered))
            reviewer = SimpleNamespace(
                review_step=Mock(
                    return_value={
                        "status": "completed",
                        "decision": {"continue": True},
                    }
                )
            )
            vlm = SimpleNamespace(
                decide_recovery=Mock(return_value=_Recovery()),
                verify_transition=Mock(return_value={"success": True, "reason": "recovered"}),
            )
            config = ClosedLoopExecutionConfig(
                goal="align block",
                input_frame=failed.rgb_path,
                output_dir=root / "out",
                max_recovery_attempts=1,
                debug_flow_review=True,
            )
            with patch(
                "novaplan.closed_loop_execution._run_one_recovery_attempt",
                return_value=attempt,
            ):
                result = _run_recovery_loop(
                    step_index=0,
                    step_dir=root,
                    next_step_dir=None,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "place block", "track_object": "block"},
                    initial_failure_reason="failed",
                    config=config,
                    planner=SimpleNamespace(),
                    vlm=vlm,
                    recovery_video_client=SimpleNamespace(),
                    observation_provider=provider,
                    history=[],
                    flow_reviewer=reviewer,
                )

            self.assertEqual(result["status"], "recovered")
            reviewer.review_step.assert_called_once()
            self.assertEqual(
                reviewer.review_step.call_args.kwargs["label"],
                "Step 1 recovery 1",
            )

    def test_recovery_loop_retries_after_malformed_policy_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = _bundle(root / "failed.png", 10)
            recovered = _bundle(root / "recovered.png", 60)
            target = root / "target.png"
            _image(target, 80)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            provider = SimpleNamespace(acquire=Mock(return_value=recovered))
            vlm = SimpleNamespace(
                decide_recovery=Mock(
                    side_effect=[ValueError("missing contact finger"), _Recovery()]
                ),
                verify_transition=Mock(return_value={"success": True, "reason": "recovered"}),
            )
            attempt = {
                "status": "grounded",
                "action": {"action": "correct block"},
                "relative_ee_transforms": str(relative),
            }
            config = ClosedLoopExecutionConfig(
                goal="align block",
                input_frame=failed.rgb_path,
                output_dir=root / "out",
                max_recovery_attempts=2,
            )
            with patch(
                "novaplan.closed_loop_execution._run_one_recovery_attempt",
                return_value=attempt,
            ) as run_attempt:
                result = _run_recovery_loop(
                    step_index=0,
                    step_dir=root,
                    next_step_dir=None,
                    failed_observation=failed,
                    target_path=target,
                    previous_action={"action": "place block", "track_object": "block"},
                    initial_failure_reason="failed",
                    config=config,
                    planner=SimpleNamespace(),
                    vlm=vlm,
                    recovery_video_client=SimpleNamespace(),
                    observation_provider=provider,
                    history=[],
                )

            self.assertEqual(result["status"], "recovered")
            self.assertEqual(result["attempts"][0]["status"], "recovery_decision_failed")
            self.assertEqual(vlm.decide_recovery.call_count, 2)
            run_attempt.assert_called_once()


if __name__ == "__main__":
    unittest.main()
