#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.closed_loop_execution import (  # noqa: E402
    ClosedLoopExecutionConfig,
    _is_retryable_hand_flow_rejection,
    _materialize_grounding_observation,
    _run_execution_step,
    load_illustration_actions,
    run_closed_loop_execution,
)
from novaplan.observations import (  # noqa: E402
    FilesystemObservationProvider,
    ObservationBundle,
    RecordedTraceObservationProvider,
    load_observation_bundle,
)
from local_planning.run_closed_loop_execution import _terminal_transcript  # noqa: E402


def _write_rgb(path: Path, value: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 10, 3), value, dtype=np.uint8)).save(path)


class ObservationProviderTest(unittest.TestCase):
    def test_terminal_transcript_is_updated_while_context_is_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with _terminal_transcript(output_dir) as transcript_path:
                print("live transcript marker")
                self.assertIn("live transcript marker", transcript_path.read_text())

    def test_execution_step_failure_is_concise_and_preserves_full_subprocess_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = root / "step_001"
            step.mkdir()
            config = ClosedLoopExecutionConfig(
                goal="place block",
                input_frame=step / "start.png",
                output_dir=root / "out",
                hand_flow_interaction_epsilon=0.875,
            )
            proc = SimpleNamespace(
                returncode=1,
                stdout="[hamer-client] many diagnostic lines\nmore details\n",
                stderr="Traceback (most recent call last):\nValueError: contact failed\n",
            )

            with patch("novaplan.closed_loop_execution.subprocess.run", return_value=proc):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"ValueError: contact failed.*Full subprocess log:",
                ):
                    _run_execution_step(
                        step,
                        {"action": "place block", "track_object": "block"},
                        config,
                        0,
                    )

            log_path = root / "out" / "step_001" / "execution_step" / "execution_step_subprocess.log"
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text()
            self.assertIn("many diagnostic lines", log_text)
            self.assertIn("ValueError: contact failed", log_text)
            self.assertIn("--hand_flow_interaction_epsilon 0.875", log_text)
            self.assertIn("--debug_artifact_dir", log_text)
            self.assertIn("step_001/debug_artifacts/geometric_grounding/selected_rollout", log_text)

    def test_execution_step_failure_reports_completed_object_flow_and_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = root / "step_001"
            step.mkdir()
            out_dir = root / "out" / "step_001" / "execution_step"
            out_dir.mkdir(parents=True)
            (out_dir / "grounding_progress.json").write_text(
                json.dumps(
                    {
                        "stage": "hand_flow_computation_started",
                        "object_flow": {
                            "valid": False,
                            "should_switch_to_hand": False,
                            "max_rotation_deg": None,
                            "flow_switch_theta_deg": 45.0,
                            "valid_point_counts": [8, 2],
                            "failed_steps": [2],
                            "reason": "object flow missing reliable adjacent-frame transforms",
                        },
                        "effective_switch_to_hand": True,
                        "selected_grounding": "hand",
                        "selection_reason": "object flow missing reliable adjacent-frame transforms",
                    }
                )
            )
            config = ClosedLoopExecutionConfig(
                goal="place block",
                input_frame=step / "start.png",
                output_dir=root / "out",
            )
            proc = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="ValueError: object-mask motion never reached the interaction threshold\n",
            )

            with (
                patch("novaplan.closed_loop_execution.subprocess.run", return_value=proc),
                patch("builtins.print") as print_mock,
                self.assertRaises(RuntimeError),
            ):
                _run_execution_step(
                    step,
                    {"action": "place block", "track_object": "block"},
                    config,
                    0,
                )

            messages = [str(call.args[0]) for call in print_mock.call_args_list if call.args]
            self.assertTrue(any("metric object flow computed (valid=False)" in msg for msg in messages))
            self.assertTrue(any("selected hand-centric grounding" in msg for msg in messages))
            self.assertTrue(any("object_valid=False" in msg for msg in messages))
            self.assertTrue(any("rotation_threshold_exceeded=False" in msg for msg in messages))
            self.assertTrue(any("failed_adjacent_transforms=1" in msg for msg in messages))
            self.assertTrue(any("failed_steps=[2]" in msg for msg in messages))
            self.assertTrue(any("visible_points_at_failed_steps=[2]" in msg for msg in messages))

    def test_initial_zero_horizon_is_recorded_as_task_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "initial.png"
            _write_rgb(initial)
            planner = SimpleNamespace(
                assess_task_structure=Mock(
                    return_value={
                        "horizon": 0,
                        "planner_horizon": 0,
                        "reactive_execution_horizon": 0,
                        "is_coupled": False,
                        "plan_in_advance_allowed": False,
                    }
                )
            )

            summary = run_closed_loop_execution(
                ClosedLoopExecutionConfig(
                    goal="already complete",
                    input_frame=initial,
                    output_dir=root / "out",
                ),
                planner=planner,
            )

            self.assertEqual(summary["termination"]["status"], "task_complete")
            self.assertEqual(summary["termination"]["step"], 0)
            self.assertEqual(summary["steps"], [])

    def test_only_hand_calibration_rejections_trigger_video_regeneration(self):
        self.assertTrue(
            _is_retryable_hand_flow_rejection(
                RuntimeError("hand trajectory rejected because the projected hand leaves the image")
            )
        )
        self.assertTrue(
            _is_retryable_hand_flow_rejection(
                RuntimeError(
                    "object-mask motion never reached the interaction threshold epsilon=0.900"
                )
            )
        )
        self.assertTrue(
            _is_retryable_hand_flow_rejection(
                RuntimeError("HaMeR returned no hand meshes for the generated video")
            )
        )
        self.assertFalse(
            _is_retryable_hand_flow_rejection(
                RuntimeError("hand-flow service connection refused")
            )
        )

    def test_closed_loop_regenerates_and_reranks_after_hand_calibration_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "initial.png"
            post = root / "post.png"
            target1 = root / "target1.png"
            target2 = root / "target2.png"
            for path, value in ((initial, 10), (post, 20), (target1, 30), (target2, 40)):
                _write_rgb(path, value)
            step_dir = root / "step_001"
            step_dir.mkdir()
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            grounding = {
                "execution_summary": {
                    "selected_flow": "hand",
                    "flow_switch": {"object_flow": {}, "switch_to_hand": True},
                    "artifact_paths": {"relative_ee_transforms": str(relative)},
                }
            }
            video1 = np.zeros((2, 8, 10, 3), dtype=np.uint8)
            video2 = np.ones((2, 8, 10, 3), dtype=np.uint8)
            planner = SimpleNamespace(
                generate_execution_rollout=Mock(
                    side_effect=[
                        SimpleNamespace(video=[video1]),
                        SimpleNamespace(video=[video2]),
                    ]
                )
            )
            observation = ObservationBundle(
                rgb=np.full((8, 10, 3), 20, dtype=np.uint8),
                rgb_path=post,
                provenance="unit_test",
                is_live=True,
                step_index=0,
            )
            provider = SimpleNamespace(acquire=Mock(return_value=observation))
            vlm = SimpleNamespace(
                verify_transition=Mock(return_value={"success": True, "reason": "ok"})
            )
            config = ClosedLoopExecutionConfig(
                goal="place block",
                input_frame=initial,
                output_dir=root / "out",
                step_dirs=[step_dir],
                plan_actions=[{"action": "place block", "track_object": "block"}],
                horizon_result={"horizon": 1, "plan_in_advance_allowed": True},
                execution_mode="strategic",
                observation_provider=provider,
                max_hand_flow_regenerations=1,
            )
            saved = [
                {
                    "video": str(root / "first.mp4"),
                    "target_image": str(target1),
                    "generation_attempt": 0,
                },
                {
                    "video": str(root / "second.mp4"),
                    "target_image": str(target2),
                    "generation_attempt": 1,
                },
            ]
            with (
                patch(
                    "novaplan.closed_loop_execution._save_selected_rollout",
                    side_effect=saved,
                ),
                patch(
                    "novaplan.closed_loop_execution._run_execution_step",
                    side_effect=[
                        RuntimeError("hand contact calibration failed: candidate scale set is empty"),
                        grounding,
                    ],
                ) as run_grounding,
            ):
                summary = run_closed_loop_execution(
                    config,
                    planner=planner,
                    vlm=vlm,
                )

            self.assertEqual(planner.generate_execution_rollout.call_count, 2)
            self.assertEqual(run_grounding.call_count, 2)
            self.assertEqual(summary["steps"][0]["status"], "verified")
            self.assertEqual(
                [item["status"] for item in summary["steps"][0]["grounding_attempts"]],
                ["rejected", "grounded"],
            )

    def test_closed_loop_uses_latest_live_rgbd_for_verification_and_next_grounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "initial.png"
            live1_path = root / "live1.png"
            live2_path = root / "live2.png"
            _write_rgb(initial, 10)
            _write_rgb(live1_path, 40)
            _write_rgb(live2_path, 70)
            step_dirs = []
            for index in (1, 2):
                step_dir = root / f"step_{index:03d}"
                _write_rgb(step_dir / "start.png", 100 + index)
                _write_rgb(step_dir / "target.png", 120 + index)
                Image.fromarray(np.full((8, 10), 1000, dtype=np.uint16)).save(
                    step_dir / "start_depth.png"
                )
                (step_dir / "config.json").write_text(json.dumps({"depth_scale": 0.001}))
                step_dirs.append(step_dir)

            def live_bundle(path: Path, value: int, step: int) -> ObservationBundle:
                return ObservationBundle(
                    rgb=np.full((8, 10, 3), value, dtype=np.uint8),
                    rgb_path=path,
                    provenance="external_live_capture",
                    is_live=True,
                    step_index=step,
                    depth=np.ones((8, 10), dtype=np.float32),
                    intrinsics={"fx": 10.0, "fy": 10.0, "cx": 5.0, "cy": 4.0},
                    config={"depth_scale": 0.001},
                )

            provider = SimpleNamespace(
                acquire=Mock(
                    side_effect=[
                        live_bundle(live1_path, 40, 0),
                        live_bundle(live2_path, 70, 1),
                    ]
                )
            )
            vlm = SimpleNamespace(
                verify_transition=Mock(return_value={"success": True, "reason": "ok"})
            )
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            grounding = {
                "execution_summary": {
                    "artifact_paths": {"relative_ee_transforms": str(relative)}
                }
            }
            output = root / "out"
            config = ClosedLoopExecutionConfig(
                goal="complete two steps",
                input_frame=initial,
                output_dir=output,
                step_dirs=step_dirs,
                plan_actions=["first action", "second action"],
                horizon_result={"horizon": 2, "plan_in_advance_allowed": True},
                execution_mode="strategic",
                observation_provider=provider,
            )
            with patch(
                "novaplan.closed_loop_execution._run_execution_step",
                side_effect=[grounding, grounding],
            ) as run_grounding:
                summary = run_closed_loop_execution(config, vlm=vlm)

            self.assertEqual([step["status"] for step in summary["steps"]], ["verified", "verified"])
            np.testing.assert_array_equal(
                vlm.verify_transition.call_args_list[0].kwargs["start_image"],
                np.full((8, 10, 3), 10, dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                vlm.verify_transition.call_args_list[1].kwargs["start_image"],
                np.full((8, 10, 3), 40, dtype=np.uint8),
            )
            second_grounding_dir = Path(run_grounding.call_args_list[1].args[0])
            self.assertEqual(second_grounding_dir, output / "step_002" / "grounding_input")
            np.testing.assert_array_equal(
                np.asarray(Image.open(second_grounding_dir / "start.png").convert("RGB")),
                np.full((8, 10, 3), 40, dtype=np.uint8),
            )
            self.assertTrue((second_grounding_dir / "start_depth.png").exists())

    def test_recorded_trace_labels_next_start_as_standin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = root / "step_001"
            step2 = root / "step_002"
            step1.mkdir()
            _write_rgb(step2 / "start.png", 42)
            Image.fromarray(np.full((8, 10), 1000, dtype=np.uint16)).save(step2 / "start_depth.png")
            Image.fromarray(np.full((8, 10), 9000, dtype=np.uint16)).save(step2 / "post_depth.png")
            (step2 / "config.json").write_text(json.dumps({"depth_scale": 0.001}))

            bundle = RecordedTraceObservationProvider().acquire(
                step_index=0,
                step_dir=step1,
                next_step_dir=step2,
                output_dir=root / "out",
                relative_transforms_path=root / "relative.npy",
            )

            self.assertIsNotNone(bundle)
            self.assertEqual(bundle.provenance, "recorded_next_step_start_standin")
            self.assertFalse(bundle.is_live)
            self.assertAlmostEqual(float(bundle.depth[0, 0]), 1.0)
            self.assertTrue(bundle.metadata["illustration_only"])
            self.assertTrue(bundle.metadata["not_evidence_of_execution"])
            self.assertFalse(bundle.metadata["allow_recovery_state_reuse"])
            self.assertFalse(bundle.metadata["robot_commanded_by_novaplan"])

            recovery = RecordedTraceObservationProvider().acquire(
                step_index=0,
                step_dir=step1,
                next_step_dir=step2,
                output_dir=root / "out",
                relative_transforms_path=root / "recovery_relative.npy",
                observation_id="step_001_recovery_001",
            )
            self.assertEqual(recovery.rgb_path, bundle.rgb_path)
            self.assertEqual(recovery.provenance, "recorded_next_step_start_recovery_standin")
            self.assertEqual(recovery.metadata["observation_role"], "post_recovery_standin")
            self.assertTrue(recovery.metadata["allow_recovery_state_reuse"])

    def test_recorded_trace_uses_explicit_post_only_before_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1 = root / "step_001"
            step2 = root / "step_002"
            explicit = root / "failed_post.png"
            step1.mkdir()
            _write_rgb(explicit, 10)
            _write_rgb(step2 / "start.png", 42)

            provider = RecordedTraceObservationProvider(explicit_post_image=explicit)
            post_action = provider.acquire(
                step_index=0,
                step_dir=step1,
                next_step_dir=step2,
                output_dir=root / "out",
                relative_transforms_path=root / "relative.npy",
            )
            post_recovery = provider.acquire(
                step_index=0,
                step_dir=step1,
                next_step_dir=step2,
                output_dir=root / "out",
                relative_transforms_path=root / "recovery_relative.npy",
                observation_id="step_001_recovery_001",
            )

            self.assertEqual(post_action.rgb_path, explicit.resolve())
            self.assertEqual(post_action.provenance, "recorded_explicit_post_image")
            self.assertEqual(post_recovery.rgb_path, (step2 / "start.png").resolve())
            self.assertEqual(
                post_recovery.provenance,
                "recorded_next_step_start_recovery_standin",
            )

    def test_recorded_trace_final_step_uses_end_frame_with_matching_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = root / "step_001"
            _write_rgb(step / "end_frame_rgb.png", 61)
            Image.fromarray(np.full((8, 10), 1500, dtype=np.uint16)).save(
                step / "end_frame_depth.png"
            )
            (step / "config.json").write_text(json.dumps({"depth_scale": 0.001}))

            bundle = RecordedTraceObservationProvider().acquire(
                step_index=0,
                step_dir=step,
                next_step_dir=None,
                output_dir=root / "out",
                relative_transforms_path=None,
            )

            self.assertEqual(bundle.provenance, "recorded_current_step_end_frame_standin")
            self.assertEqual(bundle.depth_path, (step / "end_frame_depth.png").resolve())
            self.assertAlmostEqual(float(bundle.depth[0, 0]), 1.5, places=6)

    def test_illustration_action_loader_labels_example_data_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            action_path = root / "illustration_actions.json"
            action_path.write_text(
                json.dumps(
                    {
                        "actions": [
                            {"action": "place the blue block", "track_object": "blue block"}
                        ]
                    }
                )
            )

            actions = load_illustration_actions(root)

            self.assertEqual(actions[0]["source"], "recorded_example_data_action")
            self.assertTrue(actions[0]["illustration_only"])
            self.assertTrue(actions[0]["not_evidence_of_execution"])
            self.assertEqual(actions[0]["example_data_path"], str(action_path.resolve()))

    def test_reactive_illustration_action_bypasses_online_action_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "initial.png"
            step = root / "step_001"
            target = step / "end_frame_rgb.png"
            _write_rgb(initial, 10)
            _write_rgb(step / "start.png", 10)
            _write_rgb(target, 70)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            rollout = np.zeros((2, 8, 10, 3), dtype=np.uint8)
            planner = SimpleNamespace(
                plan_next_reactive_step=Mock(
                    side_effect=AssertionError("online action proposal must not run")
                ),
                generate_execution_rollout=Mock(return_value=SimpleNamespace(video=[rollout])),
            )
            vlm = SimpleNamespace(
                verify_transition=Mock(return_value={"success": True, "reason": "illustrated"})
            )
            config = ClosedLoopExecutionConfig(
                goal="sort one block",
                input_frame=initial,
                output_dir=root / "out",
                step_dirs=[step],
                horizon_result={
                    "horizon": 1,
                    "reactive_execution_horizon": 1,
                    "is_coupled": False,
                    "plan_in_advance_allowed": False,
                },
                execution_context="illustration",
                illustration_actions=[
                    {
                        "action": "place the blue block",
                        "track_object": "blue block",
                        "source": "recorded_example_data_action",
                        "illustration_only": True,
                    }
                ],
            )
            grounding = {
                "execution_summary": {
                    "artifact_paths": {"relative_ee_transforms": str(relative)}
                }
            }
            with (
                patch(
                    "novaplan.closed_loop_execution._save_selected_rollout",
                    return_value={"video": str(root / "selected.mp4"), "target_image": str(target)},
                ),
                patch("novaplan.closed_loop_execution._run_execution_step", return_value=grounding),
            ):
                summary = run_closed_loop_execution(config, planner=planner, vlm=vlm)

            planner.plan_next_reactive_step.assert_not_called()
            self.assertEqual(
                planner.generate_execution_rollout.call_args.kwargs["action"],
                "place the blue block",
            )
            self.assertEqual(summary["execution_context"], "illustration")
            self.assertTrue(summary["steps"][0]["action"]["illustration_only"])
            self.assertTrue(
                summary["steps"][0]["action"]["not_evidence_of_execution"]
            )
            self.assertTrue(
                summary["steps"][0]["observation"]["metadata"]["not_evidence_of_execution"]
            )

    def test_recorded_observations_keep_reactive_action_proposal_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = root / "start.png"
            step1 = root / "step_001"
            step2 = root / "step_002"
            target = step1 / "end_frame_rgb.png"
            _write_rgb(initial, 10)
            _write_rgb(step1 / "start.png", 10)
            _write_rgb(target, 70)
            _write_rgb(step2 / "start.png", 60)
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            rollout = np.zeros((2, 8, 10, 3), dtype=np.uint8)
            live_beam = SimpleNamespace(
                actions=["place the live-selected block"],
                track_objects=["live-selected block"],
                video=[rollout],
                score=0.8,
                task_complete=False,
                constraint_context=None,
            )
            planner = SimpleNamespace(plan_next_reactive_step=Mock(return_value=live_beam))
            vlm = SimpleNamespace(
                verify_transition=Mock(return_value={"success": True, "reason": "recorded post state"})
            )
            config = ClosedLoopExecutionConfig(
                goal="sort one block",
                input_frame=initial,
                output_dir=root / "out",
                sample_root=root,
                horizon_result={
                    "horizon": 1,
                    "reactive_execution_horizon": 1,
                    "is_coupled": False,
                    "plan_in_advance_allowed": False,
                },
                execution_context="recorded_observations",
                illustration_actions=[
                    {
                        "action": "example-data action must not be used",
                        "track_object": "example-data object",
                    }
                ],
            )
            grounding = {
                "execution_summary": {
                    "artifact_paths": {"relative_ee_transforms": str(relative)}
                }
            }
            with (
                patch(
                    "novaplan.closed_loop_execution._save_selected_rollout",
                    return_value={"video": str(root / "selected.mp4"), "target_image": str(target)},
                ),
                patch("novaplan.closed_loop_execution._run_execution_step", return_value=grounding),
            ):
                summary = run_closed_loop_execution(config, planner=planner, vlm=vlm)

            planner.plan_next_reactive_step.assert_called_once()
            self.assertEqual(summary["steps"][0]["action"]["source"], "reactive_video_language_planner")
            self.assertEqual(
                summary["steps"][0]["action"]["action"],
                "place the live-selected block",
            )
            self.assertEqual(
                summary["steps"][0]["observation"]["rgb_path"],
                str((step2 / "start.png").resolve()),
            )
            self.assertEqual(
                summary["steps"][0]["observation"]["provenance"],
                "recorded_next_step_start_standin",
            )

    def test_filesystem_provider_writes_request_then_reads_ready_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inbox = root / "observations" / "step_001"
            _write_rgb(inbox / "rgb.png", 73)
            metric_depth = np.linspace(0.431, 1.234, 80, dtype=np.float32).reshape(8, 10)
            np.save(inbox / "depth.npy", metric_depth)
            (inbox / "config.json").write_text(
                json.dumps({"intrinsics": {"fx": 1, "fy": 2, "cx": 3, "cy": 4}})
            )
            out = root / "out"

            request_path = out / "step_001" / "observation_request.json"

            def publish_ready() -> None:
                deadline = time.monotonic() + 2.0
                while not request_path.exists():
                    if time.monotonic() >= deadline:
                        return
                    time.sleep(0.005)
                request = json.loads(request_path.read_text())
                (inbox / "READY").write_text(json.dumps({"request_id": request["request_id"]}))

            producer = threading.Thread(target=publish_ready)
            producer.start()

            bundle = FilesystemObservationProvider(
                root / "observations", wait_seconds=2.0, poll_seconds=0.01
            ).acquire(
                step_index=0,
                step_dir=None,
                next_step_dir=None,
                output_dir=out,
                relative_transforms_path=root / "relative_ee_transforms.npy",
            )
            producer.join(timeout=2.0)

            self.assertIsNotNone(bundle)
            self.assertTrue(bundle.is_live)
            self.assertEqual(bundle.intrinsics["fx"], 1.0)
            grounding_dir = _materialize_grounding_observation(bundle, root / "grounding_input")
            grounding_config = json.loads((grounding_dir / "config.json").read_text())
            self.assertEqual(grounding_config["depth_scale"], 0.001)
            materialized_depth = (
                np.asarray(Image.open(grounding_dir / "start_depth.png"), dtype=np.float32)
                * grounding_config["depth_scale"]
            )
            np.testing.assert_allclose(materialized_depth, metric_depth, atol=0.0005)
            request = json.loads(request_path.read_text())
            self.assertEqual(request["status"], "observation_consumed")
            self.assertIn("relative_ee_transforms.npy", request["relative_ee_transforms"])
            self.assertEqual(bundle.metadata["request_id"], request["request_id"])

    def test_unscaled_png_depth_defaults_to_millimeters_and_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.png"
            config_path = root / "config.json"
            _write_rgb(rgb_path)
            raw_depth = np.linspace(431, 1234, 80).round().astype(np.uint16).reshape(8, 10)
            Image.fromarray(raw_depth).save(depth_path)
            config_path.write_text(
                json.dumps({"intrinsics": {"fx": 1, "fy": 2, "cx": 3, "cy": 4}})
            )
            bundle = load_observation_bundle(
                rgb_path=rgb_path,
                depth_path=depth_path,
                config_path=config_path,
                provenance="test",
                is_live=True,
                step_index=0,
            )

            np.testing.assert_allclose(bundle.depth, raw_depth * 0.001, atol=1e-6)
            grounding_dir = _materialize_grounding_observation(bundle, root / "grounding_input")
            grounding_config = json.loads((grounding_dir / "config.json").read_text())

            self.assertEqual(grounding_config["depth_scale"], 0.001)
            np.testing.assert_array_equal(
                np.asarray(Image.open(grounding_dir / "start_depth.png")),
                raw_depth,
            )

    def test_invalid_observation_depth_scales_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = ObservationBundle(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                rgb_path=root / "rgb.png",
                depth=np.ones((2, 2), dtype=np.float32),
                provenance="test",
                is_live=True,
                step_index=0,
            )

            for index, depth_scale in enumerate((0, -0.001, float("nan"), "invalid")):
                with self.subTest(depth_scale=depth_scale):
                    bundle.config = {"depth_scale": depth_scale}
                    with self.assertRaisesRegex(
                        ValueError,
                        "depth_scale must be a finite positive number",
                    ):
                        _materialize_grounding_observation(
                            bundle,
                            root / f"grounding_input_{index}",
                        )

    def test_filesystem_provider_rejects_stale_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inbox = root / "observations" / "step_001"
            _write_rgb(inbox / "rgb.png", 73)
            (inbox / "READY").write_text(json.dumps({"request_id": "from-an-old-run"}))
            bundle = FilesystemObservationProvider(root / "observations", wait_seconds=0).acquire(
                step_index=0,
                step_dir=None,
                next_step_dir=None,
                output_dir=root / "out",
                relative_transforms_path=None,
            )
            self.assertIsNone(bundle)

    def test_zero_wait_rerun_resumes_pending_request_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = root / "observations"
            output = root / "out"
            relative = root / "relative.npy"
            np.save(relative, np.eye(4)[None])
            provider = FilesystemObservationProvider(observations, wait_seconds=0)
            kwargs = dict(
                step_index=0,
                step_dir=None,
                next_step_dir=None,
                output_dir=output,
                relative_transforms_path=relative,
            )

            self.assertIsNone(provider.acquire(**kwargs))
            request_path = output / "step_001" / "observation_request.json"
            first_request = json.loads(request_path.read_text())
            inbox = observations / "step_001"
            _write_rgb(inbox / "rgb.png", 88)
            (inbox / "READY").write_text(
                json.dumps({"request_id": first_request["request_id"]})
            )

            bundle = provider.acquire(**kwargs)
            self.assertIsNotNone(bundle)
            self.assertTrue(bundle.metadata["resumed_pending_request"])
            consumed = json.loads(request_path.read_text())
            self.assertEqual(consumed["status"], "observation_consumed")

            self.assertIsNone(provider.acquire(**kwargs))
            replacement = json.loads(request_path.read_text())
            self.assertNotEqual(replacement["request_id"], first_request["request_id"])
            self.assertEqual(replacement["status"], "awaiting_external_execution")

    def test_filesystem_provider_returns_none_without_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = FilesystemObservationProvider(root / "observations", wait_seconds=0).acquire(
                step_index=1,
                step_dir=None,
                next_step_dir=None,
                output_dir=root / "out",
                relative_transforms_path=None,
            )
            self.assertIsNone(bundle)
            self.assertTrue((root / "out" / "step_002" / "observation_request.json").exists())


if __name__ == "__main__":
    unittest.main()
