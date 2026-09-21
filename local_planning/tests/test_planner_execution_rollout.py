#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import concurrent.futures
import sys
import tempfile
import threading
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.planner import Beam, NovaPlanPlanner  # noqa: E402
from novaplan.video_generation import (  # noqa: E402
    HybridVideoGenerationClient,
    VeoVideoGenerationClient,
    VideoGenerationResult,
    WanVideoGenerationClient,
)
from novaplan.veo_client import Veo3VideoAdapter  # noqa: E402
from novaplan.video_generation_client import VideoGenerationClient  # noqa: E402
from local_planning.run_closed_loop_execution import _make_recovery_video_client  # noqa: E402


class _FixedVideoModel:
    def __init__(self):
        self.calls = []

    def generate_rollouts(self, *, start_frame, action_text, num_samples, num_frames, fps, **kwargs):
        self.calls.append(
            {
                "start_frame": np.asarray(start_frame).copy(),
                "action": action_text,
                "num_samples": num_samples,
                "num_frames": num_frames,
                "fps": fps,
                "kwargs": kwargs,
            }
        )
        videos = []
        for idx in range(num_samples):
            video = np.repeat(np.asarray(start_frame)[None], num_frames, axis=0)
            video = video.copy()
            video[-1] = idx + 1
            videos.append(video)
        return videos


class _FlowChoiceVideoModel:
    def __init__(self, object_flow: np.ndarray, hand_flow: np.ndarray):
        self.object_flow = object_flow
        self.hand_flow = hand_flow

    def generate_rollouts(self, *, start_frame, num_frames, **kwargs):
        del kwargs
        video = np.repeat(np.asarray(start_frame)[None], num_frames, axis=0)
        choice = SimpleNamespace(
            selected_flow="hand",
            selected_is_hand=True,
            hand_flow=SimpleNamespace(flow_image=self.hand_flow),
            to_dict=lambda: {"selected_flow": "hand"},
        )
        return VideoGenerationResult(
            videos=[video],
            flow_images=[self.object_flow],
            flow_choices=[choice],
            video_sources=["wan22"],
        )


class _RankingVLM:
    def __init__(self):
        self.ranking_candidates = None
        self.ranking_debug_dir = None
        self.ranking_step = None

    def rank_rollouts_batch(self, *, goal, candidates, top_n, debug_dir, step):
        del goal, top_n
        self.ranking_candidates = candidates
        self.ranking_debug_dir = Path(debug_dir)
        self.ranking_step = int(step)
        winner = dict(candidates[-1])
        winner.update({"score": 0.9, "success": True, "rank_reason": "best"})
        return [winner]


class _PromptTrackingVLM(_RankingVLM):
    def __init__(self):
        super().__init__()
        self.extension_backends = []

    def extend_video_prompt(self, *, backend, action, **kwargs):
        del kwargs
        self.extension_backends.append(backend)
        return f"extended {backend}: {action}"


class _TopOnlyZeroScoreVLM:
    def propose_actions(self, **kwargs):
        del kwargs
        return [{"action": "move block", "track_object": "block", "is_finish": False}]

    def rank_rollouts_batch(self, *, candidates, **kwargs):
        del kwargs
        winner = dict(candidates[-1])
        winner.update({"score": 0.0, "success": False, "rank_reason": "least bad"})
        return [winner]


class _FinishVLM:
    def propose_actions(self, **kwargs):
        del kwargs
        return [
            {
                "action": "Hold current position securely.",
                "track_object": "human hand",
                "is_finish": True,
                "constraint_context": "goal complete",
            }
        ]


class _WanFLF:
    def __init__(self):
        self.last_frame = None

    def generate_rollouts(self, **kwargs):
        self.last_frame = kwargs.get("last_frame")
        video = np.stack([kwargs["start_frame"], kwargs["last_frame"]])
        return VideoGenerationResult(videos=[video], video_sources=["wan22"])


class _VeoFLF:
    def __init__(self):
        self.last_frame = None

    def generate_rollouts(self, **kwargs):
        self.last_frame = kwargs.get("last_frame")
        video = np.stack([kwargs["start_frame"], kwargs["last_frame"]])
        callback = kwargs.get("on_video")
        if callback:
            callback(0, video)
        return [video]


class _FailingVeo:
    def generate_rollouts(self, **kwargs):
        del kwargs
        raise RuntimeError("reauthentication required")


class _UnavailableSelectionFlow:
    def __init__(self):
        self.calls = 0

    def ensure_selection_flow_contract(self):
        self.calls += 1
        raise RuntimeError("video selection service unavailable")


class _MarkerSelectionFlow:
    def ensure_selection_flow_contract(self):
        return None

    def extract_flow(self, *, videos, **kwargs):
        del kwargs
        marker = int(np.asarray(videos[0])[0, 0, 0, 0])
        return SimpleNamespace(
            flow_images=[np.full((2, 2, 3), marker, dtype=np.uint8)],
            flow_bytes=[],
            coords_3d=[],
            visibilities=[],
            flow_choices=[],
        )


class _PartialVeoBatch(list):
    sample_indices = (0, 2)
    sample_errors = {1: "transient Vertex failure"}


class _PartialVeo:
    def generate_rollouts(self, **kwargs):
        videos = [
            np.full((2, 4, 5, 3), 10, dtype=np.uint8),
            np.full((2, 4, 5, 3), 12, dtype=np.uint8),
        ]
        callback = kwargs.get("on_video")
        if callback is not None:
            callback(0, videos[0])
            callback(2, videos[1])
        return _PartialVeoBatch(videos)


class _PromptCaptureWan:
    def __init__(self):
        self.action_text = None

    def generate_rollouts(self, **kwargs):
        self.action_text = kwargs["action_text"]
        video = np.repeat(np.asarray(kwargs["start_frame"])[None], kwargs["num_frames"], axis=0)
        return VideoGenerationResult(videos=[video], video_sources=["wan22"])


class _PromptCaptureVeo:
    def __init__(self):
        self.action_text = None

    def generate_rollouts(self, **kwargs):
        self.action_text = kwargs["action_text"]
        video = np.repeat(np.asarray(kwargs["start_frame"])[None], kwargs["num_frames"], axis=0)
        callback = kwargs.get("on_video")
        if callback is not None:
            callback(0, video)
        return [video]


class PlannerExecutionRolloutTest(unittest.TestCase):
    def test_video_generation_public_names_and_legacy_aliases_match(self):
        self.assertIs(VideoGenerationClient, WanVideoGenerationClient)
        self.assertIs(Veo3VideoAdapter, VeoVideoGenerationClient)
        self.assertEqual(
            HybridVideoGenerationClient.__module__,
            "novaplan.video_generation.hybrid",
        )

    def test_veo_default_uses_current_vertex_endpoint(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            adapter = VeoVideoGenerationClient(mock=True)
        self.assertEqual(adapter._model, "veo-3.1-generate-001")
        self.assertEqual(adapter._location, "global")

    def test_veo_model_and_location_can_be_overridden(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VEO3_MODEL": "veo-3.1-fast-generate-001",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            },
            clear=True,
        ):
            adapter = VeoVideoGenerationClient(mock=True)
        self.assertEqual(adapter._model, "veo-3.1-fast-generate-001")
        self.assertEqual(adapter._location, "us-central1")

    def test_veo_vertex_client_uses_bounded_request_timeout(self):
        with mock.patch.dict(
            "os.environ",
            {"GOOGLE_CLOUD_PROJECT": "test-project"},
            clear=True,
        ), mock.patch("novaplan.video_generation.veo.genai.Client") as client:
            VeoVideoGenerationClient(mock=False, request_timeout_s=7.5)

        http_options = client.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, 7500)

    def test_veo_operation_polling_has_deadline(self):
        adapter = VeoVideoGenerationClient(
            mock=True,
            operation_timeout_s=1.0,
            poll_interval_s=0.1,
        )
        get_operation = mock.Mock()
        adapter._client = SimpleNamespace(
            operations=SimpleNamespace(get=get_operation)
        )
        operation = SimpleNamespace(done=False)

        with mock.patch(
            "novaplan.video_generation.veo.time.monotonic",
            side_effect=[0.0, 0.2, 1.0],
        ), mock.patch("novaplan.video_generation.veo.time.sleep"):
            with self.assertRaisesRegex(TimeoutError, "timed out after 1 seconds"):
                adapter._wait_for_operation(operation, sample_index=0)

        get_operation.assert_not_called()

    def test_veo_keeps_successful_samples_when_one_generation_fails(self):
        adapter = VeoVideoGenerationClient(mock=True)
        adapter.mock = False

        def generate_videos(*, config, **kwargs):
            del kwargs
            sample_seed = int(config.seed)
            if sample_seed == 11:
                raise RuntimeError("transient Vertex failure")
            video = SimpleNamespace(
                video_bytes=str(sample_seed).encode("ascii"),
                uri=None,
            )
            return SimpleNamespace(
                done=True,
                result=SimpleNamespace(
                    generated_videos=[SimpleNamespace(video=video)]
                ),
                status="completed",
            )

        adapter._client = SimpleNamespace(
            models=SimpleNamespace(generate_videos=generate_videos),
            operations=SimpleNamespace(get=mock.Mock()),
        )
        callback_indices = []

        def decode_marker(payload):
            marker = int(payload.decode("ascii"))
            return np.full((2, 3, 4, 3), marker, dtype=np.uint8)

        with mock.patch.object(
            adapter,
            "_mp4_to_numpy",
            side_effect=decode_marker,
        ):
            videos = adapter.generate_rollouts(
                start_frame=np.zeros((3, 4, 3), dtype=np.uint8),
                action_text="move once",
                num_samples=3,
                num_frames=2,
                seed=10,
                on_video=lambda index, video: callback_indices.append(index),
            )

        self.assertEqual(videos.sample_indices, (0, 2))
        self.assertIn("transient Vertex failure", videos.sample_errors[1])
        self.assertEqual(
            [int(video[0, 0, 0, 0]) for video in videos],
            [10, 12],
        )
        self.assertEqual(sorted(callback_indices), [0, 2])

    def test_veo_http_result_download_has_timeout_and_status_check(self):
        adapter = VeoVideoGenerationClient(mock=True, download_timeout_s=9.0)
        response = mock.Mock(content=b"video")
        with mock.patch(
            "novaplan.video_generation.veo.requests.get",
            return_value=response,
        ) as get:
            payload = adapter._download_video_bytes("https://example.test/video.mp4")

        self.assertEqual(payload, b"video")
        get.assert_called_once_with(
            "https://example.test/video.mp4",
            timeout=9.0,
        )
        response.raise_for_status.assert_called_once_with()

    def test_veo_temporal_resampling_returns_exact_planner_length(self):
        source = np.repeat(
            np.arange(97, dtype=np.uint8)[:, None, None, None],
            3,
            axis=3,
        )

        sampled = VeoVideoGenerationClient._resample_video_frames(source, 41)
        expected_indices = np.rint(np.linspace(0, 96, num=41)).astype(np.int64)

        self.assertEqual(sampled.shape, (41, 1, 1, 3))
        np.testing.assert_array_equal(sampled[:, 0, 0, 0], expected_indices)
        np.testing.assert_array_equal(sampled[0], source[0])
        np.testing.assert_array_equal(sampled[-1], source[-1])

    def test_direct_planner_defaults_match_paper_and_execution_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = NovaPlanPlanner(
                vlm=_FinishVLM(),
                t2v=_FixedVideoModel(),
                use_prompt_extension=False,
                debug_dir=Path(tmp),
            )

            self.assertEqual(planner.beam_size, 2)
            self.assertEqual(planner.num_action_per_beam, 2)
            self.assertEqual(planner.num_video_per_action, 4)
            self.assertEqual(planner.execution_num_action_per_step, 1)
            self.assertEqual(planner.execution_num_video_per_action, 8)
            self.assertEqual(planner.segment_T, 41)
            self.assertEqual(planner.fps, 16)

    def test_recovery_backend_selects_one_backend_from_available_hybrid_stack(self):
        wan = _WanFLF()
        veo = _VeoFLF()
        hybrid = HybridVideoGenerationClient(wan_client=wan, veo_client=veo)
        args = SimpleNamespace(recovery_video_backend="veo3", video_backend="both")
        self.assertIs(_make_recovery_video_client(args, hybrid, None), veo)
        args.recovery_video_backend = "wan22"
        self.assertIs(_make_recovery_video_client(args, hybrid, None), wan)
        args.recovery_video_backend = "both"
        self.assertIs(_make_recovery_video_client(args, hybrid, None), hybrid)

    def test_hybrid_extends_wan_in_chinese_path_and_veo_in_english_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            wan = _PromptCaptureWan()
            veo = _PromptCaptureVeo()
            vlm = _PromptTrackingVLM()
            planner = NovaPlanPlanner(
                vlm=vlm,
                t2v=HybridVideoGenerationClient(wan_client=wan, veo_client=veo),
                execution_num_video_per_action=1,
                segment_T=2,
                exploit_filter=False,
                enable_flow=False,
                use_prompt_extension=True,
                debug_dir=Path(tmp),
            )

            action = "place the blue block in the blue dish"
            planner.generate_execution_rollout(
                np.zeros((4, 5, 3), dtype=np.uint8),
                "sort the blocks",
                action=action,
                track_object="blue block",
            )

            self.assertEqual(vlm.extension_backends, ["wan22", "veo3"])
            self.assertEqual(wan.action_text, f"extended wan22: {action}")
            self.assertEqual(veo.action_text, f"extended veo3: {action}")

    def test_algorithm_two_s_min_cannot_outrank_returned_zero_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = NovaPlanPlanner(
                vlm=_TopOnlyZeroScoreVLM(),
                t2v=_FixedVideoModel(),
                beam_size=1,
                num_action_per_beam=1,
                num_video_per_action=2,
                segment_T=2,
                exploit_filter=False,
                use_prompt_extension=False,
                debug_dir=Path(tmp),
            )
            selected = planner.step(
                [Beam(score=0.0, frame=np.zeros((4, 5, 3), dtype=np.uint8))],
                "move block",
            )[0]

            self.assertEqual(selected.score, 0.0)
            self.assertEqual(int(selected.video[-1][-1, 0, 0, 0]), 2)

    def test_reactive_finish_returns_completion_without_video_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _FixedVideoModel()
            planner = NovaPlanPlanner(
                vlm=_FinishVLM(),
                t2v=model,
                execution_num_action_per_step=1,
                execution_num_video_per_action=2,
                exploit_filter=False,
                use_prompt_extension=False,
                debug_dir=Path(tmp),
            )
            result = planner.plan_next_reactive_step(
                np.zeros((4, 5, 3), dtype=np.uint8),
                "already complete",
            )

            self.assertTrue(result.task_complete)
            self.assertEqual(result.actions, [])
            self.assertEqual(model.calls, [])

    def test_fixed_strategic_action_regenerates_from_latest_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _FixedVideoModel()
            vlm = _RankingVLM()
            planner = NovaPlanPlanner(
                vlm=vlm,
                t2v=model,
                beam_size=2,
                num_action_per_beam=3,
                num_video_per_action=4,
                execution_num_video_per_action=2,
                segment_T=3,
                fps=8,
                exploit_filter=False,
                use_prompt_extension=False,
                debug_dir=Path(tmp),
            )
            latest = np.full((6, 8, 3), 17, dtype=np.uint8)
            status_events = []
            saved_debug_names = []
            planner.status_callback = lambda event, payload: status_events.append((event, payload))
            planner.save_debug_videos = True
            planner._save_debug_video = (
                lambda video, filename, metadata="": saved_debug_names.append(filename)
            )

            result = planner.generate_execution_rollout(
                latest,
                "stack the blocks",
                action="place the blue block on the red block",
                track_object="blue block",
                history=["pick up the blue block"],
            )

            self.assertEqual(len(model.calls), 1)
            np.testing.assert_array_equal(model.calls[0]["start_frame"], latest)
            self.assertEqual(model.calls[0]["num_samples"], 2)
            self.assertEqual(result.actions[-1], "place the blue block on the red block")
            self.assertEqual(result.track_objects[-1], "blue block")
            self.assertEqual(int(result.video[-1][-1, 0, 0, 0]), 2)
            self.assertEqual(vlm.ranking_candidates[0]["track_object"], "blue block")
            generation_event = next(
                payload for event, payload in status_events
                if event == "video_generation_completed"
            )
            self.assertEqual(generation_event["generated_total"], 2)
            selection_event = next(payload for event, payload in status_events if event == "video_selected")
            self.assertEqual(selection_event["candidate_id"], 1)
            self.assertEqual(selection_event["candidate_count"], 2)
            self.assertEqual(selection_event["backend"], "unknown")
            self.assertEqual(selection_event["score"], 0.9)
            self.assertEqual(
                vlm.ranking_debug_dir,
                Path(tmp)
                / "step_001"
                / "debug_artifacts"
                / "video_rollout_selection"
                / "selected_rollout",
            )
            self.assertEqual(vlm.ranking_step, 1)

            saved_debug_names.clear()
            planner.generate_execution_rollout(
                latest,
                "stack the blocks",
                action="place the blue block on the red block",
                track_object="blue block",
                generation_attempt=2,
                debug_step_index=1,
            )
            self.assertEqual(
                vlm.ranking_debug_dir,
                Path(tmp)
                / "step_001"
                / "debug_artifacts"
                / "video_rollout_selection"
                / "regeneration_002",
            )
            self.assertEqual(vlm.ranking_step, 1)
            self.assertTrue(saved_debug_names)
            self.assertTrue(
                all(name.startswith("step_1_video_") for name in saved_debug_names)
            )

    def test_parallel_action_rollouts_keep_unique_video_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = NovaPlanPlanner(
                vlm=_RankingVLM(),
                t2v=_FixedVideoModel(),
                num_video_per_action=1,
                segment_T=2,
                exploit_filter=False,
                save_debug_videos=True,
                use_prompt_extension=False,
                debug_dir=Path(tmp),
            )
            save_barrier = threading.Barrier(2)
            saved_names = []
            saved_names_lock = threading.Lock()

            def _delayed_save(video, filename, metadata=""):
                del video, metadata
                save_barrier.wait(timeout=5)
                with saved_names_lock:
                    saved_names.append(filename)

            planner._save_debug_video = _delayed_save
            beam = Beam(score=0.0, frame=np.zeros((4, 5, 3), dtype=np.uint8))
            tasks = [
                {
                    "beam_idx": idx,
                    "beam": beam,
                    "action": f"action {idx}",
                    "track_object": "block",
                    "step_counter": 1,
                    "beam_counter": idx,
                    "action_idx": 0,
                }
                for idx in range(2)
            ]

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results = list(
                    executor.map(
                        lambda task: planner._process_action_complete(task, "move block")[0],
                        tasks,
                    )
                )

            self.assertEqual(sorted(item["video_index"] for item in results), [1, 2])
            self.assertEqual(
                sorted(saved_names),
                ["step_1_video_1.mp4", "step_1_video_2.mp4"],
            )

    def test_rollout_ranking_keeps_object_flow_before_hand_grounding_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            object_flow = np.full((4, 5, 3), 17, dtype=np.uint8)
            hand_flow = np.full((4, 5, 3), 201, dtype=np.uint8)
            vlm = _RankingVLM()
            planner = NovaPlanPlanner(
                vlm=vlm,
                t2v=_FlowChoiceVideoModel(object_flow, hand_flow),
                execution_num_video_per_action=1,
                segment_T=2,
                exploit_filter=False,
                use_prompt_extension=False,
                debug_dir=Path(tmp),
            )

            planner.generate_execution_rollout(
                np.zeros((4, 5, 3), dtype=np.uint8),
                "move block",
                action="move the block",
                track_object="block",
            )

            np.testing.assert_array_equal(
                vlm.ranking_candidates[0]["flow_image"],
                object_flow,
            )

    def test_hybrid_forwards_last_frame_to_wan_and_veo(self):
        wan = _WanFLF()
        veo = _VeoFLF()
        hybrid = HybridVideoGenerationClient(wan_client=wan, veo_client=veo)
        start = np.zeros((4, 5, 3), dtype=np.uint8)
        goal = np.full((4, 5, 3), 255, dtype=np.uint8)

        result = hybrid.generate_rollouts(
            start_frame=start,
            last_frame=goal,
            action_text="move once",
            num_samples=1,
            num_frames=2,
            fps=8,
        )

        np.testing.assert_array_equal(wan.last_frame, goal)
        np.testing.assert_array_equal(veo.last_frame, goal)
        self.assertEqual(result.video_sources, ["wan22", "veo3"])

    def test_hybrid_reports_partial_backend_failure(self):
        hybrid = HybridVideoGenerationClient(wan_client=_WanFLF(), veo_client=_FailingVeo())
        start = np.zeros((4, 5, 3), dtype=np.uint8)
        goal = np.full((4, 5, 3), 255, dtype=np.uint8)

        result = hybrid.generate_rollouts(
            start_frame=start,
            last_frame=goal,
            action_text="move once",
            num_samples=1,
            num_frames=2,
            fps=8,
        )

        self.assertEqual(result.video_sources, ["wan22"])
        self.assertEqual(result.requested_backends, ["wan22", "veo3"])
        self.assertEqual(result.requested_samples_per_backend, 1)
        self.assertEqual(result.backend_errors, {"veo3": "reauthentication required"})

    def test_hybrid_aligns_flow_with_partial_veo_successes(self):
        hybrid = HybridVideoGenerationClient(
            wan_client=_WanFLF(),
            veo_client=_PartialVeo(),
            flow_client=_MarkerSelectionFlow(),
        )
        start = np.zeros((4, 5, 3), dtype=np.uint8)
        goal = np.full((4, 5, 3), 255, dtype=np.uint8)

        result = hybrid.generate_rollouts(
            start_frame=start,
            last_frame=goal,
            action_text="move once",
            num_samples=3,
            num_frames=2,
            fps=8,
            enable_flow=True,
            mask_prompt="block",
        )

        self.assertEqual(result.video_sources, ["wan22", "veo3", "veo3"])
        self.assertIsNone(result.flow_images[0])
        self.assertEqual(int(result.flow_images[1][0, 0, 0]), 10)
        self.assertEqual(int(result.flow_images[2][0, 0, 0]), 12)
        self.assertIn("partial batch (2/3 successful)", result.backend_errors["veo3"])

    def test_hybrid_preflights_selection_flow_before_generating_either_backend(self):
        wan = _WanFLF()
        veo = _VeoFLF()
        flow = _UnavailableSelectionFlow()
        hybrid = HybridVideoGenerationClient(
            wan_client=wan,
            veo_client=veo,
            flow_client=flow,
        )

        with self.assertRaisesRegex(RuntimeError, "selection service unavailable"):
            hybrid.generate_rollouts(
                start_frame=np.zeros((4, 5, 3), dtype=np.uint8),
                action_text="move once",
                num_samples=1,
                num_frames=2,
                fps=8,
                enable_flow=True,
                mask_prompt="block",
            )

        self.assertEqual(flow.calls, 1)
        self.assertIsNone(wan.last_frame)
        self.assertIsNone(veo.last_frame)

    def test_mock_veo_preserves_first_and_last_conditioning_frames(self):
        start = np.zeros((7, 9, 3), dtype=np.uint8)
        goal = np.full((7, 9, 3), 211, dtype=np.uint8)
        video = VeoVideoGenerationClient(mock=True).generate_rollouts(
            start_frame=start,
            last_frame=goal,
            action_text="one contact motion",
            num_samples=1,
            num_frames=4,
        )[0]

        np.testing.assert_array_equal(video[0], start)
        np.testing.assert_array_equal(video[-1], goal)


if __name__ == "__main__":
    unittest.main()
