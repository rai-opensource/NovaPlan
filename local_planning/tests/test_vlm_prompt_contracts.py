#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.llm_client import DEFAULT_OPENAI_VLM_MODEL, VLMAdapter
from novaplan.closed_loop_execution import _select_action
from novaplan.vlm_prompts import (
    PROMPT_CONTRACT_VERSION,
    PROMPT_REGISTRY,
    build_action_proposal_prompt,
    build_direct_video_prompt,
    build_recovery_prompt,
    build_rollout_ranking_prompt,
    build_task_structure_prompt,
    build_transition_verification_prompt,
    build_video_negative_prompt,
    build_video_prompt_extension_instructions,
    validate_action_proposal_payload,
    validate_ranking_payload,
)


class VLMPromptContractTest(unittest.TestCase):
    def test_runtime_model_default(self):
        self.assertEqual(DEFAULT_OPENAI_VLM_MODEL, "gpt-5.6")

    def test_none_and_max_are_real_reasoning_efforts(self):
        self.assertEqual(VLMAdapter._normalize_reasoning_effort("none"), "none")
        self.assertEqual(VLMAdapter._normalize_reasoning_effort("max"), "max")
        self.assertIsNone(VLMAdapter._normalize_reasoning_effort("model-default"))

    def test_registry_covers_every_live_prompt_family(self):
        self.assertEqual(
            set(PROMPT_REGISTRY),
            {
                "task_structure",
                "action_proposal",
                "video_prompt_extension",
                "rollout_ranking",
                "transition_verification",
                "recovery_policy",
                "rollout_score",
            },
        )
        for contract in PROMPT_REGISTRY.values():
            self.assertEqual(contract.version, PROMPT_CONTRACT_VERSION)
            self.assertTrue(contract.provenance)
            self.assertTrue(contract.response_schema)

    def test_task_structure_marks_hidden_state_as_reactive(self):
        prompt = build_task_structure_prompt(
            goal="open the drawer and retrieve the hidden mug",
            min_horizon=1,
            max_horizon=8,
        )
        self.assertIn("initially hidden object is REACTIVE", prompt)
        self.assertIn("plan_in_advance_allowed=false", prompt)

    def test_task_structure_shared_resources_do_not_create_coupling(self):
        prompt = build_task_structure_prompt(
            goal="put each block in its matching container",
            min_horizon=1,
            max_horizon=8,
        )
        self.assertIn("Sharing one human hand, robot, workspace", prompt)
        self.assertIn("ordering_dependencies=[] and is_coupled=false", prompt)

    def test_horizon_routing_is_derived_from_ordering_dependencies(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"visual_analysis":{},"subtasks":[{"description":"place yellow"},'
                            '{"description":"place red"},{"description":"place blue"}],'
                            '"ordering_dependencies":[],"is_coupled":true,'
                            '"plan_in_advance_allowed":true,"horizon":3,'
                            '"coupling_reason":"shared hand and workspace only; any order",'
                            '"reasoning":"three independent placements"}'
                        )
                    )
                )
            ]
        )
        adapter._model_for_task = lambda task=None: "unit-test"
        adapter._reasoning_effort_for_task = lambda task=None: "low"

        result = adapter.estimate_horizon(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "put each block in its matching container",
        )

        self.assertFalse(result["is_coupled"])
        self.assertFalse(result["plan_in_advance_allowed"])
        self.assertEqual(result["planner_horizon"], 1)
        self.assertEqual(result["reactive_execution_horizon"], 3)
        self.assertEqual(result["execution_mode"], "reactive_step_by_step")

    def test_action_prompt_carries_full_history_horizon_and_constraints(self):
        prompt = build_action_proposal_prompt(
            goal="sort the blocks",
            num_actions=2,
            previous_action="ignored when full history exists",
            action_history=["Place the blue block", "Place the red block"],
            steps_remaining=3,
            constraint_context="green block must precede the bridge",
        )
        self.assertIn("1. Place the blue block", prompt)
        self.assertIn("2. Place the red block", prompt)
        self.assertIn("REMAINING EXECUTION HORIZON: 3", prompt)
        self.assertIn("You have time. Prioritize precision and safety.", prompt)
        self.assertIn("green block must precede the bridge", prompt)
        self.assertIn("EXACTLY 2 proposal objects", prompt)
        self.assertIn("A-then-B and B-then-A", prompt)

        urgent = build_action_proposal_prompt(
            goal="finish assembly",
            num_actions=1,
            action_history=["Place base"],
            steps_remaining=2,
        )
        self.assertIn("Time is critical. Actions must be aggressive.", urgent)

    def test_action_schema_rejects_padding_or_missing_fields(self):
        valid = {
            "phase": "Approaching",
            "dependency_analysis": "independent visible targets",
            "valid_objects": ["A", "B"],
            "proposals": [
                {"action": "Place A", "track_object": "A", "reasoning": "visible"},
                {"action": "Place B", "track_object": "B", "reasoning": "independent"},
            ]
        }
        self.assertEqual(len(validate_action_proposal_payload(valid, expected_count=2)), 2)
        with self.assertRaisesRegex(ValueError, "expected exactly 2"):
            validate_action_proposal_payload({**valid, "proposals": valid["proposals"][:1]}, expected_count=2)
        with self.assertRaisesRegex(ValueError, "empty required fields"):
            validate_action_proposal_payload(
                {
                    **valid,
                    "proposals": [{"action": "Place A", "track_object": "A", "reasoning": ""}],
                },
                expected_count=1,
            )

    def test_video_extension_is_backend_and_frame_semantics_specific(self):
        wan = build_video_prompt_extension_instructions(
            backend="wan22",
            action="Place the block",
            goal="sort blocks",
            track_object="block",
            has_last_frame=False,
        )
        veo = build_video_prompt_extension_instructions(
            backend="veo3",
            action="Place the block",
            goal="sort blocks",
            track_object="block",
            has_last_frame=True,
        )
        self.assertEqual((wan.backend, wan.language), ("wan22", "zh"))
        self.assertIn("一只干净普通的人类右手", wan.system_prompt)
        self.assertIn("机械臂", wan.negative_prompt)
        self.assertEqual((veo.backend, veo.language), ("veo3", "en"))
        self.assertIn("START and GOAL images", veo.system_prompt)
        self.assertIn("multiple hands", veo.negative_prompt)
        direct_veo = build_direct_video_prompt(
            backend="veo3",
            action="Place the block",
        )
        self.assertIn("Only one clean ordinary human right hand", direct_veo)
        self.assertIn("[ACTION] Place the block", direct_veo)

    def test_paper_ranking_and_verification_schemas_are_stable(self):
        ranking_prompt = build_rollout_ranking_prompt(
            goal="insert the block",
            action_descriptions=['ID 0: "Insert"', 'ID 1: "Push away"'],
        )
        self.assertIn("score=0.0", ranking_prompt)
        self.assertIn("score<=0.1", ranking_prompt)
        self.assertIn("score<=0.2", ranking_prompt)
        self.assertIn("Rank 2 robot rollouts", ranking_prompt)
        self.assertNotIn("Actor Check", ranking_prompt)
        rankings = validate_ranking_payload(
            {
                "rankings": [
                    {"candidate_id": 0, "success": True, "score": 0.8, "reason": "seated"},
                    {"candidate_id": 1, "success": False, "score": 0.0, "reason": "wrong motion"},
                ]
            },
            candidate_count=2,
        )
        self.assertEqual({item["candidate_id"] for item in rankings}, {0, 1})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_ranking_payload(
                {
                    "rankings": [
                        {"candidate_id": 0, "success": True, "score": 0.8, "reason": "ok"},
                        {"candidate_id": 0, "success": False, "score": 0.0, "reason": "bad"},
                    ]
                },
                candidate_count=2,
            )
        with self.assertRaisesRegex(ValueError, "marked successful"):
            validate_ranking_payload(
                {
                    "rankings": [
                        {"candidate_id": 0, "success": True, "score": 0.2, "reason": "contradiction"},
                    ]
                },
                candidate_count=1,
            )
        with self.assertRaisesRegex(ValueError, "marked unsuccessful"):
            validate_ranking_payload(
                {
                    "rankings": [
                        {"candidate_id": 0, "success": False, "score": 0.9, "reason": "contradiction"},
                    ]
                },
                candidate_count=1,
            )

        verification = build_transition_verification_prompt(action="insert", goal="assemble")
        self.assertIn('{"success": boolean, "reason":', verification)
        self.assertNotIn("correction_action", verification)

    def test_recovery_prompt_keeps_paper_top_level_and_internal_finger_in_prompt_p(self):
        prompt = build_recovery_prompt(
            goal="seat the block",
            previous_action="Insert the block",
            failure_reason="slightly tilted",
            object_prompt="block",
        )
        self.assertIn("prompt_P.enabled=false", prompt)
        self.assertIn('"recovery_action":', prompt)
        self.assertIn("first-last-frame recovery rollouts", prompt)
        self.assertIn("Do not merely copy the failed previous action", prompt)
        self.assertIn("CONTACT FINGER: <finger>", prompt)
        self.assertIn("PIXEL: [x, y]", prompt)
        self.assertNotIn('"contact_finger":', prompt)
        recovery_schema = PROMPT_REGISTRY["recovery_policy"].response_schema
        self.assertIn("recovery_action", recovery_schema["required"])
        self.assertFalse(recovery_schema["additionalProperties"])

    def test_vlm_call_site_uses_backend_specific_extension_contract(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="a generated prompt"))]
            )

        adapter._openai_chat_completion = fake_completion
        adapter._model_for_task = lambda task=None: "unit-test"
        adapter._reasoning_effort_for_task = lambda task=None: None
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        wan_prompt = adapter.extend_video_prompt(image, "Place block", backend="wan22")
        veo_prompt = adapter.extend_video_prompt(image, "Place block", backend="veo3", last_image=image)

        self.assertIn("中文", calls[0]["messages"][0]["content"])
        self.assertIn("START and GOAL", calls[1]["messages"][0]["content"])
        image_parts = [part for part in calls[1]["messages"][1]["content"] if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 2)
        self.assertIn("【硬约束", wan_prompt)
        self.assertIn("[HARD CONSTRAINTS", veo_prompt)
        self.assertIn("one named fingertip", veo_prompt)
        self.assertIn("多余的手指", build_video_negative_prompt("wan22"))
        self.assertIn("extra fingers", build_video_negative_prompt("veo3"))

    def test_video_extension_sanitizes_leaked_mechanical_actor_in_scene_only(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[SCENE DESCRIPTION] A robotic arm moves the block."))]
        )
        adapter._model_for_task = lambda task=None: "unit-test"
        adapter._reasoning_effort_for_task = lambda task=None: None
        prompt = adapter.extend_video_prompt(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "Place block",
            backend="veo3",
        )
        scene = prompt.split("[SCENE DESCRIPTION]", 1)[1]
        self.assertNotIn("robotic arm", scene.lower())
        self.assertIn("human right hand", scene.lower())

    def test_batch_ranker_requires_flow_and_restores_original_id(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=(
                    '{"rankings":['
                    '{"candidate_id":101,"success":true,"score":0.9,"reason":"best"}]}'
                )))]
            )

        adapter._openai_chat_completion = fake_completion
        adapter._model_for_task = lambda task=None: "unit-test"
        adapter._reasoning_effort_for_task = lambda task=None: None
        video = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        ranked = adapter.rank_rollouts_batch(
            goal="place block",
            candidates=[
                {"candidate_id": 101, "action": "Place block", "rollout": video, "flow_image": video[0]},
                {"candidate_id": 202, "action": "Place block", "rollout": video, "flow_image": None},
            ],
            top_n=1,
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["candidate_id"], 101)
        self.assertEqual(ranked[0]["grid_candidate_id"], 101)
        self.assertEqual(ranked[0]["score"], 0.9)
        prompt_text = captured["messages"][0]["content"][0]["text"]
        self.assertIn("ID 101", prompt_text)
        self.assertNotIn("ID 202", prompt_text)
        self.assertNotIn("no_flow", prompt_text)

    def test_batch_ranker_returns_empty_when_no_candidate_has_flow(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: self.fail(
            "flowless candidates must not be sent to the VLM"
        )
        video = np.zeros((2, 8, 8, 3), dtype=np.uint8)

        ranked = adapter.rank_rollouts_batch(
            goal="place block",
            candidates=[
                {"candidate_id": 101, "action": "Place block", "rollout": video, "flow_image": None},
            ],
            top_n=1,
        )

        self.assertEqual(ranked, [])

    def test_action_cleanup_and_strict_legacy_score_parser(self):
        cleaned, finished = VLMAdapter._normalize_proposed_action("FINISH: Hold position")
        self.assertTrue(finished)
        self.assertEqual(cleaned, "Hold current position securely.")
        cleaned, finished = VLMAdapter._normalize_proposed_action(
            "The robot right hand grasp the blue block"
        )
        self.assertFalse(finished)
        self.assertEqual(cleaned, "Grasp the blue block")
        with self.assertRaisesRegex(ValueError, "single-hand"):
            VLMAdapter._normalize_proposed_action("Use both hands to lift the block")

        adapter = VLMAdapter.__new__(VLMAdapter)
        self.assertAlmostEqual(
            adapter._parse_score('{"confidence":0.8,"steps_to_goal":2}'),
            0.6,
        )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            adapter._parse_score("{}")

    def test_closed_loop_fallback_passes_full_history_and_constraint_context(self):
        class RecordingVLM:
            def __init__(self):
                self.kwargs = None

            def propose_actions(self, **kwargs):
                self.kwargs = kwargs
                return [{"action": "Place green", "track_object": "green"}]

        vlm = RecordingVLM()
        action = _select_action(
            step_idx=2,
            step_dir=None,
            plan_actions=[],
            prefer_reactive_vlm=True,
            vlm=vlm,
            current_image=np.zeros((8, 8, 3), dtype=np.uint8),
            goal="assemble",
            history=["Open base", "Place blue"],
            horizon_remaining=2,
            constraint_context='{"dependency":"green first"}',
            fallback_action=None,
            fallback_track_object="object",
        )
        self.assertEqual(action["action"], "Place green")
        self.assertEqual(vlm.kwargs["action_history"], ["Open base", "Place blue"])
        self.assertEqual(vlm.kwargs["constraint_context"], '{"dependency":"green first"}')

    def test_vlm_action_call_rejects_wrong_count_instead_of_fabricating_motion(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"phase":"Approaching","dependency_analysis":"none",'
                            '"valid_objects":["block"],"proposals":['
                            '{"action":"Place block","track_object":"block","reasoning":"visible"}]}'
                        )
                    )
                )
            ]
        )
        adapter._model_for_task = lambda task=None: "unit-test"
        adapter._reasoning_effort_for_task = lambda task=None: None
        with self.assertRaisesRegex(RuntimeError, "expected exactly 2"):
            adapter.propose_actions(
                np.zeros((8, 8, 3), dtype=np.uint8),
                "sort blocks",
                2,
                action_history=["Place red"],
                constraint_context="blue remains",
            )

    def test_vlm_action_result_exposes_deterministic_constraint_context(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"phase":"Interacting","dependency_analysis":"green must be first",'
                            '"valid_objects":["green block"],"proposals":['
                            '{"action":"Place green block","track_object":"green block",'
                            '"reasoning":"it is the prerequisite"}]}'
                        )
                    )
                )
            ]
        )
        adapter._model_for_task = lambda task=None: "unit-test"
        adapter._reasoning_effort_for_task = lambda task=None: None
        proposal = adapter.propose_actions(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "assemble",
            1,
            action_history=["Open the base"],
        )[0]
        self.assertEqual(
            proposal["constraint_context"],
            '{"dependency_analysis":"green must be first","phase":"Interacting",'
            '"valid_objects":["green block"]}',
        )

    def test_legacy_action_parser_never_fabricates_move_forward(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        with self.assertRaises(ValueError):
            adapter._parse_actions('{"actions": []}', 1)


if __name__ == "__main__":
    unittest.main()
