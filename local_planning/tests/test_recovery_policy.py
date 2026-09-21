#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import sys
import unittest
import base64
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.recovery_policy import (
    GRASP,
    NON_PREHENSILE,
    RecoveryContext,
    choose_recovery_strategy,
    draw_solid_red_star,
    recovery_decision_from_model_text,
)
from novaplan.llm_client import VLMAdapter


class RecoveryPolicyTest(unittest.TestCase):
    def test_model_text_non_prehensile_maps_to_hand_flow(self):
        context = RecoveryContext(goal="align the block", object_prompt="yellow block")
        decision = recovery_decision_from_model_text(
            r"""
            {"mode": "push", "object_prompt": "yellow block",
             "contact_prompt": "right edge of yellow block",
             "recovery_prompt": "CONTACT FINGER: index\nnudge the yellow block left",
             "contact_point_2d": [120, 80], "reason": "small correction",
             "confidence": 0.8}
            """,
            context,
        )

        self.assertEqual(decision.mode, NON_PREHENSILE)
        self.assertTrue(decision.should_run_hand_flow)
        self.assertEqual(decision.contact_point_2d, [120.0, 80.0])
        self.assertEqual(decision.recovery_mode, decision.mode)
        self.assertEqual(decision.to_dict()["recovery_action"], "nudge the yellow block left")

    def test_paper_schema_non_prehensile_keeps_annotation_and_prompt(self):
        context = RecoveryContext(goal="seat the purple block", object_prompt="purple block")
        decision = recovery_decision_from_model_text(
            r"""
            {
              "recovery_mode": "non_prehensile",
              "recovery_action": "Poke the lower-right side of the purple block once.",
              "mode_justification": {
                "discrepancy_summary": "The block is slightly tilted.",
                "why_this_mode": "A single poke can seat it."
              },
              "annotation": {
                "enabled": true,
                "object_name": "purple block",
                "contact_point_definition": "lower right side wall; PIXEL: [120, 80]",
                "edit_spec": {
                  "marker": "solid_red_star",
                  "anchor": "center_of_star_is_contact_point",
                  "placement": "lower right side wall"
                },
                "annotated_image_png_base64": "abcd"
              },
              "prompt_P": {
                "enabled": true,
                "title": "Index-finger poke",
                "text": "CONTACT FINGER: index\nUse START as the first frame and GOAL as the last frame."
              }
            }
            """,
            context,
        )

        paper = decision.to_paper_dict()
        self.assertEqual(paper["recovery_mode"], NON_PREHENSILE)
        self.assertEqual(
            paper["recovery_action"],
            "Poke the lower-right side of the purple block once.",
        )
        self.assertTrue(paper["annotation"]["enabled"])
        self.assertEqual(paper["annotation"]["edit_spec"]["marker"], "solid_red_star")
        self.assertEqual(paper["prompt_P"]["title"], "Index-finger poke")

    def test_contact_point_can_generate_red_star_annotation_fallback(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "current.png"
            Image.new("RGB", (64, 64), "white").save(image_path)
            context = RecoveryContext(
                goal="align the block",
                object_prompt="block",
                current_image_path=str(image_path),
            )
            decision = recovery_decision_from_model_text(
                r"""
                {"mode": "non_prehensile", "object_prompt": "block",
                 "contact_prompt": "right edge", "contact_point_2d": [32, 32],
                 "recovery_prompt": "CONTACT FINGER: index\npoke once", "reason": "small correction"}
                """,
                context,
            )

            self.assertTrue(decision.annotated_image_png_base64)
            self.assertTrue(decision.to_paper_dict()["annotation"]["enabled"])

    def test_contact_pixel_can_be_recovered_from_red_star_annotation(self):
        with TemporaryDirectory() as tmp:
            original_path = Path(tmp) / "current.png"
            original = Image.new("RGB", (64, 64), "white")
            ImageDraw.Draw(original).rectangle((45, 5, 62, 22), fill=(255, 0, 0))
            original.save(original_path)
            annotated = draw_solid_red_star(original, [20, 30], radius=8)
            buffer = io.BytesIO()
            annotated.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            context = RecoveryContext(
                goal="align the block",
                object_prompt="block",
                current_image_path=str(original_path),
            )
            response = {
                "recovery_mode": "non_prehensile",
                "recovery_action": "Poke the right face of the block once.",
                "annotation": {
                    "enabled": True,
                    "object_name": "block",
                    "contact_point_definition": "right face",
                    "annotated_image_png_base64": encoded,
                },
                "prompt_P": {
                    "enabled": True,
                    "title": "single poke",
                    "text": "CONTACT FINGER: index\nACTION DESCRIPTION: poke once",
                },
            }
            import json

            decision = recovery_decision_from_model_text(json.dumps(response), context)
            self.assertAlmostEqual(decision.contact_point_2d[0], 20.0, delta=1.0)
            self.assertAlmostEqual(decision.contact_point_2d[1], 30.0, delta=1.0)

    def test_existing_red_source_object_is_not_mistaken_for_annotation(self):
        with TemporaryDirectory() as tmp:
            original_path = Path(tmp) / "current.png"
            original = Image.new("RGB", (64, 64), "white")
            ImageDraw.Draw(original).rectangle((12, 12, 30, 30), fill=(255, 0, 0))
            original.save(original_path)
            buffer = io.BytesIO()
            original.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            context = RecoveryContext(
                goal="align the red block",
                object_prompt="red block",
                current_image_path=str(original_path),
            )
            response = {
                "recovery_mode": "non_prehensile",
                "recovery_action": "Poke the right face of the red block once.",
                "annotation": {
                    "enabled": True,
                    "object_name": "red block",
                    "contact_point_definition": "right face",
                    "annotated_image_png_base64": encoded,
                },
                "prompt_P": {
                    "enabled": True,
                    "title": "single poke",
                    "text": "CONTACT FINGER: index\nACTION DESCRIPTION: poke once",
                },
            }
            import json

            with self.assertRaisesRegex(ValueError, "usable contact pixel"):
                recovery_decision_from_model_text(json.dumps(response), context)

    def test_heuristic_defaults_ambiguous_failures_to_grasp(self):
        context = RecoveryContext(
            goal="put the mug on the coaster",
            previous_action="pick up the mug",
            failure_reason="the object slipped and was dropped",
            object_prompt="mug",
        )
        decision = choose_recovery_strategy(context)

        self.assertEqual(decision.mode, GRASP)
        self.assertTrue(decision.use_standard_grasp_pipeline)
        self.assertFalse(decision.to_paper_dict()["prompt_P"]["enabled"])
        self.assertIsNone(decision.to_paper_dict()["prompt_P"]["text"])
        self.assertIn("regrasp", decision.recovery_action.lower())

    def test_heuristic_refuses_ungrounded_non_prehensile_command(self):
        context = RecoveryContext(
            goal="align the block with the target",
            previous_action="slide the block",
            failure_reason="needs a small correction and slight nudge",
            object_prompt="block",
        )
        decision = choose_recovery_strategy(context)

        self.assertEqual(decision.mode, GRASP)
        self.assertTrue(decision.use_standard_grasp_pipeline)
        self.assertIn("cannot ground", decision.reason)

    def test_vlm_recovery_hook_parses_model_json_without_live_api(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"mode":"non_prehensile","object_prompt":"block","contact_point_2d":[10,20],"recovery_prompt":"CONTACT FINGER: index\\npoke the block once","reason":"small correction"}'
                    )
                )
            ]
        )
        adapter._model_for_task = lambda task=None: "unit-test-model"
        adapter._reasoning_effort_for_task = lambda task=None: "low"
        context = RecoveryContext(
            goal="align the block",
            previous_action="slide the block",
            failure_reason="slightly short",
            object_prompt="block",
        )

        decision = VLMAdapter.decide_recovery(adapter, context, fallback_to_heuristic=False)

        self.assertEqual(decision.mode, NON_PREHENSILE)
        self.assertEqual(decision.metadata["model"], "unit-test-model")
        self.assertEqual(decision.metadata["reasoning_effort"], "low")

    def test_non_prehensile_rejects_missing_or_ambiguous_contact_finger(self):
        context = RecoveryContext(goal="align block", object_prompt="block")
        missing_prompt = (
            '{"mode":"non_prehensile","recovery_action":"poke once",'
            '"contact_point_2d":[10,20]}'
        )
        with self.assertRaisesRegex(ValueError, "explicit prompt_P.text"):
            recovery_decision_from_model_text(missing_prompt, context)

        missing = (
            '{"mode":"non_prehensile","recovery_action":"poke once",'
            '"contact_point_2d":[10,20],'
            '"recovery_prompt":"poke once"}'
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            recovery_decision_from_model_text(missing, context)

        ambiguous = (
            '{"mode":"non_prehensile","recovery_action":"poke once",'
            '"contact_point_2d":[10,20],'
            '"recovery_prompt":"CONTACT FINGER: index\\nUse the thumb too"}'
        )
        with self.assertRaisesRegex(ValueError, "multiple"):
            recovery_decision_from_model_text(ambiguous, context)

    def test_grasp_model_response_forces_paper_null_annotation_and_prompt(self):
        context = RecoveryContext(
            goal="place mug",
            previous_action="Grasp and place the mug",
            object_prompt="mug",
        )
        decision = recovery_decision_from_model_text(
            """
            {
              "recovery_mode": "grasp",
              "recovery_action": "Regrasp the mug and place it on the coaster.",
              "mode_justification": {
                "discrepancy_summary": "mug dropped",
                "why_this_mode": "stable control is needed"
              },
              "annotation": {"enabled": true, "object_name": "mug"},
              "prompt_P": {"enabled": true, "title": "wrong", "text": "wrong"}
            }
            """,
            context,
        )
        paper = decision.to_paper_dict()
        self.assertFalse(paper["annotation"]["enabled"])
        self.assertIsNone(paper["annotation"]["edit_spec"])
        self.assertEqual(paper["prompt_P"], {"enabled": False, "title": None, "text": None})
        self.assertEqual(decision.recovery_prompt, "")
        self.assertEqual(
            decision.recovery_action,
            "Regrasp the mug and place it on the coaster.",
        )

    def test_current_schema_rejects_missing_recovery_action_instead_of_reusing_failed_action(self):
        context = RecoveryContext(
            goal="place mug",
            previous_action="Place the mug on the coaster.",
            object_prompt="mug",
        )
        response = """
        {
          "recovery_mode": "grasp",
          "mode_justification": {
            "discrepancy_summary": "mug dropped",
            "why_this_mode": "stable control is needed"
          },
          "annotation": {"enabled": false},
          "prompt_P": {"enabled": false, "title": null, "text": null}
        }
        """

        with self.assertRaisesRegex(ValueError, "non-empty recovery_action"):
            recovery_decision_from_model_text(response, context)

    def test_current_schema_rejects_non_string_recovery_action(self):
        context = RecoveryContext(goal="place mug", object_prompt="mug")
        response = """
        {
          "recovery_mode": "grasp",
          "recovery_action": {"verb": "regrasp"},
          "mode_justification": {
            "discrepancy_summary": "mug dropped",
            "why_this_mode": "stable control is needed"
          },
          "annotation": {"enabled": false},
          "prompt_P": {"enabled": false, "title": null, "text": null}
        }
        """

        with self.assertRaisesRegex(ValueError, "recovery_action must be a string"):
            recovery_decision_from_model_text(response, context)

    def test_vlm_recovery_hook_can_fall_back_to_heuristic(self):
        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._openai_chat_completion = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
        context = RecoveryContext(
            goal="put the mug on the coaster",
            previous_action="pick up the mug",
            failure_reason="object slipped and was dropped",
            object_prompt="mug",
        )

        decision = VLMAdapter.decide_recovery(adapter, context, fallback_to_heuristic=True)

        self.assertEqual(decision.mode, GRASP)
        self.assertTrue(decision.use_standard_grasp_pipeline)


if __name__ == "__main__":
    unittest.main()
