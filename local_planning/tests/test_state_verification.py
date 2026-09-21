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
from unittest.mock import Mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.closed_loop_execution import _verify_step  # noqa: E402
from novaplan.llm_client import VLMAdapter  # noqa: E402


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _decode_data_url(url: str) -> np.ndarray:
    encoded = url.split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


class StateVerificationTest(unittest.TestCase):
    def _adapter(self, content: str) -> tuple[VLMAdapter, dict]:
        adapter = VLMAdapter.__new__(VLMAdapter)
        captured: dict = {}

        def call(**kwargs):
            captured.update(kwargs)
            return _response(content)

        adapter._openai_chat_completion = call
        adapter._model_for_task = lambda task=None: "unit-test-verifier"
        adapter._reasoning_effort_for_task = lambda task=None: "low"
        return adapter, captured

    def test_three_image_verification_preserves_start_current_target_order(self):
        adapter, captured = self._adapter(
            '{"success": true, "reason": "the observed transition matches the target"}'
        )
        start = np.full((8, 10, 3), 10, dtype=np.uint8)
        current = np.full((8, 10, 3), 20, dtype=np.uint8)
        target = np.full((8, 10, 3), 30, dtype=np.uint8)

        result = VLMAdapter.verify_transition(
            adapter,
            start_image=start,
            current_image=current,
            target_image=target,
            action="Place the block in the bin.",
            goal="Sort the block.",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "unit-test-verifier")
        content = captured["messages"][0]["content"]
        self.assertIn("Place the block in the bin.", content[0]["text"])
        image_values = [int(_decode_data_url(item["image_url"]["url"])[0, 0, 0]) for item in content[1:]]
        self.assertEqual(image_values, [10, 20, 30])

    def test_invalid_verification_schema_fails_closed(self):
        adapter, _ = self._adapter(
            '{"success": "yes", "reason": "not a valid boolean"}'
        )
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        result = VLMAdapter.verify_transition(
            adapter,
            start_image=frame,
            current_image=frame,
            target_image=frame,
            action="Move the block.",
            goal="Place the block.",
        )

        self.assertFalse(result["success"])
        self.assertIn("must be a boolean", result["reason"])

    def test_closed_loop_verification_records_three_image_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / name for name in ("start.png", "current.png", "target.png")]
            for index, path in enumerate(paths):
                Image.fromarray(np.full((5, 6, 3), index * 20, dtype=np.uint8)).save(path)
            vlm = SimpleNamespace(
                verify_transition=Mock(return_value={"success": True, "reason": "complete"})
            )

            result = _verify_step(
                vlm=vlm,
                start_path=paths[0],
                post_path=paths[1],
                target_path=paths[2],
                action="Place the block.",
                goal="Sort the block.",
                disabled=False,
            )

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["success"])
            self.assertEqual(result["start_image"], str(paths[0]))
            self.assertEqual(result["post_image"], str(paths[1]))
            self.assertEqual(result["target_image"], str(paths[2]))
            vlm.verify_transition.assert_called_once()


if __name__ == "__main__":
    unittest.main()
