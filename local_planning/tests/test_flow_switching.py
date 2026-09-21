#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.flow_switching import (
    HandFlowCandidate,
    evaluate_object_flow,
    select_flow_reference,
)


def _rot_z(deg: float) -> np.ndarray:
    theta = math.radians(deg)
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _points() -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, -0.25, 0.75],
        ],
        dtype=np.float64,
    )


class FlowSwitchingTest(unittest.TestCase):
    def test_object_flow_kept_for_smooth_rotation(self):
        p0 = _points()
        p1 = ( _rot_z(20.0) @ p0.T).T
        evaluation = evaluate_object_flow(np.stack([p0, p1], axis=0), flow_switch_theta_deg=45.0)
        decision = select_flow_reference(evaluation)
        self.assertTrue(evaluation.valid)
        self.assertFalse(evaluation.should_switch_to_hand)
        self.assertEqual(decision.selected_flow, "object")
        self.assertFalse(decision.to_dict()["switch_to_hand"])
        self.assertAlmostEqual(evaluation.max_rotation_deg, 20.0, places=5)

    def test_hand_flow_selected_for_large_rotation_when_available(self):
        p0 = _points()
        p1 = (_rot_z(60.0) @ p0.T).T
        evaluation = evaluate_object_flow(np.stack([p0, p1], axis=0), flow_switch_theta_deg=45.0)
        decision = select_flow_reference(evaluation, HandFlowCandidate(valid=True))
        self.assertTrue(evaluation.valid)
        self.assertTrue(evaluation.should_switch_to_hand)
        self.assertEqual(decision.selected_flow, "hand")
        self.assertTrue(decision.to_dict()["switch_to_hand"])

    def test_hand_flow_selected_for_invalid_object_fallback_when_available(self):
        p0 = _points()
        p1 = p0.copy()
        vis = np.zeros((2, p0.shape[0]), dtype=bool)
        evaluation = evaluate_object_flow(
            np.stack([p0, p1], axis=0),
            visibilities=vis,
            min_visible_points=3,
        )
        decision = select_flow_reference(evaluation, HandFlowCandidate(valid=True))
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.should_switch_to_hand)
        self.assertEqual(decision.selected_flow, "hand")
        self.assertTrue(decision.to_dict()["switch_to_hand"])

    def test_visibility_dropout_invalidates_object_flow(self):
        p0 = _points()
        p1 = p0.copy()
        vis = np.zeros((2, p0.shape[0]), dtype=bool)
        evaluation = evaluate_object_flow(
            np.stack([p0, p1], axis=0),
            visibilities=vis,
            min_visible_points=3,
        )
        decision = select_flow_reference(evaluation)
        self.assertFalse(evaluation.valid)
        self.assertEqual(decision.selected_flow, "none")


if __name__ == "__main__":
    unittest.main()
