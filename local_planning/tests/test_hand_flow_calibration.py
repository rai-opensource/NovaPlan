#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.hand_flow.calibration import (  # noqa: E402
    _nearest_mask_pixel,
    calibrate_mesh_sequence_to_depth,
    calibrate_mesh_sequence_to_object_contact,
    calibrate_mesh_sequence_to_prompted_contact,
    canonical_contact_finger,
    hand_pose_from_semantic,
    interaction_interval_from_object_masks,
    palm_frame_from_semantic,
    scale_pixel_between_resolutions,
    semantic_keypoints,
    transform_semantic_xyz,
)


class HandFlowCalibrationTest(unittest.TestCase):
    def test_nearest_mask_pixel_enforces_circular_projection_radius(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[30, 30] = True
        self.assertIsNone(_nearest_mask_pixel(mask, 15, 15, radius=15))

        mask[24, 24] = True
        nearest = _nearest_mask_pixel(mask, 15, 15, radius=15)
        self.assertEqual(nearest[:2], (24, 24))
        self.assertAlmostEqual(nearest[2], np.sqrt(162.0))

    def test_contact_pixel_scales_from_video_to_tapip_resolution(self):
        mapped = scale_pixel_between_resolutions([640, 360], [720, 1280], [543, 724])
        np.testing.assert_allclose(mapped, [362.0, 271.5], atol=1e-9)

    def test_interaction_interval_uses_new_support_outside_initial_mask(self):
        masks = np.zeros((4, 5, 5), dtype=np.float32)
        masks[0, :2, :2] = 1.0
        masks[1, :2, :2] = 1.0
        masks[1, 3, 3] = 1.0
        masks[2, :2, :2] = 1.0
        masks[2, 2, :3] = 1.0
        masks[3, 2, :4] = 1.0
        # Frames 2 and 3 add at least 75% of the initial-mask area outside M_t0.
        self.assertEqual(
            interaction_interval_from_object_masks(masks, epsilon=0.75),
            (2, 3),
        )

    def test_interaction_interval_does_not_treat_occlusion_as_object_motion(self):
        masks = np.zeros((3, 4, 4), dtype=np.float32)
        masks[0, :3, :3] = 1.0
        masks[1, :2, :2] = 1.0
        masks[2, :1, :1] = 1.0
        with self.assertRaisesRegex(ValueError, "never reached"):
            interaction_interval_from_object_masks(masks, epsilon=0.5)

    def test_interaction_interval_rejects_unreached_threshold(self):
        masks = np.ones((3, 4, 4), dtype=np.float32)
        with self.assertRaisesRegex(
            ValueError,
            r"max_new_support_ratio=0\.000 < epsilon=0\.900",
        ):
            interaction_interval_from_object_masks(masks, epsilon=0.9)

    def test_interaction_interval_rejection_reports_observed_ratio(self):
        masks = np.zeros((2, 4, 4), dtype=np.float32)
        masks[0, :2, :2] = 1.0
        masks[1, :2, 1:3] = 1.0

        with self.assertRaisesRegex(
            ValueError,
            r"max_new_support_ratio=0\.500 < epsilon=0\.900",
        ):
            interaction_interval_from_object_masks(masks, epsilon=0.9)

    def test_depth_projection_recovers_isotropic_scale(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        verts = np.array(
            [
                [-0.20, -0.20, 1.0],
                [0.20, -0.20, 1.0],
                [-0.20, 0.20, 1.0],
                [0.20, 0.20, 1.0],
                [0.0, 0.0, 1.1],
                [0.10, 0.0, 1.05],
                [0.0, 0.10, 1.05],
                [-0.10, 0.0, 1.05],
                [0.0, -0.10, 1.05],
            ],
            dtype=np.float64,
        )
        scale = 1.4
        depth = np.zeros((100, 100), dtype=np.float64)
        uv = np.rint(np.column_stack([intr[0, 0] * verts[:, 0] / verts[:, 2] + intr[0, 2], intr[1, 1] * verts[:, 1] / verts[:, 2] + intr[1, 2]])).astype(int)
        for (u, v), z in zip(uv, verts[:, 2]):
            depth[v, u] = z * scale

        result = calibrate_mesh_sequence_to_depth(
            verts[None],
            np.array([0]),
            depth[None],
            intr[None],
            min_points=5,
            min_projection_bbox_px=10.0,
        )

        self.assertEqual(result.status[0], "ok")
        self.assertAlmostEqual(float(result.scales[0]), scale, places=6)
        np.testing.assert_allclose(result.vertices[0], verts * scale, atol=1e-6)

    def test_semantic_pose_uses_calibrated_contact_fingertip(self):
        keypoints = np.zeros((21, 3), dtype=np.float64)
        keypoints[0] = [0.0, 0.0, 0.0]
        keypoints[5] = [-0.05, 0.08, 0.0]
        keypoints[8] = [0.10, 0.20, 0.30]
        keypoints[9] = [0.0, 0.10, 0.0]
        keypoints[17] = [0.05, 0.08, 0.0]
        semantic = {
            "fingers": {"index": {"tip": keypoints[8].tolist()}},
            "keypoints_3d": keypoints.tolist(),
        }

        transformed = transform_semantic_xyz(semantic, 2.0, np.array([1.0, 0.0, -1.0]))
        pose = hand_pose_from_semantic(transformed, np.zeros((4, 3)), contact_finger="index")

        np.testing.assert_allclose(pose[:3, 3], np.array([1.2, 0.4, -0.4]), atol=1e-6)
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)

    def test_nested_semantic_schema_builds_canonical_palm_pose(self):
        semantic = {
            "wrist": [0.0, 0.0, 0.0],
            "fingers": {
                "thumb": {"mcp": [-0.04, 0.03, 0.01]},
                "index": {
                    "mcp": [-0.035, 0.08, 0.005],
                    "tip": [-0.04, 0.15, 0.02],
                },
                "middle": {"mcp": [0.0, 0.10, 0.0]},
                "ring": {"mcp": [0.03, 0.08, -0.004]},
                "pinky": {"mcp": [0.05, 0.05, -0.008]},
            },
        }

        keypoints = semantic_keypoints(semantic)
        self.assertEqual(keypoints.shape, (21, 3))
        np.testing.assert_allclose(keypoints[0], semantic["wrist"])
        np.testing.assert_allclose(keypoints[5], semantic["fingers"]["index"]["mcp"])
        np.testing.assert_allclose(keypoints[8], semantic["fingers"]["index"]["tip"])
        self.assertTrue(np.isnan(keypoints[2]).all())

        rotation = palm_frame_from_semantic(semantic)
        self.assertIsNotNone(rotation)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

        pose = hand_pose_from_semantic(
            semantic,
            np.zeros((4, 3), dtype=np.float64),
            contact_finger="index",
        )
        np.testing.assert_allclose(pose[:3, :3], rotation, atol=1e-12)
        np.testing.assert_allclose(pose[:3, 3], semantic["fingers"]["index"]["tip"])

    def test_nested_semantic_missing_or_nonfinite_tips_preserve_vertex_fallback(self):
        vertices = np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            dtype=np.float64,
        )

        for tip in (None, [np.nan, np.nan, np.nan]):
            index = {"mcp": [-0.04, 0.08, 0.0]}
            if tip is not None:
                index["tip"] = tip
            semantic = {
                "wrist": [0.0, 0.0, 0.0],
                "fingers": {
                    "index": index,
                    "middle": {"mcp": [0.0, 0.10, 0.0]},
                    "pinky": {"mcp": [0.05, 0.05, 0.0]},
                },
            }

            keypoints = semantic_keypoints(semantic)
            self.assertTrue(np.isnan(keypoints[8]).all())
            pose = hand_pose_from_semantic(
                semantic,
                vertices,
                contact_finger="index",
            )

            np.testing.assert_allclose(pose[:3, 3], np.median(vertices, axis=0))

    def test_nonfinite_nested_tip_uses_valid_array_tip(self):
        keypoints = np.zeros((21, 3), dtype=np.float64)
        keypoints[8] = [0.1, 0.2, 0.3]
        semantic = {
            "keypoints_3d": keypoints.tolist(),
            "fingers": {"index": {"tip": [np.nan, np.nan, np.nan]}},
        }

        pose = hand_pose_from_semantic(
            semantic,
            np.ones((2, 3), dtype=np.float64),
            contact_finger="index",
        )

        np.testing.assert_allclose(pose[:3, 3], keypoints[8])

    def test_array_semantic_schema_keeps_precedence_over_nested_schema(self):
        expected = np.arange(21 * 3, dtype=np.float64).reshape(21, 3)
        semantic = {
            "keypoints_3d": expected.tolist(),
            "wrist": [-1.0, -1.0, -1.0],
            "fingers": {"index": {"mcp": [-2.0, -2.0, -2.0]}},
        }

        np.testing.assert_array_equal(semantic_keypoints(semantic), expected)

    def test_object_contact_calibration_recovers_scale_and_translation(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        vertices = np.array([[[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.01, 0.0, 1.0]]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        depth = np.zeros((1, 100, 100), dtype=np.float64)
        depth[0, 50, 52] = 2.0
        mask = np.zeros((1, 100, 100), dtype=np.float32)
        mask[0, 50, 52] = 1.0
        semantic = [
            {
                "fingers": {
                    "index": {
                        "mcp": [0.0, 0.0, 0.8],
                        "tip": [0.0, 0.0, 1.0],
                    }
                }
            }
        ]

        result = calibrate_mesh_sequence_to_object_contact(
            vertices,
            faces,
            semantic,
            np.array([0]),
            depth,
            intr[None],
            mask,
            contact_finger="index",
            projection_radius=3,
        )

        self.assertEqual(result.summary["method"], "object_contact_scale_translation")
        self.assertAlmostEqual(float(result.scales[0]), 2.0, places=6)
        np.testing.assert_allclose(result.offsets[0], np.array([0.04, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(result.vertices[0, 0], np.array([0.04, 0.0, 2.0]), atol=1e-6)

    def test_palm_frame_fits_plane_using_all_mcp_joints(self):
        keypoints = np.zeros((21, 3), dtype=np.float64)
        keypoints[0] = [0.0, 0.0, 0.0]
        keypoints[1] = [-1.0, 0.0, 0.0]
        keypoints[5] = [-1.0, 1.0, 0.2]
        keypoints[9] = [0.0, 1.0, 0.0]
        keypoints[13] = [1.0, 0.0, 0.0]
        keypoints[17] = [1.0, -0.8, -0.1]

        rotation = palm_frame_from_semantic({"keypoints_3d": keypoints.tolist()})

        self.assertIsNotNone(rotation)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        normal = rotation[:, 2]
        middle_direction = keypoints[9] - keypoints[0]
        expected_x = middle_direction - np.dot(middle_direction, normal) * normal
        expected_x /= np.linalg.norm(expected_x)
        np.testing.assert_allclose(rotation[:, 0], expected_x, atol=1e-9)
        anatomical_normal = np.cross(
            keypoints[5] - keypoints[0],
            keypoints[17] - keypoints[0],
        )
        self.assertGreater(float(np.dot(normal, anatomical_normal)), 0.0)
        self.assertGreater(abs(float(normal[2])), 0.99)

    def test_strict_contact_rejects_failed_onset_instead_of_using_release_anchor(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        base_vertices = np.array([[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.01, 0.0, 1.0]])
        vertices = np.stack([base_vertices, base_vertices], axis=0)
        depths = np.zeros((2, 100, 100), dtype=np.float64)
        masks = np.zeros((2, 100, 100), dtype=np.float32)
        depths[1, 50, 52] = 2.0
        masks[1, 50, 52] = 1.0
        semantics = [
            {"fingers": {"index": {"mcp": [0.0, 0.0, 0.8], "tip": [0.0, 0.0, 1.0]}}},
            {"fingers": {"index": {"mcp": [0.0, 0.0, 0.8], "tip": [0.0, 0.0, 1.0]}}},
        ]

        with self.assertRaisesRegex(ValueError, "start=empty_object_mask"):
            calibrate_mesh_sequence_to_object_contact(
                vertices,
                np.array([[0, 1, 2]], dtype=np.int32),
                semantics,
                np.array([0, 1]),
                depths,
                intr[None],
                masks,
                movement_start_frame=0,
                movement_end_frame=1,
                contact_finger="index",
                projection_radius=3,
                strict_contact=True,
            )

    def test_strict_contact_resolves_release_before_hand_withdrawal(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        contact_vertices = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.01, 0.0, 1.0]]
        )
        withdrawn_vertices = contact_vertices + np.array([0.30, 0.0, 0.0])
        vertices = np.stack(
            [contact_vertices, contact_vertices, contact_vertices, withdrawn_vertices],
            axis=0,
        )
        depths = np.zeros((4, 100, 100), dtype=np.float64)
        masks = np.zeros((4, 100, 100), dtype=np.float32)
        depths[:, 50, 52] = 2.0
        masks[:, 50, 52] = 1.0
        semantics = [
            {"fingers": {"index": {"mcp": [x, 0.0, 0.8], "tip": [x, 0.0, 1.0]}}}
            for x in (0.0, 0.0, 0.0, 0.30)
        ]

        result = calibrate_mesh_sequence_to_object_contact(
            vertices,
            np.array([[0, 1, 2]], dtype=np.int32),
            semantics,
            np.arange(4),
            depths,
            intr[None],
            masks,
            movement_start_frame=0,
            movement_end_frame=3,
            contact_finger="index",
            projection_radius=3,
            strict_contact=True,
        )

        summary = result.summary
        self.assertEqual(summary["movement_end_frame"], 2)
        self.assertEqual(summary["requested_movement_end_frame"], 3)
        self.assertEqual(summary["requested_movement_end_hand_frame"], 3)
        self.assertEqual(summary["release_contact_frame"], 2)
        self.assertTrue(summary["release_contact_search_used"])
        self.assertEqual(summary["release_contact_search_attempts"], 1)
        self.assertEqual(
            summary["contact_calibration_end_requested"]["status"],
            "no_fingertip_object_contact",
        )
        self.assertEqual(summary["contact_calibration_end"]["status"], "ok")

    def test_strict_contact_rejects_when_only_onset_has_contact(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        contact_vertices = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.01, 0.0, 1.0]]
        )
        withdrawn_vertices = contact_vertices + np.array([0.30, 0.0, 0.0])
        vertices = np.stack(
            [contact_vertices, withdrawn_vertices, withdrawn_vertices],
            axis=0,
        )
        depths = np.zeros((3, 100, 100), dtype=np.float64)
        masks = np.zeros((3, 100, 100), dtype=np.float32)
        depths[:, 50, 52] = 2.0
        masks[:, 50, 52] = 1.0
        semantics = [
            {"fingers": {"index": {"mcp": [x, 0.0, 0.8], "tip": [x, 0.0, 1.0]}}}
            for x in (0.0, 0.30, 0.30)
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"start=ok end=no_fingertip_object_contact; .*release_search_attempts=1",
        ):
            calibrate_mesh_sequence_to_object_contact(
                vertices,
                np.array([[0, 1, 2]], dtype=np.int32),
                semantics,
                np.arange(3),
                depths,
                intr[None],
                masks,
                movement_start_frame=0,
                movement_end_frame=2,
                contact_finger="index",
                projection_radius=3,
                strict_contact=True,
            )

    def test_contact_interval_maps_sparse_hamer_frames_inward(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        contact_vertices = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.01, 0.0, 1.0]]
        )
        withdrawn_vertices = contact_vertices + np.array([0.30, 0.0, 0.0])
        vertices = np.stack(
            [withdrawn_vertices, contact_vertices, contact_vertices, withdrawn_vertices],
            axis=0,
        )
        frame_indices = np.array([0, 4, 6, 8])
        depths = np.zeros((9, 100, 100), dtype=np.float64)
        masks = np.zeros((9, 100, 100), dtype=np.float32)
        depths[:, 50, 52] = 2.0
        masks[:, 50, 52] = 1.0
        semantics = [
            {"fingers": {"index": {"mcp": [x, 0.0, 0.8], "tip": [x, 0.0, 1.0]}}}
            for x in (0.30, 0.0, 0.0, 0.30)
        ]

        result = calibrate_mesh_sequence_to_object_contact(
            vertices,
            np.array([[0, 1, 2]], dtype=np.int32),
            semantics,
            frame_indices,
            depths,
            intr[None],
            masks,
            movement_start_frame=3,
            movement_end_frame=7,
            contact_finger="index",
            projection_radius=3,
            strict_contact=True,
        )

        summary = result.summary
        self.assertEqual(summary["movement_start_frame"], 4)
        self.assertEqual(summary["movement_end_frame"], 6)
        self.assertEqual(summary["requested_movement_end_frame"], 7)
        self.assertEqual(summary["requested_movement_end_hand_frame"], 6)
        self.assertFalse(summary["release_contact_search_used"])

    def test_release_search_does_not_switch_to_another_finger(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        contact_vertices = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.01, 1.0], [0.01, 0.0, 1.0]]
        )
        away_vertices = contact_vertices + np.array([0.30, 0.0, 0.0])
        vertices = np.repeat(
            np.concatenate([contact_vertices, away_vertices], axis=0)[None],
            4,
            axis=0,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        depths = np.zeros((4, 100, 100), dtype=np.float64)
        masks = np.zeros((4, 100, 100), dtype=np.float32)
        depths[:, 50, 52] = 2.0
        masks[:, 50, 52] = 1.0
        semantics = [
            {
                "fingers": {
                    "index": {"mcp": [index_x, 0.0, 0.8], "tip": [index_x, 0.0, 1.0]},
                    "middle": {"mcp": [middle_x, 0.0, 0.8], "tip": [middle_x, 0.0, 1.0]},
                }
            }
            for index_x, middle_x in (
                (0.0, 0.30),
                (0.30, 0.30),
                (0.30, 0.0),
                (0.30, 0.30),
            )
        ]

        with self.assertRaisesRegex(ValueError, r"release_search_attempts=2"):
            calibrate_mesh_sequence_to_object_contact(
                vertices,
                faces,
                semantics,
                np.arange(4),
                depths,
                intr[None],
                masks,
                movement_start_frame=0,
                movement_end_frame=3,
                contact_finger="index",
                projection_radius=3,
                strict_contact=True,
            )

    def test_prompted_contact_anchors_requested_fingertip_and_release_drift(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        base_vertices = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]])
        vertices = np.stack([base_vertices, base_vertices, base_vertices], axis=0)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        semantics = []
        for x in (0.0, 0.01, 0.02):
            semantics.append({"fingers": {"index": {"tip": [x, 0.0, 1.0]}}})
        depths = np.zeros((3, 100, 100), dtype=np.float64)
        masks = np.zeros((3, 100, 100), dtype=np.float32)
        for frame, u in enumerate((52, 53, 55)):
            depths[frame, 50, u] = 2.0
            masks[frame, 50, u] = 1.0

        result = calibrate_mesh_sequence_to_prompted_contact(
            vertices,
            faces,
            semantics,
            np.array([0, 1, 2]),
            depths,
            intr[None],
            masks,
            contact_finger="index_tip",
            contact_point_2d=[52, 50],
            movement_start_frame=0,
            movement_end_frame=2,
            projection_radius=4,
            release_delta_m=0.02,
        )

        self.assertEqual(result.summary["method"], "prompted_fingertip_contact")
        self.assertEqual(result.summary["contact_finger"], "index")
        np.testing.assert_allclose(result.offsets[0], [0.04, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result.offsets[-1], [0.06, 0.0, 0.0], atol=1e-6)

    def test_prompted_contact_resolves_release_before_withdrawal(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        base_vertices = np.array(
            [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0]]
        )
        vertices = np.repeat(base_vertices[None], 4, axis=0)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        semantics = [
            {"fingers": {"index": {"tip": [x, 0.0, 1.0]}}}
            for x in (0.0, 0.01, 0.02, 0.30)
        ]
        depths = np.zeros((4, 100, 100), dtype=np.float64)
        masks = np.zeros((4, 100, 100), dtype=np.float32)
        for frame, u in enumerate((52, 53, 55, 55)):
            depths[frame, 50, u] = 2.0
            masks[frame, 50, u] = 1.0

        result = calibrate_mesh_sequence_to_prompted_contact(
            vertices,
            faces,
            semantics,
            np.arange(4),
            depths,
            intr[None],
            masks,
            contact_finger="index",
            contact_point_2d=[52, 50],
            movement_start_frame=0,
            movement_end_frame=3,
            projection_radius=4,
            release_delta_m=0.02,
        )

        summary = result.summary
        self.assertEqual(summary["movement_end_frame"], 2)
        self.assertEqual(summary["requested_movement_end_frame"], 3)
        self.assertEqual(summary["release_contact_frame"], 2)
        self.assertTrue(summary["release_contact_search_used"])
        self.assertEqual(summary["release_contact_search_attempts"], 1)
        self.assertEqual(
            summary["release_contact_requested_status"],
            "no_metric_object_surface_contact",
        )

    def test_prompted_contact_rejects_ambiguous_finger(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            canonical_contact_finger("fingertip")

    def test_prompted_contact_rejects_hand_mostly_outside_frame(self):
        intr = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        vertices = np.array([[[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [11.0, 0.0, 1.0]]])
        depths = np.zeros((1, 100, 100), dtype=np.float64)
        masks = np.zeros((1, 100, 100), dtype=np.float32)
        depths[0, 50, 50] = 1.0
        masks[0, 50, 50] = 1.0
        semantics = [{"fingers": {"index": {"tip": [0.0, 0.0, 1.0]}}}]
        with self.assertRaisesRegex(ValueError, "leaves the image"):
            calibrate_mesh_sequence_to_prompted_contact(
                vertices,
                np.array([[0, 1, 2]], dtype=np.int32),
                semantics,
                np.array([0]),
                depths,
                intr[None],
                masks,
                contact_finger="index",
                contact_point_2d=[50, 50],
                max_outside_fraction=0.30,
            )


if __name__ == "__main__":
    unittest.main()
