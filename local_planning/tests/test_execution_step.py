#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.execution_step import (  # noqa: E402
    FlowTrackData,
    compile_flow_execution_step,
    load_flow_tracks,
    relative_transforms_from_poses,
    save_execution_step,
)
from novaplan.object_flow import normalize_flow_arrays  # noqa: E402
from novaplan.hand_flow.calibration import palm_frame_from_semantic  # noqa: E402
from local_planning.run_execution_step import (  # noqa: E402
    _hand_pose_with_reference_palm_orientation,
    _trim_hand_pose_track,
)
from local_planning.visualization.viser_flow_review import (  # noqa: E402
    FlowLayer,
    FlowReviewServer,
    _build_current_trails,
    _load_flow_tracks as load_review_flow_tracks,
    _rigid_tracks_from_transforms,
    _spatial_track_order,
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


def _pose(xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return pose


class ExecutionStepTest(unittest.TestCase):
    def test_point_major_flow_and_visibility_are_normalized_together(self):
        coords_tn3 = np.arange(3 * 6 * 3, dtype=np.float64).reshape(3, 6, 3)
        visibility_tn = np.arange(3 * 6, dtype=np.float64).reshape(3, 6)

        coords, visibility = normalize_flow_arrays(
            np.swapaxes(coords_tn3, 0, 1),
            visibility_tn.T,
            layout="point_major",
        )

        np.testing.assert_array_equal(coords, coords_tn3)
        np.testing.assert_array_equal(visibility, visibility_tn)
        raw_flow = FlowTrackData(
            coords=np.swapaxes(coords_tn3, 0, 1),
            visibilities=visibility_tn.T,
            layout="point_major",
        )
        self.assertEqual(raw_flow.num_frames, 3)
        self.assertEqual(raw_flow.num_points, 6)

    def test_point_major_flow_layout_is_inferred_from_frame_indices(self):
        coords_tn3 = np.arange(10 * 8 * 3, dtype=np.float64).reshape(10, 8, 3)
        visibility_tn = np.ones((10, 8), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "point_major.npz"
            np.savez(
                path,
                coords=np.swapaxes(coords_tn3, 0, 1),
                visibilities=visibility_tn.T,
                frame_indices=np.arange(10),
            )

            loaded = load_flow_tracks(path)

        np.testing.assert_array_equal(loaded.coords, coords_tn3)
        np.testing.assert_array_equal(loaded.visibilities, visibility_tn)
        self.assertEqual(loaded.layout, "time_major")

    def test_time_major_flow_with_more_frames_than_points_stays_time_major(self):
        rng = np.random.RandomState(7)
        points = rng.normal(size=(8, 3))
        translations = np.arange(10, dtype=np.float64)[:, None] * np.array([[0.01, -0.02, 0.03]])
        coords_tn3 = points[None, :, :] + translations[:, None, :]
        visibility_tn = np.ones((10, 8), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_time_major.npz"
            np.savez(path, coords=coords_tn3, visibilities=visibility_tn)

            loaded = load_flow_tracks(path)
            review_coords, review_vis, _ = load_review_flow_tracks(path)
            result = compile_flow_execution_step(
                loaded,
                selected_flow="object",
                min_visible_points=3,
            )

        np.testing.assert_array_equal(loaded.coords, coords_tn3)
        np.testing.assert_array_equal(review_coords, coords_tn3)
        np.testing.assert_array_equal(review_vis, visibility_tn)
        self.assertEqual(result.ee_poses.shape, (10, 4, 4))
        np.testing.assert_allclose(result.ee_poses[:, :3, 3], translations, atol=1e-6)

    def test_unmarked_time_major_no_visibility_stays_time_major_when_t_exceeds_n(self):
        rng = np.random.RandomState(17)
        points = rng.normal(size=(8, 3))
        translations = np.arange(10, dtype=np.float64)[:, None] * np.array([[0.01, -0.02, 0.03]])
        coords_tn3 = points[None, :, :] + translations[:, None, :]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "time_major_without_visibility.npz"
            np.savez(path, coords=coords_tn3)

            loaded = load_flow_tracks(path)
            result = compile_flow_execution_step(
                loaded,
                selected_flow="object",
                min_visible_points=3,
            )

        np.testing.assert_array_equal(loaded.coords, coords_tn3)
        self.assertEqual(result.ee_poses.shape, (10, 4, 4))
        np.testing.assert_allclose(
            result.ee_poses[:, :3, 3],
            translations,
            atol=1e-6,
        )

    def test_invalid_and_conflicting_flow_layouts_are_rejected(self):
        coords = np.zeros((10, 8, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "layout must be one of"):
            normalize_flow_arrays(coords, layout="frames_first")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conflicting_layout.npz"
            np.savez(
                path,
                coords=np.swapaxes(coords, 0, 1),
                flow_layout=np.array("time_major"),
                frame_indices=np.arange(10),
            )
            with self.assertRaisesRegex(ValueError, "Conflicting flow layout evidence"):
                load_flow_tracks(path)

    def test_object_flow_returns_fitted_object_transforms(self):
        points = _points()
        translations = [np.array([0.0, 0.0, 0.0]), np.array([0.10, 0.0, 0.0]), np.array([0.25, 0.05, 0.0])]
        coords = np.stack([points + t for t in translations], axis=0)
        vis = np.ones(coords.shape[:2], dtype=bool)

        result = compile_flow_execution_step(
            FlowTrackData(coords=coords, visibilities=vis),
            selected_flow="auto",
            min_visible_points=3,
        )

        self.assertEqual(result.selected_flow, "object")
        np.testing.assert_allclose(result.ee_poses[:, :3, 3], np.array(translations), atol=1e-6)

    def test_object_flow_ransac_rejects_geometric_outliers_without_smoothing(self):
        points = _points()
        translation = np.array([0.12, -0.03, 0.04])
        moved = points + translation
        moved[:2] += np.array([0.6, -0.4, 0.5])
        coords = np.stack([points, moved], axis=0)

        result = compile_flow_execution_step(
            FlowTrackData(coords=coords),
            selected_flow="object",
            min_visible_points=3,
        )

        np.testing.assert_allclose(result.ee_poses[1, :3, 3], translation, atol=1e-6)
        self.assertLess(result.object_motion.inlier_ratios[1], 1.0)
        self.assertFalse(result.metadata["object_motion_temporal_smoothing"])

    def test_hand_flow_selected_for_large_object_rotation(self):
        points = _points()
        coords = np.stack([points, (_rot_z(70.0) @ points.T).T], axis=0)
        vis = np.ones(coords.shape[:2], dtype=bool)
        hand_poses = np.stack([_pose((0.0, 0.0, 0.0)), _pose((0.0, 0.3, 0.0))], axis=0)

        result = compile_flow_execution_step(
            FlowTrackData(coords=coords, visibilities=vis),
            hand_poses=hand_poses,
            selected_flow="auto",
            min_visible_points=3,
        )

        self.assertEqual(result.selected_flow, "hand")
        np.testing.assert_allclose(result.ee_poses, hand_poses, atol=1e-6)

    def test_save_execution_outputs(self):
        points = _points()
        coords = np.stack([points, points + np.array([0.1, 0.0, 0.0])], axis=0)
        result = compile_flow_execution_step(
            FlowTrackData(coords=coords),
            selected_flow="object",
            min_visible_points=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_execution_step(result, Path(tmp))
            self.assertTrue(saved["relative_ee_transforms"].exists())
            self.assertTrue(saved["relative_ee_frame_indices"].exists())
            self.assertTrue(saved["execution_step"].exists())
            self.assertTrue(saved["summary"].exists())
            with np.load(saved["execution_step"], allow_pickle=False) as bundle:
                self.assertEqual(
                    set(bundle.files),
                    {
                        "relative_ee_transforms",
                        "relative_ee_frame_indices",
                        "selected_flow",
                        "object_motion_transforms",
                        "object_motion_relative_transforms",
                    },
                )
            self.assertEqual(
                {path.name for path in Path(tmp).iterdir()},
                {
                    "relative_ee_transforms.npy",
                    "relative_ee_frame_indices.npy",
                    "execution_step.npz",
                    "execution_step.json",
                },
            )

    def test_adjacent_relative_transforms_are_saved(self):
        points = _points()
        step_a = points + np.array([0.1, 0.0, 0.0])
        step_b = step_a + np.array([0.0, 0.2, 0.0])
        coords = np.stack([points, step_a, step_b], axis=0)
        result = compile_flow_execution_step(
            FlowTrackData(coords=coords),
            selected_flow="object",
            min_visible_points=3,
        )

        rel = result.object_motion.relative_transforms
        self.assertEqual(rel.shape, (3, 4, 4))
        np.testing.assert_allclose(rel[1, :3, 3], np.array([0.1, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(rel[2, :3, 3], np.array([0.0, 0.2, 0.0]), atol=1e-6)

    def test_hand_relative_transforms_preserve_source_frame_gaps(self):
        points = _points()
        coords = np.stack([points, (_rot_z(70.0) @ points.T).T], axis=0)
        hand_poses = np.stack(
            [_pose((0.0, 0.0, 0.0)), _pose((0.0, 0.2, 0.0)), _pose((0.1, 0.2, 0.0))],
            axis=0,
        )
        result = compile_flow_execution_step(
            FlowTrackData(coords=coords),
            hand_poses=hand_poses,
            hand_frame_indices=np.array([0, 2, 5]),
            selected_flow="hand",
            min_visible_points=3,
        )

        self.assertEqual(result.frame_indices.tolist(), [0, 2, 5])
        self.assertEqual(result.to_dict()["frame_gaps"], [2, 3])
        np.testing.assert_allclose(result.relative_ee_transforms[1, :3, 3], [0.0, 0.2, 0.0])
        np.testing.assert_allclose(result.relative_ee_transforms[2, :3, 3], [0.1, 0.0, 0.0])

    def test_partial_palm_landmark_keeps_sequence_orientation_sign(self):
        keypoints = np.zeros((21, 3), dtype=np.float64)
        keypoints[1] = [-0.045, 0.025, 0.0]
        keypoints[5] = [-0.035, 0.075, 0.0]
        keypoints[9] = [0.0, 0.090, 0.0]
        keypoints[12] = [0.0, 0.140, 0.0]
        keypoints[13] = [0.030, 0.075, 0.0]
        keypoints[17] = [0.050, 0.045, 0.0]
        complete = {"keypoints_3d": keypoints.tolist()}
        partial_keypoints = keypoints.copy()
        partial_keypoints[5] = np.nan
        partial = {"keypoints_3d": partial_keypoints.tolist()}

        raw_complete = palm_frame_from_semantic(complete)
        raw_partial = palm_frame_from_semantic(partial)
        for rotation in (raw_complete, raw_partial):
            self.assertIsNotNone(rotation)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

        ambiguous_opposite = raw_complete.copy()
        ambiguous_opposite[:, 1:3] *= -1.0
        with patch(
            "local_planning.run_execution_step.palm_frame_from_semantic",
            side_effect=[raw_complete, ambiguous_opposite],
        ):
            pose_0, reference = _hand_pose_with_reference_palm_orientation(
                complete,
                np.zeros((1, 3), dtype=np.float64),
                contact_finger="middle",
                reference_rotation=None,
            )
            pose_1, reference_after = _hand_pose_with_reference_palm_orientation(
                partial,
                np.zeros((1, 3), dtype=np.float64),
                contact_finger="middle",
                reference_rotation=reference,
            )
        relative = relative_transforms_from_poses(np.stack([pose_0, pose_1]))

        np.testing.assert_allclose(reference_after, reference, atol=1e-12)
        np.testing.assert_allclose(relative[1, :3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(pose_1[:3, :3])), 1.0, places=12)

    def test_hand_pose_track_excludes_approach_and_withdrawal(self):
        frame_indices = np.array([5, 7, 8, 10, 12])
        poses = np.stack([_pose((float(frame), 0.0, 0.0)) for frame in frame_indices])

        trimmed_poses, trimmed_indices = _trim_hand_pose_track(
            poses,
            frame_indices,
            {"movement_start_frame": 7, "movement_end_frame": 10},
        )

        self.assertEqual(trimmed_indices.tolist(), [7, 8, 10])
        np.testing.assert_allclose(trimmed_poses[:, 0, 3], [7.0, 8.0, 10.0])

    def test_hand_pose_track_rejects_single_frame_interaction(self):
        poses = np.stack([_pose(), _pose((0.1, 0.0, 0.0))])

        with self.assertRaisesRegex(ValueError, "fewer than two HaMeR poses"):
            _trim_hand_pose_track(
                poses,
                np.array([5, 9]),
                {"movement_start_frame": 5, "movement_end_frame": 5},
            )

    def test_hand_mesh_is_hidden_on_frames_without_hamer_output(self):
        frame_indices = np.array([2, 4, 7])
        shared = {
            "name": "hand",
            "coords": np.zeros((3, 3, 3), dtype=np.float64),
            "vis": None,
            "point_origins": None,
            "colors": np.zeros((3, 3), dtype=np.uint8),
            "frame_indices": frame_indices,
            "display_order": np.arange(3),
        }
        mesh_layer = FlowLayer(
            **shared,
            mesh_faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        trajectory_layer = FlowLayer(**shared)

        self.assertEqual(FlowReviewServer._layer_index_for_frame(mesh_layer, 2), 0)
        self.assertIsNone(FlowReviewServer._layer_index_for_frame(mesh_layer, 3))
        self.assertEqual(FlowReviewServer._layer_index_for_frame(mesh_layer, 4), 1)
        self.assertIsNone(FlowReviewServer._layer_index_for_frame(mesh_layer, 8))
        self.assertEqual(FlowReviewServer._layer_index_for_frame(trajectory_layer, 3), 0)
        self.assertEqual(FlowReviewServer._layer_index_for_frame(trajectory_layer, 8), 2)

    def test_relative_transform_uses_left_multiplication(self):
        poses = np.stack([_pose(), _pose((0.1, 0.0, 0.0))])
        rel = relative_transforms_from_poses(poses)
        np.testing.assert_allclose(rel[0], np.eye(4))
        np.testing.assert_allclose(rel[1, :3, 3], [0.1, 0.0, 0.0])

    def test_flow_trails_are_cumulative_and_respect_point_origins(self):
        frame_offsets = (
            np.arange(6, dtype=np.float64)[:, None, None]
            * np.array([[[0.01, 0.0, 0.0]]])
        )
        coords = np.repeat(frame_offsets, 2, axis=1)
        coords[:, 1, 1] = 0.1
        vis = np.ones((6, 2), dtype=bool)
        origins = np.array([0, 2], dtype=np.int64)
        colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        full_segments, _ = _build_current_trails(
            coords,
            vis,
            origins,
            colors,
            frame_idx=5,
        )

        self.assertEqual(len(full_segments), 8)

    def test_rigid_visualization_uses_fitted_transforms_not_raw_jitter(self):
        base = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float64)
        raw = np.stack([base, base + [0.04, -0.03, 0.08]], axis=0)
        transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
        transforms[1, :3, 3] = [0.01, 0.0, 0.0]

        rigid = _rigid_tracks_from_transforms(raw, transforms)

        np.testing.assert_allclose(rigid[0], base)
        np.testing.assert_allclose(rigid[1], base + [0.01, 0.0, 0.0])

    def test_spatial_track_order_is_deterministic_and_covers_extent(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        valid = np.ones(4, dtype=bool)

        first = _spatial_track_order(points, valid)
        second = _spatial_track_order(points, valid)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first[:2]), {0, 3})

    def test_armed_flow_review_starts_from_frame_zero(self):
        reviewer = FlowReviewServer.__new__(FlowReviewServer)
        reviewer._playback_armed = True
        reviewer.current_frame = 9
        reviewer.num_frames = 12
        reviewer.is_playing = False
        reviewer.frame_slider = SimpleNamespace(value=9.0)
        reviewer.update_visuals = Mock()

        reviewer._begin_armed_playback()

        self.assertFalse(reviewer._playback_armed)
        self.assertEqual(reviewer.current_frame, 0)
        self.assertEqual(reviewer.frame_slider.value, 0.0)
        reviewer.update_visuals.assert_called_once_with()
        self.assertTrue(reviewer.is_playing)

    def test_playback_frame_redraws_without_slider_callback(self):
        reviewer = FlowReviewServer.__new__(FlowReviewServer)
        reviewer.current_frame = 0
        reviewer.num_frames = 12
        reviewer.frame_slider = SimpleNamespace(value=0.0)
        reviewer.update_visuals = Mock()

        reviewer._render_playback_frame(4)

        self.assertEqual(reviewer.current_frame, 4)
        self.assertEqual(reviewer.frame_slider.value, 4.0)
        reviewer.update_visuals.assert_called_once_with()

    def test_playback_loops_to_frame_zero_after_final_frame(self):
        reviewer = FlowReviewServer.__new__(FlowReviewServer)
        reviewer.current_frame = 11
        reviewer.num_frames = 12
        reviewer.frame_slider = SimpleNamespace(value=11.0)
        reviewer.update_visuals = Mock()

        reviewer._advance_playback_frame()

        self.assertEqual(reviewer.current_frame, 0)
        self.assertEqual(reviewer.frame_slider.value, 0.0)
        reviewer.update_visuals.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
