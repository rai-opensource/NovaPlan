#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Local tests for the hand-flow client contract (HaMeR backend).

These tests do not require a running hand-flow service. They validate the parts of
the public client that parse returned meshes and write the local hand-flow
artifacts consumed by visualization/debugging scripts.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from novaplan.hand_flow.hamer_client import (
    _collect_real_meshes,
    _copy_reference_mesh,
    _save_meshes_to_out_folder,
)


def _mesh(frame_idx: int = 0, hand_id: int = 0) -> dict:
    return {
        "frame_idx": frame_idx,
        "hand_id": hand_id,
        "is_right": True,
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "faces": [[0, 1, 2]],
        "semantic_xyz": {"fingers": {"index": {"tip": [0.0, 0.0, 0.0]}}},
    }


class HandFlowExtractionClientTest(unittest.TestCase):
    def test_collect_real_meshes_accepts_frames_or_top_level(self):
        duplicate = _mesh(frame_idx=0, hand_id=0)
        unique = _mesh(frame_idx=1, hand_id=0)
        resp = {
            "ok": True,
            "frames": [{"real_meshes": [duplicate]}],
            "real_meshes": [duplicate, unique],
        }

        meshes, video_bytes = _collect_real_meshes(resp)

        self.assertEqual(len(meshes), 2)
        self.assertEqual(video_bytes, b"")
        self.assertEqual({m["frame_idx"] for m in meshes}, {0, 1})

    def test_save_meshes_writes_npz_dump_with_absolute_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = _save_meshes_to_out_folder([_mesh(frame_idx=2)], tmp, start_frame=10)
            self.assertEqual(len(info["npz_paths"]), 1)
            npz_path = Path(info["npz_paths"][0])
            self.assertTrue(npz_path.exists())

            with np.load(npz_path) as saved:
                self.assertEqual(int(saved["frame_idx"]), 12)
                self.assertEqual(saved["vertices"].shape, (3, 3))
                self.assertEqual(saved["faces"].shape, (1, 3))

    def test_reference_npz_is_saved_without_optional_open3d(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "novaplan.hand_flow.hamer_client._O3D_OK",
            False,
        ):
            info = _save_meshes_to_out_folder([_mesh(frame_idx=2)], tmp, start_frame=10)
            copied = _copy_reference_mesh(
                info["dump_dir"],
                Path(info["npz_paths"][0]).name,
                tmp,
                output_stem="reference",
            )

            self.assertTrue(Path(copied["npz"]).exists())
            self.assertIsNone(copied["ply"])
            self.assertFalse((Path(tmp) / "reference.ply").exists())


if __name__ == "__main__":
    unittest.main()
