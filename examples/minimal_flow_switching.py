# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Minimal local example for NovaPlan object-vs-hand flow switching."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from novaplan.flow_switching import HandFlowCandidate, select_flow_reference_from_arrays


def main() -> None:
    # Six visible points observed for two frames. The second frame is a 60
    # degree rotation around z, so the default 45 degree threshold requests hand
    # flow.
    """Run the command-line entry point."""
    theta = np.deg2rad(60.0)
    rot_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points_0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.2, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=np.float64,
    )
    points_1 = points_0 @ rot_z.T
    coords_3d = np.stack([points_0, points_1], axis=0)
    visibilities = np.ones(coords_3d.shape[:2], dtype=bool)

    decision = select_flow_reference_from_arrays(
        coords_3d,
        visibilities,
        hand_flow=HandFlowCandidate(valid=True, trajectory=coords_3d[:, :1]),
        layout="time_major",
        flow_switch_theta_deg=45.0,
    )
    print(decision.to_dict())


if __name__ == "__main__":
    main()
