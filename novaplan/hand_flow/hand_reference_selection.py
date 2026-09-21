#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
pick_dominant_hand_mesh.py

Problem:
  In a directory you may have multiple hand-mesh candidates per frame, e.g.
    grasp{t}{id}.ply   (t = frame index, id = candidate index)

Assumption:
  Only ONE real hand exists in the whole video.

Goal:
  For each frame t, pick the "dominant" candidate hand mesh by maximizing
  the count of vertices with `is_hand` (optionally thresholded).

This script:
  1) Scans a folder for grasp*.ply
  2) Groups by frame index t
  3) Loads each candidate mesh and counts `is_hand`
  4) Picks best per frame
  5) Prints summary + optionally writes selection JSON and/or symlinks.

Dependencies:
  - numpy
  - (recommended) plyfile  : add to pixi.toml if needed
  - (fallback) trimesh     : pixi install -e local-planning-hand

Notes:
  - We read PLY vertex properties directly (plyfile) so we can access custom fields like `is_hand`.
  - If `is_hand` is float/prob, set --threshold (default 0.5).
  - If `is_hand` is 0/1 int, threshold works fine too.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

def find_dominant_hand(dir):
    """Find dominant hand."""
    right = []
    right_fs = []
    left = []
    left_fs = []
    for pred_f in [x for x in os.listdir(dir) if x.endswith(".npz")]:
        pred = np.load(os.path.join(dir, pred_f))
        ts = int(pred_f.split('_')[0].replace("frame", ""))
        if pred["is_right"]:
            right.append(ts)
            right_fs.append(pred_f)
        else:
            left.append(ts)
            left_fs.append(pred_f)
    if len(left) > len(right):
        return left, left_fs
    return right, right_fs

def find_movement_start_frame(flow_file, 
                              agg: str = "p90", 
                              smooth: int = 1, 
                              baseline_frames: int = 3,
                              k: float = 6.0,
                              consecutive: int = 3):
    """Find movement start frame."""
    X = np.load(flow_file)["coords"]
    # frame-to-frame displacement per point: (T-1, N)
    d = np.linalg.norm(X[1:] - X[:-1], axis=2)

    # aggregate to a motion signal per frame step: (T-1,)
    if agg == "median":
        s = np.median(d, axis=1)
    elif agg == "rms":
        s = np.sqrt(np.mean(d * d, axis=1))
    elif agg == "p90":
        s = np.quantile(d, 0.90, axis=1)
    else:
        raise ValueError("agg must be 'median', 'rms', or 'p90'")

    # optional smoothing
    if smooth and smooth > 1:
        w = int(smooth)
        kernel = np.ones(w, dtype=np.float64) / w
        s = np.convolve(s, kernel, mode="same")

    # robust baseline from early frames (on s, which corresponds to t=1..T-1)
    b = s[: max(1, min(baseline_frames, s.shape[0]))]
    med = float(np.median(b))
    mad = float(np.median(np.abs(b - med))) + 1e-12  # robust scale
    thr = med + k * 1.4826 * mad  # 1.4826*MAD ~ std for Gaussian

    # find first index where s stays above thr for `consecutive` steps
    above = (s > thr)
    if consecutive <= 1:
        idx = np.argmax(above) if np.any(above) else -1
        return (idx + 1) if idx >= 0 else -1  # +1 because s[t-1] corresponds to frame t

    run = 0
    for i in range(above.shape[0]):
        run = run + 1 if above[i] else 0
        if run >= consecutive:
            step_idx = i - consecutive + 1
            return step_idx + 1  # convert step index to frame index

    return -1

def find_hand_reference(dir, idx, *, verbose: bool = False):
    """
    Find the hand reference mesh closest to the target frame index.
    
    Args:
        dir: Directory containing frame*.npz files (with absolute frame numbers)
        idx: Target frame index (absolute)
    
    Returns:
        Filename of the best matching hand reference
    """
    ts, fs = find_dominant_hand(dir)
    # ts now contains absolute frame numbers, so no need to add start_t
    off_ts = [abs(x - idx) for x in ts]
    best_fs = fs[np.argmin(off_ts)]

    if verbose:
        print(
            f"[hand-reference] offsets={off_ts} frames={ts} "
            f"target={idx} selected={best_fs}"
        )
    return best_fs

def main():
    """Run the command-line entry point."""
    ap = argparse.ArgumentParser(description="Select the dominant hand mesh nearest motion onset.")
    ap.add_argument(
        "--mesh_dir",
        type=Path,
        required=True,
        help="Directory containing calibrated frame*_hand*.npz meshes.",
    )
    ap.add_argument("--flow_file", type=Path, required=True, help="Flow NPZ used to detect motion onset.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not args.mesh_dir.is_dir():
        ap.error(f"--mesh_dir is not a directory: {args.mesh_dir}")
    if not args.flow_file.is_file():
        ap.error(f"--flow_file does not exist: {args.flow_file}")
    movement_start = find_movement_start_frame(args.flow_file)
    if movement_start < 0:
        ap.error(f"no sustained motion was detected in {args.flow_file}")
    print(find_hand_reference(args.mesh_dir, movement_start, verbose=args.verbose))

if __name__ == "__main__":
    main()
