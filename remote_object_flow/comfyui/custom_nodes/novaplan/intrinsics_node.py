# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Small camera intrinsics utility node for NovaPlan ComfyUI workflows."""

from __future__ import annotations

import torch


class IntrinsicsNode:
    """Build a 3x3 pinhole camera intrinsics matrix from fx, fy, cx, cy."""

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "fx": ("FLOAT", {"default": 645.8, "min": 0.0, "step": 0.01}),
                "fy": ("FLOAT", {"default": 645.8, "min": 0.0, "step": 0.01}),
                "cx": ("FLOAT", {"default": 644.6, "min": 0.0, "step": 0.01}),
                "cy": ("FLOAT", {"default": 364.9, "min": 0.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INTRINSICS",)
    RETURN_NAMES = ("intrinsics",)
    FUNCTION = "build"
    CATEGORY = "NovaPlan/Camera"

    def build(self, fx: float, fy: float, cx: float, cy: float):
        """Construct a camera-intrinsics matrix from scalar parameters."""
        intrinsics = torch.tensor(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return (intrinsics,)


NODE_CLASS_MAPPINGS = {"IntrinsicsNode": IntrinsicsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"IntrinsicsNode": "Camera Intrinsics"}

