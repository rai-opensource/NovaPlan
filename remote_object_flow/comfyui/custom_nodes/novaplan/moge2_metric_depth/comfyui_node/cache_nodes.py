# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Neutral in-memory and file-backed utilities for metric depth tensors."""

import os
import time

import folder_paths
import numpy as np
import torch


_DEPTH_CACHE = {}
_VERBOSE = os.environ.get("NOVAPLAN_VERBOSE_LOGS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _debug(message: str) -> None:
    if _VERBOSE:
        print(message, flush=True)


class CacheRawDepthNode:
    """Cache calibrated metric depth in the ComfyUI worker process."""

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "raw_depth": ("IMAGE",),
                "cache_id": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("raw_depth",)
    OUTPUT_NODE = True
    FUNCTION = "cache_depth"
    CATEGORY = "NovaPlan/Metric Depth"

    def cache_depth(self, raw_depth, cache_id):
        """Cache a raw depth tensor and return its cache key."""
        started = time.perf_counter()
        _DEPTH_CACHE[cache_id] = raw_depth.clone()
        _debug(
            f"[CacheRawDepthNode] Cached {cache_id!r} with shape "
            f"{tuple(raw_depth.shape)} in {time.perf_counter() - started:.3f}s"
        )
        return {"ui": {"cached": [cache_id]}, "result": (raw_depth,)}


class LoadCachedRawDepthNode:
    """Load calibrated metric depth cached by ``CacheRawDepthNode``."""

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {"required": {"cache_id": ("STRING", {"default": "default"})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("raw_depth",)
    FUNCTION = "load_depth"
    CATEGORY = "NovaPlan/Metric Depth"

    def load_depth(self, cache_id):
        """Load and return a raw depth tensor."""
        if cache_id not in _DEPTH_CACHE:
            raise ValueError(
                f"Depth cache ID {cache_id!r} not found. "
                f"Available IDs: {list(_DEPTH_CACHE)}"
            )
        return (_DEPTH_CACHE[cache_id],)


class SaveRawDepthNode:
    """Save a metric-depth tensor as a NumPy array for diagnostics."""

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "raw_depth": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "metric_depth"}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_depth"
    CATEGORY = "NovaPlan/Metric Depth"

    def save_depth(self, raw_depth, filename_prefix):
        """Save depth."""
        output_dir = folder_paths.get_output_directory()
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.npy"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        np.save(filepath, raw_depth.cpu().numpy())
        return {
            "ui": {
                "npy": [
                    {"filename": filename, "subfolder": "", "type": "output"}
                ]
            }
        }


class LoadRawDepthFromFileNode:
    """Load metric depth from a NumPy array in the ComfyUI input directory."""

    @classmethod
    def INPUT_TYPES(cls):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "depth_file": ("STRING", {"default": "depth.npy"}),
                "depth_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0001,
                        "max": 1000.0,
                        "step": 0.0001,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("raw_depth",)
    FUNCTION = "load_depth"
    CATEGORY = "NovaPlan/Metric Depth"

    def load_depth(self, depth_file, depth_scale):
        """Load and return a raw depth tensor."""
        filepath = os.path.join(folder_paths.get_input_directory(), depth_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metric-depth file not found: {filepath}")

        depth = np.load(filepath)
        if depth.ndim == 2:
            depth = depth[None, :, :, None]
        elif depth.ndim == 3:
            depth = depth[:, :, :, None]
        elif depth.ndim != 4 or depth.shape[-1] != 1:
            raise ValueError(
                f"Unexpected metric-depth shape {depth.shape}; expected [H,W], "
                "[T,H,W], or [T,H,W,1]"
            )

        return (torch.from_numpy(depth.astype(np.float32) * depth_scale),)


NODE_CLASS_MAPPINGS = {
    "CacheRawDepthNode": CacheRawDepthNode,
    "LoadCachedRawDepthNode": LoadCachedRawDepthNode,
    "SaveRawDepthNode": SaveRawDepthNode,
    "LoadRawDepthFromFileNode": LoadRawDepthFromFileNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheRawDepthNode": "Cache Metric Depth",
    "LoadCachedRawDepthNode": "Load Cached Metric Depth",
    "SaveRawDepthNode": "Save Metric Depth (.npy)",
    "LoadRawDepthFromFileNode": "Load Metric Depth (.npy)",
}
