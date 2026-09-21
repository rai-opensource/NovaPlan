#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
Self-contained utility functions for client scripts.
No external dependencies beyond standard library and numpy.
"""

import json
from typing import Any, Dict

import numpy as np
import colorsys


def decode_semantic_xyz_json(x: Any) -> Dict[str, Any]:
    """
    Decode semantic_xyz_json field from various formats.
    
    Accepts:
      - numpy scalar bytes (dtype='|S...') / np.bytes_
      - python bytes
      - python str
      - already-parsed dict
      - np.ndarray containing a single bytes/str/dict
    
    Returns parsed JSON dict.
    
    Args:
        x: Semantic JSON field in any supported format
    
    Returns:
        dict: Parsed JSON dictionary
    """
    if x is None:
        raise ValueError("semantic json field is None")

    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise ValueError(f"Expected scalar semantic json, got array shape={x.shape}")
        x = x.item()

    if isinstance(x, dict):
        return x

    if isinstance(x, (bytes, np.bytes_)):
        s = bytes(x).decode("utf-8")
        # allow empty string gracefully
        if len(s.strip()) == 0:
            raise ValueError("semantic json field is empty bytes string")
        return json.loads(s)

    if isinstance(x, str):
        if len(x.strip()) == 0:
            raise ValueError("semantic json field is empty string")
        return json.loads(x)

    raise ValueError(f"Unsupported semantic json field type: {type(x)}")


def make_rank_colors(
    values: np.ndarray,
    *,
    best_is_red: bool = True,
    hue_max: float = 0.75,          # 0.0=red, 0.75~purple
    saturation: float = 1.0,
    value: float = 1.0,
    stable: bool = True,
) -> np.ndarray:
    """
    Generate colors for ranked values using HSV color space.
    
    Creates a rainbow of colors where each value gets a color based on its rank.
    By default, best (highest) values get red, worst get purple.
    
    Args:
        values: (N,) array of values to rank
        best_is_red: If True, best values get red (hue=0), worst get purple (hue_max).
                     If False, reversed.
        hue_max: Maximum hue value (0.0=red, 0.33=green, 0.66=blue, 0.75=purple)
        saturation: HSV saturation (0.0 to 1.0)
        value: HSV value/brightness (0.0 to 1.0)
        stable: Use stable sort for ranking
    
    Returns:
        (N, 3) uint8 array of RGB colors, where colors[i] corresponds to values[i]
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = v.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    if n == 1:
        hues = np.array([0.0], dtype=np.float64)
    else:
        hues = hue_max * (np.arange(n, dtype=np.float64) / (n - 1))

    rank_colors = np.empty((n, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(float(h), float(saturation), float(value))
        rank_colors[i] = (int(r * 255), int(g * 255), int(b * 255))

    kind = "stable" if stable else "quicksort"
    order = np.argsort(v, kind=kind)
    if best_is_red:
        order = order[::-1]

    colors = np.empty((n, 3), dtype=np.uint8)
    colors[order] = rank_colors
    return colors


def rainbow_colors(T: int, *, hue_start: float = 0.0, hue_end: float = 0.75) -> np.ndarray:
    """
    Generate a rainbow of colors sweeping from hue_start to hue_end.
    
    Default: red -> orange -> yellow -> green -> cyan -> blue -> purple.
    
    Args:
        T: Number of colors to generate
        hue_start: Starting hue (0.0=red, 1/3=green, 2/3=blue)
        hue_end: Ending hue (default 0.75=purple)
    
    Returns:
        (T, 3) uint8 array of RGB colors
    """
    T = int(T)
    if T <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if T == 1:
        r, g, b = colorsys.hsv_to_rgb(hue_start, 1.0, 1.0)
        return np.array([[int(r * 255), int(g * 255), int(b * 255)]], dtype=np.uint8)

    hues = np.linspace(hue_start, hue_end, T, dtype=np.float64)
    cols = np.empty((T, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
        cols[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return cols
