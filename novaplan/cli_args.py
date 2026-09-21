# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Shared argparse validators for NovaPlan command-line entry points."""

from __future__ import annotations

import argparse
import re
from urllib.parse import urlparse


def positive_int(value: str) -> int:
    """Parse and validate an integer argument."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse and validate an integer argument."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    """Parse and validate a floating-point argument."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    """Parse and validate a floating-point argument."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def unit_interval(value: str) -> float:
    """Parse and validate a unit-interval value."""
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 inclusive")
    return parsed


def rotation_degrees(value: str) -> float:
    """Parse and validate a bounded rotation in degrees."""
    parsed = float(value)
    if not 0.0 <= parsed <= 180.0:
        raise argparse.ArgumentTypeError("must be between 0 and 180 degrees inclusive")
    return parsed


def tcp_port(value: str) -> int:
    """Parse and validate a TCP port number."""
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be a TCP port between 1 and 65535")
    return parsed


def frame_limit(value: str) -> int:
    """Parse and validate an optional frame limit."""
    parsed = int(value)
    if parsed == -1 or parsed > 0:
        return parsed
    raise argparse.ArgumentTypeError("must be -1 (unlimited) or an integer greater than zero")


def image_size(value: str) -> str:
    """Parse and validate an image-size argument."""
    match = re.fullmatch(r"([1-9][0-9]*)[*xX]([1-9][0-9]*)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("must use WIDTH*HEIGHT, for example 1280*720")
    return f"{int(match.group(1))}*{int(match.group(2))}"


def http_url(value: str) -> str:
    """Parse and validate an HTTP service URL."""
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("must be an absolute http:// or https:// URL")
    return value.strip().rstrip("/")


def rgb_channel(value: str) -> int:
    """Parse and validate an RGB channel value."""
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("must be an integer between 0 and 255")
    return parsed


def validate_horizon_bounds(
    parser: argparse.ArgumentParser,
    *,
    minimum: int,
    maximum: int,
) -> None:
    """Validate horizon bounds."""
    if minimum > maximum:
        parser.error("--horizon_count_min cannot exceed --horizon_count_max")
