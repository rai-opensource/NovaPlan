#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Ensure the RAFT checkpoint required by NovaPlan CVD is locally available."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_ARCHIVE_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
MIN_CHECKPOINT_BYTES = 1_000_000


def _is_checkpoint(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_CHECKPOINT_BYTES
    except OSError:
        return False


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
        if not _is_checkpoint(temporary):
            raise RuntimeError(f"RAFT checkpoint is unexpectedly small: {source}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _find_existing(search_root: Path, destination: Path) -> Path | None:
    if not search_root.is_dir():
        return None
    destination_resolved = destination.resolve()
    for candidate in search_root.rglob("raft-things.pth"):
        if candidate.resolve() != destination_resolved and _is_checkpoint(candidate):
            return candidate
    return None


def _download_from_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip") as archive:
            urllib.request.urlretrieve(url, archive.name)
            with zipfile.ZipFile(archive.name) as models:
                members = [
                    name
                    for name in models.namelist()
                    if Path(name).name == "raft-things.pth"
                ]
                if len(members) != 1:
                    raise RuntimeError(
                        "Expected exactly one raft-things.pth in the official "
                        f"RAFT model archive, found {members}"
                    )
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, delete=False
                ) as output:
                    temporary = Path(output.name)
                    with models.open(members[0]) as source:
                        shutil.copyfileobj(source, output)
        if not _is_checkpoint(temporary):
            raise RuntimeError(
                f"Downloaded RAFT checkpoint is unexpectedly small: {temporary}"
            )
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_checkpoint(
    destination: Path,
    *,
    source: Path | None,
    search_root: Path | None,
    archive_url: str,
) -> str:
    """Ensure that the required RAFT checkpoint is locally available."""
    if _is_checkpoint(destination):
        return f"already present: {destination}"

    if source is not None:
        if not _is_checkpoint(source):
            raise FileNotFoundError(
                f"RAFT_CHECKPOINT_SOURCE is missing or invalid: {source}"
            )
        _copy_atomic(source, destination)
        return f"copied from {source} to {destination}"

    if search_root is not None:
        existing = _find_existing(search_root, destination)
        if existing is not None:
            _copy_atomic(existing, destination)
            return f"reused {existing} at {destination}"

    _download_from_archive(archive_url, destination)
    return f"downloaded to {destination}"


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    args = parser.parse_args()

    result = ensure_checkpoint(
        args.destination,
        source=args.source,
        search_root=args.search_root,
        archive_url=args.archive_url,
    )
    print(f"[ensure-raft-checkpoint] {result}", flush=True)


if __name__ == "__main__":
    main()
