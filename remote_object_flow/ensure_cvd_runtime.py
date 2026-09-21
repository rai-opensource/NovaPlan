#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Ensure the minimal RAFT source runtime used by optional CVD is available."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path


MEGASAM_CVD_REF = "a27b4e633c5cc0828a62ed943ef9f6505705fd3f"
DEFAULT_SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/mega-sam/mega-sam/"
    f"{MEGASAM_CVD_REF}/cvd_opt"
)
UPSTREAM_FILES = (
    Path("core/raft.py"),
    Path("core/corr.py"),
    Path("core/extractor.py"),
    Path("core/update.py"),
    Path("core/utils/utils.py"),
)
GENERATED_FILES = (Path("core/__init__.py"), Path("core/utils/__init__.py"))
LEGAL_FILES = (Path("LICENSE"), Path("RAFT_LICENSE"))
REQUIRED_FILES = UPSTREAM_FILES + GENERATED_FILES + LEGAL_FILES
LEGAL_SOURCE_DIR = (
    Path(__file__).resolve().parent
    / "comfyui/custom_nodes/novaplan/moge2_metric_depth"
)
MODIFICATION_NOTICE = (
    "# Modified by Robotics and AI Institute LLC for NovaPlan: "
    "package-relative imports.\n"
)
PINNED_SHA256 = {
    Path("core/raft.py"): "d081f76f4796e01715b7a92d54bf48bfe650c9bfc7766dda8f1b483024f3c53c",
    Path("core/corr.py"): "ddc0c55be2f1d16a92ff543fa9cc7ead43062a5af7f557e1f0642bf49d500e0a",
    Path("core/extractor.py"): "25181e46dc4e1ebbf19269080ced1e01619fe68452654465b29b7616976864b3",
    Path("core/update.py"): "fe1dd199bba01296aec62b77f11737c1b6527b81e6137a862a1fb705d6981a44",
    Path("core/utils/utils.py"): "7b9080f7c7131424edb377c87950a353ead930b30852c9ef72c1b3236b65b825",
}


IMPORT_REWRITES = {
    Path("core/raft.py"): (
        ("from corr import ", "from .corr import "),
        ("from extractor import ", "from .extractor import "),
        ("from update import ", "from .update import "),
        ("from utils.utils import ", "from .utils.utils import "),
    ),
    Path("core/corr.py"): (
        ("from utils.utils import ", "from .utils.utils import "),
    ),
}


def _is_upstream_runtime(path: Path) -> bool:
    if not all((path / relative).is_file() for relative in UPSTREAM_FILES):
        return False
    try:
        raft_source = (path / "core/raft.py").read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return "return coords1 - coords0, flow_up, net" in raft_source


def _is_runtime(path: Path) -> bool:
    if not _is_upstream_runtime(path):
        return False
    if not all(
        (path / relative).is_file()
        for relative in GENERATED_FILES + LEGAL_FILES
    ):
        return False
    try:
        raft_source = (path / "core/raft.py").read_text()
        return (
            "from .corr import " in raft_source
            and MODIFICATION_NOTICE.strip() in raft_source
        )
    except (OSError, UnicodeDecodeError):
        return False


def _prepare_private_package(runtime: Path) -> None:
    for relative, replacements in IMPORT_REWRITES.items():
        path = runtime / relative
        source = path.read_text()
        for old, new in replacements:
            source = source.replace(old, new)
        if MODIFICATION_NOTICE.strip() not in source:
            source = MODIFICATION_NOTICE + source
        path.write_text(source)
    for relative in GENERATED_FILES:
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Private NovaPlan CVD RAFT runtime."""\n')
    for relative in LEGAL_FILES:
        source = LEGAL_SOURCE_DIR / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing bundled CVD legal file: {source}")
        shutil.copyfile(source, runtime / relative)


def _normalize_source(source: Path) -> Path | None:
    for candidate in (source, source / "cvd_opt", source.parent):
        candidate = candidate.resolve()
        if _is_upstream_runtime(candidate):
            return candidate
    return None


def _find_existing(search_root: Path, destination: Path) -> Path | None:
    if not search_root.is_dir():
        return None
    destination_resolved = destination.resolve()
    for raft_file in search_root.rglob("raft.py"):
        if raft_file.parent.name != "core":
            continue
        candidate = raft_file.parent.parent
        if candidate.resolve() != destination_resolved and _is_upstream_runtime(candidate):
            return candidate
    return None


def _copy_runtime(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for relative in UPSTREAM_FILES:
        source_file = source / relative
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination_file.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                with source_file.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
            temporary.replace(destination_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    _prepare_private_package(destination)


def _download_runtime(source_base_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staging = Path(directory) / "cvd_opt"
        for relative in UPSTREAM_FILES:
            output = staging / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            url = f"{source_base_url.rstrip('/')}/{relative.as_posix()}"
            urllib.request.urlretrieve(url, output)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            if digest != PINNED_SHA256[relative]:
                raise RuntimeError(
                    f"CVD source checksum mismatch for {relative}: {digest}"
                )
        if not _is_upstream_runtime(staging):
            raise RuntimeError("Downloaded CVD source runtime is incomplete")
        _copy_runtime(staging, destination)


def ensure_runtime(
    destination: Path,
    *,
    source: Path | None,
    search_root: Path | None,
    source_base_url: str,
) -> str:
    """Ensure that the minimal licensed CVD runtime is locally available."""
    if _is_runtime(destination):
        return f"already present: {destination}"
    if _is_upstream_runtime(destination):
        _prepare_private_package(destination)
        return f"repaired private-package imports at {destination}"

    if source is not None:
        normalized = _normalize_source(source)
        if normalized is None:
            raise FileNotFoundError(
                "CVD_RUNTIME_SOURCE does not contain MegaSAM cvd_opt/core: "
                f"{source}"
            )
        _copy_runtime(normalized, destination)
        return f"copied from {normalized} to {destination}"

    if search_root is not None:
        existing = _find_existing(search_root, destination)
        if existing is not None:
            _copy_runtime(existing, destination)
            return f"reused {existing} at {destination}"

    _download_runtime(source_base_url, destination)
    return f"downloaded pinned CVD runtime to {destination}"


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--search-root", type=Path)
    parser.add_argument("--source-base-url", default=DEFAULT_SOURCE_BASE_URL)
    args = parser.parse_args()

    result = ensure_runtime(
        args.destination,
        source=args.source,
        search_root=args.search_root,
        source_base_url=args.source_base_url,
    )
    print(f"[ensure-cvd-runtime] {result}", flush=True)


if __name__ == "__main__":
    main()
