# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Check whether the active PyTorch build can compile TAPIP3D PointOps."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple


_CUDA_RELEASE = re.compile(r"\brelease\s+(\d+\.\d+)\b")


def parse_nvcc_cuda_version(output: str) -> Optional[str]:
    """Parse a CUDA version from NVCC output."""
    match = _CUDA_RELEASE.search(output)
    return match.group(1) if match else None


def assess_cuda_compatibility(
    torch_cuda: Optional[str],
    nvcc_cuda: Optional[str],
    nvcc_path: Path,
) -> Tuple[bool, str]:
    """Assess whether PyTorch and NVCC use compatible CUDA versions."""
    if not torch_cuda:
        return False, "the active PyTorch build does not include CUDA support"
    if not nvcc_cuda:
        return False, f"could not determine the CUDA version from {nvcc_path}"

    torch_major = torch_cuda.split(".", 1)[0]
    nvcc_major = nvcc_cuda.split(".", 1)[0]
    if torch_major != nvcc_major:
        return (
            False,
            f"PyTorch CUDA {torch_cuda} is incompatible with nvcc {nvcc_cuda} "
            f"at {nvcc_path}",
        )

    detail = f"PyTorch CUDA {torch_cuda} with nvcc {nvcc_cuda} at {nvcc_path}"
    if torch_cuda != nvcc_cuda:
        detail += " (same major version; PyTorch permits this minor-version difference)"
    return True, detail


def detect_active_cuda_toolchain() -> Tuple[Optional[str], Optional[str], Path]:
    """Detect active cuda toolchain."""
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    torch_cuda = torch.version.cuda
    cuda_home = Path(CUDA_HOME) if CUDA_HOME else Path("<not found>")
    nvcc_path = cuda_home / "bin" / "nvcc" if CUDA_HOME else cuda_home
    if not CUDA_HOME or not nvcc_path.is_file():
        return torch_cuda, None, nvcc_path

    completed = subprocess.run(
        [str(nvcc_path), "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return torch_cuda, None, nvcc_path
    output = f"{completed.stdout}\n{completed.stderr}"
    return torch_cuda, parse_nvcc_cuda_version(output), nvcc_path


def main() -> int:
    """Run the command-line entry point."""
    try:
        torch_cuda, nvcc_cuda, nvcc_path = detect_active_cuda_toolchain()
        compatible, detail = assess_cuda_compatibility(
            torch_cuda,
            nvcc_cuda,
            nvcc_path,
        )
    except Exception as exc:
        print(f"could not inspect the active PyTorch/CUDA toolchain: {exc}")
        return 1

    print(detail)
    return 0 if compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
