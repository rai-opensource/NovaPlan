#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

# One-time setup for the NovaPlan hand-flow service (HaMeR backend).
#
# Primary upstream links:
# - HaMeR project: https://geopavlakos.github.io/hamer/
# - HaMeR code: https://github.com/geopavlakos/hamer
# - MANO model registration/download: https://mano.is.tue.mpg.de/
#
# Lower-level setup command, run from the repository root:
#   pixi run -e hand-flow-host setup-hand-flow-host
#
# Default model downloads:
#   DOWNLOAD_HAMER_MODELS=1 pixi run -e hand-flow-host setup-hand-flow-host
#
# MANO_RIGHT.pkl is license-gated. Download it from the MANO website and either
# copy it to $HAMER_DIR/_DATA/data/mano/MANO_RIGHT.pkl yourself or pass:
#   MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl pixi run -e hand-flow-host setup-hand-flow-host
#
# For a one-command host setup and launch, prefer:
#   MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl ./remote_hand_flow/bootstrap_hand_flow_host.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_HAMER_DIR="$SCRIPT_DIR/.external/hamer"
HAMER_DIR="${HAMER_DIR:-$DEFAULT_HAMER_DIR}"
HAMER_REPO="${HAMER_REPO:-https://github.com/geopavlakos/hamer.git}"
# Revision validated by the public adapter. Set HAMER_REF='' explicitly only
# when testing another upstream revision.
HAMER_REF="${HAMER_REF-3a01849f4148352e9260b69bf28b65d1671a4905}"
HAMER_PYTHON="${HAMER_PYTHON:-python}"
HAMER_TORCH_INDEX_URL="${HAMER_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
MANO_RIGHT_SOURCE="${MANO_RIGHT_SOURCE:-}"
HAMER_DISABLE_USER_SITE="${HAMER_DISABLE_USER_SITE:-1}"
HAMER_NO_BUILD_ISOLATION="${HAMER_NO_BUILD_ISOLATION:-1}"
HAMER_SETUPTOOLS_SPEC="${HAMER_SETUPTOOLS_SPEC:-setuptools<80}"
HAMER_NUMPY_SPEC="${HAMER_NUMPY_SPEC:-numpy>=1.26.0,<2}"
HAMER_OPENCV_SPEC="${HAMER_OPENCV_SPEC:-opencv-python>=4.8.0,<5}"

INSTALL_HAMER="${INSTALL_HAMER:-1}"
INSTALL_HAMER_REQUIREMENTS="${INSTALL_HAMER_REQUIREMENTS:-1}"
INSTALL_HAMER_TORCH="${INSTALL_HAMER_TORCH:-1}"
DOWNLOAD_HAMER_MODELS="${DOWNLOAD_HAMER_MODELS:-1}"
REQUIRE_MANO="${REQUIRE_MANO:-1}"

log() {
  printf '[setup-hand-flow-host] %s\n' "$*"
}

pip_install() {
  local env_args=()
  if [[ "$HAMER_DISABLE_USER_SITE" == "1" ]]; then
    env_args+=(PYTHONNOUSERSITE=1)
  fi
  env "${env_args[@]}" PYTHONPATH= "$HAMER_PYTHON" -m pip install "$@"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 1
  fi
}

is_nonempty_dir() {
  [[ -d "$1" ]] && find "$1" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

is_data_only_hamer_dir() {
  [[ -d "$1" ]] || return 1
  local unexpected
  unexpected="$(find "$1" -mindepth 1 -maxdepth 1 ! -name "_DATA" -print -quit)"
  [[ -z "$unexpected" ]]
}

checkout_hamer_ref() {
  local checkout_dir="$1"
  if [[ -n "$HAMER_REF" ]]; then
    log "Checking out HaMeR ref: $HAMER_REF"
    git -C "$checkout_dir" checkout "$HAMER_REF"
    git -C "$checkout_dir" submodule update --init --recursive
  fi
}

clone_hamer() {
  if [[ -f "$HAMER_DIR/demo.py" && -d "$HAMER_DIR/hamer" ]]; then
    log "HaMeR already present: $HAMER_DIR"
    if [[ -n "$HAMER_REF" && -d "$HAMER_DIR/.git" ]]; then
      checkout_hamer_ref "$HAMER_DIR"
    elif [[ -n "$HAMER_REF" ]]; then
      log "Existing HaMeR is not a git checkout; verify compatibility with $HAMER_REF manually."
    fi
    return
  fi

  if [[ "$INSTALL_HAMER" != "1" ]]; then
    log "INSTALL_HAMER=0, skipping HaMeR clone."
    return
  fi

  need_cmd git
  if is_nonempty_dir "$HAMER_DIR"; then
    if is_data_only_hamer_dir "$HAMER_DIR"; then
      local tmp_dir
      tmp_dir="${HAMER_DIR}.clone_tmp.$$"
      if [[ -e "$tmp_dir" ]]; then
        printf 'Temporary clone path already exists: %s\n' "$tmp_dir" >&2
        exit 1
      fi

      log "HAMER_DIR contains only the NovaPlan data scaffold; cloning HaMeR into a temporary sibling and merging."
      git clone --recursive "$HAMER_REPO" "$tmp_dir"
      checkout_hamer_ref "$tmp_dir"
      cp -a "$tmp_dir/." "$HAMER_DIR/"
      rm -rf "$tmp_dir"
      return
    fi

    printf 'HAMER_DIR exists but does not look like a HaMeR checkout: %s\n' "$HAMER_DIR" >&2
    printf 'The setup can resume only when this directory is empty or contains only _DATA/.\n' >&2
    printf 'Move unrelated files elsewhere, or set HAMER_DIR to another path.\n' >&2
    exit 1
  fi

  mkdir -p "$(dirname "$HAMER_DIR")"
  log "Cloning HaMeR into $HAMER_DIR"
  git clone --recursive "$HAMER_REPO" "$HAMER_DIR"
  checkout_hamer_ref "$HAMER_DIR"
}

install_hamer_requirements() {
  if [[ "$INSTALL_HAMER_REQUIREMENTS" != "1" ]]; then
    log "INSTALL_HAMER_REQUIREMENTS=0, skipping Python package install."
    return
  fi

  log "Installing server packages into the active Pixi env."
  pip_install --upgrade pip "$HAMER_SETUPTOOLS_SPEC" wheel
  pip_install "fastapi>=0.111.0" "uvicorn>=0.29.0" "pydantic>=2.6.0" "$HAMER_NUMPY_SPEC" "$HAMER_OPENCV_SPEC"

  if [[ "$INSTALL_HAMER_TORCH" == "1" ]]; then
    log "Installing PyTorch from $HAMER_TORCH_INDEX_URL"
    pip_install torch torchvision --index-url "$HAMER_TORCH_INDEX_URL"
  else
    log "INSTALL_HAMER_TORCH=0, assuming torch/torchvision are already installed."
  fi

  log "Installing HaMeR and all extras."
  local hamer_pip_args=()
  if [[ "$HAMER_NO_BUILD_ISOLATION" == "1" ]]; then
    hamer_pip_args+=(--no-build-isolation)
  fi
  (cd "$HAMER_DIR" && pip_install "${hamer_pip_args[@]}" -e ".[all]")

  if [[ -d "$HAMER_DIR/third-party/ViTPose" ]]; then
    log "Installing HaMeR third-party ViTPose."
    pip_install -v -e "$HAMER_DIR/third-party/ViTPose"
  else
    printf 'ViTPose submodule not found: %s\n' "$HAMER_DIR/third-party/ViTPose" >&2
    printf 'Run: git -C %s submodule update --init --recursive\n' "$HAMER_DIR" >&2
    exit 1
  fi
}

download_hamer_models() {
  if [[ "$DOWNLOAD_HAMER_MODELS" != "1" ]]; then
    log "DOWNLOAD_HAMER_MODELS=0, skipping HaMeR demo checkpoint download."
    return
  fi

  if [[ ! -f "$HAMER_DIR/fetch_demo_data.sh" ]]; then
    printf 'Missing HaMeR fetch_demo_data.sh under %s\n' "$HAMER_DIR" >&2
    exit 1
  fi

  log "Downloading HaMeR demo checkpoint bundle."
  (cd "$HAMER_DIR" && bash fetch_demo_data.sh)
}

place_mano_model() {
  local mano_dir="$HAMER_DIR/_DATA/data/mano"
  local mano_target="$mano_dir/MANO_RIGHT.pkl"

  mkdir -p "$mano_dir"
  if [[ -s "$mano_target" ]]; then
    log "MANO_RIGHT.pkl already present: $mano_target"
    return
  fi

  if [[ -n "$MANO_RIGHT_SOURCE" ]]; then
    if [[ ! -f "$MANO_RIGHT_SOURCE" ]]; then
      printf 'MANO_RIGHT_SOURCE does not exist: %s\n' "$MANO_RIGHT_SOURCE" >&2
      exit 1
    fi
    log "Copying MANO_RIGHT.pkl from $MANO_RIGHT_SOURCE"
    cp "$MANO_RIGHT_SOURCE" "$mano_target"
    return
  fi

  log "MANO_RIGHT.pkl is not installed yet."
  log "Download it from https://mano.is.tue.mpg.de/ after accepting the MANO license,"
  log "then copy it to:"
  log "  $mano_target"
  log "or rerun with:"
  log "  MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl pixi run -e hand-flow-host setup-hand-flow-host"
  if [[ "$REQUIRE_MANO" == "1" ]]; then
    printf 'Missing required MANO model: %s\n' "$mano_target" >&2
    exit 2
  fi
}

verify_imports() {
  if [[ "$INSTALL_HAMER_REQUIREMENTS" != "1" ]]; then
    log "INSTALL_HAMER_REQUIREMENTS=0, skipping import verification."
    return
  fi

  log "Verifying core imports."
  env \
    PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}" \
    PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/novaplan-matplotlib}" \
    PYTHONPATH="$HAMER_DIR" \
    "$HAMER_PYTHON" - <<'PY'
import fastapi
import uvicorn
import torch
import hamer
from vitpose_model import ViTPoseModel
print("Hand-flow service backend imports OK")
PY
}

print_summary() {
  cat <<EOF

Hand-flow service setup summary
-------------------------------
Backend: HaMeR
HAMER_DIR: $HAMER_DIR
Checkpoint slot:
  $HAMER_DIR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
MANO slot:
  $HAMER_DIR/_DATA/data/mano/MANO_RIGHT.pkl

Launch after the checkpoint and MANO file are present:
  export HAMER_DIR=$HAMER_DIR
  pixi run -e hand-flow-host launch-hand-flow-server

EOF
}

clone_hamer
install_hamer_requirements
download_hamer_models
place_mano_model
verify_imports
print_summary
