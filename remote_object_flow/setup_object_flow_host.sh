#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

# One-time setup for the NovaPlan object-flow host.
#
# Primary upstream links:
# - ComfyUI: https://github.com/comfyanonymous/ComfyUI
# - TAPIP3D source: https://github.com/zbw001/TAPIP3D
# - TAPIP3D checkpoint used by the NovaPlan adapter:
#   https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
# - SAM3: https://github.com/facebookresearch/sam3
# - MoGe2 model used by the bundled nodes:
#   https://huggingface.co/Ruicheng/moge-2-vitl-normal
# - Optional CVD RAFT runtime:
#   https://github.com/mega-sam/mega-sam/tree/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/cvd_opt
#
# Recommended runtime layout:
#   COMFYUI_DIR=<NovaPlan checkout>/.runtime/comfyui
#     main.py
#     input/
#     output/
#     saved_results/
#     custom_nodes/
#       novaplan/       # copied from this release
#         .external/
#           TAPIP3D/    # pinned official upstream checkout
#     models/
#
# Run from the repository root:
#   pixi run -e object-flow-host setup-object-flow-host
#
# The required TAPIP3D checkpoint downloads by default. To prepare credentials
# for model libraries that populate Hugging Face caches on first use:
#   huggingface-cli login
# To reuse a pre-provisioned/offline checkpoint instead, set
# DOWNLOAD_OBJECT_FLOW_MODELS=0 and TAPIP3D_CHECKPOINT_URL/asset paths as documented.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COMFYUI_DIR="${COMFYUI_DIR:-$REPO_ROOT/.runtime/comfyui}"
COMFYUI_REPO="${COMFYUI_REPO:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_REF="${COMFYUI_REF-v0.3.68}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-python}"

NOVAPLAN_SOURCE_DIR="${NOVAPLAN_SOURCE_DIR:-$SCRIPT_DIR/comfyui/custom_nodes/novaplan}"
NOVAPLAN_TARGET_DIR="${NOVAPLAN_TARGET_DIR:-$COMFYUI_DIR/custom_nodes/novaplan}"
RESULT_OUTPUT_DIR="${RESULT_OUTPUT_DIR:-$COMFYUI_DIR/saved_results}"

INSTALL_COMFYUI="${INSTALL_COMFYUI:-1}"
INSTALL_COMFYUI_REQUIREMENTS="${INSTALL_COMFYUI_REQUIREMENTS:-1}"
INSTALL_OBJECT_FLOW_PIP_PACKAGES="${INSTALL_OBJECT_FLOW_PIP_PACKAGES:-1}"
INSTALL_TAPIP3D_SOURCE="${INSTALL_TAPIP3D_SOURCE:-1}"
INSTALL_TAPIP3D_POINTOPS="${INSTALL_TAPIP3D_POINTOPS:-auto}"
DOWNLOAD_OBJECT_FLOW_MODELS="${DOWNLOAD_OBJECT_FLOW_MODELS:-1}"
INSTALL_CVD_RUNTIME="${INSTALL_CVD_RUNTIME:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_SAGE_ATTN="${INSTALL_SAGE_ATTN:-1}"
FLASH_ATTN_PACKAGE="${FLASH_ATTN_PACKAGE:-flash-attn}"
SAGE_ATTN_PACKAGE="${SAGE_ATTN_PACKAGE:-sageattention}"

SAM3_REF="${SAM3_REF:-46957e47805eaa273f4aa7bbbd25a88bca9108ce}"
MOGE_REF="${MOGE_REF:-925b8ed835a7a9cdb7578ba15c658a0afc969030}"
SAM3_INSTALL_SPEC="${SAM3_INSTALL_SPEC:-git+https://github.com/facebookresearch/sam3.git@${SAM3_REF}}"
MOGE_INSTALL_SPEC="${MOGE_INSTALL_SPEC:-git+https://github.com/microsoft/MoGe.git@${MOGE_REF}}"
TAPIP3D_REPO="${TAPIP3D_REPO:-https://github.com/zbw001/TAPIP3D.git}"
TAPIP3D_REF="${TAPIP3D_REF:-4cb7e69a1687f67d56ec3e506768f51f2c581b46}"
NOVAPLAN_TAPIP3D_DIR="${NOVAPLAN_TAPIP3D_DIR:-$NOVAPLAN_TARGET_DIR/.external/TAPIP3D}"
TAPIP3D_CHECKPOINT_URL="${TAPIP3D_CHECKPOINT_URL:-https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth}"
RAFT_CHECKPOINT_SOURCE="${RAFT_CHECKPOINT_SOURCE:-}"
RAFT_MODELS_URL="${RAFT_MODELS_URL:-https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip}"
NOVAPLAN_CVD_RUNTIME_DIR="${NOVAPLAN_CVD_RUNTIME_DIR:-$NOVAPLAN_TARGET_DIR/.external/mega-sam/cvd_opt}"
CVD_RUNTIME_SOURCE="${CVD_RUNTIME_SOURCE:-}"
CVD_SOURCE_BASE_URL="${CVD_SOURCE_BASE_URL:-https://raw.githubusercontent.com/mega-sam/mega-sam/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/cvd_opt}"

log() {
  printf '[setup-object-flow-host] %s\n' "$*"
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

clone_comfyui() {
  if [[ -f "$COMFYUI_DIR/main.py" ]]; then
    log "ComfyUI already present: $COMFYUI_DIR"
    if [[ -n "$COMFYUI_REF" && -d "$COMFYUI_DIR/.git" ]]; then
      log "Checking out tested ComfyUI ref: $COMFYUI_REF"
      git -C "$COMFYUI_DIR" checkout "$COMFYUI_REF"
    elif [[ -n "$COMFYUI_REF" ]]; then
      log "Existing ComfyUI is not a git checkout; verify compatibility with $COMFYUI_REF manually."
    fi
    return
  fi

  if [[ "$INSTALL_COMFYUI" != "1" ]]; then
    log "INSTALL_COMFYUI=0, skipping ComfyUI clone."
    return
  fi

  need_cmd git
  if is_nonempty_dir "$COMFYUI_DIR"; then
    printf 'COMFYUI_DIR exists but does not contain main.py: %s\n' "$COMFYUI_DIR" >&2
    printf 'Use an empty COMFYUI_DIR, or install ComfyUI there before rerunning.\n' >&2
    exit 1
  fi

  mkdir -p "$(dirname "$COMFYUI_DIR")"
  log "Cloning ComfyUI into $COMFYUI_DIR"
  git clone "$COMFYUI_REPO" "$COMFYUI_DIR"

  if [[ -n "$COMFYUI_REF" ]]; then
    log "Checking out tested ComfyUI ref: $COMFYUI_REF"
    git -C "$COMFYUI_DIR" checkout "$COMFYUI_REF"
  fi
}

install_comfyui_requirements() {
  if [[ "$INSTALL_COMFYUI_REQUIREMENTS" != "1" ]]; then
    log "INSTALL_COMFYUI_REQUIREMENTS=0, skipping ComfyUI requirements."
    return
  fi

  if [[ ! -f "$COMFYUI_DIR/requirements.txt" ]]; then
    log "No ComfyUI requirements.txt found; skipping pip install."
    return
  fi

  log "Installing ComfyUI Python requirements into the active Pixi env."
  "$COMFYUI_PYTHON" -m pip install -r "$COMFYUI_DIR/requirements.txt"
  "$COMFYUI_PYTHON" -m pip install "huggingface_hub>=0.23.0"
}

install_attention_backend() {
  case "$ATTENTION_BACKEND" in
    sage)
      if [[ "$INSTALL_SAGE_ATTN" != "1" ]]; then
        log "INSTALL_SAGE_ATTN=0, skipping sageattention install."
        return
      fi
      if "$COMFYUI_PYTHON" -c "import sageattention" >/dev/null 2>&1; then
        log "sageattention already installed in the active Pixi env."
        return
      fi
      log "Installing sageattention into the active Pixi env."
      "$COMFYUI_PYTHON" -m pip install "$SAGE_ATTN_PACKAGE"
      "$COMFYUI_PYTHON" -c "import sageattention" >/dev/null
      ;;
    flash)
      if [[ "$INSTALL_FLASH_ATTN" != "1" ]]; then
        log "INSTALL_FLASH_ATTN=0, skipping flash-attn install."
        return
      fi
      if "$COMFYUI_PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
        log "flash-attn already installed in the active Pixi env."
        return
      fi
      log "Installing flash-attn into the active Pixi env. This can take several minutes."
      "$COMFYUI_PYTHON" -m pip install "$FLASH_ATTN_PACKAGE" --no-build-isolation
      "$COMFYUI_PYTHON" -c "import flash_attn; print(flash_attn.__version__)" >/dev/null
      ;;
    pytorch|none)
      log "ATTENTION_BACKEND=$ATTENTION_BACKEND, no extra attention package install needed."
      ;;
    *)
      printf "Unsupported ATTENTION_BACKEND: %s\nUse one of: sage, flash, pytorch, none.\n" "$ATTENTION_BACKEND" >&2
      exit 1
      ;;
  esac
}

create_layout() {
  log "Creating ComfyUI runtime directories under $COMFYUI_DIR"
  mkdir -p \
    "$COMFYUI_DIR/input" \
    "$COMFYUI_DIR/output" \
    "$COMFYUI_DIR/temp" \
    "$COMFYUI_DIR/user" \
    "$RESULT_OUTPUT_DIR" \
    "$COMFYUI_DIR/custom_nodes" \
    "$NOVAPLAN_TARGET_DIR/.external" \
    "$COMFYUI_DIR/models/checkpoints" \
    "$COMFYUI_DIR/models/sam3"
}

clone_tapip3d() {
  local required_file="$NOVAPLAN_TAPIP3D_DIR/utils/inference_utils.py"
  local expected_revision="$TAPIP3D_REF"

  if [[ "$INSTALL_TAPIP3D_SOURCE" != "1" ]]; then
    if [[ ! -f "$required_file" ]]; then
      printf 'INSTALL_TAPIP3D_SOURCE=0 but no official TAPIP3D checkout was found at %s\n' "$NOVAPLAN_TAPIP3D_DIR" >&2
      exit 1
    fi
    log "Using operator-provided TAPIP3D checkout: $NOVAPLAN_TAPIP3D_DIR"
    return
  fi

  need_cmd git
  if [[ -d "$NOVAPLAN_TAPIP3D_DIR/.git" ]]; then
    local current_revision
    current_revision="$(git -C "$NOVAPLAN_TAPIP3D_DIR" rev-parse HEAD)"
    if [[ "$current_revision" != "$expected_revision" ]]; then
      log "Updating official TAPIP3D checkout to pinned revision $expected_revision"
      git -C "$NOVAPLAN_TAPIP3D_DIR" fetch --depth 1 origin "$expected_revision"
      git -C "$NOVAPLAN_TAPIP3D_DIR" checkout --detach "$expected_revision"
    else
      log "Official TAPIP3D checkout already pinned at $expected_revision"
    fi
  else
    if is_nonempty_dir "$NOVAPLAN_TAPIP3D_DIR"; then
      printf 'TAPIP3D target exists but is not a git checkout: %s\n' "$NOVAPLAN_TAPIP3D_DIR" >&2
      printf 'Move it aside or set NOVAPLAN_TAPIP3D_DIR to an empty location.\n' >&2
      exit 1
    fi

    local clone_tmp="${NOVAPLAN_TAPIP3D_DIR}.clone_tmp.$$"
    if [[ -e "$clone_tmp" ]]; then
      printf 'Temporary TAPIP3D clone path already exists: %s\n' "$clone_tmp" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$NOVAPLAN_TAPIP3D_DIR")"
    log "Cloning official TAPIP3D source into $NOVAPLAN_TAPIP3D_DIR"
    git clone "$TAPIP3D_REPO" "$clone_tmp"
    git -C "$clone_tmp" checkout --detach "$expected_revision"
    mv "$clone_tmp" "$NOVAPLAN_TAPIP3D_DIR"
  fi

  if [[ ! -f "$required_file" || ! -f "$NOVAPLAN_TAPIP3D_DIR/LICENSE" ]]; then
    printf 'Pinned TAPIP3D checkout is incomplete: %s\n' "$NOVAPLAN_TAPIP3D_DIR" >&2
    exit 1
  fi
}

remove_legacy_tapip3d_bundle() {
  local legacy_dir="$NOVAPLAN_TARGET_DIR/TAPIP3D"
  if [[ ! -d "$legacy_dir" || "$legacy_dir" == "$NOVAPLAN_TAPIP3D_DIR" ]]; then
    return
  fi

  local legacy_checkpoint="$legacy_dir/checkpoints/tapip3d_final.pth"
  local checkpoint="$NOVAPLAN_TAPIP3D_DIR/checkpoints/tapip3d_final.pth"
  if [[ -s "$legacy_checkpoint" && ! -s "$checkpoint" ]]; then
    mkdir -p "$(dirname "$checkpoint")"
    cp "$legacy_checkpoint" "$checkpoint"
    log "Migrated the existing TAPIP3D checkpoint from $legacy_checkpoint"
  fi

  log "Removing the obsolete bundled TAPIP3D deployment at $legacy_dir"
  rm -rf "$legacy_dir"
}

copy_novaplan_nodes() {
  if [[ ! -d "$NOVAPLAN_SOURCE_DIR" ]]; then
    printf 'Bundled NovaPlan custom nodes not found: %s\n' "$NOVAPLAN_SOURCE_DIR" >&2
    exit 1
  fi

  log "Copying NovaPlan custom nodes to $NOVAPLAN_TARGET_DIR"
  mkdir -p "$NOVAPLAN_TARGET_DIR"
  local new_raft="$NOVAPLAN_TARGET_DIR/moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth"

  if [[ ! -s "$new_raft" ]]; then
    local discovered_raft
    discovered_raft="$(find "$COMFYUI_DIR" -type f -name raft-things.pth -print -quit 2>/dev/null || true)"
    if [[ -n "$discovered_raft" && "$discovered_raft" != "$new_raft" ]]; then
      mkdir -p "$(dirname "$new_raft")"
      cp "$discovered_raft" "$new_raft"
      log "Migrated the discovered RAFT checkpoint from $discovered_raft"
    fi
  fi
  remove_legacy_tapip3d_bundle
  if [[ "$NOVAPLAN_SOURCE_DIR" != "$NOVAPLAN_TARGET_DIR" ]]; then
    rm -rf "$NOVAPLAN_TARGET_DIR/tapip3d_adapter"
  fi
  cp -R "$NOVAPLAN_SOURCE_DIR/." "$NOVAPLAN_TARGET_DIR/"

  mkdir -p \
    "$NOVAPLAN_TAPIP3D_DIR/checkpoints" \
    "$NOVAPLAN_TARGET_DIR/moge2_metric_depth/comfyui_node/checkpoints" \
    "$NOVAPLAN_TARGET_DIR/demo_data"
  printf '%s\n' "$NOVAPLAN_TAPIP3D_DIR" > "$NOVAPLAN_TARGET_DIR/tapip3d_adapter/tapip3d_root.txt"
}

install_object_flow_python_packages() {
  if [[ "$INSTALL_OBJECT_FLOW_PIP_PACKAGES" != "1" ]]; then
    log "INSTALL_OBJECT_FLOW_PIP_PACKAGES=0, skipping object-flow-specific pip packages."
    return
  fi

  log "Installing object-flow Python packages into the active Pixi env."
  "$COMFYUI_PYTHON" -m pip install \
    "numpy>=1.26,<2" \
    "accelerate" \
    "addict" \
    "av" \
    "einops" \
    "e3nn" \
    "evo" \
    "huggingface_hub>=0.23.0" \
    "imageio" \
    "hydra-core" \
    "kornia" \
    "matplotlib" \
    "moviepy==1.0.3" \
    "omegaconf" \
    "opencv-python>=4.8,<5" \
    "pandas" \
    "pycocotools" \
    "scikit-image" \
    "scikit-learn" \
    "python-box[all]~=7.0" \
    "plyfile>=1.0,<1.1.4" \
    "pillow_heif" \
    "pycolmap" \
    "rich" \
    "scipy" \
    "safetensors" \
    "timm" \
    "tqdm" \
    "trimesh" \
    "transformers" \
    "typed-argument-parser"

  log "Installing SAM3 under Meta's separate SAM License from $SAM3_INSTALL_SPEC"
  "$COMFYUI_PYTHON" -m pip install "$SAM3_INSTALL_SPEC"

  log "Installing MoGe from $MOGE_INSTALL_SPEC"
  "$COMFYUI_PYTHON" -m pip install "$MOGE_INSTALL_SPEC"

  # SAM3 requires NumPy 1.x. Reassert the compatible set after VCS installs so
  # pip cannot silently leave the shared runtime with mutually incompatible
  # NumPy, OpenCV, and Plyfile versions.
  "$COMFYUI_PYTHON" -m pip install \
    "numpy>=1.26,<2" \
    "opencv-python>=4.8,<5" \
    "plyfile>=1.0,<1.1.4"
  "$COMFYUI_PYTHON" -m pip check
}

install_tapip3d_pointops() {
  local source_dir="$NOVAPLAN_TAPIP3D_DIR/third_party/pointops2"
  case "$INSTALL_TAPIP3D_POINTOPS" in
    auto|0|1) ;;
    *)
      printf 'Unsupported INSTALL_TAPIP3D_POINTOPS value: %s\nUse one of: auto, 0, 1.\n' \
        "$INSTALL_TAPIP3D_POINTOPS" >&2
      exit 1
      ;;
  esac

  if [[ "$INSTALL_TAPIP3D_POINTOPS" == "0" ]]; then
    log "INSTALL_TAPIP3D_POINTOPS=0; using NovaPlan's slower PyTorch KNN adapter."
    return
  fi
  if "$COMFYUI_PYTHON" -c \
    "import pointops2, pointops2_cuda; assert callable(getattr(pointops2_cuda, 'knnquery_cuda', None))" \
    >/dev/null 2>&1; then
    log "TAPIP3D pointops2 CUDA extension is already installed."
    return
  fi

  local compatibility
  if compatibility="$("$COMFYUI_PYTHON" "$SCRIPT_DIR/check_pointops_cuda.py")"; then
    log "Pointops2 build check: $compatibility"
  else
    if [[ "$INSTALL_TAPIP3D_POINTOPS" == "1" ]]; then
      printf 'Cannot build required TAPIP3D pointops2: %s\n' "$compatibility" >&2
      printf 'Install an nvcc toolkit with the same CUDA major version as PyTorch, or use auto/0 for the fallback.\n' >&2
      exit 1
    fi
    log "Skipping pointops2 build: $compatibility"
    log "Using NovaPlan's slower PyTorch KNN adapter."
    return
  fi

  if [[ ! -f "$source_dir/setup.py" ]]; then
    printf 'Missing pointops2 source in the official TAPIP3D checkout: %s\n' "$source_dir" >&2
    exit 1
  fi

  log "Building TAPIP3D pointops2 CUDA extension."
  if ! (
    cd "$source_dir"
    "$COMFYUI_PYTHON" -m pip install -v --no-build-isolation .
  ); then
    if [[ "$INSTALL_TAPIP3D_POINTOPS" == "1" ]]; then
      printf 'Required TAPIP3D pointops2 build failed. Check the PyTorch/CUDA toolkit versions above.\n' >&2
      exit 1
    fi
    log "pointops2 build failed in auto mode; using NovaPlan's PyTorch KNN adapter."
    return
  fi
  if ! "$COMFYUI_PYTHON" -c \
    "import pointops2, pointops2_cuda; assert callable(getattr(pointops2_cuda, 'knnquery_cuda', None))" \
    >/dev/null 2>&1; then
    if [[ "$INSTALL_TAPIP3D_POINTOPS" == "1" ]]; then
      printf 'pointops2 installed but pointops2_cuda could not be imported.\n' >&2
      exit 1
    fi
    log "pointops2 CUDA import failed in auto mode; using NovaPlan's PyTorch KNN adapter."
  fi
}

download_url() {
  local url="$1"
  local dest="$2"

  if [[ -s "$dest" ]]; then
    log "File already present: $dest"
    return
  fi

  mkdir -p "$(dirname "$dest")"
  log "Downloading $url"
  "$COMFYUI_PYTHON" - "$url" "$dest" <<'PY'
import sys
from pathlib import Path
from urllib.request import urlopen

url, dest = sys.argv[1:3]
dest_path = Path(dest)
dest_path.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url) as response, dest_path.open("wb") as out:
    out.write(response.read())
print(dest_path)
PY
}

download_object_flow_models() {
  if [[ "$DOWNLOAD_OBJECT_FLOW_MODELS" != "1" ]]; then
    log "Skipping the required TAPIP3D checkpoint download because DOWNLOAD_OBJECT_FLOW_MODELS=$DOWNLOAD_OBJECT_FLOW_MODELS."
    return
  fi

  download_url \
    "$TAPIP3D_CHECKPOINT_URL" \
    "$NOVAPLAN_TAPIP3D_DIR/checkpoints/tapip3d_final.pth"

  log "SAM3 and MoGe2 are loaded by their libraries."
  log "The first model call may download from Hugging Face into HF_HOME/cache."
}

provision_cvd_runtime() {
  if [[ "$INSTALL_CVD_RUNTIME" != "1" ]]; then
    log "INSTALL_CVD_RUNTIME=0, skipping optional CVD runtime provisioning."
    return
  fi

  local runtime_command=(
    "$COMFYUI_PYTHON"
    "$SCRIPT_DIR/ensure_cvd_runtime.py"
    --destination "$NOVAPLAN_CVD_RUNTIME_DIR"
    --search-root "$COMFYUI_DIR"
    --source-base-url "$CVD_SOURCE_BASE_URL"
  )
  if [[ -n "$CVD_RUNTIME_SOURCE" ]]; then
    runtime_command+=(--source "$CVD_RUNTIME_SOURCE")
  fi
  "${runtime_command[@]}"

  local dest="$NOVAPLAN_TARGET_DIR/moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth"
  local checkpoint_command=(
    "$COMFYUI_PYTHON"
    "$SCRIPT_DIR/ensure_raft_checkpoint.py"
    --destination "$dest"
    --search-root "$COMFYUI_DIR"
    --archive-url "$RAFT_MODELS_URL"
  )
  if [[ -n "$RAFT_CHECKPOINT_SOURCE" ]]; then
    checkpoint_command+=(--source "$RAFT_CHECKPOINT_SOURCE")
  fi
  "${checkpoint_command[@]}"
}

verify_required_assets() {
  if [[ ! -f "$NOVAPLAN_TAPIP3D_DIR/utils/inference_utils.py" || ! -f "$NOVAPLAN_TAPIP3D_DIR/LICENSE" ]]; then
    printf 'Official TAPIP3D checkout is missing or incomplete: %s\n' "$NOVAPLAN_TAPIP3D_DIR" >&2
    exit 1
  fi

  if [[ "$INSTALL_CVD_RUNTIME" != "1" ]]; then
    return
  fi

  local raft_source="$NOVAPLAN_CVD_RUNTIME_DIR/core/raft.py"
  local raft_checkpoint="$NOVAPLAN_TARGET_DIR/moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth"
  local cvd_license="$NOVAPLAN_CVD_RUNTIME_DIR/LICENSE"
  local raft_license="$NOVAPLAN_CVD_RUNTIME_DIR/RAFT_LICENSE"
  if [[ ! -s "$raft_source" || ! -s "$raft_checkpoint" || ! -s "$cvd_license" || ! -s "$raft_license" ]]; then
    cat >&2 <<EOF
Could not provision the optional CVD runtime:
  source:     $raft_source
  checkpoint: $raft_checkpoint
  license:    $cvd_license
  RAFT license: $raft_license

Set INSTALL_CVD_RUNTIME=0 only when every request will use enable_cvd=false.
EOF
    exit 1
  fi
}

verify_tapip3d_adapter() {
  local adapter_path="$NOVAPLAN_TARGET_DIR/tapip3d_adapter/upstream.py"
  if [[ ! -f "$adapter_path" ]]; then
    printf 'TAPIP3D adapter is missing: %s\n' "$adapter_path" >&2
    exit 1
  fi

  log "Verifying the deployed TAPIP3D adapter."
  local verification_command=(
    "$COMFYUI_PYTHON"
    "$SCRIPT_DIR/verify_tapip3d_adapter.py"
    --adapter "$adapter_path"
    --tapip3d-root "$NOVAPLAN_TAPIP3D_DIR"
  )
  if [[ "$INSTALL_TAPIP3D_POINTOPS" == "1" ]]; then
    verification_command+=(--require-pointops-cuda)
  fi
  "${verification_command[@]}"
}

print_summary() {
  cat <<EOF

Object-flow host setup complete.

ComfyUI runtime:
  COMFYUI_DIR=$COMFYUI_DIR
  COMFYUI_INPUT_DIR=$COMFYUI_DIR/input
  COMFYUI_OUTPUT_DIR=$COMFYUI_DIR/output
  RESULT_OUTPUT_DIR=$RESULT_OUTPUT_DIR

NovaPlan custom nodes:
  $NOVAPLAN_TARGET_DIR

Important checkpoint slots:
  TAPIP3D:
    source:     $NOVAPLAN_TAPIP3D_DIR
    revision:   $TAPIP3D_REF
    checkpoint: $NOVAPLAN_TAPIP3D_DIR/checkpoints/tapip3d_final.pth
  RAFT for the default CVD path:
    $NOVAPLAN_TARGET_DIR/moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth
  Minimal CVD source runtime:
    $NOVAPLAN_CVD_RUNTIME_DIR

Model links to review before redistribution:
  SAM3: https://github.com/facebookresearch/sam3
  TAPIP3D: https://tapip3d.github.io/
  MoGe2: https://huggingface.co/Ruicheng/moge-2-vitl-normal

Start workers:
  export COMFYUI_DIR=$COMFYUI_DIR
  export RESULT_OUTPUT_DIR=$RESULT_OUTPUT_DIR
  pixi run -e object-flow-host launch-object-flow-main
  pixi run -e object-flow-host launch-object-flow-server

EOF
}

clone_comfyui
create_layout
clone_tapip3d
copy_novaplan_nodes
install_comfyui_requirements
install_attention_backend
install_object_flow_python_packages
install_tapip3d_pointops
download_object_flow_models
provision_cvd_runtime
verify_required_assets
verify_tapip3d_adapter
print_summary
