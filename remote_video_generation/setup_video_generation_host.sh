#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

# One-time setup for the NovaPlan video-generation host.
#
# Primary upstream links:
# - ComfyUI: https://github.com/comfyanonymous/ComfyUI
# - ComfyUI Wan 2.2 guide: https://docs.comfy.org/tutorials/video/wan/wan2_2
# - Wan 2.2 code/model family: https://github.com/Wan-Video/Wan2.2
# - Wan 2.2 I2V model card: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
#
# Recommended runtime layout:
#   COMFYUI_DIR=<NovaPlan checkout>/comfyui (existing deployment), or
#   COMFYUI_DIR=<NovaPlan checkout>/.runtime/comfyui (fresh deployment)
#     main.py
#     input/
#     output/
#     models/
#       diffusion_models/
#       text_encoders/
#       vae/
#       clip_vision/
#     custom_nodes/
#
# Run from the repository root:
#   pixi run -e video-host setup-video-host
#
# Optional model download:
#   huggingface-cli login
#   DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/runtime_paths.sh"

COMFYUI_DIR="$(resolve_video_comfyui_dir "$REPO_ROOT")"
COMFYUI_REPO="${COMFYUI_REPO:-https://github.com/comfyanonymous/ComfyUI.git}"
# The paper runtime vendored ComfyUI 0.3.68. Set COMFYUI_REF='' explicitly to
# follow another already-tested ComfyUI revision.
COMFYUI_REF="${COMFYUI_REF-v0.3.68}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-python}"

INSTALL_COMFYUI="${INSTALL_COMFYUI:-1}"
INSTALL_COMFYUI_REQUIREMENTS="${INSTALL_COMFYUI_REQUIREMENTS:-1}"
DOWNLOAD_WAN_MODELS="${DOWNLOAD_WAN_MODELS:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_SAGE_ATTN="${INSTALL_SAGE_ATTN:-1}"
FLASH_ATTN_PACKAGE="${FLASH_ATTN_PACKAGE:-flash-attn}"
SAGE_ATTN_PACKAGE="${SAGE_ATTN_PACKAGE:-sageattention}"
INSTALL_VIDEO_FLOW_PACKAGES="${INSTALL_VIDEO_FLOW_PACKAGES:-1}"
DOWNLOAD_COTRACKER_MODEL="${DOWNLOAD_COTRACKER_MODEL:-1}"
SAM3_REF="${SAM3_REF:-46957e47805eaa273f4aa7bbbd25a88bca9108ce}"
SAM3_INSTALL_SPEC="${SAM3_INSTALL_SPEC:-git+https://github.com/facebookresearch/sam3.git@${SAM3_REF}#egg=sam3}"
COTRACKER_TORCH_HUB_REPO="${COTRACKER_TORCH_HUB_REPO:-facebookresearch/co-tracker:main}"
COTRACKER_HF_REPO="${COTRACKER_HF_REPO:-facebook/cotracker3}"
VIDEO_CUSTOM_NODES_SOURCE="${VIDEO_CUSTOM_NODES_SOURCE:-$REPO_ROOT/remote_video_generation/custom_nodes/novaplan}"
FLOW_SAM3_NODE_SOURCE="${FLOW_SAM3_NODE_SOURCE:-$REPO_ROOT/remote_object_flow/comfyui/custom_nodes/novaplan/sam3}"

WAN_COMFY_REPO="${WAN_COMFY_REPO:-Comfy-Org/Wan_2.2_ComfyUI_Repackaged}"
WAN21_COMFY_REPO="${WAN21_COMFY_REPO:-Comfy-Org/Wan_2.1_ComfyUI_repackaged}"

log() {
  printf '[setup-video-host] %s\n' "$*"
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
    log "INSTALL_COMFYUI_REQUIREMENTS=0, skipping pip requirements."
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

install_video_flow_packages() {
  if [[ "$INSTALL_VIDEO_FLOW_PACKAGES" != "1" ]]; then
    log "INSTALL_VIDEO_FLOW_PACKAGES=0, skipping SAM3/CoTracker setup."
    return
  fi

  log "Installing video-host 2D flow Python packages into the active Pixi env."
  "$COMFYUI_PYTHON" -m pip install \
    "decord" \
    "einops" \
    "huggingface_hub>=0.23.0" \
    "matplotlib" \
    "opencv-python" \
    "pandas" \
    "pycocotools" \
    "scikit-image" \
    "scikit-learn"

  if "$COMFYUI_PYTHON" -c "import sam3" >/dev/null 2>&1; then
    log "sam3 already installed in the active Pixi env."
  else
    log "Installing SAM3 under Meta's separate SAM License from $SAM3_INSTALL_SPEC"
    "$COMFYUI_PYTHON" -m pip install "$SAM3_INSTALL_SPEC"
  fi

  log "Caching CoTracker torch hub repo: $COTRACKER_TORCH_HUB_REPO"
  "$COMFYUI_PYTHON" - "$COTRACKER_TORCH_HUB_REPO" <<PY
import sys
import torch

repo = sys.argv[1]
torch.hub.load(repo, "cotracker3_offline", pretrained=False, trust_repo=True)
PY
}

create_layout() {
  log "Creating ComfyUI runtime directories under $COMFYUI_DIR"
  mkdir -p \
    "$COMFYUI_DIR/input" \
    "$COMFYUI_DIR/output" \
    "$COMFYUI_DIR/temp" \
    "$COMFYUI_DIR/user" \
    "$COMFYUI_DIR/custom_nodes" \
    "$COMFYUI_DIR/custom_nodes/novaplan/cotracker3" \
    "$COMFYUI_DIR/models/diffusion_models" \
    "$COMFYUI_DIR/models/unet" \
    "$COMFYUI_DIR/models/text_encoders" \
    "$COMFYUI_DIR/models/vae" \
    "$COMFYUI_DIR/models/loras" \
    "$COMFYUI_DIR/models/clip_vision"
}

sync_video_custom_nodes() {
  sync_video_runtime_custom_nodes "$REPO_ROOT" "$COMFYUI_DIR"
}

download_hf_file() {
  local repo_id="$1"
  local repo_file="$2"
  local dest="$3"

  if [[ -s "$dest" ]]; then
    log "Model already present: $dest"
    return
  fi

  mkdir -p "$(dirname "$dest")"
  log "Downloading hf://$repo_id/$repo_file"
  "$COMFYUI_PYTHON" - "$repo_id" "$repo_file" "$dest" <<'PY'
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id, repo_file, dest = sys.argv[1:4]
path = hf_hub_download(repo_id=repo_id, filename=repo_file)
dest_path = Path(dest)
dest_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(path, dest_path)
print(dest_path)
PY
}

download_wan_models() {
  if [[ "$DOWNLOAD_WAN_MODELS" != "1" ]]; then
    log "Skipping Wan 2.2 downloads. Set DOWNLOAD_WAN_MODELS=1 after accepting model licenses."
    return
  fi

  download_hf_file \
    "$WAN_COMFY_REPO" \
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "$COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"

  download_hf_file \
    "$WAN_COMFY_REPO" \
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "$COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

  download_hf_file \
    "$WAN21_COMFY_REPO" \
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$COMFYUI_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

  download_hf_file \
    "$WAN21_COMFY_REPO" \
    "split_files/vae/wan_2.1_vae.safetensors" \
    "$COMFYUI_DIR/models/vae/wan_2.1_vae.safetensors"

  download_hf_file \
    "$WAN_COMFY_REPO" \
    "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
    "$COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"

  download_hf_file \
    "$WAN_COMFY_REPO" \
    "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
    "$COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
}

download_cotracker_model() {
  if [[ "$DOWNLOAD_COTRACKER_MODEL" != "1" ]]; then
    log "Skipping CoTracker3 checkpoint download. Set DOWNLOAD_COTRACKER_MODEL=1 to enable."
    return
  fi

  log "CoTracker3 code and weights are governed by CC BY-NC 4.0."
  log "Downloading the default checkpoint for research/noncommercial use."
  download_hf_file \
    "$COTRACKER_HF_REPO" \
    "scaled_offline.pth" \
    "$COMFYUI_DIR/custom_nodes/novaplan/cotracker3/scaled_offline.pth"
}

print_summary() {
  cat <<EOF

Video-generation host setup complete.

ComfyUI runtime:
  COMFYUI_DIR=$COMFYUI_DIR
  COMFYUI_INPUT_DIR=$COMFYUI_DIR/input
  COMFYUI_OUTPUT_DIR=$COMFYUI_DIR/output

Expected Wan 2.2 model slots:
  $COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  $COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  $COMFYUI_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  $COMFYUI_DIR/models/vae/wan_2.1_vae.safetensors
  $COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
  $COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

Expected 2D flow model slot:
  $COMFYUI_DIR/custom_nodes/novaplan/cotracker3/scaled_offline.pth

Start workers:
  export COMFYUI_DIR=$COMFYUI_DIR
  pixi run -e video-host launch-video-main
  pixi run -e video-host launch-video-server

EOF
}

clone_comfyui
create_layout
sync_video_custom_nodes
install_comfyui_requirements
install_attention_backend
install_video_flow_packages
download_wan_models
download_cotracker_model
print_summary
