#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/runtime_paths.sh"
source "$REPO_ROOT/scripts/comfyui_worker_lifecycle.sh"

COMFYUI_DIR="$(resolve_video_comfyui_dir "$REPO_ROOT")"
COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-$COMFYUI_DIR/input}"
COMFYUI_OUTPUT_DIR="${COMFYUI_OUTPUT_DIR:-$COMFYUI_DIR/output}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-python}"
COMFYUI_LISTEN="${COMFYUI_LISTEN:-0.0.0.0}"
COMFYUI_WORKER_START_PORT="${COMFYUI_WORKER_START_PORT:-8188}"
COMFYUI_NUM_WORKERS="${COMFYUI_NUM_WORKERS:-8}"
REQUIRE_VIDEO_ASSETS="${REQUIRE_VIDEO_ASSETS:-1}"
COMFYUI_EXTRA_ARGS="${COMFYUI_EXTRA_ARGS:---disable-auto-launch --enable-cors-header --fast --supports-fp8-compute}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage}"
VIDEO_VRAM_MODE="${VIDEO_VRAM_MODE:-managed}"

case "$ATTENTION_BACKEND" in
  sage)
    COMFYUI_EXTRA_ARGS="$COMFYUI_EXTRA_ARGS --use-sage-attention"
    ;;
  flash)
    COMFYUI_EXTRA_ARGS="$COMFYUI_EXTRA_ARGS --use-flash-attention"
    ;;
  pytorch)
    COMFYUI_EXTRA_ARGS="$COMFYUI_EXTRA_ARGS --use-pytorch-cross-attention"
    ;;
  none)
    ;;
  *)
    printf "Unsupported ATTENTION_BACKEND: %s\nUse one of: sage, flash, pytorch, none.\n" "$ATTENTION_BACKEND" >&2
    exit 1
    ;;
esac

case "$VIDEO_VRAM_MODE" in
  highvram)
    COMFYUI_EXTRA_ARGS="$COMFYUI_EXTRA_ARGS --highvram"
    ;;
  managed)
    ;;
  *)
    printf 'Unsupported VIDEO_VRAM_MODE: %s\nUse one of: highvram, managed.\n' "$VIDEO_VRAM_MODE" >&2
    exit 1
    ;;
esac

mkdir -p "$REPO_ROOT/logs"

echo "Synchronizing NovaPlan custom nodes into $COMFYUI_DIR before launch."
sync_video_runtime_custom_nodes "$REPO_ROOT" "$COMFYUI_DIR"

echo "Stopping existing video-generation processes before launch."
stop_matching_processes \
  "video-generation queue server" \
  '[r]emote_video_generation/video_generation_server.py'
stop_recorded_comfyui_workers \
  "video" \
  "$REPO_ROOT/logs/video_worker_*.pid"
for ((i = 0; i < COMFYUI_NUM_WORKERS; i++)); do
  port=$((COMFYUI_WORKER_START_PORT + i))
  stop_comfyui_worker \
    "video" \
    "$port" \
    "$REPO_ROOT/logs/video_worker_${i}.pid"
done

if [[ ! -f "$COMFYUI_DIR/main.py" ]]; then
  echo "ComfyUI is not installed at $COMFYUI_DIR" >&2
  echo "Run: DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host" >&2
  exit 1
fi

if [[ "$REQUIRE_VIDEO_ASSETS" == "1" ]]; then
  required_assets=(
    "$COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    "$COMFYUI_DIR/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    "$COMFYUI_DIR/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "$COMFYUI_DIR/models/vae/wan_2.1_vae.safetensors"
    "$COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    "$COMFYUI_DIR/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    "$COMFYUI_DIR/custom_nodes/novaplan/cotracker3/scaled_offline.pth"
  )
  missing_assets=()
  for asset in "${required_assets[@]}"; do
    [[ -s "$asset" ]] || missing_assets+=("$asset")
  done
  if (( ${#missing_assets[@]} > 0 )); then
    printf 'Missing required video-host asset(s):\n' >&2
    printf '  %s\n' "${missing_assets[@]}" >&2
    echo "Run: DOWNLOAD_WAN_MODELS=1 pixi run -e video-host setup-video-host" >&2
    exit 1
  fi
fi

visible_gpu_count="$($COMFYUI_PYTHON -c 'import torch; print(torch.cuda.device_count())')"
if (( visible_gpu_count < COMFYUI_NUM_WORKERS )); then
  echo "Requested $COMFYUI_NUM_WORKERS workers but PyTorch sees only $visible_gpu_count CUDA device(s)." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launching video-generation ComfyUI workers from $COMFYUI_DIR (VRAM mode: $VIDEO_VRAM_MODE)"
for ((i = 0; i < COMFYUI_NUM_WORKERS; i++)); do
  port=$((COMFYUI_WORKER_START_PORT + i))
  log_path="$REPO_ROOT/logs/video_worker_${i}.log"
  echo "  worker $i: GPU $i, port $port, log $log_path"
  CUDA_VISIBLE_DEVICES="$i" setsid "$COMFYUI_PYTHON" "$COMFYUI_DIR/main.py" \
    --port "$port" \
    --listen "$COMFYUI_LISTEN" \
    --output-directory "$COMFYUI_OUTPUT_DIR" \
    --input-directory "$COMFYUI_INPUT_DIR" \
    --database-url "sqlite:///$COMFYUI_DIR/user/comfyui_worker_${i}.db" \
    $COMFYUI_EXTRA_ARGS > "$log_path" 2>&1 < /dev/null &
  worker_pid="$!"
  record_comfyui_worker_pid "$worker_pid" "$REPO_ROOT/logs/video_worker_${i}.pid"
  disown "$worker_pid" || true
done

echo "Started $COMFYUI_NUM_WORKERS video-generation worker(s)."
