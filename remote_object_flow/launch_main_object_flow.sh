#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/comfyui_worker_lifecycle.sh"

COMFYUI_DIR="${COMFYUI_DIR:-$REPO_ROOT/.runtime/comfyui}"
COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-$COMFYUI_DIR/input}"
COMFYUI_OUTPUT_DIR="${COMFYUI_OUTPUT_DIR:-$COMFYUI_DIR/output}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-python}"
COMFYUI_LISTEN="${COMFYUI_LISTEN:-0.0.0.0}"
COMFYUI_START_PORT="${COMFYUI_START_PORT:-8187}"
COMFYUI_NUM_WORKERS="${COMFYUI_NUM_WORKERS:-1}"
REQUIRE_OBJECT_FLOW_ASSETS="${REQUIRE_OBJECT_FLOW_ASSETS:-1}"
COMFYUI_EXTRA_ARGS="${COMFYUI_EXTRA_ARGS:---disable-auto-launch --enable-cors-header --fast --supports-fp8-compute}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-sage}"
INSTALL_CVD_RUNTIME="${INSTALL_CVD_RUNTIME:-${REQUIRE_CVD_CHECKPOINT:-1}}"
NOVAPLAN_TAPIP3D_DIR="${NOVAPLAN_TAPIP3D_DIR:-$COMFYUI_DIR/custom_nodes/novaplan/.external/TAPIP3D}"
NOVAPLAN_CVD_RUNTIME_DIR="${NOVAPLAN_CVD_RUNTIME_DIR:-$COMFYUI_DIR/custom_nodes/novaplan/.external/mega-sam/cvd_opt}"
CVD_RUNTIME_SOURCE="${CVD_RUNTIME_SOURCE:-}"
CVD_SOURCE_BASE_URL="${CVD_SOURCE_BASE_URL:-https://raw.githubusercontent.com/mega-sam/mega-sam/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/cvd_opt}"
RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-$COMFYUI_DIR/custom_nodes/novaplan/moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth}"
RAFT_CHECKPOINT_SOURCE="${RAFT_CHECKPOINT_SOURCE:-}"
RAFT_MODELS_URL="${RAFT_MODELS_URL:-https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip}"

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


mkdir -p "$REPO_ROOT/logs"

stop_flow_workers() {
  local worker_index=0
  local port
  local end_port
  end_port=$((COMFYUI_START_PORT + COMFYUI_NUM_WORKERS - 1))
  for ((port = COMFYUI_START_PORT; port <= end_port; port++)); do
    stop_comfyui_worker \
      "flow" \
      "$port" \
      "$REPO_ROOT/logs/flow_worker_${worker_index}.pid"
    worker_index=$((worker_index + 1))
  done
}

echo "Stopping existing object-flow processes before launch."
stop_matching_processes \
  "object-flow queue server" \
  '[r]emote_object_flow/object_flow_server.py'
stop_recorded_comfyui_workers \
  "flow" \
  "$REPO_ROOT/logs/flow_worker_*.pid"
stop_flow_workers

if [[ ! -f "$COMFYUI_DIR/main.py" ]]; then
  echo "ComfyUI is not installed at $COMFYUI_DIR" >&2
  echo "Run: pixi run -e object-flow-host setup-object-flow-host" >&2
  exit 1
fi

if [[ "$REQUIRE_OBJECT_FLOW_ASSETS" == "1" ]]; then
  tapip_checkpoint="$NOVAPLAN_TAPIP3D_DIR/checkpoints/tapip3d_final.pth"
  if [[ ! -f "$NOVAPLAN_TAPIP3D_DIR/utils/inference_utils.py" || ! -f "$NOVAPLAN_TAPIP3D_DIR/LICENSE" ]]; then
    echo "Missing official TAPIP3D checkout: $NOVAPLAN_TAPIP3D_DIR" >&2
    echo "Run: pixi run -e object-flow-host setup-object-flow-host" >&2
    exit 1
  fi
  if [[ ! -s "$tapip_checkpoint" ]]; then
    echo "Missing TAPIP3D checkpoint: $tapip_checkpoint" >&2
    echo "Run: DOWNLOAD_OBJECT_FLOW_MODELS=1 pixi run -e object-flow-host setup-object-flow-host" >&2
    exit 1
  fi
fi

export NOVAPLAN_TAPIP3D_DIR

visible_gpu_count="$($COMFYUI_PYTHON -c 'import torch; print(torch.cuda.device_count())')"
if (( visible_gpu_count < COMFYUI_NUM_WORKERS )); then
  echo "Requested $COMFYUI_NUM_WORKERS workers but PyTorch sees only $visible_gpu_count CUDA device(s)." >&2
  exit 1
fi

ensure_optional_cvd_runtime() {
  if [[ "$INSTALL_CVD_RUNTIME" != "1" ]]; then
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

  local checkpoint_command=(
    "$COMFYUI_PYTHON"
    "$SCRIPT_DIR/ensure_raft_checkpoint.py"
    --destination "$RAFT_CHECKPOINT"
    --search-root "$COMFYUI_DIR"
    --archive-url "$RAFT_MODELS_URL"
  )
  if [[ -n "$RAFT_CHECKPOINT_SOURCE" ]]; then
    checkpoint_command+=(--source "$RAFT_CHECKPOINT_SOURCE")
  fi
  "${checkpoint_command[@]}"
}

ensure_optional_cvd_runtime

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NOVAPLAN_CVD_RUNTIME_DIR
export RAFT_CHECKPOINT

echo "Launching object-flow ComfyUI workers from $COMFYUI_DIR"
for ((i = 0; i < COMFYUI_NUM_WORKERS; i++)); do
  port=$((COMFYUI_START_PORT + i))
  log_path="$REPO_ROOT/logs/flow_worker_${i}.log"
  echo "  worker $i: GPU $i, port $port, log $log_path"
  CUDA_VISIBLE_DEVICES="$i" setsid "$COMFYUI_PYTHON" "$COMFYUI_DIR/main.py" \
    --port "$port" \
    --listen "$COMFYUI_LISTEN" \
    --output-directory "$COMFYUI_OUTPUT_DIR" \
    --input-directory "$COMFYUI_INPUT_DIR" \
    $COMFYUI_EXTRA_ARGS > "$log_path" 2>&1 < /dev/null &
  worker_pid="$!"
  record_comfyui_worker_pid "$worker_pid" "$REPO_ROOT/logs/flow_worker_${i}.pid"
  disown "$worker_pid" || true
done

echo "Started $COMFYUI_NUM_WORKERS object-flow worker(s)."
