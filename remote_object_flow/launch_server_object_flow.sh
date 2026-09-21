#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

export PORT="${PORT:-7001}"
export KILL_OLD_PROCESSES="${KILL_OLD_PROCESSES:-1}"
export COMFYUI_HOST="${COMFYUI_HOST:-http://127.0.0.1}"
export COMFYUI_START_PORT="${COMFYUI_START_PORT:-8187}"
export COMFYUI_MASTER_PORT="${COMFYUI_MASTER_PORT:-$COMFYUI_START_PORT}"
export COMFYUI_NUM_WORKERS="${COMFYUI_NUM_WORKERS:-1}"
export COMFYUI_DIR="${COMFYUI_DIR:-$REPO_ROOT/.runtime/comfyui}"
export COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-$COMFYUI_DIR/input}"
export COMFYUI_OUTPUT_DIR="${COMFYUI_OUTPUT_DIR:-$COMFYUI_DIR/output}"
export RESULT_OUTPUT_DIR="${RESULT_OUTPUT_DIR:-$COMFYUI_DIR/saved_results}"
export NOVAPLAN_DEMO_ROOT="${NOVAPLAN_DEMO_ROOT:-$COMFYUI_DIR/custom_nodes/novaplan/demo_data}"
export FLOW_SWITCH_THETA_DEG="${FLOW_SWITCH_THETA_DEG:-45}"
export MODEL_PINNING_ENABLED="${MODEL_PINNING_ENABLED:-1}"
export WAIT_FOR_WORKERS="${WAIT_FOR_WORKERS:-1}"
export WORKER_READY_TIMEOUT="${WORKER_READY_TIMEOUT:-300}"
export WORKER_READY_INTERVAL="${WORKER_READY_INTERVAL:-2}"
export INSTALL_CVD_RUNTIME="${INSTALL_CVD_RUNTIME:-${REQUIRE_CVD_CHECKPOINT:-1}}"
export RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-$COMFYUI_DIR/custom_nodes/novaplan/moge2_metric_depth/comfyui_node/checkpoints/raft-things.pth}"
export RAFT_CHECKPOINT_SOURCE="${RAFT_CHECKPOINT_SOURCE:-}"
export RAFT_MODELS_URL="${RAFT_MODELS_URL:-https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip}"
export NOVAPLAN_CVD_RUNTIME_DIR="${NOVAPLAN_CVD_RUNTIME_DIR:-$COMFYUI_DIR/custom_nodes/novaplan/.external/mega-sam/cvd_opt}"
export CVD_RUNTIME_SOURCE="${CVD_RUNTIME_SOURCE:-}"
export CVD_SOURCE_BASE_URL="${CVD_SOURCE_BASE_URL:-https://raw.githubusercontent.com/mega-sam/mega-sam/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/cvd_opt}"

kill_listening_port() {
  local port="$1"
  local pids
  pids="$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n "s/.*pid=\([0-9]\+\).*/\1/p" | sort -u)"
  if [[ -n "$pids" ]]; then
    echo "Killing process(es) listening on port $port: $pids"
    kill $pids || true
  fi
}

wait_for_worker() {
  local port="$1"
  local url="${COMFYUI_HOST}:${port}/system_stats"
  local deadline
  deadline=$((SECONDS + WORKER_READY_TIMEOUT))

  echo "Waiting for ComfyUI worker on $url"
  while (( SECONDS < deadline )); do
    if python - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        if 200 <= response.status < 500:
            raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
    then
      echo "ComfyUI worker is ready on port $port"
      return 0
    fi
    sleep "$WORKER_READY_INTERVAL"
  done

  echo "Timed out waiting for ComfyUI worker on port $port" >&2
  return 1
}

verify_worker_nodes() {
  local port="$1"
  local base_url="${COMFYUI_HOST}:${port}"
  local worker_index=$((port - COMFYUI_START_PORT))
  local worker_log="$REPO_ROOT/logs/flow_worker_${worker_index}.log"

  echo "Verifying NovaPlan custom-node registration on $base_url"
  if ! python - "$base_url" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

base_url = sys.argv[1].rstrip("/")
required = (
    "MoGe2CVDMetricDepthNode",
    "CacheRawDepthNode",
    "LoadCachedRawDepthNode",
    "IntrinsicsNode",
    "Sam3VideoNode",
    "TAPIP3DNode",
)
missing = []
for node_type in required:
    url = f"{base_url}/object_info/{urllib.parse.quote(node_type)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"Could not query {node_type}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if node_type not in payload:
        missing.append(node_type)

if missing:
    print("Missing required ComfyUI node types: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  then
    echo "Object-flow worker custom-node registration is incomplete." >&2
    if [[ -f "$worker_log" ]]; then
      echo "Recent worker log ($worker_log):" >&2
      tail -n 80 "$worker_log" >&2
    fi
    echo "Run 'pixi run -e object-flow-host setup-object-flow-host', then restart the object-flow worker." >&2
    return 1
  fi
}

wait_for_workers() {
  if [[ "$WAIT_FOR_WORKERS" != "1" ]]; then
    echo "Skipping ComfyUI worker readiness wait."
    return 0
  fi

  local port
  local end_port
  end_port=$((COMFYUI_START_PORT + COMFYUI_NUM_WORKERS - 1))
  for ((port = COMFYUI_START_PORT; port <= end_port; port++)); do
    wait_for_worker "$port"
    verify_worker_nodes "$port"
  done
}

ensure_required_assets() {
  if [[ "$INSTALL_CVD_RUNTIME" != "1" ]]; then
    return
  fi

  local runtime_command=(
    python
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
    python
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

ensure_required_assets
unset CUDA_VISIBLE_DEVICES

if [[ "$KILL_OLD_PROCESSES" == "1" ]]; then
  pkill -f "remote_object_flow/object_flow_server.py" || true
  kill_listening_port "$PORT"
  sleep 1
fi

wait_for_workers

log_path="$REPO_ROOT/logs/object_flow_server.log"
echo "Launching object-flow server on port $PORT"
setsid python remote_object_flow/object_flow_server.py > "$log_path" 2>&1 < /dev/null &
server_pid="$!"
disown "$server_pid" || true
echo "Server started. PID: $server_pid. Log: $log_path"
