#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/runtime_paths.sh"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

export PORT="${PORT:-7000}"
export KILL_OLD_PROCESSES="${KILL_OLD_PROCESSES:-1}"
export COMFYUI_HOST="${COMFYUI_HOST:-http://127.0.0.1}"
export COMFYUI_WORKER_START_PORT="${COMFYUI_WORKER_START_PORT:-8188}"
export COMFYUI_NUM_WORKERS="${COMFYUI_NUM_WORKERS:-8}"
export COMFYUI_WORKER_END_PORT="${COMFYUI_WORKER_END_PORT:-$((COMFYUI_WORKER_START_PORT + COMFYUI_NUM_WORKERS))}"
export COMFYUI_DIR="$(resolve_video_comfyui_dir "$REPO_ROOT")"
export COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-$COMFYUI_DIR/input}"
export COMFYUI_OUTPUT_DIR="${COMFYUI_OUTPUT_DIR:-$COMFYUI_DIR/output}"
export WAIT_FOR_WORKERS="${WAIT_FOR_WORKERS:-1}"
export WORKER_READY_TIMEOUT="${WORKER_READY_TIMEOUT:-300}"
export WORKER_READY_INTERVAL="${WORKER_READY_INTERVAL:-2}"

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
  local deadline=$((SECONDS + WORKER_READY_TIMEOUT))

  while (( SECONDS < deadline )); do
    if python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read()' "$url" >/dev/null 2>&1; then
      echo "Worker ready: $url"
      return 0
    fi
    sleep "$WORKER_READY_INTERVAL"
  done

  echo "Timed out waiting for ComfyUI worker: $url" >&2
  return 1
}

verify_worker_nodes() {
  local port="$1"
  local base_url="${COMFYUI_HOST}:${port}"

  echo "Verifying video-generation node registration on $base_url"
  python - "$base_url" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

base_url = sys.argv[1].rstrip("/")
model_ownership_contract = "novaplan-model-ownership-v1"
required = {
    "WanImageToVideo": None,
    "WanFirstLastFrameToVideo": None,
    "Sam3VideoNode": model_ownership_contract,
    "CoTracker3Node": model_ownership_contract,
}
missing = []
incompatible = []
for node_type, description_contract in required.items():
    url = f"{base_url}/object_info/{urllib.parse.quote(node_type)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"Could not query {node_type}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if node_type not in payload:
        missing.append(node_type)
        continue
    description = payload[node_type].get("description", "")
    if description_contract and description_contract not in description:
        incompatible.append(node_type)

if missing:
    print("Missing required ComfyUI node types: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
if incompatible:
    print(
        "Stale or incompatible NovaPlan node implementations: "
        + ", ".join(incompatible)
        + ". Restart workers with launch_main_video_generation.sh to sync them.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

wait_for_workers() {
  if [[ "$WAIT_FOR_WORKERS" != "1" ]]; then
    return
  fi

  local end_port=$((COMFYUI_WORKER_END_PORT - 1))
  echo "Waiting for ComfyUI workers on ports ${COMFYUI_WORKER_START_PORT}-${end_port}..."
  for ((port = COMFYUI_WORKER_START_PORT; port < COMFYUI_WORKER_END_PORT; port++)); do
    wait_for_worker "$port"
    if ! verify_worker_nodes "$port"; then
      echo "Video worker custom-node registration is incomplete." >&2
      echo "Run 'pixi run -e video-host setup-video-host', then restart the video workers." >&2
      return 1
    fi
  done
}

if (( COMFYUI_WORKER_END_PORT - COMFYUI_WORKER_START_PORT != COMFYUI_NUM_WORKERS )); then
  echo "COMFYUI_WORKER_END_PORT must equal START_PORT + COMFYUI_NUM_WORKERS." >&2
  exit 1
fi

kill_video_server_processes() {
  local pids
  pids="$(pgrep -f 'remote_video_generation/video_generation_server.py' || true)"
  if [[ -n "$pids" ]]; then
    echo "Killing existing video-generation server process(es): $pids"
    kill $pids || true
    sleep 2
    local remaining
    remaining="$(pgrep -f 'remote_video_generation/video_generation_server.py' || true)"
    if [[ -n "$remaining" ]]; then
      echo "Force killing remaining video-generation server process(es): $remaining"
      kill -9 $remaining || true
    fi
  fi
}

unset CUDA_VISIBLE_DEVICES

if [[ "$KILL_OLD_PROCESSES" == "1" ]]; then
  kill_video_server_processes
  kill_listening_port "$PORT"
  sleep 1
fi

wait_for_workers

log_path="$REPO_ROOT/logs/video_generation_server.log"
echo "Launching video-generation queue server on port $PORT"
setsid python remote_video_generation/video_generation_server.py > "$log_path" 2>&1 < /dev/null &
server_pid="$!"
disown "$server_pid" || true
echo "Server started. PID: $server_pid. Log: $log_path"
