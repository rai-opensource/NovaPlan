#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

# End-to-end workstation smoke test for NovaPlan hand-flow extraction.
#
# Run after the hand-flow service is serving /predict:
#   NOVAPLAN_HAND_FLOW_SERVER_URL=http://127.0.0.1:8080/predict ./local_planning/run_hand_flow_client_sample.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SAMPLE_DIR="${SAMPLE_DIR:-example_data/color_sorting/step_001}"
HAND_FLOW_SERVER_URL="${NOVAPLAN_HAND_FLOW_SERVER_URL:-http://127.0.0.1:8080/predict}"
HAND_FLOW_INPUT_FILE="${HAND_FLOW_INPUT_FILE:-}"
HAND_FLOW_OUT_FOLDER="${HAND_FLOW_OUT_FOLDER:-$REPO_ROOT/runs/verification/hand_flow/hamer_outputs}"
INSTALL_LOCAL_HAND_ENV="${INSTALL_LOCAL_HAND_ENV:-1}"
RUN_HAND_CONTRACT_TEST="${RUN_HAND_CONTRACT_TEST:-1}"
HAND_FLOW_ALL_FRAMES="${HAND_FLOW_ALL_FRAMES:-1}"
HAND_FLOW_READY_TIMEOUT="${HAND_FLOW_READY_TIMEOUT:-60}"
HAND_FLOW_READY_INTERVAL="${HAND_FLOW_READY_INTERVAL:-2}"
RESOLVED_SAMPLE_DIR=""
RESOLVED_HAND_FLOW_INPUT_FILE=""

log() {
  printf '[run-hand-flow-client-sample] %s\n' "$*"
}

require_pixi() {
  if ! command -v pixi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Missing required command: pixi

Install Pixi first:
  curl -fsSL https://pixi.sh/install.sh | bash
EOF
    exit 1
  fi
}

check_sample_inputs() {
  local sample="$REPO_ROOT/$SAMPLE_DIR"
  if [[ "$SAMPLE_DIR" = /* ]]; then
    sample="$SAMPLE_DIR"
  fi
  sample="$(cd "$sample" && pwd)"

  if [[ ! -f "$sample/rgb_video_16fps.mp4" ]]; then
    printf 'Missing sample video: %s\n' "$sample/rgb_video_16fps.mp4" >&2
    exit 1
  fi

  RESOLVED_SAMPLE_DIR="$sample"
  RESOLVED_HAND_FLOW_INPUT_FILE="$HAND_FLOW_INPUT_FILE"
  if [[ -n "$RESOLVED_HAND_FLOW_INPUT_FILE" && "$RESOLVED_HAND_FLOW_INPUT_FILE" != /* ]]; then
    RESOLVED_HAND_FLOW_INPUT_FILE="$REPO_ROOT/$RESOLVED_HAND_FLOW_INPUT_FILE"
  fi

  if [[ -z "$RESOLVED_HAND_FLOW_INPUT_FILE" ]]; then
    if [[ -f "$REPO_ROOT/runs/verification/hand_flow/flow/tapip3d_output.npz" ]]; then
      RESOLVED_HAND_FLOW_INPUT_FILE="$REPO_ROOT/runs/verification/hand_flow/flow/tapip3d_output.npz"
    elif [[ -f "$sample/test_results_human_hand/tapip3d_output.npz" ]]; then
      RESOLVED_HAND_FLOW_INPUT_FILE="$sample/test_results_human_hand/tapip3d_output.npz"
    elif [[ -f "$sample/test_results/tapip3d_output.npz" ]]; then
      RESOLVED_HAND_FLOW_INPUT_FILE="$sample/test_results/tapip3d_output.npz"
    else
      RESOLVED_HAND_FLOW_INPUT_FILE="$(
        find "$sample/test_results_human_hand" "$sample/test_results" "$sample" \
          -maxdepth 1 -name 'tapip3d_output_*.npz' -print -quit 2>/dev/null || true
      )"
    fi
  fi

  if [[ -z "$RESOLVED_HAND_FLOW_INPUT_FILE" || ! -f "$RESOLVED_HAND_FLOW_INPUT_FILE" ]]; then
    printf 'Missing flow NPZ for hand-flow motion/reference selection under: %s\n' "$sample" >&2
    printf 'Expected test_results_human_hand/tapip3d_output.npz or test_results/tapip3d_output*.npz.\n' >&2
    printf 'To generate the hand-flow bundle first, run:\n' >&2
    printf '  NOVAPLAN_OBJECT_FLOW_SERVER_URL=http://FLOW_HOST:7001 ./local_planning/run_hand_flow_extraction_sample.sh\n' >&2
    exit 1
  fi
}

health_url_from_predict_url() {
  python - "$HAND_FLOW_SERVER_URL" <<'PY'
import sys
from urllib.parse import urlsplit, urlunsplit

parts = urlsplit(sys.argv[1])
print(urlunsplit((parts.scheme, parts.netloc, "/health", "", "")))
PY
}

wait_for_hand_flow() {
  local health_url="$1"
  local deadline
  deadline=$((SECONDS + HAND_FLOW_READY_TIMEOUT))

  log "Waiting for hand-flow service: $health_url"
  while (( SECONDS < deadline )); do
    if python - "$health_url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        if 200 <= response.status < 500:
            print(response.read().decode("utf-8", errors="replace"))
            raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep "$HAND_FLOW_READY_INTERVAL"
  done

  printf 'Timed out waiting for hand-flow service: %s\n' "$health_url" >&2
  exit 1
}

cd "$REPO_ROOT"
require_pixi
check_sample_inputs

if [[ "$INSTALL_LOCAL_HAND_ENV" == "1" ]]; then
  log "Installing Pixi environment: local-planning-hand"
  pixi install -e local-planning-hand
fi

if [[ "$RUN_HAND_CONTRACT_TEST" == "1" ]]; then
  log "Running no-server hand-client contract test."
  pixi run -e local-planning test-hand-flow-client
fi

wait_for_hand_flow "$(health_url_from_predict_url)"

log "Using sample directory: $RESOLVED_SAMPLE_DIR"
log "Using flow file: $RESOLVED_HAND_FLOW_INPUT_FILE"

cmd=(
  pixi run -e local-planning-hand hand-flow-client
  --sample_dir "$RESOLVED_SAMPLE_DIR"
  --url "$HAND_FLOW_SERVER_URL"
  --flow_file "$RESOLVED_HAND_FLOW_INPUT_FILE"
)
cmd+=(--out_folder "$HAND_FLOW_OUT_FOLDER")
if [[ "$HAND_FLOW_ALL_FRAMES" == "1" ]]; then
  cmd+=(--all_frames)
fi

log "Running the HaMeR backend client."
"${cmd[@]}"

out_dir="$HAND_FLOW_OUT_FOLDER"
log "Output directory: $out_dir"
find "$out_dir" -maxdepth 2 -type f | sort | head -20
