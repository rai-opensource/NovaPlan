#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

# Generate the per-step segmentation/depth bundle used by hand-flow grounding.
#
# Run after the remote object-flow server is reachable:
#   NOVAPLAN_OBJECT_FLOW_SERVER_URL=http://127.0.0.1:7001 ./local_planning/run_hand_flow_extraction_sample.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SAMPLE_DIR="${SAMPLE_DIR:-example_data/color_sorting/step_001}"
OBJECT_FLOW_SERVER_URL="${NOVAPLAN_OBJECT_FLOW_SERVER_URL:-http://127.0.0.1:7001}"
HAND_FLOW_MASK_PROMPT="${HAND_FLOW_MASK_PROMPT:-hand}"
HAND_FLOW_RESULTS_DIR="${HAND_FLOW_RESULTS_DIR:-}"
HAND_FLOW_DEPTH_SOURCE_JOB_ID="${HAND_FLOW_DEPTH_SOURCE_JOB_ID:-}"
HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR="${HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR:-}"
HAND_FLOW_LEGACY_MASK_STEM="${HAND_FLOW_LEGACY_MASK_STEM:-color_block_720p_test}"
HAND_FLOW_DEBUG_DEPTH_SUFFIX="${HAND_FLOW_DEBUG_DEPTH_SUFFIX:-depth_step}"
HAND_FLOW_DEBUG_FLOW_SUFFIX="${HAND_FLOW_DEBUG_FLOW_SUFFIX:-red_step}"
HAND_FLOW_DRY_RUN="${HAND_FLOW_DRY_RUN:-0}"
INSTALL_LOCAL_ENV="${INSTALL_LOCAL_ENV:-1}"

log() {
  printf '[run-hand-flow-extraction-sample] %s\n' "$*"
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

resolve_sample_dir() {
  local sample="$REPO_ROOT/$SAMPLE_DIR"
  if [[ "$SAMPLE_DIR" = /* ]]; then
    sample="$SAMPLE_DIR"
  fi
  if [[ ! -d "$sample" ]]; then
    printf 'Missing sample directory: %s\n' "$sample" >&2
    exit 1
  fi
  cd "$sample" && pwd
}

read_depth_job_id() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
depth_job_id = data.get("depth_job_id")
if isinstance(depth_job_id, str) and depth_job_id:
    print(depth_job_id)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

cd "$REPO_ROOT"
require_pixi

SAMPLE_ABS="$(resolve_sample_dir)"
EXAMPLE_DIR="$(dirname "$SAMPLE_ABS")"
STEP="$(basename "$SAMPLE_ABS")"

if [[ -z "$HAND_FLOW_RESULTS_DIR" ]]; then
  HAND_FLOW_RESULTS_DIR="$REPO_ROOT/runs/verification/hand_flow/flow"
elif [[ "$HAND_FLOW_RESULTS_DIR" != /* ]]; then
  HAND_FLOW_RESULTS_DIR="$REPO_ROOT/$HAND_FLOW_RESULTS_DIR"
fi

if [[ -z "$HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR" ]]; then
  HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR="$SAMPLE_ABS/test_results"
elif [[ "$HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR" != /* ]]; then
  HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR="$REPO_ROOT/$HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR"
fi

if [[ "$HAND_FLOW_DEPTH_SOURCE_JOB_ID" == "auto" ]]; then
  job_ids_path="$HAND_FLOW_DEPTH_SOURCE_RESULTS_DIR/flow_job_ids.json"
  if [[ ! -f "$job_ids_path" ]]; then
    printf 'Cannot auto-reuse depth: missing %s\n' "$job_ids_path" >&2
    printf 'Run object-flow extraction with the updated script first, or pass HAND_FLOW_DEPTH_SOURCE_JOB_ID=<id>.\n' >&2
    exit 1
  fi
  HAND_FLOW_DEPTH_SOURCE_JOB_ID="$(read_depth_job_id "$job_ids_path")"
fi

if [[ "$INSTALL_LOCAL_ENV" == "1" ]]; then
  log "Installing Pixi environment: local-planning"
  pixi install -e local-planning
fi

cmd=(
  pixi run -e local-planning python remote_object_flow/tests/test_object_flow_server.py
  --server "$OBJECT_FLOW_SERVER_URL"
  --example_dir "$EXAMPLE_DIR"
  --step "$STEP"
  --mask_prompt "$HAND_FLOW_MASK_PROMPT"
  --output_dir "$HAND_FLOW_RESULTS_DIR"
  --auxiliary_arrays
  --sam3_debug_video
  --no_flow_image
  --legacy_step_layout
  --legacy_mask_stem "$HAND_FLOW_LEGACY_MASK_STEM"
  --legacy_debug_depth_suffix "$HAND_FLOW_DEBUG_DEPTH_SUFFIX"
  --legacy_debug_flow_suffix "$HAND_FLOW_DEBUG_FLOW_SUFFIX"
)

if [[ "$HAND_FLOW_DRY_RUN" == "1" ]]; then
  cmd+=(--dry_run)
fi

if [[ -n "$HAND_FLOW_DEPTH_SOURCE_JOB_ID" ]]; then
  cmd+=(--depth_source_job_id "$HAND_FLOW_DEPTH_SOURCE_JOB_ID")
fi

log "Sample directory: $SAMPLE_ABS"
log "Object-flow server: $OBJECT_FLOW_SERVER_URL"
log "Mask prompt: $HAND_FLOW_MASK_PROMPT"
if [[ -n "$HAND_FLOW_DEPTH_SOURCE_JOB_ID" ]]; then
  log "Depth source job: $HAND_FLOW_DEPTH_SOURCE_JOB_ID"
fi
log "Output directory: $HAND_FLOW_RESULTS_DIR"
log "Running hand-flow extraction."
"${cmd[@]}"

if [[ "$HAND_FLOW_DRY_RUN" != "1" ]]; then
  log "Output files:"
  find "$HAND_FLOW_RESULTS_DIR" -maxdepth 2 -type f | sort | head -20
fi
