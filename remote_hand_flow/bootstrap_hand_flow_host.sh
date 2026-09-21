#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

set -euo pipefail

# End-to-end NovaPlan hand-flow service bootstrap.
#
# Run this on the workstation or GPU host that will serve hand flow:
#   MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl ./remote_hand_flow/bootstrap_hand_flow_host.sh
#
# The script installs the Pixi hand-flow-host env, clones/installs the upstream
# HaMeR backend, downloads its public checkpoint bundle, copies MANO_RIGHT.pkl
# when MANO_RIGHT_SOURCE is provided, launches the NovaPlan service, and checks
# /health. MANO_RIGHT.pkl is license-gated and cannot be downloaded by script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORIGINAL_CWD="$(pwd)"

DEFAULT_HAMER_DIR="$SCRIPT_DIR/.external/hamer"
HAMER_DIR="${HAMER_DIR:-$DEFAULT_HAMER_DIR}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
HAND_FLOW_SERVER_HEALTH_URL="${HAND_FLOW_SERVER_HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
DOWNLOAD_HAMER_MODELS="${DOWNLOAD_HAMER_MODELS:-1}"
START_HAND_FLOW_SERVER="${START_HAND_FLOW_SERVER:-1}"
KILL_OLD_PROCESSES="${KILL_OLD_PROCESSES:-1}"
RUN_HAND_FLOW_HEALTH_CHECK="${RUN_HAND_FLOW_HEALTH_CHECK:-1}"
HAND_FLOW_READY_TIMEOUT="${HAND_FLOW_READY_TIMEOUT:-180}"
HAND_FLOW_READY_INTERVAL="${HAND_FLOW_READY_INTERVAL:-2}"
PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/novaplan-matplotlib}"

if [[ -n "${MANO_RIGHT_SOURCE:-}" && "$MANO_RIGHT_SOURCE" != /* ]]; then
  MANO_RIGHT_SOURCE="$ORIGINAL_CWD/$MANO_RIGHT_SOURCE"
  export MANO_RIGHT_SOURCE
fi

log() {
  printf '[bootstrap-hand-flow-host] %s\n' "$*"
}

require_pixi() {
  if ! command -v pixi >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Missing required command: pixi

Install Pixi first:
  curl -fsSL https://pixi.sh/install.sh | bash

Then restart the shell or load the shell hook printed by the installer.
EOF
    exit 1
  fi
}

check_mano_input() {
  local mano_target="$HAMER_DIR/_DATA/data/mano/MANO_RIGHT.pkl"
  if [[ -s "$mano_target" ]]; then
    log "MANO_RIGHT.pkl already present: $mano_target"
    return
  fi

  if [[ -n "${MANO_RIGHT_SOURCE:-}" && -f "$MANO_RIGHT_SOURCE" ]]; then
    log "Will copy MANO_RIGHT.pkl from MANO_RIGHT_SOURCE=$MANO_RIGHT_SOURCE"
    return
  fi

  cat >&2 <<EOF
MANO_RIGHT.pkl is required before the hand-flow service's HaMeR backend can be ready.

1. Register and accept the MANO license:
   https://mano.is.tue.mpg.de/
2. Download MANO_RIGHT.pkl.
3. Rerun this bootstrap with:
   MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl ./remote_hand_flow/bootstrap_hand_flow_host.sh

Or copy it directly to:
  $mano_target
EOF
  exit 2
}

wait_for_health() {
  local deadline
  deadline=$((SECONDS + HAND_FLOW_READY_TIMEOUT))

  log "Waiting for hand-flow service: $HAND_FLOW_SERVER_HEALTH_URL"
  while (( SECONDS < deadline )); do
    if python - "$HAND_FLOW_SERVER_HEALTH_URL" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as response:
        body = response.read().decode("utf-8", errors="replace")
        if 200 <= response.status < 500:
            print(body)
            raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
    then
      log "Hand-flow service is reachable."
      return 0
    fi
    sleep "$HAND_FLOW_READY_INTERVAL"
  done

  printf 'Timed out waiting for hand-flow service: %s\n' "$HAND_FLOW_SERVER_HEALTH_URL" >&2
  printf 'Check logs/hand_flow_server.log for details.\n' >&2
  return 1
}

cd "$REPO_ROOT"
require_pixi
check_mano_input

log "Installing Pixi environment: hand-flow-host"
pixi install -e hand-flow-host

log "Running hand-flow backend setup."
export HAMER_DIR DOWNLOAD_HAMER_MODELS PYTHONNOUSERSITE PYOPENGL_PLATFORM MPLCONFIGDIR
pixi run -e hand-flow-host setup-hand-flow-host

if [[ "$START_HAND_FLOW_SERVER" != "1" ]]; then
  log "START_HAND_FLOW_SERVER=0, setup complete without launch."
  exit 0
fi

log "Launching hand-flow service."
if [[ "$KILL_OLD_PROCESSES" == "1" ]]; then
  log "KILL_OLD_PROCESSES=1, launch will stop stale hand-flow service processes and clear port $PORT first."
fi
export PORT HOST HAMER_DIR KILL_OLD_PROCESSES PYTHONNOUSERSITE PYOPENGL_PLATFORM MPLCONFIGDIR
pixi run -e hand-flow-host launch-hand-flow-server

if [[ "$RUN_HAND_FLOW_HEALTH_CHECK" == "1" ]]; then
  wait_for_health
fi

cat <<EOF

NovaPlan hand-flow service is ready.
Server:
  http://127.0.0.1:$PORT
Client URL:
  http://127.0.0.1:$PORT/predict
Log:
  $REPO_ROOT/logs/hand_flow_server.log

EOF
