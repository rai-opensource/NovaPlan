#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

_comfyui_worker_matches() {
  local pid="$1"
  local port="$2"
  local cmdline
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"main.py"* && "$cmdline" == *"--port $port"* ]]
}

_stop_pid_list() {
  local label="$1"
  shift
  local -a pids=("$@")
  (( ${#pids[@]} > 0 )) || return 0

  echo "Stopping stale $label process(es): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  for _ in {1..20}; do
    local any_alive=0
    local pid
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 0.1
  done

  local -a remaining=()
  local pid
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && remaining+=("$pid")
  done
  if (( ${#remaining[@]} > 0 )); then
    echo "Force stopping stale $label process(es): ${remaining[*]}"
    kill -9 "${remaining[@]}" 2>/dev/null || true
  fi
}

stop_recorded_comfyui_workers() {
  local role="$1"
  local pid_glob="$2"
  local pid_file
  local pid
  local -a pids=()
  local -A seen=()

  while IFS= read -r pid_file; do
    [[ -s "$pid_file" ]] || {
      rm -f "$pid_file"
      continue
    }
    read -r pid < "$pid_file" || true
    if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]]; then
      local cmdline
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
      if [[ "$cmdline" == *"main.py"* && -z "${seen[$pid]:-}" ]]; then
        seen["$pid"]=1
        pids+=("$pid")
      fi
    fi
    rm -f "$pid_file"
  done < <(compgen -G "$pid_glob" || true)

  _stop_pid_list "$role ComfyUI worker" "${pids[@]}"
}

stop_matching_processes() {
  local role="$1"
  local pattern="$2"
  local pid
  local -a pids=()
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] && pids+=("$pid")
  done < <(pgrep -f "$pattern" || true)
  _stop_pid_list "$role" "${pids[@]}"
}

stop_comfyui_worker() {
  local role="$1"
  local port="$2"
  local pid_file="$3"
  local pid
  local -a pids=()
  local -A seen=()

  if [[ -s "$pid_file" ]]; then
    read -r pid < "$pid_file" || true
    if [[ "$pid" =~ ^[0-9]+$ ]] && _comfyui_worker_matches "$pid" "$port"; then
      seen["$pid"]=1
      pids+=("$pid")
    fi
  fi

  while read -r pid; do
    if [[ "$pid" =~ ^[0-9]+$ && -z "${seen[$pid]:-}" ]]; then
      seen["$pid"]=1
      pids+=("$pid")
    fi
  done < <(pgrep -f '[m]ain.py.*--port[[:space:]]+'"$port"'([[:space:]]|$)' || true)

  while read -r pid; do
    if [[ "$pid" =~ ^[0-9]+$ && -z "${seen[$pid]:-}" ]]; then
      seen["$pid"]=1
      pids+=("$pid")
    fi
  done < <(
    ss -ltnp "sport = :$port" 2>/dev/null \
      | sed -n "s/.*pid=\([0-9]\+\).*/\1/p" \
      | sort -u
  )

  if (( ${#pids[@]} > 0 )); then
    _stop_pid_list "$role ComfyUI worker on port $port" "${pids[@]}"
  fi

  rm -f "$pid_file"
}

record_comfyui_worker_pid() {
  local pid="$1"
  local pid_file="$2"
  mkdir -p "$(dirname "$pid_file")"
  printf '%s\n' "$pid" > "$pid_file"
}
