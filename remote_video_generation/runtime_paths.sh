#!/usr/bin/env bash
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# Resolve the video-host ComfyUI while preserving an existing repository-local
# installation. An explicit COMFYUI_DIR always has highest priority.
resolve_video_comfyui_dir() {
  local repo_root="$1"
  if [[ -n "${COMFYUI_DIR:-}" ]]; then
    printf '%s\n' "$COMFYUI_DIR"
  elif [[ -f "$repo_root/comfyui/main.py" ]]; then
    printf '%s\n' "$repo_root/comfyui"
  elif [[ -f "$repo_root/.runtime/comfyui/main.py" ]]; then
    printf '%s\n' "$repo_root/.runtime/comfyui"
  else
    printf '%s\n' "$repo_root/.runtime/comfyui"
  fi
}

# Keep the project-owned video/object-flow nodes in the resolved ComfyUI runtime in
# lockstep with this checkout. ComfyUI imports the copies under custom_nodes/,
# not the canonical files in remote_video_generation/.
sync_video_runtime_custom_nodes() {
  local repo_root="$1"
  local comfyui_dir="$2"
  local video_source="${VIDEO_CUSTOM_NODES_SOURCE:-$repo_root/remote_video_generation/custom_nodes/novaplan}"
  local flow_sam_source="${FLOW_SAM3_NODE_SOURCE:-$repo_root/remote_object_flow/comfyui/custom_nodes/novaplan/sam3}"
  local target="$comfyui_dir/custom_nodes/novaplan"
  local sam_source="$flow_sam_source"

  if [[ "${SYNC_VIDEO_CUSTOM_NODES:-1}" != "1" ]]; then
    printf '[video-custom-nodes] Automatic sync disabled by SYNC_VIDEO_CUSTOM_NODES.\n'
    return
  fi

  if [[ -d "$target/tapip3d_adapter" && -d "$target/.external/TAPIP3D" && -d "$target/moge2_metric_depth" ]]; then
    if [[ ! -f "$target/__init__.py" ]]; then
      printf 'Incomplete shared NovaPlan custom-node bundle at %s\n' "$target" >&2
      printf 'Run: pixi run -e object-flow-host setup-object-flow-host\n' >&2
      return 1
    fi
    printf '[video-custom-nodes] Preserving the full object-flow bundle and refreshing shared video nodes.\n'
  else
    printf '[video-custom-nodes] Refreshing the video-only NovaPlan node bundle.\n'
    mkdir -p "$target"
    cp -a "$video_source/__init__.py" "$target/__init__.py"
  fi

  local required_source
  for required_source in \
    "$sam_source/comfyui_node/__init__.py" \
    "$sam_source/comfyui_node/nodes.py" \
    "$video_source/cotracker3/__init__.py" \
    "$video_source/cotracker3/nodes.py"; do
    if [[ ! -f "$required_source" ]]; then
      printf 'Missing NovaPlan custom-node source: %s\n' "$required_source" >&2
      return 1
    fi
  done

  mkdir -p "$target/sam3/comfyui_node" "$target/cotracker3"
  cp -a "$sam_source/comfyui_node/__init__.py" "$target/sam3/comfyui_node/__init__.py"
  cp -a "$sam_source/comfyui_node/nodes.py" "$target/sam3/comfyui_node/nodes.py"
  cp -a "$video_source/cotracker3/__init__.py" "$target/cotracker3/__init__.py"
  cp -a "$video_source/cotracker3/nodes.py" "$target/cotracker3/nodes.py"
}
