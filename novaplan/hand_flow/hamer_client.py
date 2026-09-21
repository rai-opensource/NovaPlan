#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""
Client for NovaPlan's hand-flow service using its HaMeR backend contract.

- No relative imports
- Uses urllib (no requests dependency)
- Sends base64-embedded video bytes so local and remote hosts behave identically
- Parses response
- NEW:
  - Saves returned REAL meshes (vertices+faces) to --out_folder/real_mesh_dump as .npz and .ply

Server response expectation (per mesh dict):
  {
    "frame_idx": int,
    "hand_id": int,
    "vertices": [[x,y,z], ...],
    "faces": [[i,j,k], ...],
    "semantic_xyz": { "index_tip": [u,v] or null, ... }
  }

Notes:
- Open3D is optional (only needed for .ply output).
"""

import argparse
import base64
import glob
import io
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
import numpy as np

try:
    from ..cli_args import frame_limit, http_url, positive_float, positive_int
    from .hand_reference_selection import find_hand_reference, find_movement_start_frame
except ImportError:
    from novaplan.cli_args import frame_limit, http_url, positive_float, positive_int
    from hand_reference_selection import find_hand_reference, find_movement_start_frame

# Open3D optional
try:
    import open3d as o3d
    _O3D_OK = True
except Exception:
    o3d = None
    _O3D_OK = False


K_REAL = np.array(
    [[643.9169,    0.,      649.77606],
     [  0.,      643.32336, 365.96713],
     [  0.,        0.,        1.     ]],
    dtype=np.float32
)

# -------------------------
# HTTP
# -------------------------
def _http_post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTPError {e.code}: {raw}") from e
    except URLError as e:
        raise RuntimeError(f"URLError: {e}") from e


def _b64_of_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _npy_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return _b64_of_bytes(buf.getvalue())


def _read_file_b64(
    path: str,
    start_t: Optional[Union[int, float]] = None,
    end_t: Optional[Union[int, float]] = None,
    *,
    npy_fps: Optional[float] = None,   # only used for .npy time->index conversion
) -> str:
    """
    ALWAYS returns base64 of an in-memory .npy (i.e. array bytes), never raw file bytes.

    - For .npy:
        * If npy_fps is provided: start_t/end_t are seconds -> indices via round(t*npy_fps)
        * Else: start_t/end_t are interpreted as indices
        * If start_t/end_t are None: returns the full array
    - For videos:
        * Decodes frames into (T,H,W,3) uint8 (OpenCV BGR)
        * start_t/end_t are seconds (video fps read from container)
        * If start_t/end_t are None: decodes the full video
    """
    ext = os.path.splitext(path)[1].lower()

    # ----------------------------
    # .npy : (optional) slice along axis=0 -> .npy bytes -> base64
    # ----------------------------
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        T = int(arr.shape[0])

        def _to_idx(t: Optional[Union[int, float]]) -> Optional[int]:
            if t is None:
                return None
            if npy_fps is not None:
                return int(round(float(t) * float(npy_fps)))
            return int(t)

        s = 0 if start_t is None else _to_idx(start_t)
        e = T if end_t is None else _to_idx(end_t)

        s = max(0, min(T, int(s)))
        e = max(0, min(T, int(e)))
        if e < s:
            s, e = e, s

        sliced = np.asarray(arr[s:e])
        return _npy_to_b64(sliced)

    # ----------------------------
    # video : decode -> (T,H,W,3) ndarray -> .npy bytes -> base64
    # ----------------------------
    if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not vid_fps or vid_fps <= 0:
            vid_fps = 30.0

        # start_t/end_t are seconds for video
        s_f = 0 if start_t is None else start_t
        e_f = None if end_t is None else end_t
        if e_f is not None and e_f < s_f:
            s_f, e_f = e_f, s_f

        if s_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(s_f))

        frames = []
        cur = s_f
        while True:
            if e_f is not None and cur >= e_f:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            rgb = rgb[..., ::-1] * 255
            frames.append(rgb.astype(np.uint8))  # (H,W,3) uint8, BGR
            cur += 1

        cap.release()

        if frames:
            arr = np.stack(frames, axis=0)  # (T,H,W,3)
        else:
            arr = np.empty((0, 0, 0, 3), dtype=np.uint8)

        return _npy_to_b64(arr)

    raise ValueError(f"Unsupported file extension: {ext} (path={path})")

# -------------------------
# Mesh helpers
# -------------------------
def _mesh_from_arrays(vertices, faces):
    if not _O3D_OK:
        raise RuntimeError("open3d is not available. Install open3d to write .ply meshes.")
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (V,3), got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {F.shape}")

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(V)
    m.triangles = o3d.utility.Vector3iVector(F)
    m.compute_vertex_normals()
    return m


def _collect_real_meshes(resp: dict) -> Tuple[List[Dict[str, Any]], bytes]:
    """
    Returns:
      (uniq_real_mesh_dicts, real_video_bytes)

    Meshes:
      - Accepts either resp["frames"][i]["real_meshes"] or resp["real_meshes"]
      - De-dups by (frame_idx, hand_id)

    Video bytes:
      - Accepts resp["real_video_b64"] (preferred)
      - Fallback: resp["real_video_bytes_b64"]
      - Returns b"" if absent
    """
    out: List[Dict[str, Any]] = []

    frames = resp.get("frames", None)
    if isinstance(frames, list):
        for fr in frames:
            ms = fr.get("real_meshes", [])
            if isinstance(ms, list):
                out.extend(ms)

    ms2 = resp.get("real_meshes", None)
    if isinstance(ms2, list):
        out.extend(ms2)

    seen = set()
    uniq: List[Dict[str, Any]] = []
    for m in out:
        fi = int(m.get("frame_idx", -1))
        hi = int(m.get("hand_id", -1))
        key = (fi, hi)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)

    # video
    vb64 = resp.get("real_video_b64") or resp.get("real_video_bytes_b64") or ""
    if isinstance(vb64, str) and vb64.strip():
        try:
            vbytes = base64.b64decode(vb64.encode("utf-8"), validate=False)
        except Exception:
            vbytes = b""
    else:
        vbytes = b""

    return uniq, vbytes

def _save_meshes_to_out_folder(
    mesh_dicts: list,
    out_folder: str,
    start_frame: int = 0,
    *,
    verbose: bool = False,
) -> dict:
    """
    Saves meshes under:
      out_folder/real_mesh_dump/frameXXXXXX_handYY.npz   (always)
      out_folder/real_mesh_dump/frameXXXXXX_handYY.ply   (if open3d available)
    
    Args:
        mesh_dicts: List of mesh dictionaries from server response
        out_folder: Output folder path
        start_frame: Starting frame number to add to relative frame indices (default 0)
    """
    dump_dir = os.path.join(out_folder, "real_mesh_dump")
    os.makedirs(dump_dir, exist_ok=True)

    npz_paths = []
    ply_paths = []

    for md in mesh_dicts:
        # frame_idx from server is relative to start_frame, convert to absolute
        relative_fi = int(md.get("frame_idx", -1))
        absolute_fi = relative_fi + start_frame
        hi = int(md.get("hand_id", 0))
        is_right = int(md.get("is_right", 1))

        verts = np.asarray(md.get("vertices", []), dtype=np.float32)
        faces = np.asarray(md.get("faces", []), dtype=np.int32)
        semantic_xyz = md.get("semantic_xyz", {}) if isinstance(md.get("semantic_xyz", {}), dict) else {}

        if verts.size == 0 or faces.size == 0:
            continue

        stem = f"frame{absolute_fi:06d}_hand{hi:02d}"
        npz_path = os.path.join(dump_dir, stem + ".npz")
        payload = {
            "frame_idx": np.int64(absolute_fi),  # Save absolute frame index
            "hand_id": np.int64(hi),
            "is_right": np.int64(is_right),
            "vertices": verts.astype(np.float32),
            "faces": faces.astype(np.int32),
            "semantic_xyz_json": np.bytes_(json.dumps(semantic_xyz)),
        }
        for key in (
            "verts_cam",
            "vertices_weak",
            "verts_local",
            "verts_rot",
            "K_REAL",
            "K_WEAK",
            "R_pnp",
            "cam_t",
            "t_pnp",
            "t_weak",
        ):
            if key in md and md.get(key) is not None:
                payload[key] = np.asarray(md[key], dtype=np.float32)
        for key in ("ok_pnp", "pnp_reprojection_error"):
            if key in md and md.get(key) is not None:
                payload[key] = np.asarray(md[key])
        if isinstance(md.get("semantic_xyz_weak"), dict):
            payload["semantic_xyz_weak_json"] = np.bytes_(json.dumps(md["semantic_xyz_weak"]))
        np.savez_compressed(npz_path, **payload)
        npz_paths.append(npz_path)

        if _O3D_OK:
            try:
                mesh = _mesh_from_arrays(verts, faces)
                ply_path = os.path.join(dump_dir, stem + ".ply")
                o3d.io.write_triangle_mesh(ply_path, mesh, write_ascii=False, compressed=False)
                ply_paths.append(ply_path)
            except Exception as e:
                print(f"[mesh-dump] failed to write ply for {stem}: {e}")

    if verbose:
        print(f"[mesh-dump] wrote {len(npz_paths)} npz meshes to {dump_dir}")
    if _O3D_OK and verbose:
        print(f"[mesh-dump] wrote {len(ply_paths)} ply meshes to {dump_dir}")

    return {"dump_dir": dump_dir, "npz_paths": npz_paths, "ply_paths": ply_paths}


def _copy_reference_mesh(
    dump_dir: str,
    mesh_filename: str,
    out_folder: str,
    *,
    output_stem: str,
) -> Dict[str, Optional[str]]:
    """Copy the required NPZ reference and an optional sibling PLY."""

    source_npz = os.path.join(dump_dir, mesh_filename)
    target_npz = os.path.join(out_folder, output_stem + ".npz")
    shutil.copy2(source_npz, target_npz)

    source_ply = os.path.join(
        dump_dir,
        os.path.splitext(mesh_filename)[0] + ".ply",
    )
    target_ply = os.path.join(out_folder, output_stem + ".ply")
    copied_ply = None
    if os.path.exists(source_ply):
        shutil.copy2(source_ply, target_ply)
        copied_ply = target_ply

    return {"npz": target_npz, "ply": copied_ply}


def _save_real_video_bytes(video_bytes: bytes, out_folder: str, name: str = "real.mp4") -> str:
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, name)
    if not video_bytes:
        return ""
    with open(out_path, "wb") as f:
        f.write(video_bytes)
    return out_path


def _clear_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        child = os.path.join(path, name)
        if os.path.isdir(child) and not os.path.islink(child):
            shutil.rmtree(child)
        else:
            os.remove(child)


def _find_flow_file(sample_dir: str) -> str:
    candidates = [
        os.path.join(sample_dir, "test_results_human_hand", "tapip3d_output.npz"),
        os.path.join(sample_dir, "test_results", "tapip3d_output.npz"),
        os.path.join(sample_dir, "tapip3d_output.npz"),
        os.path.join(sample_dir, "flow.npz"),
    ]
    candidates.extend(sorted(glob.glob(os.path.join(sample_dir, "test_results_human_hand", "tapip3d_output_*.npz"))))
    candidates.extend(sorted(glob.glob(os.path.join(sample_dir, "test_results", "tapip3d_output_*.npz"))))
    candidates.extend(sorted(glob.glob(os.path.join(sample_dir, "tapip3d_output_*.npz"))))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise ValueError(
        "Could not find flow file. Expected test_results_human_hand/tapip3d_output.npz, "
        "test_results/tapip3d_output.npz, test_results/tapip3d_output_<video>.npz, "
        "or pass --flow_file /path/to/tapip3d_output.npz."
    )


def _flow_image_size_from_npz(data) -> Optional[Tuple[int, int]]:
    if "video" in data:
        video = data["video"]
        if video.ndim == 4:
            if video.shape[1] in {1, 3, 4}:
                return int(video.shape[3]), int(video.shape[2])
            return int(video.shape[2]), int(video.shape[1])
    if "depths" in data:
        depths = data["depths"]
        if depths.ndim >= 3:
            return int(depths.shape[-1]), int(depths.shape[-2])
    return None


def _video_image_size(path: str) -> Optional[Tuple[int, int]]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(path, mmap_mode="r")
            if arr.ndim == 4:
                if arr.shape[1] in {1, 3, 4}:
                    return int(arr.shape[3]), int(arr.shape[2])
                return int(arr.shape[2]), int(arr.shape[1])
            return None

        if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            import cv2

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if width > 0 and height > 0:
                return width, height
    except Exception:
        return None
    return None


def _scale_intrinsics_to_size(
    intrinsics: np.ndarray,
    *,
    source_size: Optional[Tuple[int, int]],
    target_size: Optional[Tuple[int, int]],
    verbose: bool = False,
) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    if source_size is None or target_size is None or source_size == target_size:
        return intrinsics
    source_w, source_h = source_size
    target_w, target_h = target_size
    if source_w <= 0 or source_h <= 0 or target_w <= 0 or target_h <= 0:
        return intrinsics
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    intrinsics[..., 0, 0] *= sx
    intrinsics[..., 0, 2] *= sx
    intrinsics[..., 1, 1] *= sy
    intrinsics[..., 1, 2] *= sy
    if verbose:
        print(
            "[hamer-client] scaled flow intrinsics "
            f"from {source_w}x{source_h} to RGB {target_w}x{target_h} "
            f"(sx={sx:.4f}, sy={sy:.4f})"
        )
    return intrinsics


def _load_flow_intrinsics(
    flow_file: str,
    start_t=None,
    end_t=None,
    target_size: Optional[Tuple[int, int]] = None,
    *,
    verbose: bool = False,
):
    try:
        with np.load(flow_file, allow_pickle=False) as data:
            if "intrinsics" not in data:
                return None
            intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
            source_size = _flow_image_size_from_npz(data)
    except Exception as exc:
        print(f"[client] WARNING: could not load intrinsics from {flow_file}: {exc}")
        return None

    intrinsics = _scale_intrinsics_to_size(
        intrinsics,
        source_size=source_size,
        target_size=target_size,
        verbose=verbose,
    )
    if intrinsics.ndim == 3 and start_t is not None:
        s = max(0, min(intrinsics.shape[0], int(start_t)))
        e = intrinsics.shape[0] if end_t is None else max(0, min(intrinsics.shape[0], int(end_t)))
        if e < s:
            s, e = e, s
        intrinsics = intrinsics[s:e]
    if intrinsics.ndim == 3 and intrinsics.shape[0] == 0:
        return None
    if intrinsics.shape == (3, 3) or (intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3)):
        return intrinsics.tolist()
    print(f"[client] WARNING: ignoring unsupported intrinsics shape from {flow_file}: {intrinsics.shape}")
    return None


# -------------------------
# Main
# -------------------------
def main():
    """Run the command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--url",
        type=http_url,
        default=os.environ.get("NOVAPLAN_HAND_FLOW_SERVER_URL", "http://127.0.0.1:8080/predict"),
        help="Hand-flow service /predict URL. Defaults to NOVAPLAN_HAND_FLOW_SERVER_URL or localhost.",
    )
    
    # Required: sample directory (infers all other paths)
    ap.add_argument("--sample_dir", type=str, required=True,
                    help="Sample directory. Infers rgb, out_folder, and flow_file relative to it.")
    ap.add_argument(
        "--rgb_path",
        type=str,
        default=None,
        help="Optional RGB video path override. Useful for running HaMeR on a selected generated rollout.",
    )
    ap.add_argument(
        "--out_folder",
        type=str,
        default=None,
        help="Output folder for HaMeR artifacts. Defaults to sample_dir/test_results/hamer_outputs.",
    )
    ap.add_argument(
        "--flow_file",
        type=str,
        default=None,
        help=(
            "Optional explicit TAPIP3D flow npz for motion/reference selection. "
            "Prefer sample_dir/test_results_human_hand/tapip3d_output.npz for hand workflows."
        ),
    )
    
    ap.add_argument("--batch_size", type=positive_int, default=8)
    ap.add_argument("--rescale_factor", type=positive_float, default=2.0)
    ap.add_argument("--stride", type=positive_int, default=1)
    ap.add_argument("--max_frames", type=frame_limit, default=-1)
    ap.add_argument("--fps", type=positive_float, default=30.0)

    ap.add_argument("--persist_outputs", action="store_true", help="server keeps outputs on disk")
    ap.add_argument(
        "--save_meshes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save returned calibrated meshes under out_folder.",
    )
    ap.add_argument("--all_frames", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Print detailed request and artifact diagnostics.")
    args = ap.parse_args()
    
    # Infer all paths from sample_dir
    sample_dir = os.path.abspath(os.path.expanduser(args.sample_dir))
    if not os.path.isdir(sample_dir):
        raise ValueError(f"sample_dir does not exist: {sample_dir}")
    
    # Infer rgb path
    if args.rgb_path:
        args.rgb = os.path.abspath(os.path.expanduser(args.rgb_path))
        if not os.path.exists(args.rgb):
            raise ValueError(f"rgb_path does not exist: {args.rgb}")
    elif os.path.exists(os.path.join(sample_dir, "rgb_video_16fps.mp4")):
        args.rgb = os.path.join(sample_dir, "rgb_video_16fps.mp4")
    else:
        # Try alternative names
        args.rgb = ""
        for alt_name in ["rgb_video.mp4", "rgb.mp4", "video.mp4"]:
            alt_path = os.path.join(sample_dir, alt_name)
            if os.path.exists(alt_path):
                args.rgb = alt_path
                break
        if not args.rgb:
            raise ValueError(f"Could not find RGB video in {sample_dir}. Expected rgb_video_16fps.mp4 or similar.")
    
    
    if args.out_folder:
        args.out_folder = os.path.abspath(os.path.expanduser(args.out_folder))
    else:
        args.out_folder = os.path.join(sample_dir, "test_results", "hamer_outputs")
    
    if args.flow_file:
        args.flow_file = os.path.abspath(os.path.expanduser(args.flow_file))
        if not os.path.exists(args.flow_file):
            raise ValueError(f"flow_file does not exist: {args.flow_file}")
    else:
        args.flow_file = _find_flow_file(sample_dir)
    
    os.makedirs(args.out_folder, exist_ok=True)
    if args.verbose:
        print("[hamer-client] flow_file:", args.flow_file)
    
    # Check if reference files already exist
    reference_npz_path = os.path.join(args.out_folder, "reference.npz")
    reference_ply_path = os.path.join(args.out_folder, "reference.ply")
    reference_exists = os.path.exists(reference_npz_path)
    
    if reference_exists and not args.all_frames:
        print("[client] WARNING: Reference files already exist at:")
        print(f"  {reference_npz_path}")
        if os.path.exists(reference_ply_path):
            print(f"  {reference_ply_path}")
        print("[client] Skipping request since --all_frames is not set.")
        print("[client] To regenerate, either:")
        print("  1. Delete the reference files, or")
        print("  2. Use --all_frames flag to process all frames")
        return
    
    motion_t = find_movement_start_frame(flow_file=args.flow_file)
    if not args.all_frames:
        start_t, end_t = max(0, motion_t-5), motion_t+5
    else:
        start_t, end_t = None, None
    
    payload = {
        "out_folder": args.out_folder,
        "batch_size": int(args.batch_size),
        "rescale_factor": float(args.rescale_factor),
        "stride": int(args.stride),
        "max_frames": int(args.max_frames),
        "fps": float(args.fps),
        "persist_outputs": bool(args.persist_outputs),

        # The server may ignore this field; it is retained for API compatibility.
        "return_mesh_arrays": True,
    }
    camera_intrinsics = _load_flow_intrinsics(
        args.flow_file,
        start_t=start_t,
        end_t=end_t,
        target_size=_video_image_size(args.rgb),
        verbose=args.verbose,
    )
    if camera_intrinsics is not None:
        payload["camera_intrinsics"] = camera_intrinsics
        if args.verbose:
            print("[hamer-client] sending camera intrinsics from flow file")
    else:
        print("[client] WARNING: no flow intrinsics sent; server will use an image-size focal fallback")

    # --- ALWAYS send videos as base64-encoded bytes ---
    # Use keys that match the simplified server handler:
    #   RGB: rgb_videob64 (or rgb_video_b64)
    payload["rgb_videob64"] = _read_file_b64(args.rgb, start_t, end_t)

    resp = _http_post_json(args.url, payload, timeout=3600.0)

    if not resp.get("ok", False):
        print("Server error:", resp)
        sys.exit(1)

    if args.verbose:
        print("[hamer-client] server response ok")
        print(" out_folder:", resp.get("out_folder"))
        print(" weak_root :", resp.get("weak_root"))
        print(" real_root :", resp.get("real_root"))
        print(" num_frames:", resp.get("num_frames"))

    real_mesh_dicts, real_video_bytes = _collect_real_meshes(resp)
    if args.verbose:
        print(f"[hamer-client] received real meshes: {len(real_mesh_dicts)}")

    if not real_mesh_dicts:
        raise RuntimeError("HaMeR returned no hand meshes for the generated video")

    if real_mesh_dicts and args.verbose:
        m0 = real_mesh_dicts[0]
        v0 = np.asarray(m0.get("vertices", []))
        f0 = np.asarray(m0.get("faces", []))
        sem0 = m0.get("semantic_xyz", {})
        print(f"[client] sample mesh: frame={m0.get('frame_idx')} hand={m0.get('hand_id')} V={v0.shape} F={f0.shape}")
        if isinstance(sem0, dict):
            keys = list(sem0.keys())[:8]
            print(f"[client] sample semantic keys: {keys}")


    _clear_directory(args.out_folder)
    # Save meshes for future visualization
    if args.save_meshes:
        start_frame = start_t if start_t is not None else 0
        info = _save_meshes_to_out_folder(
            real_mesh_dicts,
            args.out_folder,
            start_frame=start_frame,
            verbose=args.verbose,
        )

        best_fs = find_hand_reference(info["dump_dir"], motion_t, verbose=args.verbose)
        _copy_reference_mesh(
            info["dump_dir"],
            best_fs,
            args.out_folder,
            output_stem="reference",
        )
        saved_frame_indices = []
        for npz_path in info["npz_paths"]:
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    saved_frame_indices.append(int(data["frame_idx"]))
            except Exception:
                continue
        movement_start_frame = int(motion_t) if motion_t is not None and int(motion_t) >= 0 else (
            min(saved_frame_indices) if saved_frame_indices else 0
        )
        movement_end_frame = max(saved_frame_indices) if saved_frame_indices else movement_start_frame
        try:
            end_fs = find_hand_reference(
                info["dump_dir"],
                movement_end_frame,
                verbose=args.verbose,
            )
            _copy_reference_mesh(
                info["dump_dir"],
                end_fs,
                args.out_folder,
                output_stem="reference_end",
            )
        except Exception as exc:
            print(f"[client] WARNING: failed to save reference_end files: {exc}")

        meta_path = os.path.join(args.out_folder, "meta.json")
        meta = {
            "movement_start_frame": int(movement_start_frame),
            "grasp_frame_idx": int(movement_start_frame),
            "movement_end_frame": int(movement_end_frame),
            "is_right": None,
        }
        try:
            with np.load(os.path.join(args.out_folder, "reference.npz"), allow_pickle=False) as reference_npz:
                if "is_right" in reference_npz:
                    meta["is_right"] = bool(int(reference_npz["is_right"]))
        except Exception:
            pass
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if args.verbose:
            print(f"[hamer-client] wrote meta: {meta_path}")
        
    video_path = _save_real_video_bytes(real_video_bytes, args.out_folder, name="real.mp4")
    if video_path and args.verbose:
        print("[hamer-client] saved real video:", video_path)
    print(
        f"[hamer-client] completed: frames={resp.get('num_frames')} "
        f"meshes={len(real_mesh_dicts)} output={args.out_folder}"
    )

if __name__ == "__main__":
    main()
