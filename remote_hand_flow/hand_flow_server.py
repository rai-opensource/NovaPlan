#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Serve NovaPlan hand flow through a HaMeR reconstruction backend.

The FastAPI service, request/response contract, persistence, and NovaPlan
calibration metadata are NovaPlan code. The detector, crop inference, and
weak-perspective camera conversion sequence is adapted from HaMeR ``demo.py``
at revision 3a01849f4148352e9260b69bf28b65d1671a4905. HaMeR is MIT licensed;
see ``remote_hand_flow/HAMER_LICENSE.md``.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from novaplan.cli_args import tcp_port
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

DEFAULT_HAMER_DIR = REPO_ROOT / "remote_hand_flow" / ".external" / "hamer"
HAMER_DIR = Path(os.environ.get("HAMER_DIR", str(DEFAULT_HAMER_DIR))).expanduser()

if HAMER_DIR.exists() and str(HAMER_DIR) not in sys.path:
    sys.path.insert(0, str(HAMER_DIR))

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


class PredictRequest(BaseModel):
    """Validate one HaMeR hand-flow prediction request."""
    rgb_videob64: Optional[str] = Field(default=None, description="Base64-encoded numpy.save video array.")
    rgb_video_b64: Optional[str] = Field(default=None, description="Alias for rgb_videob64.")
    out_folder: Optional[str] = Field(default=None, description="Compatibility field echoed in the response.")
    batch_size: int = Field(default=8, ge=1)
    rescale_factor: float = Field(default=2.0, gt=0)
    stride: int = Field(default=1, ge=1)
    max_frames: int = Field(default=-1, ge=-1)
    fps: float = Field(default=30.0, gt=0)
    persist_outputs: bool = Field(default=False)
    return_mesh_arrays: bool = Field(default=True)
    body_detector: Literal["vitdet", "regnety"] = Field(default="vitdet", description="HaMeR body detector.")
    camera_intrinsics: Optional[Any] = Field(
        default=None,
        description="Optional [3,3], [T,3,3], [fx,fy,cx,cy], or [T,4] real camera intrinsics.",
    )
    intrinsics: Optional[Any] = Field(default=None, description="Alias for camera_intrinsics.")

    @model_validator(mode="after")
    def validate_video_and_frame_limit(self):
        """Validate the encoded video and optional frame limit."""
        if not self.rgb_videob64 and not self.rgb_video_b64:
            raise ValueError("rgb_videob64 is required")
        if self.max_frames == 0:
            raise ValueError("max_frames must be -1 (unlimited) or greater than zero")
        return self


app = FastAPI(title="NovaPlan Hand-Flow Service")
_runtime_lock = threading.Lock()
_runtime: Optional["HamerRuntime"] = None


def _decode_video_array(video_b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(video_b64.encode("utf-8"), validate=False)
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode rgb_videob64 as numpy array: {exc}") from exc

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise HTTPException(status_code=400, detail=f"Expected video shape (T,H,W,3), got {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(arr)


def _serializable_xyz(point: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(point, dtype=np.float32).reshape(3)]


def _intrinsics_matrix_from_flat(flat: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.float32).reshape(4)
    fx, fy, cx, cy = [float(x) for x in flat]
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _coerce_camera_intrinsics(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (3, 3):
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return arr
    if arr.shape == (4,):
        return _intrinsics_matrix_from_flat(arr)
    if arr.ndim == 2 and arr.shape[1] == 4:
        return np.stack([_intrinsics_matrix_from_flat(row) for row in arr], axis=0)
    raise HTTPException(status_code=400, detail=f"camera_intrinsics has unsupported shape: {arr.shape}")


def _get_runtime(body_detector: str) -> "HamerRuntime":
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = HamerRuntime(body_detector=body_detector)
        elif _runtime.body_detector != body_detector:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"HaMeR runtime already loaded with body_detector={_runtime.body_detector!r}; "
                    f"restart the server to use {body_detector!r}."
                ),
            )
        return _runtime


class HamerRuntime:
    """Load HaMeR once and execute hand-mesh inference requests."""
    def __init__(self, *, body_detector: str) -> None:
        if not HAMER_DIR.exists():
            raise RuntimeError(f"HAMER_DIR does not exist: {HAMER_DIR}")

        self.body_detector = body_detector
        self.checkpoint = os.environ.get("HAMER_CHECKPOINT", "")
        self.auto_download = os.environ.get("HAMER_AUTO_DOWNLOAD_MODELS", "0").lower() in {"1", "true", "yes", "on"}

        import cv2
        import torch
        import hamer
        from hamer.configs import CACHE_DIR_HAMER
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
        from hamer.utils import recursive_to
        from hamer.utils.renderer import Renderer, cam_crop_to_full
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        self.cv2 = cv2
        self.torch = torch
        self.hamer = hamer
        self.ViTDetDataset = ViTDetDataset
        self.recursive_to = recursive_to
        self.Renderer = Renderer
        self.cam_crop_to_full = cam_crop_to_full
        self.DefaultPredictor_Lazy = DefaultPredictor_Lazy

        if self.auto_download:
            download_models(CACHE_DIR_HAMER)

        checkpoint = self.checkpoint or DEFAULT_CHECKPOINT
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model, self.model_cfg = load_hamer(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.detector = self._load_detector(body_detector)
        self.cpm = ViTPoseModel(self.device)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self.faces_right = np.asarray(self.renderer.faces, dtype=np.int32)
        self.faces_left = np.asarray(self.renderer.faces_left, dtype=np.int32)

    def _load_detector(self, body_detector: str) -> Any:
        if body_detector == "vitdet":
            from detectron2.config import LazyConfig

            cfg_path = Path(self.hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            return self.DefaultPredictor_Lazy(detectron2_cfg)

        if body_detector == "regnety":
            from detectron2 import model_zoo

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
                trained=True,
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            return self.DefaultPredictor_Lazy(detectron2_cfg)

        raise ValueError("body_detector must be 'vitdet' or 'regnety'")

    def predict(
        self,
        frames_bgr: np.ndarray,
        req: PredictRequest,
        *,
        camera_intrinsics: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Run HaMeR prediction for one request."""
        if req.stride > 1:
            frames_bgr = frames_bgr[:: req.stride]
            if camera_intrinsics is not None and camera_intrinsics.ndim == 3:
                camera_intrinsics = camera_intrinsics[:: req.stride]
        if req.max_frames >= 0:
            frames_bgr = frames_bgr[: req.max_frames]
            if camera_intrinsics is not None and camera_intrinsics.ndim == 3:
                camera_intrinsics = camera_intrinsics[: req.max_frames]

        meshes: List[Dict[str, Any]] = []
        for local_idx, frame in enumerate(frames_bgr):
            frame_idx = local_idx * req.stride
            meshes.extend(
                self._predict_frame(
                    frame_idx=frame_idx,
                    img_cv2=np.ascontiguousarray(frame),
                    batch_size=req.batch_size,
                    rescale_factor=req.rescale_factor,
                    real_intrinsics=self._intrinsics_for_frame(camera_intrinsics, local_idx, frame),
                )
            )
        return meshes

    @staticmethod
    def _default_intrinsics_for_frame(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        focal = 0.8 * float(max(width, height))
        return np.asarray(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _intrinsics_for_frame(
        self,
        intrinsics: Optional[np.ndarray],
        local_idx: int,
        frame: np.ndarray,
    ) -> np.ndarray:
        if intrinsics is None:
            return self._default_intrinsics_for_frame(frame)
        arr = np.asarray(intrinsics, dtype=np.float32)
        if arr.ndim == 2:
            return arr
        idx = max(0, min(int(local_idx), arr.shape[0] - 1))
        return arr[idx]

    def _detect_hands(self, img_cv2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        det_out = self.detector(img_cv2)
        img_rgb = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].detach().cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].detach().cpu().numpy()

        if pred_bboxes.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        vitposes_out = self.cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            for keyp, right_flag in ((left_hand_keyp, 0), (right_hand_keyp, 1)):
                valid = keyp[:, 2] > 0.5
                if int(np.sum(valid)) > 3:
                    bbox = [
                        keyp[valid, 0].min(),
                        keyp[valid, 1].min(),
                        keyp[valid, 0].max(),
                        keyp[valid, 1].max(),
                    ]
                    bboxes.append(bbox)
                    is_right.append(right_flag)

        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        return np.asarray(bboxes, dtype=np.float32), np.asarray(is_right, dtype=np.float32)

    def _predict_frame(
        self,
        *,
        frame_idx: int,
        img_cv2: np.ndarray,
        batch_size: int,
        rescale_factor: float,
        real_intrinsics: np.ndarray,
    ) -> List[Dict[str, Any]]:
        boxes, right = self._detect_hands(img_cv2)
        if boxes.shape[0] == 0:
            return []

        dataset = self.ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
        dataloader = self.torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        frame_meshes: List[Dict[str, Any]] = []
        for batch in dataloader:
            batch = self.recursive_to(batch, self.device)
            with self.torch.no_grad():
                out = self.model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max(dim=1).values
            pred_cam_t_full = self.cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            ).detach().cpu().numpy()

            batch_count = int(batch["img"].shape[0])
            for n in range(batch_count):
                is_right = int(np.asarray(batch["right"][n].detach().cpu()).item())
                hand_id = int(np.asarray(batch["personid"][n].detach().cpu()).item())
                cam_t = pred_cam_t_full[n].astype(np.float32)

                verts_local = out["pred_vertices"][n].detach().cpu().numpy().astype(np.float32)
                verts_local[:, 0] = (2 * is_right - 1) * verts_local[:, 0]
                keypoints_local = out["pred_keypoints_3d"][n].detach().cpu().numpy().astype(np.float32)
                keypoints_local[:, 0] = (2 * is_right - 1) * keypoints_local[:, 0]

                img_w, img_h = [float(x) for x in img_size[n].detach().cpu().numpy().reshape(2)]
                focal_weak = float(scaled_focal_length[n].detach().cpu().item())
                k_weak = np.asarray(
                    [[focal_weak, 0.0, img_w / 2.0], [0.0, focal_weak, img_h / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
                k_real = np.asarray(real_intrinsics, dtype=np.float32)

                verts_weak = verts_local + cam_t.reshape(1, 3)
                keypoints_weak = keypoints_local + cam_t.reshape(1, 3)
                pnp = self._solve_real_camera_pose(
                    verts_local=verts_local,
                    verts_weak=verts_weak,
                    keypoints_local=keypoints_local,
                    keypoints_weak=keypoints_weak,
                    k_weak=k_weak,
                    k_real=k_real,
                )
                if pnp["ok"]:
                    rotation = pnp["R"].astype(np.float32)
                    t_pnp = pnp["t"].astype(np.float32)
                    verts = (rotation @ verts_local.T).T + t_pnp.reshape(1, 3)
                    keypoints = (rotation @ keypoints_local.T).T + t_pnp.reshape(1, 3)
                else:
                    rotation = np.eye(3, dtype=np.float32)
                    t_pnp = cam_t.astype(np.float32)
                    verts = verts_weak
                    keypoints = keypoints_weak

                faces = self.faces_right if is_right else self.faces_left
                frame_meshes.append(
                    {
                        "frame_idx": int(frame_idx),
                        "hand_id": hand_id,
                        "is_right": bool(is_right),
                        "vertices": verts.tolist(),
                        "verts_cam": verts.tolist(),
                        "vertices_weak": verts_weak.tolist(),
                        "verts_local": verts_local.tolist(),
                        "verts_rot": ((rotation @ verts_local.T).T).astype(np.float32).tolist(),
                        "faces": faces.tolist(),
                        "semantic_xyz": self._semantic_xyz(keypoints),
                        "semantic_xyz_weak": self._semantic_xyz(keypoints_weak),
                        "cam_t": _serializable_xyz(t_pnp),
                        "t_weak": _serializable_xyz(cam_t),
                        "t_pnp": _serializable_xyz(t_pnp),
                        "R_pnp": rotation.astype(np.float32).tolist(),
                        "ok_pnp": bool(pnp["ok"]),
                        "pnp_reprojection_error": float(pnp.get("reprojection_error", np.nan)),
                        "K_REAL": k_real.astype(np.float32).tolist(),
                        "K_WEAK": k_weak.astype(np.float32).tolist(),
                    }
                )

        return frame_meshes

    def _project_points_np(self, points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        z = points[:, 2]
        uv = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
        valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
        uv[valid, 0] = intrinsics[0, 0] * points[valid, 0] / z[valid] + intrinsics[0, 2]
        uv[valid, 1] = intrinsics[1, 1] * points[valid, 1] / z[valid] + intrinsics[1, 2]
        return uv

    def _solve_real_camera_pose(
        self,
        *,
        verts_local: np.ndarray,
        verts_weak: np.ndarray,
        keypoints_local: np.ndarray,
        keypoints_weak: np.ndarray,
        k_weak: np.ndarray,
        k_real: np.ndarray,
    ) -> Dict[str, Any]:
        # HaMeR's crop/full-camera output is weak-camera geometry. Project it
        # with K_WEAK, then recover the equivalent full camera pose under the
        # actual rollout intrinsics with PnP.
        weak_uv_verts = self._project_points_np(verts_weak, k_weak)
        weak_uv_joints = self._project_points_np(keypoints_weak, k_weak)
        vert_stride = max(1, int(np.ceil(len(verts_local) / 120)))
        object_points = np.concatenate([keypoints_local, verts_local[::vert_stride]], axis=0).astype(np.float32)
        image_points = np.concatenate([weak_uv_joints, weak_uv_verts[::vert_stride]], axis=0).astype(np.float32)
        valid = np.isfinite(object_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
        object_points = object_points[valid]
        image_points = image_points[valid]
        if len(object_points) < 6:
            return {"ok": False, "R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32)}

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        try:
            ok, rvec, tvec, inliers = self.cv2.solvePnPRansac(
                object_points,
                image_points,
                k_real.astype(np.float32),
                dist_coeffs,
                iterationsCount=100,
                reprojectionError=4.0,
                confidence=0.99,
                flags=self.cv2.SOLVEPNP_EPNP,
            )
            if ok and inliers is not None and len(inliers) >= 6:
                inlier_idx = inliers.reshape(-1)
                ok_refined, rvec, tvec = self.cv2.solvePnP(
                    object_points[inlier_idx],
                    image_points[inlier_idx],
                    k_real.astype(np.float32),
                    dist_coeffs,
                    rvec,
                    tvec,
                    useExtrinsicGuess=True,
                    flags=self.cv2.SOLVEPNP_ITERATIVE,
                )
                ok = bool(ok_refined)
            elif not ok:
                ok, rvec, tvec = self.cv2.solvePnP(
                    object_points,
                    image_points,
                    k_real.astype(np.float32),
                    dist_coeffs,
                    flags=self.cv2.SOLVEPNP_EPNP,
                )
                inliers = None
        except Exception:
            return {"ok": False, "R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32)}

        if not ok:
            return {"ok": False, "R": np.eye(3, dtype=np.float32), "t": np.zeros(3, dtype=np.float32)}

        rotation, _ = self.cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float32).reshape(3)
        projected, _ = self.cv2.projectPoints(object_points, rvec, tvec, k_real.astype(np.float32), dist_coeffs)
        projected = projected.reshape(-1, 2)
        reproj_error = float(np.median(np.linalg.norm(projected - image_points, axis=1)))
        return {
            "ok": bool(np.isfinite(rotation).all() and np.isfinite(t).all() and t[2] > 0),
            "R": rotation.astype(np.float32),
            "t": t,
            "reprojection_error": reproj_error,
        }

    def _semantic_xyz(self, keypoints: np.ndarray) -> Dict[str, Any]:
        finger_joint_indices = {
            "thumb": {"mcp": 1, "pip": 2, "dip": 3, "tip": 4},
            "index": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
            "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
            "ring": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
            "pinky": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
        }
        fingers = {
            name: {
                joint_name: _serializable_xyz(keypoints[idx])
                for joint_name, idx in joint_indices.items()
                if idx < keypoints.shape[0]
            }
            for name, joint_indices in finger_joint_indices.items()
        }
        semantic: Dict[str, Any] = {
            "frame": "cam",
            "wrist": _serializable_xyz(keypoints[0]) if keypoints.shape[0] > 0 else None,
            "fingers": fingers,
            "keypoints_3d": keypoints.astype(np.float32).tolist(),
        }
        if "tip" in fingers.get("index", {}):
            semantic["index_tip"] = fingers["index"]["tip"]
        return semantic


@app.get("/health")
def health() -> Dict[str, Any]:
    """Return service health and runtime readiness information."""
    return {
        "ok": True,
        "service": "novaplan-hand-flow",
        "backend": "hamer",
        "hamer_dir": str(HAMER_DIR),
        "runtime_loaded": _runtime is not None,
        "default_body_detector": os.environ.get("HAMER_BODY_DETECTOR", "vitdet"),
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """Run HaMeR prediction for one request."""
    video_b64 = req.rgb_videob64 or req.rgb_video_b64
    if not video_b64:
        raise HTTPException(status_code=400, detail="Missing rgb_videob64")

    t0 = time.time()
    frames = _decode_video_array(video_b64)
    raw_intrinsics = req.camera_intrinsics if req.camera_intrinsics is not None else req.intrinsics
    camera_intrinsics = _coerce_camera_intrinsics(raw_intrinsics)
    detector = os.environ.get("HAMER_BODY_DETECTOR", req.body_detector)
    try:
        runtime = _get_runtime(detector)
        meshes = runtime.predict(frames, req, camera_intrinsics=camera_intrinsics)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HaMeR prediction failed: {exc}") from exc

    return {
        "ok": True,
        "out_folder": req.out_folder,
        "num_frames": int(frames.shape[0]),
        "real_meshes": meshes if req.return_mesh_arrays else [],
        "timing": {"total_s": round(time.time() - t0, 3)},
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Serve NovaPlan hand flow using the upstream HaMeR backend.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=tcp_port, default=os.environ.get("PORT", "8080"))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
