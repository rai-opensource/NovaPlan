# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Robotics and AI Institute LLC for NovaPlan. Changes include
# MoGe2 metric-depth inputs, native camera transforms, pure-PyTorch gradient
# and blur operations, and loading a minimal RAFT runtime in a private package.

"""Consistent video depth optimization adapted from MegaSAM CVD.

Upstream source:
https://github.com/mega-sam/mega-sam/tree/a27b4e633c5cc0828a62ed943ef9f6505705fd3f/cvd_opt
"""

import os
import importlib.util
import sys
from pathlib import Path
from argparse import Namespace
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# --- Kornia Replacements (to avoid flash-attn ABI issues) ---

def gaussian_blur2d(input: torch.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[float, float]) -> torch.Tensor:
    """
    Compute Gaussian blur using pure PyTorch conv2d.
    input: [B, C, H, W]
    """
    k_size = kernel_size[0]
    s = sigma[0]
    
    # Create 1D Gaussian kernel
    x = torch.arange(k_size, device=input.device, dtype=input.dtype) - k_size // 2
    kernel_1d = torch.exp(-0.5 * (x / s) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Reshape for separable convolution: (C, 1, K, 1) and (C, 1, 1, K)
    C = input.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, k_size).repeat(C, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, k_size, 1).repeat(C, 1, 1, 1)
    
    # Pad to maintain size
    pad = k_size // 2
    
    # Separable convolution
    # 1. Convolve vertical
    blurred = F.conv2d(input, kernel_y, padding=(pad, 0), groups=C)
    # 2. Convolve horizontal
    blurred = F.conv2d(blurred, kernel_x, padding=(0, pad), groups=C)
    
    return blurred

def spatial_gradient(input: torch.Tensor, mode: str = 'sobel', normalized: bool = True) -> torch.Tensor:
    """
    Compute the first order image derivative in both x and y using pure PyTorch.
    input: [B, C, H, W]
    output: [B, C, 2, H, W] where dim 2 is (dy, dx)
    """
    B, C, H, W = input.shape
    
    if mode == 'sobel':
        # Sobel kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=input.device, dtype=input.dtype)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=input.device, dtype=input.dtype)
        if normalized:
            kernel_x = kernel_x / 8.0
            kernel_y = kernel_y / 8.0
    else: # diff
        kernel_x = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], device=input.device, dtype=input.dtype) * 0.5
        kernel_y = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], device=input.device, dtype=input.dtype) * 0.5

    # Prepare kernels for conv2d: [C, 1, 3, 3]
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    
    # Apply padding
    input_padded = F.pad(input, (1, 1, 1, 1), mode='replicate')
    
    # Compute gradients
    grad_x = F.conv2d(input_padded, kernel_x, groups=C)
    grad_y = F.conv2d(input_padded, kernel_y, groups=C)
    
    # Stack: [B, C, 2, H, W] -> (dy, dx) order matches kornia
    return torch.stack([grad_y, grad_x], dim=2)

# -----------------------------------------------------------

# Only MegaSAM's small cvd_opt/core RAFT runtime is used. Camera tracking,
# depth models, and the rest of MegaSAM are not part of the NovaPlan pipeline.
NOVAPLAN_NODE_ROOT = Path(__file__).resolve().parents[2]
CVD_OPT_DIR = Path(
    os.environ.get(
        "NOVAPLAN_CVD_RUNTIME_DIR",
        NOVAPLAN_NODE_ROOT / ".external" / "mega-sam" / "cvd_opt",
    )
)
RAFT_CORE_DIR = CVD_OPT_DIR / "core"
REQUIRED_RAFT_SOURCE_FILES = (
    RAFT_CORE_DIR / "__init__.py",
    RAFT_CORE_DIR / "raft.py",
    RAFT_CORE_DIR / "corr.py",
    RAFT_CORE_DIR / "extractor.py",
    RAFT_CORE_DIR / "update.py",
    RAFT_CORE_DIR / "utils" / "utils.py",
    RAFT_CORE_DIR / "utils" / "__init__.py",
)
RAFT_PACKAGE_NAME = "_novaplan_cvd_raft"

# Expected RAFT checkpoint location
RAFT_CHECKPOINT = Path(
    os.environ.get(
        "RAFT_CHECKPOINT",
        Path(__file__).parent / "checkpoints" / "raft-things.pth",
    )
)


def _load_raft_class():
    """Load RAFT in a private package namespace to avoid ComfyUI collisions."""
    check_cvd_runtime()
    if RAFT_PACKAGE_NAME not in sys.modules:
        init_path = RAFT_CORE_DIR / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            RAFT_PACKAGE_NAME,
            init_path,
            submodule_search_locations=[str(RAFT_CORE_DIR)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load CVD RAFT package from {init_path}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[RAFT_PACKAGE_NAME] = package
        spec.loader.exec_module(package)

    raft_name = f"{RAFT_PACKAGE_NAME}.raft"
    if raft_name not in sys.modules:
        raft_path = RAFT_CORE_DIR / "raft.py"
        spec = importlib.util.spec_from_file_location(raft_name, raft_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load CVD RAFT module from {raft_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[raft_name] = module
        spec.loader.exec_module(module)
    return sys.modules[raft_name].RAFT


def check_cvd_runtime() -> str:
    """Validate the optional source runtime used when CVD is requested."""
    missing = [str(path) for path in REQUIRED_RAFT_SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Optional CVD source runtime is incomplete. Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun: pixi run -e object-flow-host setup-object-flow-host"
        )
    return str(CVD_OPT_DIR)


def check_raft_checkpoint() -> str:
    """Check if RAFT checkpoint exists and return path, or raise informative error."""
    if not RAFT_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"RAFT checkpoint not found at {RAFT_CHECKPOINT}\n"
            "Run: pixi run -e object-flow-host setup-object-flow-host\n"
            "Or provide RAFT_CHECKPOINT_SOURCE to setup/launch."
        )
    return str(RAFT_CHECKPOINT)


_RAFT_CACHE = {}


class _RAFTArgs(Namespace):
    """MegaSAM's RAFT checks argparse-style options with membership syntax."""

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

def _load_raft_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load RAFT optical flow model."""
    # Check cache first
    cache_key = (checkpoint_path, str(device))
    if cache_key in _RAFT_CACHE:
        print("[CVD] Using cached RAFT model", flush=True)
        return _RAFT_CACHE[cache_key]

    print(f"[CVD] Loading RAFT from {checkpoint_path}", flush=True)
    RAFT = _load_raft_class()
    
    # Create args namespace that RAFT expects
    args = _RAFTArgs(
        small=False,
        mixed_precision=True,
        dropout=0,
        alternate_corr=False,
    )
    
    model = nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.module
    model.to(device)
    model.eval()
    
    # Cache the loaded model
    _RAFT_CACHE[cache_key] = model
    
    return model


def warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp image according to optical flow."""
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    
    res = cv2.remap(
        img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def resize_flow(flow: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Resize flow to target dimensions."""
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow = flow.copy()
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)
    return flow


class BackprojectDepth(nn.Module):
    """Backproject 2D pixels to 3D using depth and intrinsics."""
    
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        
        xx, yy = torch.meshgrid(
            torch.arange(self.width),
            torch.arange(self.height),
            indexing="xy",
        )
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5
        
        # Add homogeneous coordinate
        ones = torch.ones_like(pix_coords_2hw[0:1])
        pix_coords_3hw = torch.cat([pix_coords_2hw, ones], dim=0)
        pix_coords_13N = pix_coords_3hw.flatten(1).unsqueeze(0)
        
        self.register_buffer("pix_coords_13N", pix_coords_13N)
    
    def forward(self, depth_b1hw: torch.Tensor, invK_b44: torch.Tensor) -> torch.Tensor:
        """Backproject depth map to 3D points."""
        cam_points_b3N = torch.matmul(
            invK_b44[:, :3, :3], self.pix_coords_13N.float().to(depth_b1hw.device)
        )
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        
        # Add homogeneous coordinate
        ones = torch.ones_like(cam_points_b3N[:, 0:1, :])
        cam_points_b4N = torch.cat([cam_points_b3N, ones], dim=1)
        return cam_points_b4N


class NormalGenerator(nn.Module):
    """Estimate surface normals from depth maps."""
    
    def __init__(self, height: int, width: int, smoothing_kernel_size: int = 5, smoothing_kernel_std: float = 2.0):
        super().__init__()
        self.height = height
        self.width = width
        self.backproject = BackprojectDepth(height, width)
        self.kernel_size = smoothing_kernel_size
        self.std = smoothing_kernel_std
    
    def forward(self, depth_b1hw: torch.Tensor, invK_b44: torch.Tensor) -> torch.Tensor:
        """Estimate normals from depth map."""
        
        # Smooth depth
        depth_smooth_b1hw = gaussian_blur2d(
            depth_b1hw,
            (self.kernel_size, self.kernel_size),
            (self.std, self.std),
        )
        
        # Backproject to 3D
        cam_points_b4N = self.backproject(depth_smooth_b1hw, invK_b44)
        cam_points_b3hw = cam_points_b4N[:, :3].view(-1, 3, self.height, self.width)
        
        # Compute spatial gradients
        gradients_b32hw = spatial_gradient(cam_points_b3hw)
        
        # Cross product of gradients gives normal
        return F.normalize(
            torch.cross(
                gradients_b32hw[:, :, 0],
                gradients_b32hw[:, :, 1],
                dim=1,
            ),
            dim=1,
        )


def gradient_loss(gt: torch.Tensor, pred: torch.Tensor, invalid_masks: torch.Tensor) -> torch.Tensor:
    """Gradient consistency loss between predicted and ground truth depth."""
    diff = pred - gt
    v_gradient = torch.abs(diff[..., 0:-2, 1:-1] - diff[..., 2:, 1:-1])
    h_gradient = torch.abs(diff[..., 1:-1, 0:-2] - diff[..., 1:-1, 2:])
    
    pred_grad = torch.abs(pred[..., 0:-2, 1:-1] - pred[..., 2:, 1:-1]) + \
                torch.abs(pred[..., 1:-1, 0:-2] - pred[..., 1:-1, 2:])
    gt_grad = torch.abs(gt[..., 0:-2, 1:-1] - gt[..., 2:, 1:-1]) + \
              torch.abs(gt[..., 1:-1, 0:-2] - gt[..., 1:-1, 2:])
    
    grad_diff = torch.abs(pred_grad - gt_grad)
    nearby_mask = (torch.exp(gt[..., 1:-1, 1:-1]) > 1.0).float().detach()
    weight = 1.0 - torch.exp(-(grad_diff * 5.0)).detach()
    weight *= nearby_mask * (~invalid_masks[..., 1:-1, 1:-1]).float()
    
    g_loss = torch.mean(h_gradient * weight) + torch.mean(v_gradient * weight)
    return g_loss


def si_loss(gt: torch.Tensor, pred: torch.Tensor, invalid_masks: torch.Tensor) -> torch.Tensor:
    """Scale-invariant loss."""
    log_gt = torch.log(torch.clamp(gt, 1e-3, 1e3)).view(gt.shape[0], -1)
    log_pred = torch.log(torch.clamp(pred, 1e-3, 1e3)).view(pred.shape[0], -1)
    log_diff = log_gt - log_pred
    log_diff = log_diff * (~invalid_masks).float().view(gt.shape[0], -1)
    
    num_pixels = gt.shape[-2] * gt.shape[-1]
    data_loss = torch.sum(log_diff**2, dim=-1) / num_pixels - \
                torch.sum(log_diff, dim=-1) ** 2 / (num_pixels**2)
    return torch.mean(data_loss)


def sobel_fg_alpha(disp: torch.Tensor, mode: str = "sobel", beta: float = 10.0) -> torch.Tensor:
    """Compute edge-aware weights using Sobel gradients."""
    
    sobel_grad = spatial_gradient(disp, mode=mode, normalized=False)
    sobel_mag = torch.sqrt(sobel_grad[:, :, 0] ** 2 + sobel_grad[:, :, 1] ** 2)
    alpha = torch.exp(-1.0 * beta * sobel_mag).detach()
    return alpha


ALPHA_MOTION = 0.25
RESIZE_FACTOR = 0.5# 1.0


def consistency_loss(
    cam_c2w: torch.Tensor,
    K: torch.Tensor,
    K_inv: torch.Tensor,
    disp_data: torch.Tensor,
    init_disp: torch.Tensor,
    uncertainty: torch.Tensor,
    flows: torch.Tensor,
    flow_masks: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    compute_normals: list,
    fg_alpha: torch.Tensor,
    invalid_masks: torch.Tensor,
    w_ratio: float = 1.0,
    w_flow: float = 0.2,
    w_si: float = 1.0,
    w_grad: float = 2.0,
    w_normal: float = 4.0,
) -> torch.Tensor:
    """
    Compute the full CVD consistency loss.
    
    Components:
    - Flow reprojection loss
    - Depth ratio consistency loss  
    - Scale-invariant prior loss
    - Multi-scale gradient loss
    - Normal consistency loss
    """
    _, H, W = disp_data.shape
    
    # Create pixel coordinate grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W)
    yy = yy.view(1, 1, H, W)
    grid = torch.cat((xx, yy), 1).float().to(disp_data.device).permute(0, 2, 3, 1)
    
    loss_flow = 0.0
    loss_d_ratio = 0.0
    
    flows_step = flows.permute(0, 2, 3, 1)
    flow_masks_step = flow_masks.permute(0, 2, 3, 1).squeeze(-1)
    
    # Camera transformation from frame i to frame j
    cam_1to2 = torch.bmm(
        torch.linalg.inv(torch.index_select(cam_c2w, dim=0, index=jj)),
        torch.index_select(cam_c2w, dim=0, index=ii),
    )
    
    # Warp disparity from target frame
    pixel_locations = grid + flows_step
    resize_factor = torch.tensor([W - 1.0, H - 1.0]).to(disp_data.device)[None, None, None, ...]
    normalized_pixel_locations = 2 * (pixel_locations / resize_factor) - 1.0
    
    disp_sampled = F.grid_sample(
        torch.index_select(disp_data, dim=0, index=jj)[:, None, ...],
        normalized_pixel_locations,
        align_corners=True,
    )
    
    invalid_masks_sampled = F.grid_sample(
        torch.index_select(invalid_masks.to(torch.float32), dim=0, index=jj)[:, None, ...],
        normalized_pixel_locations,
        align_corners=True,
    ) > 1e-4
    
    uu = torch.index_select(uncertainty, dim=0, index=ii).squeeze(1)
    
    grid_h = torch.cat([grid, torch.ones_like(grid[..., 0:1])], dim=-1).unsqueeze(-1)
    
    # Depth of reference view
    ref_depth = 1.0 / torch.clamp(torch.index_select(disp_data, dim=0, index=ii), 1e-3, 1e3)
    flow_invalid_masks = torch.index_select(invalid_masks, dim=0, index=ii) | invalid_masks_sampled.squeeze(1)
    
    pts_3d_ref = ref_depth[..., None, None] * (K_inv[None, None, None] @ grid_h)
    rot = cam_1to2[:, None, None, :3, :3]
    trans = cam_1to2[:, None, None, :3, 3:4]
    
    pts_3d_tgt = (rot @ pts_3d_ref) + trans
    depth_tgt = pts_3d_tgt[:, :, :, 2:3, 0]
    disp_tgt = 1.0 / torch.clamp(depth_tgt, 0.1, 1e3)
    
    # Flow consistency loss
    pts_2D_tgt = K[None, None, None] @ pts_3d_tgt
    flow_masks_step_ = flow_masks_step * (pts_2D_tgt[:, :, :, 2, 0] > 0.1) * (~flow_invalid_masks).float()
    pts_2D_tgt = pts_2D_tgt[:, :, :, :2, 0] / torch.clamp(pts_2D_tgt[:, :, :, 2:, 0], 1e-3, 1e3)
    
    disp_sampled = torch.clamp(disp_sampled, 1e-3, 1e2)
    disp_tgt = torch.clamp(disp_tgt, 1e-3, 1e2)
    
    ratio = torch.maximum(
        disp_sampled.squeeze() / disp_tgt.squeeze(),
        disp_tgt.squeeze() / disp_sampled.squeeze(),
    )
    ratio_error = torch.abs(ratio - 1.0)
    
    loss_d_ratio += torch.sum(
        (ratio_error * uu + ALPHA_MOTION * torch.log(1.0 / uu)) * flow_masks_step_
    ) / (torch.sum(flow_masks_step_) + 1e-8)
    
    flow_error = torch.abs(pts_2D_tgt - pixel_locations)
    loss_flow += torch.sum(
        (flow_error * uu[..., None] + ALPHA_MOTION * torch.log(1.0 / uu[..., None]))
        * flow_masks_step_[..., None]
    ) / (torch.sum(flow_masks_step_) * 2.0 + 1e-8)
    
    # Prior loss
    loss_prior = si_loss(init_disp, disp_data, invalid_masks)
    KK = torch.inverse(K_inv)
    
    # Normal consistency
    disp_data_ds = disp_data[:, None, ...]
    init_disp_ds = init_disp[:, None, ...]
    K_rescale = KK.clone()
    K_inv_rescale = torch.inverse(K_rescale)
    
    pred_normal = compute_normals[0](
        1.0 / torch.clamp(disp_data_ds, 1e-3, 1e3), K_inv_rescale[None]
    )
    init_normal = compute_normals[0](
        1.0 / torch.clamp(init_disp_ds, 1e-3, 1e3), K_inv_rescale[None]
    )
    fg_alpha_masked = fg_alpha * (~invalid_masks).float()
    loss_normal = torch.mean(
        fg_alpha_masked * (1.0 - torch.sum(pred_normal * init_normal, dim=1))
    )
    
    # Multi-scale gradient loss
    loss_grad = 0.0
    for scale in range(4):
        interval = 2 ** scale
        disp_data_ds = F.interpolate(
            disp_data[:, None, ...],
            scale_factor=(1.0 / interval, 1.0 / interval),
            mode="nearest-exact",
        )
        init_disp_ds = F.interpolate(
            init_disp[:, None, ...],
            scale_factor=(1.0 / interval, 1.0 / interval),
            mode="nearest-exact",
        )
        resized_invalid_masks = F.interpolate(
            invalid_masks[:, None].to(torch.uint8),
            scale_factor=(1.0 / interval, 1.0 / interval),
            mode="nearest-exact",
        ).to(torch.bool)
        loss_grad += gradient_loss(
            torch.log(disp_data_ds), torch.log(init_disp_ds), invalid_masks=resized_invalid_masks
        )
    
    return (
        w_ratio * loss_d_ratio
        + w_si * loss_prior
        + w_flow * loss_flow
        + w_normal * loss_normal
        + loss_grad * w_grad
    )


class CVDOptimizer:
    """
    Consistent Video Depth Optimizer.
    
    Optimizes depth maps for temporal consistency using optical flow constraints.
    """
    
    def __init__(self, device: torch.device, resolution: int = 384 * 512):
        """
        Initialize CVD optimizer.
        
        Args:
            device: PyTorch device
            resolution: Target resolution in pixels for processing (default 384*512=196608)
        """
        self.device = device
        self.resolution = resolution
        self.raft_model = None
    
    def _ensure_raft_loaded(self):
        """Lazy-load RAFT model."""
        if self.raft_model is None:
            checkpoint_path = check_raft_checkpoint()
            self.raft_model = _load_raft_model(checkpoint_path, self.device)
    
    def compute_optical_flow(
        self, 
        images: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute optical flow between frame pairs at multiple temporal scales.
        
        Args:
            images: [N, H, W, 3] uint8 numpy array
            
        Returns:
            flows: [P, 2, H_small, W_small] tensor
            flow_masks: [P, 1, H_small, W_small] tensor
            ii: [P] source frame indices
            jj: [P] target frame indices
        """
        self._ensure_raft_loaded()
        
        N, h0, w0, _ = images.shape
        
        # Resize to target resolution
        h1 = int(h0 * np.sqrt(self.resolution / (h0 * w0)))
        w1 = int(w0 * np.sqrt(self.resolution / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        
        # Preprocess images
        img_data = []
        for t in range(N):
            image = cv2.resize(images[t], (w1, h1))
            image = image.transpose(2, 0, 1)  # [3, H, W]
            img_data.append(image)
        img_data = np.array(img_data)  # [N, 3, H, W]
        
        ii_list = []
        jj_list = []
        flows_arr_up = []
        masks_arr_up = []
        
        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}
        
        # Multi-scale temporal steps
        for step in [1, 2, 4, 8, 15]:
            if step >= N:
                continue
                
            for i in range(max(0, -step), N - max(0, step)):
                image1 = torch.as_tensor(np.ascontiguousarray(img_data[i:i+1])).float().to(self.device)
                image2 = torch.as_tensor(np.ascontiguousarray(img_data[i+step:i+step+1])).float().to(self.device)
                
                ii_list.append(i)
                jj_list.append(i + step)
                
                with torch.no_grad():
                    # Initialize from previous scale if available
                    if np.abs(step) > 1 and i in flows_arr_low_fwd and (i + step) in flows_arr_low_bwd:
                        flow_init = np.stack(
                            [flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0
                        )
                        flow_init = torch.as_tensor(np.ascontiguousarray(flow_init)).float().to(self.device).permute(0, 3, 1, 2)
                    else:
                        flow_init = None
                    
                    flow_low, flow_up, _ = self.raft_model(
                        torch.cat([image1, image2], dim=0),
                        torch.cat([image2, image1], dim=0),
                        iters=22,
                        test_mode=True,
                        flow_init=flow_init,
                    )
                    
                    flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)
                    
                    # Keep flow at RAFT output resolution (don't downscale by 2)
                    flow_up_fwd = flow_up[0].cpu().numpy().transpose(1, 2, 0)
                    flow_up_bwd = flow_up[1].cpu().numpy().transpose(1, 2, 0)
                    
                    # Forward-backward consistency check
                    bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = fwd_lr_error < 1.0
                    
                    flows_arr_low_bwd[i + step] = flow_low_bwd
                    flows_arr_low_fwd[i] = flow_low_fwd
                    
                    flows_arr_up.append(flow_up_fwd)
                    masks_arr_up.append(fwd_mask_up)
        
        iijj = np.stack((ii_list, jj_list), axis=0)
        flows_high = np.array(flows_arr_up).transpose(0, 3, 1, 2)
        flow_masks_high = np.array(masks_arr_up)[:, None, ...]
        
        return (
            torch.from_numpy(np.float32(flows_high)).to(self.device),
            torch.from_numpy(flow_masks_high).float().to(self.device),
            torch.from_numpy(iijj[0]).long().to(self.device),
            torch.from_numpy(iijj[1]).long().to(self.device),
        )
    
    def optimize(
        self,
        depth_prior: torch.Tensor,
        images: np.ndarray,
        intrinsics: torch.Tensor,
        cam_poses: Optional[torch.Tensor] = None,
        motion_prob: Optional[torch.Tensor] = None,
        w_grad: float = 2.0,
        w_normal: float = 5.0,
        scale_shift_iters: int = 100,
        depth_iters: int = 400,
        freeze_shift: bool = False,
    ) -> torch.Tensor:
        """
        Run CVD optimization on depth maps.
        
        Args:
            depth_prior: [N, H, W] per-frame MoGe2 depth maps
            images: [N, H, W, 3] uint8 RGB images
            intrinsics: [3, 3] camera intrinsic matrix
            cam_poses: [N, 4, 4] camera extrinsics (c2w), or None for identity
            motion_prob: [N, H, W] motion probability masks, or None
            w_grad: Gradient loss weight (default 2.0)
            w_normal: Normal loss weight (default 5.0) 
            scale_shift_iters: Iterations for scale/shift alignment
            depth_iters: Iterations for depth refinement
            freeze_shift: If True, don't optimize shift (for metric depth like MoGe)
            
        Returns:
            optimized_depth: [N, H, W] refined depth maps
        """
        N, H_orig, W_orig = depth_prior.shape
        
        print(f"[CVD] Starting optimization: {N} frames at {H_orig}x{W_orig}", flush=True)
        
        # Step 1: Compute optical flow
        print("[CVD] Computing optical flow...", flush=True)
        flows, flow_masks, ii, jj = self.compute_optical_flow(images)
        
        # Get flow resolution
        _, _, H_flow, W_flow = flows.shape
        
        # Step 2: Prepare depth data at optimization resolution
        # Use RESIZE_FACTOR (0.5) for optimization
        H_opt = int(H_flow * RESIZE_FACTOR)
        W_opt = int(W_flow * RESIZE_FACTOR)
        
        # Resize depth to flow resolution first, then to optimization resolution
        disp_prior = 1.0 / (depth_prior + 1e-6)  # Convert to disparity
        invalid_masks_orig = depth_prior <= 0
        
        # Resize to flow resolution
        disp_data = F.interpolate(
            disp_prior[:, None, ...].to(self.device),
            size=(H_flow, W_flow),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        
        # Resize invalid masks
        invalid_masks = F.interpolate(
            invalid_masks_orig[:, None, ...].to(torch.float32).to(self.device),
            size=(H_flow, W_flow),
            mode="nearest-exact",
        ).squeeze(1) > 0.5
        
        # Further resize for optimization
        disp_data = F.interpolate(
            disp_data[:, None, ...],
            scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
            mode="bilinear",
        ).squeeze(1)
        
        invalid_masks = F.interpolate(
            invalid_masks[:, None, ...].to(torch.float32),
            scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR),
            mode="bilinear",
        ).squeeze(1) > 1e-4
        
        disp_data = disp_data + 1e-6
        init_disp = disp_data.clone()
        
        # Resize flows and masks for optimization
        flows = F.interpolate(flows, scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR), mode="bilinear")
        flows = flows * RESIZE_FACTOR  # Scale flow values too
        flow_masks = F.interpolate(flow_masks, scale_factor=(RESIZE_FACTOR, RESIZE_FACTOR), mode="nearest-exact")
        
        print(f"[CVD DEBUG] flows shape: {flows.shape}")
        print(f"[CVD DEBUG] flow_masks shape: {flow_masks.shape}")
        print(f"[CVD DEBUG] ii shape: {ii.shape}, jj shape: {jj.shape}")
        
        if ii.shape[0] == 0:
            print("[CVD Warning] No flow pairs found! Optimization will be skipped/ineffective.")
            return depth_prior
        
        # Step 3: Prepare camera poses
        if cam_poses is None:
            camera_matrices = torch.eye(
                4, device=self.device, dtype=torch.float32
            ).unsqueeze(0).repeat(N, 1, 1)
        else:
            if tuple(cam_poses.shape) != (N, 4, 4):
                raise ValueError(
                    f"cam_poses must have shape {(N, 4, 4)}, got "
                    f"{tuple(cam_poses.shape)}"
                )
            # Preserve the previous inverse-transform convention. Camera
            # transforms are fixed during CVD.
            camera_matrices = torch.linalg.inv(
                cam_poses.to(device=self.device, dtype=torch.float32)
            )
        
        # Step 4: Prepare intrinsics (scale for optimization resolution) 
        K = torch.eye(3, device=self.device)
        scale_x = W_opt / W_orig
        scale_y = H_opt / H_orig
        K[0, 0] = intrinsics[0, 0] * scale_x
        K[1, 1] = intrinsics[1, 1] * scale_y
        K[0, 2] = intrinsics[0, 2] * scale_x
        K[1, 2] = intrinsics[1, 2] * scale_y
        K_inv = torch.linalg.inv(K)
        
        # Step 5: Prepare uncertainty (motion probability)
        if motion_prob is not None:
            cvd_prob = F.interpolate(
                motion_prob[:, None, ...].to(self.device),
                size=(H_opt, W_opt),
                mode="bilinear",
            )
        else:
            # Default uniform uncertainty
            cvd_prob = torch.ones(N, 1, H_opt, W_opt, device=self.device) * 0.5
        
        cvd_prob = torch.where(cvd_prob > 0.5, torch.tensor(0.5, device=self.device), cvd_prob)
        cvd_prob = torch.clamp(cvd_prob, 1e-3, 1.0)
        uncertainty = cvd_prob.clone().detach()  # Detach before making it a leaf tensor
        
        # Edge-aware weights
        fg_alpha = sobel_fg_alpha(init_disp[:, None, ...]) > 0.2
        fg_alpha = fg_alpha.squeeze(1).float() + 0.2
        
        # Normal generator
        compute_normals = [NormalGenerator(H_opt, W_opt).to(self.device)]
        init_disp = torch.clamp(init_disp, 1e-3, 1e3).clone().detach()  # Detach for reference
        
        # ====== Stage 1: Scale/Shift Optimization ======
        if freeze_shift:
            print(f"[CVD] Stage 1: Scale optimization only ({scale_shift_iters} iters, shift frozen for metric depth)...", flush=True)
        else:
            print(f"[CVD] Stage 1: Scale/shift optimization ({scale_shift_iters} iters)...", flush=True)
        
        # Use enable_grad to ensure gradients work even if ComfyUI disabled them
        # Force enable grad globally and DISABLE inference mode
        # ComfyUI runs in inference_mode which overrides enable_grad
        with torch.inference_mode(mode=False), torch.enable_grad():
            print(f"  [DEBUG] torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            print(f"  [DEBUG] torch.is_inference_mode_enabled(): {torch.is_inference_mode_enabled()}")
            
            # Create leaf tensors INSIDE the enabled gradient context
            log_scale_ = torch.zeros(N, device=self.device, requires_grad=True)
            shift_ = torch.zeros(N, device=self.device, requires_grad=not freeze_shift)
            # Re-clone uncertainty to ensure it's tracked in this context
            uncertainty_opt = uncertainty.clone().detach().requires_grad_(True)
            
            # CRITICAL: Clone ALL external tensors inside this context to convert them 
            # from "Inference Tensors" to regular tensors that support autograd
            disp_data_opt = disp_data.clone()
            init_disp_opt = init_disp.clone()
            K_opt = K.clone()
            K_inv_opt = K_inv.clone()
            
            flows_opt = flows.clone()
            flow_masks_opt = flow_masks.clone()
            ii_opt = ii.clone()
            jj_opt = jj.clone()
            fg_alpha_opt = fg_alpha.clone()
            invalid_masks_opt = invalid_masks.clone()
            camera_matrices_opt = camera_matrices.clone()
            
            # Build optimizer params - exclude shift if frozen
            optim_params = [
                {"params": log_scale_, "lr": 1e-2},
                {"params": uncertainty_opt, "lr": 1e-2},
            ]
            if not freeze_shift:
                optim_params.insert(1, {"params": shift_, "lr": 1e-2})
            
            optim = torch.optim.Adam(optim_params)
            
            for i in range(scale_shift_iters):
                optim.zero_grad()
                scale_ = torch.exp(log_scale_)
                
                # Apply scale and optionally shift
                if freeze_shift:
                    scaled_disp = disp_data_opt * scale_[..., None, None]
                else:
                    scaled_disp = disp_data_opt * scale_[..., None, None] + shift_[..., None, None]
                
                loss = consistency_loss(
                    camera_matrices_opt, K_opt, K_inv_opt,
                    torch.clamp(scaled_disp, 1e-3, 1e3),
                    init_disp_opt,
                    torch.clamp(uncertainty_opt, 1e-4, 1e3),
                    flows_opt, flow_masks_opt, ii_opt, jj_opt,
                    compute_normals, fg_alpha_opt,
                    invalid_masks=invalid_masks_opt,
                    w_grad=w_grad, w_normal=w_normal,
                )
                
                loss.backward()
                uncertainty_opt.grad = torch.nan_to_num(uncertainty_opt.grad, nan=0.0)
                log_scale_.grad = torch.nan_to_num(log_scale_.grad, nan=0.0)
                if not freeze_shift:
                    shift_.grad = torch.nan_to_num(shift_.grad, nan=0.0)
                
                optim.step()
                
                if i % 20 == 0:
                    print(f"  Iter {i}: loss = {loss.item():.4f}", flush=True)
            
            # Update uncertainty for next stage
            uncertainty = uncertainty_opt.detach()

        # Apply learned scale/shift
        # IMPORTANT: Detach optimized parameters before mixing with inference tensors
        scale_factor = torch.exp(log_scale_.detach())[..., None, None]
        shift_factor = shift_.detach()[..., None, None] if not freeze_shift else 0.0
        
        disp_data = (disp_data * scale_factor + shift_factor).clone().detach()
        init_disp = (init_disp * scale_factor + shift_factor).clone().detach()
        init_disp = torch.clamp(init_disp, 1e-3, 1e3).clone().detach()
        
        # ====== Stage 2: Depth Optimization ======
        print(f"[CVD] Stage 2: Depth optimization ({depth_iters} iters)...", flush=True)
        
        # Use enable_grad to ensure gradients work even if ComfyUI disabled them
        with torch.inference_mode(mode=False), torch.enable_grad():
            # Create new leaf tensors for optimization
            # disp_data is the starting point, we want to optimize it further
            disp_data_opt = disp_data.clone().detach().requires_grad_(True)
            uncertainty_opt = uncertainty.clone().detach().requires_grad_(True)
            
            # Clone constants to be safe for autograd
            init_disp_opt = init_disp.clone()
            K_opt = K.clone()
            K_inv_opt = K_inv.clone()
            
            flows_opt = flows.clone()
            flow_masks_opt = flow_masks.clone()
            ii_opt = ii.clone()
            jj_opt = jj.clone()
            fg_alpha_opt = fg_alpha.clone()
            invalid_masks_opt = invalid_masks.clone()
            camera_matrices_opt = camera_matrices.clone()
            
            optim = torch.optim.Adam([
                {"params": disp_data_opt, "lr": 5e-3},
                {"params": uncertainty_opt, "lr": 5e-3},
            ])
            
            for i in range(depth_iters):
                optim.zero_grad()
                loss = consistency_loss(
                    camera_matrices_opt, K_opt, K_inv_opt,
                    torch.clamp(disp_data_opt, 1e-3, 1e3),
                    init_disp_opt,
                    torch.clamp(uncertainty_opt, 1e-4, 1e3),
                    flows_opt, flow_masks_opt, ii_opt, jj_opt,
                    compute_normals, fg_alpha_opt,
                    invalid_masks=invalid_masks_opt,
                    w_ratio=1.0, w_flow=0.2, w_si=1.0,
                    w_grad=w_grad, w_normal=w_normal,
                )
                
                loss.backward()
                disp_data_opt.grad = torch.nan_to_num(disp_data_opt.grad, nan=0.0)
                uncertainty_opt.grad = torch.nan_to_num(uncertainty_opt.grad, nan=0.0)
                
                optim.step()
                
                if i % 50 == 0:
                    print(f"  Iter {i}: loss = {loss.item():.4f}", flush=True)
        
        # Step 6: Convert back to depth and upscale
        # Use the optimized disparity from Stage 2
        disp_data_opt = F.interpolate(
            disp_data_opt[:, None, ...].detach(),
            size=(H_orig, W_orig),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        
        depths_opt = torch.clamp(1.0 / disp_data_opt, 1e-3, 1e2)
        
        # Handle invalid regions
        invalid_masks_orig = invalid_masks_orig.to(self.device)
        depths_opt[invalid_masks_orig] = 0.0
        
        print("[CVD] Optimization complete!", flush=True)
        
        return depths_opt
