# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Track 2D video points with a CoTracker3 ComfyUI node."""

import torch
import numpy as np
import os
import cv2
import torch.hub

class CoTracker3Node:
    """Expose CoTracker3 video point tracking as a ComfyUI node."""
    DESCRIPTION = "NovaPlan CoTracker3 video flow; novaplan-model-ownership-v1"

    # Class-level cache for model (keeps it in VRAM)
    _cached_model = None
    _cached_checkpoint = None
    _device = None
    
    @classmethod
    def INPUT_TYPES(s):
        """Declare the ComfyUI input specification."""
        return {
            "required": {
                "video": ("VIDEO",),
                "grid_size": ("INT", {"default": 30, "min": 1, "max": 256}),
                "checkpoint": ("STRING", {"default": "scaled_offline.pth"}),
            },
            "optional": {
                "mask": ("MASK",),
                "num_vis_points": ("INT", {"default": 0, "min": 0, "max": 10000}),  # 0 = show all
            }
        }

    RETURN_TYPES = ("IMAGE", "TENSOR", "TENSOR")
    RETURN_NAMES = ("visualized_flow_image", "pred_tracks", "pred_visibility")
    FUNCTION = "process"
    CATEGORY = "NovaPlan/CoTracker3"

    @classmethod
    def get_model(cls, checkpoint):
        """Get or create cached CoTracker3 model to keep it in VRAM"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve checkpoint path
        if not os.path.isabs(checkpoint):
            base_path = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(base_path, checkpoint)
        else:
            checkpoint_path = checkpoint
        
        # Return cached model if same checkpoint
        if cls._cached_model is not None and cls._cached_checkpoint == checkpoint_path:
            return cls._cached_model, device
        
        print("[CoTracker3] Loading model (will be cached for future requests)")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load model structure without pretrained weights
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", pretrained=False)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Handle state_dict wrapped in a dict (e.g. "model", "state_dict")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # Handle "module." prefix from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

        # Add the wrapper namespace when the checkpoint stores bare model keys.
        if "time_emb" in state_dict and "model.time_emb" not in state_dict:
            print("CoTracker3Node: Prefixing state_dict keys with 'model.' to match CoTrackerPredictor structure.")
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[f"model.{k}"] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # This cached model is inference-only.
        for param in model.parameters():
            param.requires_grad = False
        
        # Cache the model
        cls._cached_model = model
        cls._cached_checkpoint = checkpoint_path
        cls._device = device
        
        print(f"🟢 CoTracker3 Model loaded and cached on {device}")
        
        return model, device

    def process(self, video, grid_size, checkpoint, mask=None, num_vis_points=0):
        """Execute this ComfyUI node."""
        print(f"🔍 CoTracker3: grid_size={grid_size}, num_vis_points={num_vis_points}")
        
        # Get cached model
        model, device = self.get_model(checkpoint)

        # 2. Prepare Video Input
        if hasattr(video, 'get_components'):
            # It's a VideoFromComponents object
            components = video.get_components()
            video_frames = components.images # [T, H, W, 3] or [T, H, W, C]
        else:
            # It's a tensor
            video_frames = video
            
        # Ensure video is [B, T, 3, H, W] for CoTracker
        # ComfyUI video is usually [T, H, W, C] (RGB) and float 0-1 or uint8 0-255?
        # Typically ComfyUI passes [T, H, W, C] float 0-1.
        
        T, H, W, C = video_frames.shape
        
        # CoTracker expects [B, T, 3, H, W] in 0-255 range
        video_tensor = video_frames.permute(0, 3, 1, 2).unsqueeze(0).float() # [1, T, 3, H, W]
        if video_tensor.max() <= 1.0:
            video_tensor = video_tensor * 255.0
            
        video_tensor = video_tensor.to(device)
        
        # 3. Prepare Queries
        queries = None
        if mask is not None:
            # mask: [H, W] or [T, H, W]
            if mask.dim() == 3:
                mask_2d = mask[0] # Use first frame mask
            else:
                mask_2d = mask
                
            # Resize mask to current video resolution if needed? 
            # Usually mask matches video resolution in ComfyUI if they come from same source.
            # But let's verify or resize just in case? 
            # If mask shape != (H, W), resize it.
            if mask_2d.shape != (H, W):
                mask_np = mask_2d.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_2d = torch.from_numpy(mask_resized).to(device)
            else:
                mask_2d = mask_2d.to(device)

            # Sample grid_size * grid_size points
            num_points = grid_size * grid_size
            
            y, x = torch.where(mask_2d > 0.5)
            if len(y) == 0:
                print("Warning: No active pixels in mask. Fallback to grid.")
                queries = None
            else:
                if len(y) > num_points:
                    # Random sample
                    idx = torch.randperm(len(y))[:num_points]
                    y = y[idx]
                    x = x[idx]
                
                # Create queries: [1, N, 3] where last dim is (t, x, y)
                # t is 0 for all
                t_col = torch.zeros_like(x).float()
                queries = torch.stack([t_col, x.float(), y.float()], dim=1).unsqueeze(0).to(device) # [1, N, 3]

        # 4. Inference
        with torch.no_grad():
            if queries is not None:
                pred_tracks, pred_visibility = model(video_tensor, queries=queries)
            else:
                pred_tracks, pred_visibility = model(video_tensor, grid_size=grid_size)
        
        # pred_tracks: [1, T, N, 2]
        # pred_visibility: [1, T, N]
        
        # 5. Visualization
        # Render trajectories over the first frame to match TAPIP3D diagnostics.
        # Draw tracks on first frame.
        
        # Prepare data for visualization
        # video_np is [T, H, W, 3] uint8
        video_np = (video_frames.cpu().numpy() * 255).astype(np.uint8)
        
        # Use first frame as canvas
        vis_img = video_np[0].copy() # [H, W, 3] RGB
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR) # BGR for OpenCV
        
        tracks = pred_tracks[0].cpu().numpy() # [T, N, 2]
        visibilities = pred_visibility[0].cpu().numpy() # [T, N]
        
        num_tracks = tracks.shape[1]
        
        # Subsample tracks for visualization if num_vis_points is specified
        if num_vis_points > 0 and num_vis_points < num_tracks:
            vis_indices = np.linspace(0, num_tracks - 1, num_vis_points, dtype=int)
            tracks_vis = tracks[:, vis_indices, :]
            visibilities_vis = visibilities[:, vis_indices]
            num_tracks_vis = num_vis_points
        else:
            tracks_vis = tracks
            visibilities_vis = visibilities
            num_tracks_vis = num_tracks
        
        # Generate colors (Rainbow)
        hues = np.linspace(0, 179, num_tracks_vis, dtype=np.uint8)
        hsv = np.zeros((num_tracks_vis, 1, 3), dtype=np.uint8)
        hsv[:, 0, 0] = hues
        hsv[:, 0, 1] = 255
        hsv[:, 0, 2] = 255
        colors_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(num_tracks_vis, 3)
        colors_bgr = colors_rgb[:, ::-1] # [N, 3]
        
        # Draw trajectories
        # tracks_vis is [T, N, 2]
        # We draw lines connecting points for each track where visibility is good
        
        # Transpose to [N, T, 2] for easier iterating per track
        tracks_per_point = tracks_vis.transpose(1, 0, 2) # [N, T, 2]
        vis_per_point = visibilities_vis.transpose(1, 0) # [N, T]
        
        for i in range(num_tracks_vis):
            # Get valid points
            # We only draw segments where visibility > 0.5 (or some threshold)
            # Actually, typically we just draw the whole predicted track if it started visible?
            # Or draw valid segments. TAPIP3D example filters by vis > 0.5.
            
            pts = tracks_per_point[i] # [T, 2]
            vis = vis_per_point[i] # [T]
            
            # Create segments
            # A simple way is to just take all points where vis > 0.5
            valid_mask = vis > 0.5
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) < 2:
                continue
                
            valid_pts = pts[valid_indices].astype(np.int32)
            valid_pts = valid_pts.reshape((-1, 1, 2))
            
            color = colors_bgr[i].tolist()
            cv2.polylines(vis_img, [valid_pts], isClosed=False, color=color, thickness=1)
            
        # Convert back to RGB
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        
        # Convert to Tensor [1, H, W, 3] float 0-1
        vis_tensor = torch.from_numpy(vis_img).float() / 255.0
        vis_tensor = vis_tensor.unsqueeze(0)
        
        return (vis_tensor, pred_tracks, pred_visibility)

NODE_CLASS_MAPPINGS = {
    "CoTracker3Node": CoTracker3Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoTracker3Node": "CoTracker3",
}
