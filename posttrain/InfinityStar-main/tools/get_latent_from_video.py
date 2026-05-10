
import argparse
import torch
import os
import sys
import torchvision
from torchvision.transforms import functional as F_transforms

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Fix for missing CacheLimitExceeded
try:
    from torch._dynamo.exc import CacheLimitExceeded
except ImportError:
    import types
    if 'torch._dynamo.exc' not in sys.modules:
        sys.modules['torch._dynamo.exc'] = types.ModuleType('torch._dynamo.exc')
    class CacheLimitExceeded(Exception): pass
    sys.modules['torch._dynamo.exc'].CacheLimitExceeded = CacheLimitExceeded

from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import video_vae_model
from infinity.models.videovae.models.wan_bsq_vae import patchify
from infinity.models.videovae.modules import DiagonalGaussianDistribution

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--save_path", type=str, default="video_latent.pt", help="Path to save latent tensor")
    parser.add_argument("--vae_path", type=str, default="/home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main/checkpoint/infinitystar_videovae.pth")
    parser.add_argument("--vae_type", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample", action="store_true", help="Sample from distribution instead of taking mode (deterministic)")
    return parser.parse_args()

def preprocess_video(video_path, target_h=480, target_w=720):
    """
    Reads video, resizes, crops, and normalizes to [-1, 1].
    Returns: (1, 3, T, H, W)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    # vframes, aframes, info = torchvision.io.read_video(video_path, output_format="TCHW")
    # Using TCHW output format directly if supported, else permute
    vframes, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
    # read_video returns (T, H, W, C) in [0, 255]
    
    vframes = vframes.permute(0, 3, 1, 2).float() / 255.0 # (T, C, H, W) in [0, 1]
    
    # Normalize to [-1, 1]
    vframes = (vframes * 2.0) - 1.0
    
    # Resize/Center Crop to target resolution
    # Note: Simple resize for demo; in production, use careful cropping
    vframes = F_transforms.resize(vframes, [target_h, target_w])
    # If aspect ratio doesn't match, center crop
    vframes = F_transforms.center_crop(vframes, [target_h, target_w])
    
    # Add batch dimension and permute to (B, C, T, H, W)
    vframes = vframes.permute(1, 0, 2, 3).unsqueeze(0) # (1, C, T, H, W)
    
    return vframes

def encode_video(vae, video_tensor, device, sample=False):
    """
    Encodes video tensor to InfinityStar latent.
    Video Tensor: (B, C, T, H, W) in [-1, 1]
    """
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        # 1. Encode
        h = vae.encode(video_tensor)
        
        # 2. Patchify
        h_patched = patchify(h)
        
        # 3. Distribution
        posterior = DiagonalGaussianDistribution(h_patched)
        
        if sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
            
        # 4. Projection (Standard InfinityStar config uses use_feat_proj=2)
        # Note: We assume use_feat_proj=2 based on codebase analysis
        if hasattr(vae, 'proj_down'):
            z = vae.proj_down(z.permute(0,2,3,4,1)).permute(0,4,1,2,3)
            
        # 5. Scale
        z = z * vae.scale_learnable_parameters[0]
        
    return z

def main():
    args = get_args()
    
    # Mock global args for VAE loading
    global_args = argparse.Namespace()
    global_args.semantic_scale_dim = 16
    global_args.detail_scale_dim = 64
    global_args.use_learnable_dim_proj = 0
    global_args.detail_scale_min_tokens = 80
    global_args.use_feat_proj = 2
    global_args.semantic_scales = 8
    global_args.vae_type = args.vae_type
    global_args.videovae = 10
    global_args.vae_path = args.vae_path

    print(f"Loading VAE from {args.vae_path}...")
    vae = video_vae_model(
        vqgan_ckpt=args.vae_path,
        schedule_mode="dynamic",
        codebook_dim=args.vae_type,
        global_args=global_args,
        test_mode=True
    ).to(args.device)
    print("VAE Loaded.")

    print(f"Processing video: {args.video_path}")
    try:
        video_tensor = preprocess_video(args.video_path)
        print(f"Video tensor shape: {video_tensor.shape}")
        
        latent = encode_video(vae, video_tensor, args.device, sample=args.sample)
        print(f"Latent shape: {latent.shape}")
        
        torch.save(latent, args.save_path)
        print(f"Latent saved to {args.save_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
