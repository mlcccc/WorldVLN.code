
import sys
import os
import torch
import cv2
import numpy as np
import argparse
import types
import math

# 1. Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# 2. Mock CacheLimitExceeded
try:
    from torch._dynamo.exc import CacheLimitExceeded
except ImportError:
    if 'torch._dynamo.exc' not in sys.modules:
        exc_module = types.ModuleType('torch._dynamo.exc')
        sys.modules['torch._dynamo.exc'] = exc_module
    class CacheLimitExceeded(Exception): pass
    sys.modules['torch._dynamo.exc'].CacheLimitExceeded = CacheLimitExceeded

# 3. Imports
from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import video_vae_model
from infinity.models.videovae.models.wan_bsq_vae import patchify, unpatchify
from infinity.models.videovae.modules import DiagonalGaussianDistribution

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs='+', default=[
        "/home/batchcom/dataset-link/xjc/uavflowdatasim_output/0", 
        "/home/batchcom/dataset-link/xjc/uavflowdatasim_output/1"
    ], help="List of folders containing video.mp4 and video_summed_codes.npy")
    parser.add_argument("--vae_path", type=str, default='/home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main/checkpoint/infinitystar_videovae.pth')
    parser.add_argument("--vae_type", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Global args mocks for VAE loading
    args.semantic_scale_dim = 16
    args.detail_scale_dim = 64
    args.use_learnable_dim_proj = 0
    args.detail_scale_min_tokens = 80
    args.use_feat_proj = 2
    args.semantic_scales = 8
    
    return args

def read_video_cv2(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames) # (T, H, W, C)

def write_video_cv2(video_path, frames_uint8, fps=10):
    if len(frames_uint8) == 0:
        return
    T, H, W, C = frames_uint8.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    
    for i in range(T):
        frame = frames_uint8[i]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

def process_reconstruction(folder, model, device='cuda'):
    print(f"\nProcessing {folder}...")
    
    npy_path = os.path.join(folder, "video_summed_codes.npy")
    video_path = os.path.join(folder, "video.mp4")
    
    if not os.path.exists(npy_path):
        print(f"Skipping: {npy_path} not found")
        return
    if not os.path.exists(video_path):
        print(f"Skipping: {video_path} not found")
        return
        
    # 1. Load NPY (16-channel summed codes)
    try:
        z_infinity_np = np.load(npy_path)
        z_infinity = torch.from_numpy(z_infinity_np).to(device) # (1, 16, T_lat, H_lat, W_lat)
        print(f"Loaded latent shape: {z_infinity.shape}")
    except Exception as e:
        print(f"Error loading npy: {e}")
        return

    # 2. Convert back to decode-ready latent (16 -> 64 channels via patchify)
    # The VAE decode expects the patched version (64 channels) if it's the specific WanVAE model used here.
    # Wait, in verify_vae_real_video.py we saw:
    # z_infinity = unpatchify(z)
    # z_repatched = patchify(z_infinity)
    # recon = model.decode(z_repatched)
    
    with torch.no_grad():
        z_repatched = patchify(z_infinity)
        print(f"Repatched latent shape: {z_repatched.shape}")
        
        # Decode
        recon_video = model.decode(z_repatched)
        print(f"Reconstructed video tensor shape: {recon_video.shape}")

    # 3. Post-process Reconstruction
    recon_video = torch.clamp(recon_video, -1.0, 1.0)
    recon_uint8 = ((recon_video[0].permute(1, 2, 3, 0).cpu() + 1.0) * 127.5).byte()
    
    # 4. Load Original Video
    orig_frames_np = read_video_cv2(video_path)
    orig_uint8 = torch.from_numpy(orig_frames_np) # (T, H, W, C)
    print(f"Original video shape: {orig_uint8.shape}")
    
    # 5. Align Dimensions
    # Time
    T_recon = recon_uint8.shape[0]
    T_orig = orig_uint8.shape[0]
    min_T = min(T_recon, T_orig)
    recon_uint8 = recon_uint8[:min_T]
    orig_uint8 = orig_uint8[:min_T]
    
    # Spatial (Crop if needed, usually VAE outputs multiples of 16/32)
    H_recon, W_recon = recon_uint8.shape[1:3]
    H_orig, W_orig = orig_uint8.shape[1:3]
    min_H = min(H_recon, H_orig)
    min_W = min(W_recon, W_orig)
    
    recon_uint8 = recon_uint8[:, :min_H, :min_W, :]
    orig_uint8 = orig_uint8[:, :min_H, :min_W, :]
    
    # 6. Calculate Metrics
    mse = torch.mean((orig_uint8.float() - recon_uint8.float()) ** 2).item()
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(255.0 / math.sqrt(mse))
        
    print(f"Reconstruction Error (MSE): {mse:.4f}")
    print(f"Reconstruction Quality (PSNR): {psnr:.2f} dB")
    
    # 7. Save Comparison
    comp = torch.cat([orig_uint8, recon_uint8], dim=2)
    save_path = os.path.join(folder, "comparison_npy_recon.mp4")
    write_video_cv2(save_path, comp.numpy(), fps=10)
    print(f"Saved comparison to {save_path}")

def main():
    args = get_args()
    device = args.device
    print(f"Using device: {device}")
    
    print(f"Loading VAE from {args.vae_path}...")
    model = video_vae_model(
        vqgan_ckpt=args.vae_path,
        schedule_mode="dynamic",
        codebook_dim=args.vae_type,
        global_args=args,
        test_mode=True
    ).to(device)
    model.eval()
    print("Model loaded.")
    
    for folder in args.folders:
        process_reconstruction(folder, model, device)

if __name__ == "__main__":
    main()
