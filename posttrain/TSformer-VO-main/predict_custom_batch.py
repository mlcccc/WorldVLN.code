import os
import sys
import argparse
import json
from functools import partial
import re
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

# Add repo root to path if needed (assuming this script is in repo root)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.utils import euler_to_rotation, rotation_to_euler
from timesformer.models.vit import VisionTransformer

KITTI_MEAN = [0.34721234, 0.36705238, 0.36066107]
KITTI_STD = [0.30737526, 0.31515116, 0.32020183]

def build_model() -> VisionTransformer:
    return VisionTransformer(
        img_size=(192, 640),
        num_classes=18,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        num_frames=4,
        attention_type="divided_space_time",
    )

def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"Warning: Checkpoint mismatch. missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

def read_label_stats(run_config_path: str) -> Dict[str, np.ndarray]:
    with open(run_config_path, "r") as f:
        cfg = json.load(f)
    stats = cfg.get("label_stats")
    if not stats:
        raise ValueError("run_config.json missing label_stats")
    out = {}
    for k in ("mean_angles", "std_angles", "mean_t", "std_t"):
        out[k] = np.asarray(stats[k], dtype=np.float32)
    return out

def denorm_window_preds(pred_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    b = pred_norm.shape[0]
    pred = pred_norm.reshape(b, 3, 6).astype(np.float32)
    mean_a, std_a = stats["mean_angles"], stats["std_angles"]
    mean_t, std_t = stats["mean_t"], stats["std_t"]
    pred[:, :, 0:3] = pred[:, :, 0:3] * std_a[None, None, :] + mean_a[None, None, :]
    pred[:, :, 3:6] = pred[:, :, 3:6] * std_t[None, None, :] + mean_t[None, None, :]
    return pred

def aggregate_overlapping_windows(num_frames: int, window_starts: List[int], window_deltas: np.ndarray, window_size: int = 4) -> np.ndarray:
    acc = np.zeros((num_frames, 6), dtype=np.float32)
    cnt = np.zeros((num_frames,), dtype=np.int32)
    for i, s in enumerate(window_starts):
        for j in range(1, window_size):
            t = s + j
            if 0 <= t < num_frames:
                acc[t] += window_deltas[i, j - 1]
                cnt[t] += 1
    out = np.zeros((num_frames, 6), dtype=np.float32)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask, None]
    return out

def _R_from_rpy(roll: float, yaw: float, pitch: float) -> np.ndarray:
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)

def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    zyx = rotation_to_euler(R, seq="zyx")
    yaw, pitch, roll = float(zyx[0]), float(zyx[1]), float(zyx[2])
    return np.asarray([roll, yaw, pitch], dtype=np.float32)

def integrate_trajectory_se3(deltas_zyx: np.ndarray, init_rpy_rad: np.ndarray, init_pos_m: np.ndarray) -> np.ndarray:
    t = deltas_zyx.shape[0]
    traj = np.zeros((t, 6), dtype=np.float32)
    roll0, yaw0, pitch0 = float(init_rpy_rad[0]), float(init_rpy_rad[1]), float(init_rpy_rad[2])
    R = _R_from_rpy(roll0, yaw0, pitch0)
    p = init_pos_m.astype(np.float32).copy()
    traj[0, 0:3] = np.asarray([roll0, yaw0, pitch0], dtype=np.float32)
    traj[0, 3:6] = p
    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas_zyx[i, 0:3]]
        t_rel = deltas_zyx[i, 3:6].astype(np.float32)
        R_rel = np.asarray(euler_to_rotation(z=dz, y=dy, x=dx, isRadian=True, seq="zyx"), dtype=np.float32)
        p = p + (R @ t_rel)
        R = R @ R_rel
        traj[i, 0:3] = _rpy_from_R(R)
        traj[i, 3:6] = p
    return traj

def read_video_sampled(video_path: str, target_len: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # If target_len is 0 (unknown), read all frames
    if target_len <= 0:
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    # If target_len is specified, sample
    idxs = np.round(np.linspace(0, frame_count - 1, target_len)).astype(int)
    idxs_set = set(int(i) for i in idxs.tolist())
    frames = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if i in idxs_set:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    
    # Pad if necessary
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frames: {video_path}")
    if len(frames) > target_len:
        frames = frames[:target_len]
    while len(frames) < target_len:
        frames.append(frames[-1])
    return frames

def preprocess_frames(frames: List[np.ndarray]) -> List[torch.Tensor]:
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=KITTI_MEAN, std=KITTI_STD),
    ])
    return [tfm(f) for f in frames]

def run_model_on_frames(model: nn.Module, frames_t: List[torch.Tensor], stats: Dict[str, np.ndarray], device: torch.device, batch_size: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    window_size = 4
    t = len(frames_t)
    starts = list(range(0, t - window_size + 1, max(1, int(stride))))
    clips = []
    for s in starts:
        imgs = [frames_t[s + i].unsqueeze(0) for i in range(window_size)]
        x = torch.cat(imgs, dim=0).transpose(0, 1)
        clips.append(x)
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = torch.stack(clips[i : i + batch_size], dim=0).to(device)
            out = model(batch.float())
            preds.append(out.detach().cpu().numpy())
    
    if len(preds) == 0:
        return np.zeros((t, 6)), np.zeros((0, 3, 6))
        
    preds = np.concatenate(preds, axis=0)
    window_deltas = denorm_window_preds(preds, stats)
    deltas = aggregate_overlapping_windows(num_frames=t, window_starts=starts, window_deltas=window_deltas)
    return deltas, window_deltas

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _unwrap_angles_radians(angles_rad: np.ndarray) -> np.ndarray:
    out = np.empty_like(angles_rad)
    for i in range(angles_rad.shape[1]):
        out[:, i] = np.unwrap(angles_rad[:, i])
    return out

def main():
    parser = argparse.ArgumentParser(description="Custom Batch Inference for UAVFlow Test Data")
    parser.add_argument("--test_root", type=str, default="/home/dataset-assist-0/xjc/TSformer-VO-main/test_data_latent/uavflowdatasim_output")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--run_config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--translation_divisor", type=float, default=100.0, help="Divide input pose by this (100 for cm->m)")
    parser.add_argument("--angles_in_degrees", action="store_true", default=True)
    
    args = parser.parse_args()
    
    if args.run_config is None:
        args.run_config = os.path.join(os.path.dirname(args.ckpt), "run_config.json")
    
    print(f"Loading stats from {args.run_config}")
    stats = read_label_stats(args.run_config)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model()
    load_checkpoint(model, args.ckpt, device=device)
    
    # Find all trajectory directories recursively or by known structure
    # Structure: test_root/{source}/{id}/reshape_actionhead_data
    # We can walk through test_root and look for reshape_actionhead_data directories
    
    print(f"Searching for videos in {args.test_root}...")
    target_dirs = []
    for root, dirs, files in os.walk(args.test_root):
        if "reshape_actionhead_data" in dirs:
             target_dirs.append(os.path.join(root, "reshape_actionhead_data"))
        elif os.path.basename(root) == "reshape_actionhead_data":
             target_dirs.append(root)
             
    # Remove duplicates if any
    target_dirs = list(set(target_dirs))
    target_dirs.sort()
    
    print(f"Found {len(target_dirs)} target directories.")
    
    success_count = 0
    
    for dir_path in tqdm(target_dirs):
        parent_dir = os.path.dirname(dir_path)
        video_path = os.path.join(parent_dir, "video.mp4")
        json_path = os.path.join(dir_path, "preprocessed_logs.json")

        
        if not os.path.exists(video_path):
            # Try finding mp4 in the parent_dir if name is different
            found = False
            if os.path.exists(parent_dir):
                for f in os.listdir(parent_dir):
                    if f.endswith(".mp4"):
                        video_path = os.path.join(parent_dir, f)
                        found = True
                        break
            if not found:
                continue
                
        if not os.path.exists(json_path):
            continue
            
        try:
            # Load Logs
            with open(json_path, 'r') as f:
                logs = json.load(f)
            # logs is [[x,y,z,roll,pitch,yaw], ...]
            # Assuming typical length matches video duration we want to test
            
            target_len = len(logs)
            
            # Read video
            frames = read_video_sampled(video_path, target_len=target_len)
            
            if len(frames) < 4:
                continue
                
            # Run Model
            frames_t = preprocess_frames(frames)
            deltas, window_deltas = run_model_on_frames(
                model=model,
                frames_t=frames_t,
                stats=stats,
                device=device,
                batch_size=args.batch_size,
                stride=args.stride
            )
            
            # Initial Pose
            init_pose = np.array(logs[0], dtype=np.float32)
            # logs structure: [x, y, z, roll, pitch, yaw] or similar
            # Based on inference_p2p.py:
            # pose[0:3] / 100.0
            # pose[3:6] * pi/180
            
            init_xyz = init_pose[0:3]
            if args.translation_divisor != 1.0:
                init_xyz /= args.translation_divisor
                
            init_angles = init_pose[3:6]
            if args.angles_in_degrees:
                init_angles = init_angles * (np.pi / 180.0)
                
            # Reconstruct
            traj = integrate_trajectory_se3(deltas_zyx=deltas, init_rpy_rad=init_angles, init_pos_m=init_xyz)
            
            # Save Output
            # "Output trajectory data should be placed in the trajectory video folder"
            # i.e., parent_dir (where video.mp4 is)
            
            np.save(os.path.join(parent_dir, "tsformer_deltas.npy"), deltas.astype(np.float32))
            np.save(os.path.join(parent_dir, "tsformer_trajectory.npy"), traj.astype(np.float32))
            
            # Also save as JSON for easy reading
            with open(os.path.join(parent_dir, "tsformer_trajectory.json"), "w") as f:
                json.dump(traj.tolist(), f)
                
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
            continue

    print(f"Processed {success_count} videos successfully.")

if __name__ == "__main__":
    main()
