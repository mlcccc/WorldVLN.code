"""
Predict trajectory from a video file or an image sequence using TSformer-VO (4-frame / 18-dim).

What it does
1) Read frames from:
   - --video <path.mp4>   OR
   - --image_dir <dir> (sorted by filename)
2) Run 4-frame sliding windows through the model -> predict 3 per-frame deltas per window.
3) Average overlapping window predictions into per-frame deltas.
4) Integrate deltas starting from user-provided initial pose to produce an absolute trajectory.

Assumptions (important)
- This pipeline assumes the model predicts translation deltas in the SAME coordinate frame as your initial position.
  In our training setup, labels come from raw absolute positions (world frame), so deltas are world-frame and can be
  accumulated directly: p[t] = p[t-1] + dp[t].
- Angles are accumulated as simple Euler deltas: ang[t] = ang[t-1] + dang[t].
  (We do NOT convert between body/world frames here.)

Outputs
- Saves:
  - trajectory.npy  : (T, 6) float32 [angle0,angle1,angle2,x,y,z] (rad + meters)
  - deltas.npy      : (T, 6) float32 [dangle0,dangle1,dangle2,dx,dy,dz] (rad + meters), delta[0]=0
  - trajectory.json : list of lists, same as trajectory.npy

Example (cm -> m, angles in degrees):
  /opt/conda/bin/python3 predict_from_video_or_images.py \
    --ckpt checkpoints/uavflow_sim_ft_cm_exp1/checkpoint_best.pth \
    --run_config checkpoints/uavflow_sim_ft_cm_exp1/run_config.json \
    --video /path/to/video.mp4 \
    --out_dir /tmp/tsformer_traj \
    --init_pos 0,0,300 \
    --init_angles_deg 0,0,0 \
    --translation_divisor 100 \
    --device cuda:0
"""

import os
import sys

# Ensure local repo modules take precedence over site-packages (e.g. HF `datasets`).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
from functools import partial
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

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


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")


def parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated values, e.g. 0,0,3")
    return np.asarray([float(x) for x in parts], dtype=np.float32)


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


def aggregate_overlapping_windows(
    num_frames: int,
    window_starts: List[int],
    window_deltas: np.ndarray,
    window_size: int = 4,
) -> np.ndarray:
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


def integrate_trajectory(
    deltas: np.ndarray,
    init_angles_rad: np.ndarray,
    init_pos_m: np.ndarray,
) -> np.ndarray:
    """
    Integrate SE(3) increments:
    - deltas[t] is relative motion (t-1)->t, delta[0]=0
    - deltas angles are ZYX Euler of relative rotation: [dz, dy, dx]
    - deltas translation is expressed in previous frame coordinate system

    Output trajectory is absolute [roll, yaw, pitch, x, y, z].
    """
    t = deltas.shape[0]
    traj = np.zeros((t, 6), dtype=np.float32)

    roll0, yaw0, pitch0 = float(init_angles_rad[0]), float(init_angles_rad[1]), float(init_angles_rad[2])
    R = np.asarray(euler_to_rotation(z=yaw0, y=pitch0, x=roll0, isRadian=True, seq="zyx"), dtype=np.float32)
    p = init_pos_m.astype(np.float32).copy()

    traj[0, 0:3] = np.asarray([roll0, yaw0, pitch0], dtype=np.float32)
    traj[0, 3:6] = p

    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas[i, 0:3]]
        t_rel = deltas[i, 3:6].astype(np.float32)
        R_rel = np.asarray(euler_to_rotation(z=dz, y=dy, x=dx, isRadian=True, seq="zyx"), dtype=np.float32)

        p = p + (R @ t_rel)
        R = R @ R_rel

        zyx = rotation_to_euler(R, seq="zyx")  # [yaw, pitch, roll]
        traj[i, 0:3] = np.asarray([float(zyx[2]), float(zyx[0]), float(zyx[1])], dtype=np.float32)
        traj[i, 3:6] = p

    return traj


def read_frames_from_video(video_path: str, max_frames: Optional[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def read_frames_from_dir(image_dir: str, max_frames: Optional[int]) -> List[np.ndarray]:
    names = [n for n in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, n))]
    # Keep common image extensions; if user has extra files, they won't break sorting.
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    names = [n for n in names if n.lower().endswith(exts)]
    names.sort()
    if max_frames is not None:
        names = names[:max_frames]
    frames = []
    for n in names:
        p = os.path.join(image_dir, n)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint (.pth) to load")
    ap.add_argument("--run_config", type=str, required=True, help="run_config.json (for label_stats)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--image_dir", type=str, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="sliding window stride (default 1)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--angles_in_degrees", action="store_true", default=True, help="input init_angles_deg are degrees")
    ap.add_argument(
        "--translation_divisor",
        type=float,
        default=1.0,
        help="Convert input translation units by dividing (100 for cm->m). Applied to init_pos.",
    )
    ap.add_argument(
        "--init_pos",
        type=str,
        required=True,
        help="x,y,z in the SAME units as your source (e.g. cm if translation_divisor=100).",
    )
    ap.add_argument(
        "--init_angles_deg",
        type=str,
        default=None,
        help="a0,a1,a2 in degrees (e.g. 0,0,0). Used when --angles_in_degrees is set.",
    )
    ap.add_argument("--init_angles_rad", type=str, default=None, help="a0,a1,a2 in radians (e.g. 0,0,0)")
    args = ap.parse_args()

    if (args.video is None) == (args.image_dir is None):
        raise ValueError("Provide exactly one of --video or --image_dir")

    os.makedirs(args.out_dir, exist_ok=True)

    # Read frames
    if args.video is not None:
        frames = read_frames_from_video(args.video, args.max_frames)
    else:
        frames = read_frames_from_dir(args.image_dir, args.max_frames)

    if len(frames) < 4:
        raise ValueError(f"Need at least 4 frames for a 4-frame model; got {len(frames)}")

    # Initial pose
    init_pos = parse_vec3(args.init_pos).astype(np.float32)
    if args.translation_divisor != 1.0:
        init_pos = init_pos / float(args.translation_divisor)
    if args.init_angles_rad:
        init_ang = parse_vec3(args.init_angles_rad).astype(np.float32)
    else:
        if not args.init_angles_deg:
            raise ValueError("Provide --init_angles_rad or --init_angles_deg (and set --angles_in_degrees).")
        init_ang = parse_vec3(args.init_angles_deg).astype(np.float32) * (np.pi / 180.0)

    # Model + stats
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model()
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()
    stats = read_label_stats(args.run_config)

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=KITTI_MEAN, std=KITTI_STD),
        ]
    )

    # Build windows
    window_size = 4
    starts = list(range(0, len(frames) - window_size + 1, args.stride))
    clips = []
    for s in starts:
        imgs = []
        for i in range(window_size):
            img = preprocess(frames[s + i])  # (C,H,W)
            imgs.append(img.unsqueeze(0))
        x = torch.cat(imgs, dim=0)  # (T,C,H,W)
        x = x.transpose(0, 1)  # (C,T,H,W)
        clips.append(x)

    # Batched inference
    preds = []
    with torch.no_grad():
        for i in range(0, len(clips), args.batch_size):
            batch = torch.stack(clips[i : i + args.batch_size], dim=0).to(device)  # (B,C,T,H,W)
            out = model(batch.float())  # (B,18)
            preds.append(out.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (N,18) normalized
    window_deltas = denorm_window_preds(preds, stats)  # (N,3,6) in (rad + meters) based on training stats

    deltas = aggregate_overlapping_windows(num_frames=len(frames), window_starts=starts, window_deltas=window_deltas)
    traj = integrate_trajectory(deltas=deltas, init_angles_rad=init_ang, init_pos_m=init_pos)

    np.save(os.path.join(args.out_dir, "deltas.npy"), deltas.astype(np.float32))
    np.save(os.path.join(args.out_dir, "trajectory.npy"), traj.astype(np.float32))
    with open(os.path.join(args.out_dir, "trajectory.json"), "w") as f:
        json.dump(traj.tolist(), f)

    print(f"Done. Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

