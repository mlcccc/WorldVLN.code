"""
Inference on UAVFlow simulated routes (uavflowdatasim_output) using TSformer-VO.

This script mirrors `predict_poses.py` but adapts to UAVFlow route folders:
  <route_dir>/
    images/frame_000000.png ...
    raw_logs.json
    preprocessed_logs.json

Model checkpoint (must match 4-frame model / 18-dim head):
  - Original: checkpoint/checkpoint_model3_exp20.pth
  - Fine-tuned: <out_dir>/checkpoint_best.pth or checkpoint_last.pth

Outputs per route:
  - pred_delta.npy: (T, 6) float32, per-frame relative motion (rad + meters)
      delta[t] corresponds to motion from (t-1)->t, delta[0]=0
      order: [dAngle0, dAngle1, dAngle2, dTx, dTy, dTz]
  - pred_delta_windowed.npy: (num_windows, 3, 6) float32, window predictions (denorm)

Run example:
  /opt/conda/bin/python3 predict_uavflow_sim.py \\
    --data_root /home/batchcom/dataset-link/xjc/uavflowdatasim_output \\
    --ckpt checkpoints/uavflow_sim_ft_cm_exp1/checkpoint_best.pth \\
    --run_config checkpoints/uavflow_sim_ft_cm_exp1/run_config.json \\
    --out_dir checkpoints/uavflow_sim_ft_cm_exp1/infer \\
    --routes 0,1,2 \\
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.uavflow_sim import UavflowSimDataset
from datasets.utils import euler_to_rotation, rotation_to_euler
from timesformer.models.vit import VisionTransformer


KITTI_MEAN = [0.34721234, 0.36705238, 0.36066107]
KITTI_STD = [0.30737526, 0.31515116, 0.32020183]


def build_model() -> VisionTransformer:
    # Must match checkpoint_model3_exp20.pth exactly.
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    model.to(device)
    model.eval()


def parse_routes_arg(routes: str) -> Optional[List[str]]:
    """
    routes: "all" or "0,1,2" or "0001,0002"
    Returns list of route directory names (as strings), or None for all.
    """
    r = routes.strip().lower()
    if r in ("all", "*", ""):
        return None
    parts = [p.strip() for p in routes.split(",") if p.strip()]
    return parts if parts else None


def read_label_stats(run_config_path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not run_config_path:
        return None
    with open(run_config_path, "r") as f:
        cfg = json.load(f)
    stats = cfg.get("label_stats")
    if not stats:
        return None
    out = {}
    for k in ("mean_angles", "std_angles", "mean_t", "std_t"):
        if k not in stats:
            return None
        out[k] = np.asarray(stats[k], dtype=np.float32)
    return out


def denorm_window_preds(
    pred_norm: np.ndarray,
    stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    pred_norm: (B, 18) normalized output from model.
    Returns: (B, 3, 6) denormalized deltas (rad + meters).
    """
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
    """
    Average predictions for the same frame delta across all windows.

    window_deltas: (N, window_size-1, 6) denorm
      window i corresponds to start s = window_starts[i]
      predicts deltas at frame indices: s+1, s+2, s+3

    Returns delta_by_frame: (T, 6), with delta[0]=0.
    """
    assert window_size == 4
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
    # out[0] remains zeros
    return out


def integrate_trajectory_se3(deltas_zyx: np.ndarray, init_rpy_rad: np.ndarray, init_pos_m: np.ndarray) -> np.ndarray:
    """
    deltas_zyx: (T,6) [dz, dy, dx, tx, ty, tz] (rad + meters), translation in previous frame coords.
    output: (T,6) absolute [roll, yaw, pitch, x, y, z]
    """
    t = deltas_zyx.shape[0]
    traj = np.zeros((t, 6), dtype=np.float32)

    roll0, yaw0, pitch0 = float(init_rpy_rad[0]), float(init_rpy_rad[1]), float(init_rpy_rad[2])
    R = np.asarray(euler_to_rotation(z=yaw0, y=pitch0, x=roll0, isRadian=True, seq="zyx"), dtype=np.float32)
    p = init_pos_m.astype(np.float32).copy()

    traj[0, 0:3] = np.asarray([roll0, yaw0, pitch0], dtype=np.float32)
    traj[0, 3:6] = p

    for i in range(1, t):
        dz, dy, dx = [float(x) for x in deltas_zyx[i, 0:3]]
        t_rel = deltas_zyx[i, 3:6].astype(np.float32)
        R_rel = np.asarray(euler_to_rotation(z=dz, y=dy, x=dx, isRadian=True, seq="zyx"), dtype=np.float32)
        p = p + (R @ t_rel)
        R = R @ R_rel
        zyx = rotation_to_euler(R, seq="zyx")  # [yaw, pitch, roll]
        traj[i, 0:3] = np.asarray([float(zyx[2]), float(zyx[0]), float(zyx[1])], dtype=np.float32)
        traj[i, 3:6] = p
    return traj


def infer_one_route(
    model: nn.Module,
    route_dir: str,
    device: torch.device,
    stats: Dict[str, np.ndarray],
    translation_divisor: float,
    angles_in_degrees: bool,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      traj: (T,6) absolute [roll,yaw,pitch,x,y,z]
      delta_by_frame: (T,6) denorm (rad + meters) in model order [dz,dy,dx,tx,ty,tz]
      window_deltas: (N,3,6) denorm (rad + meters)
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=KITTI_MEAN, std=KITTI_STD),
        ]
    )

    # Build a dataset rooted at parent dir, then filter to this single route.
    # We reuse the dataset implementation to read images/windows consistently.
    # The label fields are ignored during inference.
    parent = os.path.dirname(route_dir.rstrip("/"))
    route_name = os.path.basename(route_dir.rstrip("/"))
    ds = UavflowSimDataset(
        root_dir=parent,
        window_size=4,
        stride=1,
        transform=preprocess,
        use_raw_for_labels=True,
        angles_in_degrees=angles_in_degrees,
        translation_divisor=translation_divisor,
        img_ext=".png",
        max_routes=None,
    )

    # Map route_name -> internal route_idx
    route_idx = None
    for i, r in enumerate(ds.routes):
        if os.path.basename(r.route_dir) == route_name:
            route_idx = i
            break
    if route_idx is None:
        raise FileNotFoundError(f"Route {route_name} not found/valid under {parent}")

    # Collect sample indices for this route in order
    sample_indices = [i for i, (rid, _s) in enumerate(ds.samples) if rid == route_idx]
    if len(sample_indices) == 0:
        raise FileNotFoundError(f"Route {route_name} has no valid windows (need >=4 frames)")

    # Also collect starts for aggregation
    window_starts = [ds.samples[i][1] for i in sample_indices]
    num_frames = ds.routes[route_idx].length

    loader = DataLoader(
        torch.utils.data.Subset(ds, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    preds_norm = []
    with torch.no_grad():
        for images, _gt in loader:
            images = images.to(device, non_blocking=True)
            out = model(images.float())
            preds_norm.append(out.detach().cpu().numpy())

    preds_norm = np.concatenate(preds_norm, axis=0)  # (N,18)
    window_deltas = denorm_window_preds(preds_norm, stats)  # (N,3,6)
    delta_by_frame = aggregate_overlapping_windows(
        num_frames=num_frames,
        window_starts=window_starts,
        window_deltas=window_deltas,
        window_size=4,
    )
    # initial pose from raw_logs.json
    raw_path = os.path.join(route_dir, "raw_logs.json")
    with open(raw_path, "r") as f:
        raw = np.asarray(json.load(f), dtype=np.float32)
    raw = raw[:num_frames]
    init_xyz = raw[0, 0:3] / float(translation_divisor)
    init_rpy = raw[0, 3:6]
    if angles_in_degrees:
        init_rpy = init_rpy * (np.pi / 180.0)

    traj = integrate_trajectory_se3(delta_by_frame, init_rpy_rad=init_rpy, init_pos_m=init_xyz)
    return traj, delta_by_frame, window_deltas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="uavflowdatasim_output root dir")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint (.pth) to load")
    p.add_argument("--run_config", type=str, default=None, help="run_config.json from training (for label_stats)")
    p.add_argument("--out_dir", type=str, required=True, help="output directory")
    p.add_argument("--routes", type=str, default="all", help='e.g. "0,1,2" or "all"')
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--angles_in_degrees", action="store_true", default=True)
    p.add_argument("--translation_divisor", type=float, default=1.0, help="100 for cm->m")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model()
    load_checkpoint(model, args.ckpt, device=device)

    stats = read_label_stats(args.run_config)
    if stats is None:
        raise RuntimeError(
            "找不到训练时的 label_stats（用于反归一化推理输出）。\n"
            "请传入 --run_config <.../run_config.json>（推荐），"
            "或者手动在脚本里指定 mean/std。"
        )

    wanted = parse_routes_arg(args.routes)
    route_names = [n for n in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, n))]
    # deterministic sort: numeric dirs first, then others
    route_names.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    if wanted is not None:
        wanted_set = set(wanted)
        route_names = [n for n in route_names if n in wanted_set]
    if len(route_names) == 0:
        raise FileNotFoundError("No routes selected/found.")

    for rn in route_names:
        route_dir = os.path.join(args.data_root, rn)
        # only accept expected structure
        if not os.path.isdir(os.path.join(route_dir, "images")):
            continue
        if not os.path.exists(os.path.join(route_dir, "raw_logs.json")):
            continue
        if not os.path.exists(os.path.join(route_dir, "preprocessed_logs.json")):
            continue

        out_route = os.path.join(args.out_dir, rn)
        os.makedirs(out_route, exist_ok=True)

        traj, delta_by_frame, window_deltas = infer_one_route(
            model=model,
            route_dir=route_dir,
            device=device,
            stats=stats,
            translation_divisor=args.translation_divisor,
            angles_in_degrees=args.angles_in_degrees,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        np.save(os.path.join(out_route, "pred_delta.npy"), delta_by_frame)
        np.save(os.path.join(out_route, "pred_delta_windowed.npy"), window_deltas)
        np.save(os.path.join(out_route, "trajectory.npy"), traj)
        with open(os.path.join(out_route, "trajectory.json"), "w") as f:
            json.dump(traj.tolist(), f)

    print(f"Done. Results saved to: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

