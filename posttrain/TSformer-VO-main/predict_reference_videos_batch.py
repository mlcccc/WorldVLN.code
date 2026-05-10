"""
Batch inference for `data/reference_videos_all_v2/*/reference.mp4` using TSformer-VO.

Data layout
- videos_root/<id>/reference.mp4
- test_jsons_dir/<id>.json (provides initial_pos and reference_path_raw, etc.)

We use test_json to:
- get initial pose (x,y,z,roll,yaw,pitch) in degrees for angles
- get expected trajectory length (len(reference_path_raw)) to decide sampling indices

Sampling strategy
- We read the whole video with OpenCV and then pick exactly `target_len` frames
  using evenly-spaced indices (linspace). This matches the JSON replay length.

Outputs
For each id, write to out_dir/<id>/:
- trajectory.npy / trajectory.json: (T,6) [roll,yaw,pitch,x,y,z] in (rad + meters)
- deltas.npy: (T,6) per-step increments, delta[0]=0
- metrics.json (optional): RMSE vs reference_path_raw (after unit+angle conversion)

Important unit handling
- `--translation_divisor` is applied to *input* positions from JSON (e.g. 100 for cm->m).
  Predicted deltas are assumed to already be in meters according to training run_config stats.

Example:
  /opt/conda/bin/python3 predict_reference_videos_batch.py \
    --ckpt checkpoints/uavflow_sim_ft_cm_exp1/checkpoint_e2.pth \
    --run_config checkpoints/uavflow_sim_ft_cm_exp1/run_config.json \
    --videos_root data/reference_videos_all_v2 \
    --test_jsons_dir data/test_jsons \
    --out_dir checkpoints/uavflow_sim_ft_cm_exp1/infer_reference_v2 \
    --device cuda:0 \
    --translation_divisor 100 \
    --compute_metrics
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
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


def _R_from_rpy(roll: float, yaw: float, pitch: float) -> np.ndarray:
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    # rotation_to_euler(seq='zyx') returns [z, y, x] = [yaw, pitch, roll]
    zyx = rotation_to_euler(R, seq="zyx")
    yaw, pitch, roll = float(zyx[0]), float(zyx[1]), float(zyx[2])
    return np.asarray([roll, yaw, pitch], dtype=np.float32)


def integrate_trajectory_se3(deltas_zyx: np.ndarray, init_rpy_rad: np.ndarray, init_pos_m: np.ndarray) -> np.ndarray:
    """
    deltas_zyx: (T,6) where each step is [dz, dy, dx, tx, ty, tz]
      - angles are ZYX Euler of relative rotation (rad)
      - translation is in previous frame coordinate system (meters)
    Output trajectory: (T,6) absolute [roll, yaw, pitch, x, y, z]
    """
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

        # compose: T_new = T * T_rel
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
    if frame_count <= 0:
        # fallback: read sequentially and count
        frames_all = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames_all.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames_all) == 0:
            raise RuntimeError(f"Empty video: {video_path}")
        if target_len <= 0 or target_len >= len(frames_all):
            return frames_all
        idxs = np.round(np.linspace(0, len(frames_all) - 1, target_len)).astype(int)
        return [frames_all[i] for i in idxs]

    if target_len <= 0 or target_len >= frame_count:
        # read all frames
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    idxs = np.round(np.linspace(0, frame_count - 1, target_len)).astype(int)
    idxs_set = set(int(i) for i in idxs.tolist())
    frames = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in idxs_set:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    # if missed due to codec weirdness, fall back by trimming/padding last
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read frames: {video_path}")
    if len(frames) > target_len:
        frames = frames[:target_len]
    while len(frames) < target_len:
        frames.append(frames[-1])
    return frames


def preprocess_frames(frames: List[np.ndarray]) -> List[torch.Tensor]:
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=KITTI_MEAN, std=KITTI_STD),
        ]
    )
    return [tfm(f) for f in frames]  # each: (C,H,W)


def run_model_on_frames(
    model: nn.Module,
    frames_t: List[torch.Tensor],
    stats: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    frames_t: list of (C,H,W)
    Returns:
      deltas: (T,6) denorm
      window_deltas: (N,3,6) denorm
    """
    window_size = 4
    t = len(frames_t)
    starts = list(range(0, t - window_size + 1, max(1, int(stride))))
    clips = []
    for s in starts:
        # stack to (C,T,H,W)
        imgs = [frames_t[s + i].unsqueeze(0) for i in range(window_size)]
        x = torch.cat(imgs, dim=0).transpose(0, 1)
        clips.append(x)

    preds = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = torch.stack(clips[i : i + batch_size], dim=0).to(device)  # (B,C,T,H,W)
            out = model(batch.float())
            preds.append(out.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # (N,18)
    window_deltas = denorm_window_preds(preds, stats)  # (N,3,6)
    deltas = aggregate_overlapping_windows(num_frames=t, window_starts=starts, window_deltas=window_deltas)
    return deltas, window_deltas


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _unwrap_angles_radians(angles_rad: np.ndarray) -> np.ndarray:
    out = np.empty_like(angles_rad)
    for i in range(angles_rad.shape[1]):
        out[:, i] = np.unwrap(angles_rad[:, i])
    return out


def read_uavflow_route_frames(route_dir: str) -> List[np.ndarray]:
    """
    Read frames from uavflowdatasim_output/<route>/images as RGB numpy arrays.
    """
    images_dir = os.path.join(route_dir, "images")
    names = [n for n in os.listdir(images_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    names.sort()
    frames = []
    for n in names:
        p = os.path.join(images_dir, n)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def read_uavflow_raw_logs(route_dir: str) -> np.ndarray:
    p = os.path.join(route_dir, "raw_logs.json")
    with open(p, "r") as f:
        raw = json.load(f)
    return np.asarray(raw, dtype=np.float32)  # (T,6): [x,y,z,a0,a1,a2] absolute


def infer_one_video_with_json(
    *,
    vid_id: str,
    video_path: str,
    json_path: str,
    model: nn.Module,
    stats: Dict[str, np.ndarray],
    device: torch.device,
    out_dir: str,
    batch_size: int,
    stride: int,
    translation_divisor: float,
    angles_in_degrees: bool,
    compute_metrics: bool,
) -> None:
    """
    Shared logic for:
    - Mode A (reference_videos_all_v2/<id>/reference.mp4)
    - Mode C (flat videos dir with filename->id mapping)
    """
    if not os.path.exists(video_path) or not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        meta = json.load(f)

    init_pos6 = meta.get("initial_pos")
    ref_path = meta.get("reference_path_raw")
    if not init_pos6 or not isinstance(init_pos6, list) or len(init_pos6) < 6:
        return
    target_len = len(ref_path) if isinstance(ref_path, list) and len(ref_path) > 0 else 0

    frames = read_video_sampled(video_path, target_len=target_len)
    if len(frames) < 4:
        return

    frames_t = preprocess_frames(frames)
    deltas, window_deltas = run_model_on_frames(
        model=model,
        frames_t=frames_t,
        stats=stats,
        device=device,
        batch_size=batch_size,
        stride=stride,
    )

    # init pose (x,y,z,roll,yaw,pitch)
    init_xyz = np.asarray(init_pos6[0:3], dtype=np.float32)
    if translation_divisor != 1.0:
        init_xyz = init_xyz / float(translation_divisor)
    init_angles = np.asarray(init_pos6[3:6], dtype=np.float32)
    if angles_in_degrees:
        init_angles = init_angles * (np.pi / 180.0)

    traj = integrate_trajectory_se3(deltas_zyx=deltas, init_rpy_rad=init_angles, init_pos_m=init_xyz)

    out_one = os.path.join(out_dir, vid_id)
    os.makedirs(out_one, exist_ok=True)
    np.save(os.path.join(out_one, "deltas.npy"), deltas.astype(np.float32))
    np.save(os.path.join(out_one, "trajectory.npy"), traj.astype(np.float32))
    np.save(os.path.join(out_one, "window_deltas.npy"), window_deltas.astype(np.float32))
    with open(os.path.join(out_one, "trajectory.json"), "w") as f:
        json.dump(traj.tolist(), f)

    if compute_metrics and isinstance(ref_path, list) and len(ref_path) >= len(traj):
        ref = np.asarray(ref_path[: len(traj)], dtype=np.float32)
        ref_xyz = ref[:, 0:3]
        if translation_divisor != 1.0:
            ref_xyz = ref_xyz / float(translation_divisor)
        ref_angles = ref[:, 3:6]
        if angles_in_degrees:
            ref_angles = ref_angles * (np.pi / 180.0)
        metrics = {
            "len": int(len(traj)),
            "rmse_xyz_m": rmse(traj[:, 3:6], ref_xyz),
            "rmse_angles_rad": rmse(traj[:, 0:3], ref_angles),
        }
        with open(os.path.join(out_one, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


def _id_from_flat_video_filename(name: str, pattern: str) -> str:
    """
    Extract <id> from a flat video filename.
    Default pattern: r'^\\d+_(.+)$' so:
      '00001_2025-03-30_11-49-14' -> '2025-03-30_11-49-14'
    If no match, returns the input name.
    """
    m = re.match(pattern, name)
    if m and m.group(1):
        return str(m.group(1))
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--run_config", type=str, required=True)
    # Mode A: reference videos
    ap.add_argument("--videos_root", type=str, default=None)
    ap.add_argument("--test_jsons_dir", type=str, default=None)
    # Mode C: flat video dir (e.g. DiffSynth outputs/*.mp4)
    ap.add_argument("--videos_flat_dir", type=str, default=None, help="扁平视频目录：里面直接是 *.mp4 文件")
    ap.add_argument(
        "--flat_video_id_pattern",
        type=str,
        default=r"^\d+_(.+)$",
        help=r"从扁平视频文件名提取 id 的正则（对去掉扩展名的 basename 匹配），默认去掉 `00001_` 这种数字前缀",
    )
    # Mode B: uavflow training routes (images + raw_logs)
    ap.add_argument("--uavflow_root", type=str, default=None, help="e.g. /home/batchcom/dataset-link/xjc/uavflowdatasim_output")
    ap.add_argument("--uavflow_first_n", type=int, default=0, help="Infer first N routes under uavflow_root (e.g. 200).")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1, help="sliding-window stride for inference")
    ap.add_argument("--translation_divisor", type=float, default=1.0, help="divide JSON positions by this (100 for cm->m)")
    ap.add_argument("--angles_in_degrees", action="store_true", default=True)
    ap.add_argument("--compute_metrics", action="store_true", default=False)
    ap.add_argument("--ids", type=str, default="all", help='Comma-separated ids or "all"')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model()
    load_checkpoint(model, args.ckpt, device=device)
    stats = read_label_stats(args.run_config)

    wanted = None
    if args.ids.strip().lower() not in ("all", "*", ""):
        wanted = set([p.strip() for p in args.ids.split(",") if p.strip()])

    # -------- Mode B: uavflow training routes inference --------
    if args.uavflow_root and args.uavflow_first_n > 0:
        route_names = [d for d in os.listdir(args.uavflow_root) if os.path.isdir(os.path.join(args.uavflow_root, d))]
        # numeric routes first
        route_names.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        route_names = route_names[: int(args.uavflow_first_n)]

        for rid in route_names:
            route_dir = os.path.join(args.uavflow_root, rid)
            images_dir = os.path.join(route_dir, "images")
            raw_path = os.path.join(route_dir, "raw_logs.json")
            if not os.path.isdir(images_dir) or not os.path.exists(raw_path):
                continue

            raw = read_uavflow_raw_logs(route_dir)
            frames = read_uavflow_route_frames(route_dir)
            t = int(min(len(raw), len(frames)))
            if t < 4:
                continue
            raw = raw[:t]
            frames = frames[:t]

            frames_t = preprocess_frames(frames)
            deltas, window_deltas = run_model_on_frames(
                model=model,
                frames_t=frames_t,
                stats=stats,
                device=device,
                batch_size=args.batch_size,
                stride=args.stride,
            )

            # init pose from raw_logs[0]: [x,y,z,a0,a1,a2]
            init_xyz = raw[0, 0:3].astype(np.float32)
            if args.translation_divisor != 1.0:
                init_xyz = init_xyz / float(args.translation_divisor)
            init_angles = raw[0, 3:6].astype(np.float32)
            if args.angles_in_degrees:
                init_angles = init_angles * (np.pi / 180.0)

            traj = integrate_trajectory_se3(deltas_zyx=deltas, init_rpy_rad=init_angles, init_pos_m=init_xyz)

            out_one = os.path.join(args.out_dir, rid)
            os.makedirs(out_one, exist_ok=True)
            np.save(os.path.join(out_one, "deltas.npy"), deltas.astype(np.float32))
            np.save(os.path.join(out_one, "trajectory.npy"), traj.astype(np.float32))
            np.save(os.path.join(out_one, "window_deltas.npy"), window_deltas.astype(np.float32))
            with open(os.path.join(out_one, "trajectory.json"), "w") as f:
                json.dump(traj.tolist(), f)

            if args.compute_metrics:
                ref_xyz = raw[:, 0:3].astype(np.float32)
                if args.translation_divisor != 1.0:
                    ref_xyz = ref_xyz / float(args.translation_divisor)
                ref_angles = raw[:, 3:6].astype(np.float32)
                if args.angles_in_degrees:
                    ref_angles = ref_angles * (np.pi / 180.0)
                ref_angles = _unwrap_angles_radians(ref_angles)
                pred_angles = _unwrap_angles_radians(traj[:, 0:3])
                metrics = {
                    "len": int(len(traj)),
                    "rmse_xyz_m": rmse(traj[:, 3:6], ref_xyz),
                    "rmse_angles_rad": rmse(pred_angles, ref_angles),
                }
                with open(os.path.join(out_one, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)

        print(f"Done. Saved to {args.out_dir}", flush=True)
        return

    # -------- Mode C: flat videos dir (DiffSynth outputs) --------
    if args.videos_flat_dir:
        if not args.test_jsons_dir:
            raise ValueError("使用 --videos_flat_dir 时必须提供 --test_jsons_dir")
        mp4s = [n for n in os.listdir(args.videos_flat_dir) if n.lower().endswith(".mp4")]
        mp4s.sort()
        for fn in mp4s:
            base = os.path.splitext(fn)[0]
            vid_id = _id_from_flat_video_filename(base, pattern=args.flat_video_id_pattern)
            if wanted is not None and vid_id not in wanted:
                continue
            video_path = os.path.join(args.videos_flat_dir, fn)
            json_path = os.path.join(args.test_jsons_dir, f"{vid_id}.json")
            infer_one_video_with_json(
                vid_id=vid_id,
                video_path=video_path,
                json_path=json_path,
                model=model,
                stats=stats,
                device=device,
                out_dir=args.out_dir,
                batch_size=args.batch_size,
                stride=args.stride,
                translation_divisor=args.translation_divisor,
                angles_in_degrees=args.angles_in_degrees,
                compute_metrics=args.compute_metrics,
            )

        print(f"Done. Saved to {args.out_dir}", flush=True)
        return

    # -------- Mode A: reference videos inference (original) --------
    if not args.videos_root or not args.test_jsons_dir:
        raise ValueError(
            "需要提供 --videos_root + --test_jsons_dir，或 --videos_flat_dir + --test_jsons_dir，或者使用 --uavflow_root + --uavflow_first_n"
        )

    ids = [d for d in os.listdir(args.videos_root) if os.path.isdir(os.path.join(args.videos_root, d))]
    ids.sort()
    if wanted is not None:
        ids = [i for i in ids if i in wanted]

    for vid_id in ids:
        video_path = os.path.join(args.videos_root, vid_id, "reference.mp4")
        json_path = os.path.join(args.test_jsons_dir, f"{vid_id}.json")
        infer_one_video_with_json(
            vid_id=vid_id,
            video_path=video_path,
            json_path=json_path,
            model=model,
            stats=stats,
            device=device,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            stride=args.stride,
            translation_divisor=args.translation_divisor,
            angles_in_degrees=args.angles_in_degrees,
            compute_metrics=args.compute_metrics,
        )

    print(f"Done. Saved to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

