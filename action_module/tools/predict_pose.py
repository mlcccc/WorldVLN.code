"""
Stage-2 latent-to-action batch inference for UAV-Flow-style route folders.

Inputs per route dir:
- latents.pt (Tensor shaped (1,64,T_lat,16,16))
- preprocessed_logs.json (list of length T, each [x,y,z,roll,yaw,pitch]; angles
  in degrees by default, translation unit controlled by --translation_divisor)

Pass --ckpt explicitly, or set --stage2_root to a directory containing run
subdirectories with checkpoint_last.pth files.

Outputs per route:
- deltas.npy: (T,6) where each step is [dz,dy,dx,tx,ty,tz] (rad + meters), delta[0]=0
- window_deltas.npy: (N,3,6) per sliding window
- trajectory.npy / trajectory.json: (T,6) absolute [roll,yaw,pitch,x,y,z] (rad + meters)
- pred_path.json / pred_actions.json: compatible-style outputs containing actions6 and integrated trajectory
- metrics.json (optional): RMSE vs preprocessed_logs after unit conversion
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_TOOLS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.utils import euler_to_rotation, rotation_to_euler  # noqa: E402
from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter  # noqa: E402
from timesformer.models.vit import VisionTransformer  # noqa: E402

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:  # noqa: E402
    from tools.train_stage2_latent2action_ddp import _add_infinitystar_to_syspath, load_infinitystar_vae
except Exception:  # pragma: no cover
    from types import SimpleNamespace

    def _add_infinitystar_to_syspath(inf_root: Optional[str], proj_root: str):
        if inf_root and os.path.isdir(inf_root):
            p = os.path.abspath(inf_root)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
        candidates = [
            os.environ.get("INFINITYSTAR_ROOT", ""),
            os.environ.get("INFINITYSTAR_HOME", ""),
            os.path.join(proj_root, "third_party", "InfinityStar-main"),
            os.path.join(proj_root, "InfinityStar-main"),
            os.path.join(os.path.dirname(proj_root), "InfinityStar-main"),
        ]
        for cand in candidates:
            if cand and os.path.isdir(cand):
                p = os.path.abspath(cand)
                if p not in sys.path:
                    sys.path.insert(0, p)
                return

    def load_infinitystar_vae(
        vae_path: str,
        vae_type: int,
        device: torch.device,
        infinitystar_root: Optional[str],
        proj_root: str,
        semantic_scale_dim: int,
        detail_scale_dim: int,
        use_learnable_dim_proj: int,
        detail_scale_min_tokens: int,
        use_feat_proj: int,
        semantic_scales: int,
    ):
        _add_infinitystar_to_syspath(infinitystar_root, proj_root=proj_root)
        from infinity.models.videovae.models.load_vae_bsq_wan_absorb_patchify import (  # type: ignore
            video_vae_model,
        )

        global_args = SimpleNamespace(
            semantic_scale_dim=int(semantic_scale_dim),
            detail_scale_dim=int(detail_scale_dim),
            use_learnable_dim_proj=int(use_learnable_dim_proj),
            detail_scale_min_tokens=int(detail_scale_min_tokens),
            use_feat_proj=int(use_feat_proj),
            semantic_scales=int(semantic_scales),
        )
        vae = video_vae_model(
            vqgan_ckpt=str(vae_path),
            schedule_mode="dynamic",
            codebook_dim=int(vae_type),
            global_args=global_args,
            test_mode=True,
        ).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        return vae


def build_tsformer() -> VisionTransformer:
    # Keep consistent with stage2 training.
    from functools import partial

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


def _find_latest_stage2_checkpoint(stage2_root: str) -> str:
    root = os.path.abspath(stage2_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"stage2_root not found: {root}")

    best_p = None
    best_m = -1.0
    for d in os.listdir(root):
        dd = os.path.join(root, d)
        if not os.path.isdir(dd):
            continue
        p = os.path.join(dd, "checkpoint_last.pth")
        if not os.path.exists(p):
            continue
        try:
            m = os.path.getmtime(p)
        except Exception:
            continue
        if m > best_m:
            best_m = m
            best_p = p

    if best_p is None:
        raise FileNotFoundError(f"no checkpoint_last.pth found under {root}")
    return best_p


def _load_latents(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "latents" in obj:
        z = obj["latents"]
    else:
        z = obj
    if not isinstance(z, torch.Tensor) or z.ndim != 5:
        raise ValueError(f"latents must be a 5D Tensor, got {type(z)} shape={getattr(z,'shape',None)} at {path}")
    return z.float().contiguous()

def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # PyTorch >= 2.6 defaults torch.load(..., weights_only=True), which can
        # reject checkpoints that store numpy metadata (for example label_stats).
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_preprocessed_traj(path: str) -> np.ndarray:
    with open(path, "r") as f:
        arr = json.load(f)
    traj = np.asarray(arr, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] < 6:
        raise ValueError(f"preprocessed traj must be (T,6+) at {path}, got shape={traj.shape}")
    return traj[:, :6]


def _find_traj_json(route_dir: str) -> str:
    """
    Regenerated/resampled routes may use different filenames.
    Prefer the legacy name when it exists.
    """
    cand = [
        os.path.join(route_dir, "preprocessed_logs.json"),
        os.path.join(route_dir, "processed_logs.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"missing trajectory json under {route_dir}: tried {cand}")


def _sample_all_window_starts(T: int, window_size: int, stride: int, device: torch.device) -> torch.Tensor:
    max_start = int(T - window_size)
    if max_start < 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.arange(0, max_start + 1, int(stride), device=device, dtype=torch.long)


def _gather_window_tokens(tokens_tnd: torch.Tensor, starts: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    tokens_tnd: (T,N,D)
    starts: (K,)
    returns patch_tokens: (K*window_size, N, D)
    """
    T, N, D = tokens_tnd.shape
    K = int(starts.shape[0])
    flat = tokens_tnd.view(T, N * D)
    t_idx = torch.arange(window_size, device=starts.device, dtype=torch.long).view(1, window_size)
    idx = starts.view(K, 1) + t_idx  # (K,window_size)
    idx2 = idx.view(K * window_size, 1).expand(K * window_size, N * D)
    g = flat.gather(0, idx2).view(K * window_size, N, D)
    return g.contiguous()


def _R_from_rpy(roll: float, yaw: float, pitch: float) -> np.ndarray:
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    zyx = rotation_to_euler(R, seq="zyx")  # [yaw,pitch,roll]
    yaw, pitch, roll = float(zyx[0]), float(zyx[1]), float(zyx[2])
    return np.asarray([roll, yaw, pitch], dtype=np.float32)


def integrate_trajectory_se3(deltas_zyx: np.ndarray, init_rpy_rad: np.ndarray, init_pos_m: np.ndarray) -> np.ndarray:
    t = int(deltas_zyx.shape[0])
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _decode_tokens_full_T(
    *,
    vae: nn.Module,
    adapter: nn.Module,
    z_ext: torch.Tensor,
    amp_enabled: bool,
) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      tokens_tnd: (T,N,D) on same device
      T: int
    """
    if z_ext.ndim != 5 or int(z_ext.shape[0]) != 1:
        raise ValueError(f"expected z_ext shape (1,64,T_lat,16,16), got {tuple(z_ext.shape)}")

    tokens_slices: List[torch.Tensor] = []
    T_acc = 0

    def hook(_module, _inp, out):
        nonlocal tokens_slices, T_acc
        hs = out[0] if isinstance(out, (tuple, list)) else out  # (B,96,t_slice,H,W)
        if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
            raise RuntimeError("up_block_3 hook output is not a 5D Tensor")
        Bh = int(hs.shape[0])
        if Bh != 1:
            raise RuntimeError(f"unexpected VAE batch in hook: hs={tuple(hs.shape)}")
        t_slice = int(hs.shape[2])
        tok, _t2, _w2 = adapter(hs)  # (Bh*t_slice,N,D)
        tok = tok.view(Bh, t_slice, tok.shape[1], tok.shape[2]).contiguous()  # (1,t_slice,N,D)
        tokens_slices.append(tok[0])
        T_acc += t_slice

    handle = vae.decoder.up_blocks[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=bool(amp_enabled) and torch.cuda.is_available()):
                _ = vae.decode(z_ext, return_dict=False)[0]
    finally:
        handle.remove()

    if len(tokens_slices) == 0 or T_acc <= 0:
        return torch.empty((0, 0, 0), device=z_ext.device, dtype=torch.float32), 0
    tokens_tnd = torch.cat(tokens_slices, dim=0).contiguous()  # (T,N,D)
    return tokens_tnd, int(tokens_tnd.shape[0])


def infer_one_route(
    *,
    route: str,
    route_dir: str,
    ckpt_path: str,
    tsformer: VisionTransformer,
    adapter: nn.Module,
    vae: nn.Module,
    device: torch.device,
    out_root: str,
    stride: int,
    translation_divisor: float,
    angles_in_degrees: bool,
    amp: bool,
    compute_metrics: bool,
    label_stats: Optional[Dict[str, np.ndarray]] = None,
):
    lat_path = os.path.join(route_dir, "latents.pt")
    if not os.path.exists(lat_path):
        return False
    try:
        traj_path = _find_traj_json(route_dir)
    except FileNotFoundError:
        return False

    z_ext = _load_latents(lat_path).to(device)
    traj_abs = _load_preprocessed_traj(traj_path)  # (T,6) = [x,y,z,roll,yaw,pitch] (deg)
    T = int(traj_abs.shape[0])
    if T < 4:
        return False

    tokens_tnd, T_dec = _decode_tokens_full_T(vae=vae, adapter=adapter, z_ext=z_ext, amp_enabled=bool(amp))
    T_use = min(int(T), int(T_dec))
    if T_use < 4:
        return False
    tokens_tnd = tokens_tnd[:T_use]
    traj_abs = traj_abs[:T_use]

    window_size = 4
    starts = _sample_all_window_starts(T=T_use, window_size=window_size, stride=int(stride), device=device)
    if starts.numel() == 0:
        return False

    patch_tokens = _gather_window_tokens(tokens_tnd, starts=starts, window_size=window_size)
    K = int(starts.shape[0])
    W_grid = 40
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=bool(amp) and torch.cuda.is_available()):
            feat = tsformer.forward_features_from_patch_tokens(patch_tokens, B=K, T=window_size, W=W_grid)
            pred = tsformer.head(feat)  # (K,18)

    pred_f = pred.detach().float()  # (K,18)
    if isinstance(label_stats, dict) and all(k in label_stats for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
        # Denormalize to (rad, meters) so downstream evaluation stays consistent.
        ma = torch.as_tensor(label_stats["mean_angles"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        sa = torch.as_tensor(label_stats["std_angles"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        mt = torch.as_tensor(label_stats["mean_t"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        st = torch.as_tensor(label_stats["std_t"], dtype=torch.float32, device=pred_f.device).view(1, 1, 3)
        p = pred_f.view(K, 3, 6)
        p[:, :, 0:3] = p[:, :, 0:3] * sa + ma
        p[:, :, 3:6] = p[:, :, 3:6] * st + mt
        window_deltas = p.cpu().numpy().astype(np.float32)
    else:
        window_deltas = pred_f.cpu().numpy().reshape(K, 3, 6).astype(np.float32)  # (K,3,6)

    # Aggregate to per-frame deltas (T,6), delta[0]=0
    deltas = np.zeros((T_use, 6), dtype=np.float32)
    acc = np.zeros((T_use, 6), dtype=np.float32)
    cnt = np.zeros((T_use,), dtype=np.int32)
    starts_np = starts.detach().cpu().numpy().astype(np.int32).tolist()
    for i, s in enumerate(starts_np):
        for j in range(1, window_size):
            t = int(s + j)
            if 0 <= t < T_use:
                acc[t] += window_deltas[i, j - 1]
                cnt[t] += 1
    mask = cnt > 0
    deltas[mask] = acc[mask] / cnt[mask, None]

    # Init pose from traj_abs[0] = [x,y,z,roll,yaw,pitch]
    init_xyz = traj_abs[0, 0:3].astype(np.float32)
    if float(translation_divisor) != 1.0:
        init_xyz = init_xyz / float(translation_divisor)
    init_rpy = traj_abs[0, 3:6].astype(np.float32)
    if bool(angles_in_degrees):
        init_rpy = init_rpy * (np.pi / 180.0)

    traj_pred = integrate_trajectory_se3(deltas_zyx=deltas, init_rpy_rad=init_rpy, init_pos_m=init_xyz)

    out_one = os.path.join(out_root, route)
    os.makedirs(out_one, exist_ok=True)
    np.save(os.path.join(out_one, "deltas.npy"), deltas.astype(np.float32))
    np.save(os.path.join(out_one, "window_deltas.npy"), window_deltas.astype(np.float32))
    np.save(os.path.join(out_one, "trajectory.npy"), traj_pred.astype(np.float32))
    with open(os.path.join(out_one, "trajectory.json"), "w") as f:
        json.dump(traj_pred.tolist(), f)

    # Write "batch_infer_*" style jsons (for downstream tooling compatibility).
    actions6 = deltas[1:].astype(np.float32)  # (T-1,6)
    pred_actions_json = os.path.join(out_one, "pred_actions.json")
    pred_path_json = os.path.join(out_one, "pred_path.json")
    with open(pred_actions_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "route": route,
                "route_dir": route_dir,
                "latents_pt": lat_path,
                "preprocessed_logs_json": traj_path,
                "ckpt": ckpt_path,
                "actions6_layout": ["rz(dz)_rad", "ry(dy)_rad", "rx(dx)_rad", "tx_m", "ty_m", "tz_m"],
                "actions6": actions6.tolist(),
                "window_size": 4,
                "stride": int(stride),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(pred_path_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "route": route,
                "route_dir": route_dir,
                "latents_pt": lat_path,
                "preprocessed_logs_json": traj_path,
                "ckpt": ckpt_path,
                "start_pose_abs": {
                    "x": float(init_xyz[0]),
                    "y": float(init_xyz[1]),
                    "z": float(init_xyz[2]),
                    "roll_rad": float(init_rpy[0]),
                    "yaw_rad": float(init_rpy[1]),
                    "pitch_rad": float(init_rpy[2]),
                },
                "poses_layout": ["roll_rad", "yaw_rad", "pitch_rad", "x_m", "y_m", "z_m"],
                "poses": traj_pred.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Additional "physical-order + physical-unit" outputs for easier alignment with GT preprocessed_logs.json:
    # - GT pose layout: [x,y,z, roll,yaw,pitch] after applying --translation_divisor, in (m,deg)
    # - Our traj_pred layout: [roll,yaw,pitch,x,y,z] in (rad,m)
    traj_xyz_m = traj_pred[:, 3:6].astype(np.float32)
    traj_rpy_deg = (traj_pred[:, 0:3] * (180.0 / np.pi)).astype(np.float32)
    traj_m_deg = np.concatenate([traj_xyz_m, traj_rpy_deg], axis=1).astype(np.float32)  # (T,6) [x,y,z,roll,yaw,pitch]
    np.save(os.path.join(out_one, "trajectory_m_deg.npy"), traj_m_deg)
    with open(os.path.join(out_one, "trajectory_m_deg.json"), "w", encoding="utf-8") as f:
        json.dump(traj_m_deg.tolist(), f, ensure_ascii=False)

    # actions6 layout conversion:
    # training/infer deltas layout per step: [dz,dy,dx, tx,ty,tz] (rad,m) where dz=dyaw, dy=dpitch, dx=droll.
    # Convert to [tx_m,ty_m,tz_m, droll_deg, dyaw_deg, dpitch_deg] to match "xyz then rpy" physical-order.
    a = actions6.astype(np.float32)
    trans_m = a[:, 3:6].astype(np.float32)  # [tx,ty,tz] in meters (still prev-frame coords)
    droll_deg = (a[:, 2] * (180.0 / np.pi)).astype(np.float32)
    dyaw_deg = (a[:, 0] * (180.0 / np.pi)).astype(np.float32)
    dpitch_deg = (a[:, 1] * (180.0 / np.pi)).astype(np.float32)
    actions6_m_deg = np.concatenate(
        [trans_m[:, 0:3], droll_deg[:, None], dyaw_deg[:, None], dpitch_deg[:, None]],
        axis=1,
    ).astype(np.float32)
    np.save(os.path.join(out_one, "actions6_m_deg.npy"), actions6_m_deg)
    with open(os.path.join(out_one, "actions6_m_deg.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "route": route,
                "actions6_layout": ["x_m", "y_m", "z_m", "roll_deg", "yaw_deg", "pitch_deg"],
                "actions6": actions6_m_deg.tolist(),
                "note": "translation is still expressed in previous-frame coordinates; only unit/layout converted.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(out_one, "infer_meta.json"), "w") as f:
        json.dump(
            {
                "route": route,
                "ckpt_path": ckpt_path,
                "created_at": datetime.now().isoformat(),
                "T_use": int(T_use),
                "stride": int(stride),
                "translation_divisor": float(translation_divisor),
                "angles_in_degrees": bool(angles_in_degrees),
            },
            f,
            indent=2,
        )

    if compute_metrics:
        ref_xyz = traj_abs[:, 0:3].astype(np.float32)
        if float(translation_divisor) != 1.0:
            ref_xyz = ref_xyz / float(translation_divisor)
        ref_rpy = traj_abs[:, 3:6].astype(np.float32)
        if bool(angles_in_degrees):
            ref_rpy = ref_rpy * (np.pi / 180.0)
        ref_traj = np.concatenate([ref_rpy, ref_xyz], axis=1).astype(np.float32)  # (T,6) [rpy,xyz]
        metrics = {
            "len": int(T_use),
            "rmse_xyz_m": rmse(traj_pred[:, 3:6], ref_traj[:, 3:6]),
            "rmse_rpy_rad": rmse(traj_pred[:, 0:3], ref_traj[:, 0:3]),
        }
        with open(os.path.join(out_one, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage2_root",
        type=str,
        default="./checkpoints/stage2_latent2action",
        help="Stage-2 checkpoint root. Used only when --ckpt is empty.",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help=(
            "Explicit Stage-2 checkpoint path. Supported examples: checkpoint_last.pth, "
            "checkpoint_pre_fulltrain_e*.pth, stage2_latent2action_combined.pt. "
            "If empty, the script searches --stage2_root for the latest checkpoint_last.pth."
        ),
    )
    ap.add_argument(
        "--data_root",
        type=str,
        default="./data/uavflow_latents",
        help="UAV-Flow-style route root. Each route directory should contain latents.pt and preprocessed_logs.json.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/stage2_latent2action",
        help="Inference output root.",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--translation_divisor", type=float, default=1.0)
    ap.add_argument("--angles_in_degrees", action="store_true", default=True)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--compute_metrics", action="store_true", default=True)
    ap.add_argument("--first_n", type=int, default=0, help="只推理前 N 条（0 表示全部）")
    ap.add_argument(
        "--routes",
        type=str,
        default="",
        help="Optional comma-separated route directory names to infer.",
    )
    # If empty/0, prefer reading from stage2 checkpoint metadata (recommended).
    ap.add_argument("--infinitystar_vae_path", type=str, default="", help="Optional override; otherwise read from Stage-2 checkpoint metadata.")
    ap.add_argument("--infinitystar_vae_type", type=int, default=0, help="Optional override; 0 means read from checkpoint metadata or use 64.")
    ap.add_argument("--infinitystar_root", type=str, default="")
    ap.add_argument("--tqdm", action="store_true", default=True, help="显示 tqdm 进度条（若环境可用）")
    args = ap.parse_args()

    ckpt_path = str(args.ckpt).strip() or _find_latest_stage2_checkpoint(str(args.stage2_root))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    os.makedirs(str(args.out_dir), exist_ok=True)
    device = torch.device(str(args.device) if (torch.cuda.is_available() and str(args.device).startswith("cuda")) else "cpu")

    ckpt = _safe_torch_load(ckpt_path)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict (combined checkpoint)")

    tsformer = build_tsformer().to(device).eval()
    adapter = Vae96ToTSformerEmbedAdapter().to(device).eval()

    # Init VAE (InfinityStar)
    inf_root = str(args.infinitystar_root).strip() or None
    _add_infinitystar_to_syspath(inf_root, proj_root=_REPO_ROOT)
    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    vae_path = (
        str(args.infinitystar_vae_path).strip()
        or str(ckpt.get("infinitystar_vae_path", "")).strip()
        or str(ckpt_args.get("infinitystar_vae_path", "")).strip()
    )
    if not vae_path:
        raise ValueError(
            "InfinityStar VAE path is required. Pass --infinitystar_vae_path "
            "or store infinitystar_vae_path in the Stage-2 checkpoint metadata."
        )
    vae_type = int(args.infinitystar_vae_type) if int(args.infinitystar_vae_type) > 0 else int(
        ckpt.get("infinitystar_vae_type", ckpt_args.get("infinitystar_vae_type", 64))
    )
    print(f"[config] InfinityStar VAE: path={vae_path} type={vae_type}", flush=True)
    vae = load_infinitystar_vae(
        vae_path=str(vae_path),
        vae_type=int(vae_type),
        device=device,
        infinitystar_root=inf_root,
        proj_root=_REPO_ROOT,
        semantic_scale_dim=int(ckpt_args.get("semantic_scale_dim", 16)),
        detail_scale_dim=int(ckpt_args.get("detail_scale_dim", 64)),
        use_learnable_dim_proj=int(ckpt_args.get("use_learnable_dim_proj", 0)),
        detail_scale_min_tokens=int(ckpt_args.get("detail_scale_min_tokens", 350)),
        use_feat_proj=int(ckpt_args.get("use_feat_proj", 2)),
        semantic_scales=int(ckpt_args.get("semantic_scales", 11)),
    )

    # Supported formats:
    # - checkpoint_last.pth: {model_state_dict, adapter_state_dict, vae_state_dict, ...}
    # - checkpoint_pre_fulltrain_e*.pth: same as above
    # - stage2_latent2action_combined.pt: {tsformer_state_dict, adapter_state_dict, vae_state_dict, model_state_dict(alias), ...}
    ts_sd = ckpt.get("model_state_dict") or ckpt.get("tsformer_state_dict")
    ad_sd = ckpt.get("adapter_state_dict") or ckpt.get("state_dict")  # fallback
    vae_sd = ckpt.get("vae_state_dict", {})
    if not isinstance(ts_sd, dict) or not isinstance(ad_sd, dict):
        raise ValueError("checkpoint missing model_state_dict/adapter_state_dict")
    missing, unexpected = tsformer.load_state_dict(ts_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] tsformer strict=False missing={missing[:10]} unexpected={unexpected[:10]}", flush=True)
    missing, unexpected = adapter.load_state_dict(ad_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] adapter strict=False missing={missing[:10]} unexpected={unexpected[:10]}", flush=True)
    if isinstance(vae_sd, dict) and len(vae_sd) > 0:
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
        if missing or unexpected:
            print(f"[warn] vae strict=False missing={missing[:10]} unexpected={unexpected[:10]}", flush=True)

    # Optional: label stats for denormalizing head outputs -> (rad, meters).
    label_stats = None
    ls = ckpt.get("label_stats")
    if isinstance(ls, dict) and all(k in ls for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
        try:
            label_stats = {
                "mean_angles": np.asarray(ls["mean_angles"], dtype=np.float32).reshape(3),
                "std_angles": np.asarray(ls["std_angles"], dtype=np.float32).reshape(3),
                "mean_t": np.asarray(ls["mean_t"], dtype=np.float32).reshape(3),
                "std_t": np.asarray(ls["std_t"], dtype=np.float32).reshape(3),
            }
            src = ckpt.get("label_stats_source") or "checkpoint"
            print(f"[config] label_stats loaded from {src}", flush=True)
        except Exception as e:
            print(f"[warn] failed to parse label_stats from checkpoint: {e}", flush=True)
            label_stats = None
    if label_stats is None:
        # Backward-compatible fallback: try to locate run_config.json next to the TSformer pretrained checkpoint
        # recorded in stage2 training args.
        try:
            args0 = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
            ts_pre = str(args0.get("tsformer_pretrained", "")).strip()
            if ts_pre:
                run_cfg = os.path.join(os.path.dirname(os.path.abspath(ts_pre)), "run_config.json")
                if os.path.isfile(run_cfg):
                    rc = json.loads(open(run_cfg, "r", encoding="utf-8").read())
                    ls2 = rc.get("label_stats") if isinstance(rc, dict) else None
                    if isinstance(ls2, dict) and all(k in ls2 for k in ("mean_angles", "std_angles", "mean_t", "std_t")):
                        label_stats = {
                            "mean_angles": np.asarray(ls2["mean_angles"], dtype=np.float32).reshape(3),
                            "std_angles": np.asarray(ls2["std_angles"], dtype=np.float32).reshape(3),
                            "mean_t": np.asarray(ls2["mean_t"], dtype=np.float32).reshape(3),
                            "std_t": np.asarray(ls2["std_t"], dtype=np.float32).reshape(3),
                        }
                        print(f"[config] label_stats loaded from run_config.json next to tsformer_pretrained: {run_cfg}", flush=True)
        except Exception as e:
            print(f"[warn] label_stats fallback load failed: {e}", flush=True)

    routes = [d for d in os.listdir(str(args.data_root)) if os.path.isdir(os.path.join(str(args.data_root), d))]
    routes.sort()
    only = str(args.routes).strip()
    if only:
        wanted = [x.strip() for x in only.split(",") if x.strip()]
        # keep order, drop duplicates
        seen = set()
        wanted2 = []
        for w in wanted:
            if w not in seen:
                wanted2.append(w)
                seen.add(w)
        missing = [w for w in wanted2 if w not in set(routes)]
        if missing:
            raise FileNotFoundError(f"--routes contains missing dirs under data_root: {missing}")
        routes = wanted2
    if int(args.first_n) > 0:
        routes = routes[: int(args.first_n)]

    ok = 0
    skipped = 0
    it = routes
    if bool(args.tqdm) and tqdm is not None:
        it = tqdm(routes, desc="infer routes", dynamic_ncols=True)
    for r in it:
        try:
            did = infer_one_route(
                route=r,
                route_dir=os.path.join(str(args.data_root), r),
                ckpt_path=ckpt_path,
                tsformer=tsformer,
                adapter=adapter,
                vae=vae,
                device=device,
                out_root=str(args.out_dir),
                stride=int(args.stride),
                translation_divisor=float(args.translation_divisor),
                angles_in_degrees=bool(args.angles_in_degrees),
                amp=bool(args.amp),
                compute_metrics=bool(args.compute_metrics),
                label_stats=label_stats,
            )
            ok += int(bool(did))
            if not did:
                skipped += 1
        except Exception as e:
            skipped += 1
            print(f"[warn] skip route={r} err={e}", flush=True)

    print(json.dumps({"ok": ok, "skipped": skipped, "ckpt": ckpt_path, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

