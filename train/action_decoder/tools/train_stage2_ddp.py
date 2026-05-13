"""
Stage-2 (latent -> action/VO) training with InfinityStar VAE + Adapter + TSformer.

Key goals:
- Support variable-length trajectories (require_T can be disabled).
- Support two collate strategies:
  - crop: crop within batch to min length then stack (fast, but modifies lengths)
  - per_sample: keep each sample original length; train by iterating samples in a step (no crop, no padding)
- Rank0-only tqdm + train.log.
"""

import argparse
import contextlib
import json
import os
import random
import sys
import time
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

# Ensure action-decoder architecture code is importable even when running via absolute path.
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAIN_ROOT = os.path.abspath(os.path.join(_TOOL_DIR, ".."))
_OPEN_ROOT = os.path.abspath(os.path.join(_TRAIN_ROOT, "..", ".."))
_ARCH_ROOT = os.path.join(_OPEN_ROOT, "Worldmodel", "action_decoder", "src")
if _ARCH_ROOT not in sys.path:
    sys.path.insert(0, _ARCH_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.latent_traj_manifest import LatentTrajManifestDataset
from datasets.utils import euler_to_rotation, rotation_to_euler
from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter
from timesformer.models.vit import VisionTransformer

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


_PROJ_ROOT = _ARCH_ROOT
_RANK0_LOG_FH = None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def rank0_print(*args, **kwargs):
    if get_rank() == 0:
        global _RANK0_LOG_FH
        print(*args, **kwargs, flush=True)
        if _RANK0_LOG_FH is not None:
            msg = " ".join(str(a) for a in args)
            _RANK0_LOG_FH.write(msg + "\n")
            _RANK0_LOG_FH.flush()


def ddp_setup() -> int:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return x
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= get_world_size()
    return y


def _allreduce_grads(param_groups):
    """Average gradients of all trainable params across all DDP ranks."""
    if not is_dist():
        return
    ws = float(get_world_size())
    for pg in param_groups:
        for p in pg["params"]:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(ws)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _try_load_label_stats(*, label_stats_json: str, tsformer_pretrained: str) -> Tuple[Optional[Dict[str, np.ndarray]], str]:
    """
    Load label normalization stats (mean/std) used by TSformer training.

    Returns:
      (stats, source_path)
      - stats keys: mean_angles,std_angles,mean_t,std_t as np.ndarray shape (3,)
      - stats is None when not found
    """
    cand = str(label_stats_json).strip()
    if cand:
        p = os.path.abspath(cand)
        if os.path.isfile(p):
            obj = _read_json(p)
            if isinstance(obj, dict) and "label_stats" in obj and isinstance(obj["label_stats"], dict):
                obj = obj["label_stats"]
            if not isinstance(obj, dict):
                raise ValueError(f"label_stats_json must be a dict or run_config with label_stats: {p}")
            out: Dict[str, np.ndarray] = {}
            for k in ("mean_angles", "std_angles", "mean_t", "std_t"):
                if k not in obj:
                    raise ValueError(f"label_stats_json missing key={k}: {p}")
                out[k] = np.asarray(obj[k], dtype=np.float32).reshape(3)
            return out, p

    # Try run_config.json next to pretrained checkpoint.
    base = os.path.dirname(os.path.abspath(str(tsformer_pretrained)))
    p2 = os.path.join(base, "run_config.json")
    if os.path.isfile(p2):
        obj = _read_json(p2)
        if isinstance(obj, dict) and "label_stats" in obj and isinstance(obj["label_stats"], dict):
            ls = obj["label_stats"]
            out = {
                "mean_angles": np.asarray(ls["mean_angles"], dtype=np.float32).reshape(3),
                "std_angles": np.asarray(ls["std_angles"], dtype=np.float32).reshape(3),
                "mean_t": np.asarray(ls["mean_t"], dtype=np.float32).reshape(3),
                "std_t": np.asarray(ls["std_t"], dtype=np.float32).reshape(3),
            }
            return out, p2

    return None, ""


def _normalize_delta_bt6(delta_bt6: torch.Tensor, stats_t: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    delta_bt6: (B,T,6) in (rad, meters). Returns normalized (B,T,6) as in UavflowSimDataset.
    """
    if delta_bt6.ndim != 3 or int(delta_bt6.shape[-1]) < 6:
        raise ValueError(f"expected delta_bt6 (B,T,6), got {tuple(delta_bt6.shape)}")
    mean_a = stats_t["mean_angles"].view(1, 1, 3)
    std_a = stats_t["std_angles"].view(1, 1, 3)
    mean_t = stats_t["mean_t"].view(1, 1, 3)
    std_t = stats_t["std_t"].view(1, 1, 3)
    out = delta_bt6.clone()
    out[..., 0:3] = (out[..., 0:3] - mean_a) / std_a
    out[..., 3:6] = (out[..., 3:6] - mean_t) / std_t
    return out


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def build_tsformer() -> VisionTransformer:
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


def _resolve_ckpt_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_tsformer(model: nn.Module, ckpt_path: str):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _resolve_ckpt_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        rank0_print(f"[warn] TSformer strict=False missing={missing[:10]} unexpected={unexpected[:10]}")


def _load_adapter_state_dict(adapter_ckpt_path: str):
    try:
        obj = torch.load(adapter_ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(adapter_ckpt_path, map_location="cpu")

    if isinstance(obj, dict) and "adapter_state_dict" in obj:
        return obj["adapter_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and any(k.startswith("patch.") or k.startswith("proj.") for k in obj.keys()):
        return obj
    raise ValueError(f"Unsupported adapter checkpoint format: {adapter_ckpt_path}")


def _add_infinitystar_to_syspath(inf_root: Optional[str], proj_root: str):
    if inf_root and os.path.isdir(inf_root):
        p = os.path.abspath(inf_root)
        if p not in sys.path:
            sys.path.insert(0, p)
        return
    open_root = os.path.abspath(os.path.join(proj_root, "..", "..", ".."))
    candidates = [
        os.environ.get("INFINITYSTAR_ROOT", ""),
        os.environ.get("INFINITYSTAR_HOME", ""),
        os.path.join(open_root, "Worldmodel", "runtime"),
        os.path.join(open_root, "Worldmodel"),
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


def traj_abs_to_delta(traj: np.ndarray, angles_in_degrees: bool, translation_divisor: float) -> np.ndarray:
    """
    traj: (T,6) = [x, y, z, roll, yaw, pitch] (angles are in degrees when angles_in_degrees=True)
    returns delta: (T,6), where delta[t] is motion (t-1 -> t), delta[0]=0
    Layout per step: [dz, dy, dx, tx, ty, tz] with ZYX euler in radians.
    """
    traj = np.asarray(traj, dtype=np.float32)
    T = int(traj.shape[0])
    out = np.zeros((T, 6), dtype=np.float32)
    if T <= 1:
        return out

    pos = traj[:, 0:3].copy() / float(translation_divisor)
    rpy = traj[:, 3:6].copy()
    if angles_in_degrees:
        rpy = rpy * (np.pi / 180.0)

    # unwrap angles (radians) to avoid discontinuities across +/-pi
    for i in range(3):
        rpy[:, i] = np.unwrap(rpy[:, i])

    Rs = []
    for t in range(T):
        # raw/preprocessed logs in this project use [roll, yaw, pitch]
        roll, yaw, pitch = float(rpy[t, 0]), float(rpy[t, 1]), float(rpy[t, 2])
        R = np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)
        Rs.append(R)

    for t in range(1, T):
        R_prev = Rs[t - 1]
        R_cur = Rs[t]
        R_rel = R_prev.T @ R_cur
        zyx = rotation_to_euler(R_rel, seq="zyx")  # [z,y,x] radians
        out[t, 0:3] = np.asarray(zyx, dtype=np.float32)
        p_rel = R_prev.T @ (pos[t] - pos[t - 1])
        out[t, 3:6] = p_rel.astype(np.float32)
    return out


def delta_to_delta(
    delta_like: np.ndarray,
    *,
    translation_divisor: float,
    dyaw_unit: str = "auto",
) -> np.ndarray:
    """
    Convert a delta-like sequence into canonical (T,6) delta in (rad, meters).

    Supported inputs:
    - (T,6):  [dz,dy,dx, tx,ty,tz] with delta[0] possibly zero
    - (T-1,6): per-step deltas, will be padded with delta[0]=0
    - (T,4) or (T-1,4): [tx,ty,tz, dyaw] (yaw-only). roll/pitch set to 0.

    Notes:
    - Rotation layout is always [dz,dy,dx] = [dyaw, dpitch, droll] in radians.
    - Translation is divided by translation_divisor (e.g. 100 for cm->m). Set to 1 if already meters.
    - dyaw_unit: auto/deg/rad (auto uses magnitude heuristic).
    """
    x = np.asarray(delta_like, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"delta_like must be 2D, got shape={x.shape}")
    Tm = int(x.shape[0])
    C = int(x.shape[1])
    if C not in (4, 6):
        raise ValueError(f"delta_like must have 4 or 6 columns, got shape={x.shape}")

    # Build per-step deltas of shape (T-1,6)
    if C == 6:
        step = x[:, :6].astype(np.float32)
    else:
        # [tx,ty,tz, dyaw]
        step = np.zeros((Tm, 6), dtype=np.float32)
        step[:, 0] = x[:, 3]  # dz=dyaw
        step[:, 3:6] = x[:, 0:3]

    # Determine if this is (T,6) with delta[0]=0 or (T-1,6)
    # Heuristic: if first row is all zeros (or near), treat as (T,6) already.
    if Tm >= 1 and float(np.max(np.abs(step[0]))) < 1e-8:
        delta = step
    else:
        delta = np.zeros((Tm + 1, 6), dtype=np.float32)
        delta[1:, :] = step

    # dyaw unit conversion
    unit = str(dyaw_unit).lower().strip() if dyaw_unit else "auto"
    if unit not in ("auto", "deg", "rad"):
        raise ValueError(f"dyaw_unit must be auto/deg/rad (got {dyaw_unit!r})")
    if unit == "auto":
        dz = delta[1:, 0]
        p95 = float(np.nanpercentile(np.abs(dz), 95)) if dz.size else 0.0
        unit = "deg" if p95 > 1.0 else "rad"
    if unit == "deg":
        delta[:, 0] = delta[:, 0] * (np.pi / 180.0)

    # translation unit conversion
    div = float(translation_divisor)
    if not np.isfinite(div) or div <= 0:
        raise ValueError(f"translation_divisor must be finite > 0 (got {translation_divisor})")
    if div != 1.0:
        delta[:, 3:6] = delta[:, 3:6] / div
    return delta.astype(np.float32)


def traj_to_delta(
    traj_or_delta: np.ndarray,
    *,
    traj_mode: str,
    angles_in_degrees: bool,
    translation_divisor: float,
    delta_dyaw_unit: str,
) -> np.ndarray:
    """
    Convert loaded trajectory json content into canonical delta (T,6) in (rad, meters).
    """
    mode = str(traj_mode).lower().strip()
    if mode == "abs_pose":
        return traj_abs_to_delta(traj_or_delta, angles_in_degrees=angles_in_degrees, translation_divisor=translation_divisor)
    if mode == "delta":
        return delta_to_delta(traj_or_delta, translation_divisor=translation_divisor, dyaw_unit=delta_dyaw_unit)
    raise ValueError(f"unknown traj_mode: {traj_mode} (expected abs_pose|delta)")


def compute_loss_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    rot_weight: float,
    trans_xy_weight: float,
    trans_z_weight: float,
    trans_vertical_index: int,
) -> torch.Tensor:
    b = int(target.shape[0])
    t = int(window_size - 1)
    pred = pred.view(b, t, 6)
    target = target.view(b, t, 6)
    pred_r, pred_t = pred[:, :, :3], pred[:, :, 3:]
    tgt_r, tgt_t = target[:, :, :3], target[:, :, 3:]

    loss = torch.zeros((), device=pred.device, dtype=torch.float32)
    if float(rot_weight) > 0:
        loss = loss + float(rot_weight) * F.mse_loss(pred_r, tgt_r)

    vi = int(trans_vertical_index)
    horiz = [0, 1, 2]
    horiz.remove(vi)
    loss_h = F.mse_loss(pred_t[:, :, horiz], tgt_t[:, :, horiz])
    loss_v = F.mse_loss(pred_t[:, :, vi], tgt_t[:, :, vi])
    loss = loss + float(trans_xy_weight) * loss_h + float(trans_z_weight) * loss_v
    return loss


def _linear_schedule(epoch: int, warmup_or_decay_epochs: int, start: float, end: float) -> float:
    if warmup_or_decay_epochs <= 0:
        return float(end)
    e = max(1, int(epoch))
    if warmup_or_decay_epochs == 1:
        return float(end)
    t = float(e - 1) / float(warmup_or_decay_epochs - 1)
    t = min(1.0, max(0.0, t))
    return float(start + t * (end - start))


def _sample_all_window_starts(B: int, T: int, window_size: int, stride: int, device: torch.device) -> torch.Tensor:
    max_start = int(T - window_size)
    if max_start < 0:
        raise ValueError(f"T={T} < window_size={window_size}")
    idx = torch.arange(0, max_start + 1, int(stride), device=device, dtype=torch.long)
    return idx.view(1, -1).repeat(B, 1).contiguous()


def _gather_window_tokens(tokens_btnd: torch.Tensor, starts_bk: torch.Tensor, window_size: int) -> torch.Tensor:
    B, T, N, D = tokens_btnd.shape
    K = int(starts_bk.shape[1])
    flat = tokens_btnd.view(B, T, N * D)
    t_idx = torch.arange(window_size, device=starts_bk.device, dtype=torch.long).view(1, 1, window_size)
    idx = starts_bk.unsqueeze(-1) + t_idx  # (B,K,window_size)
    idx2 = idx.view(B, K * window_size, 1).expand(B, K * window_size, N * D)
    g = flat.gather(1, idx2).view(B * K, window_size, N, D)
    return g.reshape(B * K * window_size, N, D).contiguous()


def _gather_window_targets(delta_bt6: torch.Tensor, starts_bk: torch.Tensor, window_size: int) -> torch.Tensor:
    B, T, _ = delta_bt6.shape
    K = int(starts_bk.shape[1])
    t_idx = torch.arange(1, window_size, device=starts_bk.device, dtype=torch.long).view(1, 1, window_size - 1)
    idx = starts_bk.unsqueeze(-1) + t_idx
    idx2 = idx.view(B, K * (window_size - 1), 1).expand(B, K * (window_size - 1), 6)
    g = delta_bt6.gather(1, idx2).view(B * K, window_size - 1, 6)
    return g.reshape(B * K, (window_size - 1) * 6).contiguous()


def _decode_tokens_full_T(
    vae_decode_module,
    adapter,
    z_ext: torch.Tensor,
    allow_adapter_grad: bool,
    allow_vae_grad: bool,
    amp_enabled: bool,
    expected_T: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      tokens: (B,T,N,D)
      T: int
    """
    B = int(z_ext.shape[0])
    m = vae_decode_module.module if isinstance(vae_decode_module, DDP) else vae_decode_module
    vae = m.vae

    uses_batch_slicing = bool(getattr(vae, "use_slicing", False)) and B > 1

    def _run_decode(z_sub: torch.Tensor, expected_B: int) -> Tuple[torch.Tensor, int]:
        tokens_slices = []
        start = 0

        def hook(_module, _inp, out):
            nonlocal start, tokens_slices
            hs = out[0] if isinstance(out, (tuple, list)) else out  # (B,96,t_slice,H,W)
            if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
                raise RuntimeError("up_block_3 hook output is not a 5D Tensor")
            Bh = int(hs.shape[0])
            if Bh != int(expected_B):
                raise RuntimeError(f"unexpected VAE batch in hook: hs={tuple(hs.shape)} expected_B={expected_B}")
            t_slice = int(hs.shape[2])
            ctx = torch.enable_grad() if allow_adapter_grad else torch.no_grad()
            with ctx:
                tok, _t2, _w2 = adapter(hs)  # (Bh*t_slice,N,D)
                tok = tok.view(Bh, t_slice, tok.shape[1], tok.shape[2]).contiguous()
            tokens_slices.append(tok)
            start += t_slice

        handle = m.vae.decoder.up_blocks[-1].register_forward_hook(hook)
        try:
            grad_ctx = contextlib.nullcontext() if allow_vae_grad else torch.no_grad()
            amp_ctx = autocast(enabled=bool(amp_enabled) and torch.cuda.is_available())
            try:
                with grad_ctx, amp_ctx:
                    _ = vae_decode_module(z_sub)
            except RuntimeError as e:
                msg = str(e)
                if "torch.cat(): expected a non-empty list of Tensors" in msg:
                    rank0_print(f"[warn] VAE decode empty dec list; skip. z_sub={tuple(z_sub.shape)} err={msg}")
                    empty = torch.empty((int(expected_B), 0, 480, 384), device=z_sub.device, dtype=torch.float32)
                    return empty, 0
                raise
        finally:
            handle.remove()

        T = int(start)
        if len(tokens_slices) == 0 or T <= 0:
            empty = torch.empty((int(expected_B), 0, 480, 384), device=z_sub.device, dtype=torch.float32)
            return empty, 0
        out = torch.cat(tokens_slices, dim=1)
        return out, T

    if uses_batch_slicing:
        outs = []
        Ts = []
        for b_idx in range(B):
            out_b, T_b = _run_decode(z_ext[b_idx : b_idx + 1], expected_B=1)
            outs.append(out_b)
            Ts.append(int(T_b))
        if len(set(Ts)) != 1 and max(Ts) > 0:
            rank0_print(f"[warn] decoded T differs across batch under slicing: {Ts}")
        out = torch.cat(outs, dim=0)
        T = int(out.shape[1])
    else:
        out, T = _run_decode(z_ext, expected_B=B)

    if expected_T is not None and int(T) != int(expected_T):
        rank0_print(f"[warn] decoded T={T} but expected_T={expected_T}")
    return out, int(T)


def collate_fn(samples, mode: str = "crop"):
    mode = str(mode).strip().lower()
    if mode == "per_sample":
        out_samples = []
        for s in samples:
            z = s["z_ext"]
            if not isinstance(z, torch.Tensor):
                z = torch.as_tensor(z)
            z = z.float().contiguous()

            t = s["traj"]
            if isinstance(t, torch.Tensor):
                tt = t.float().contiguous()
            else:
                tt = torch.from_numpy(np.asarray(t, dtype=np.float32)).contiguous()
            out_samples.append({"z_ext": z, "traj": tt, "meta": s.get("meta", {})})
        return {"samples": out_samples, "meta": [s.get("meta", {}) for s in samples]}

    # crop mode: stack after cropping to min length in batch
    z_list = []
    t_lat_list = []
    for s in samples:
        z = s["z_ext"]
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        z = z.float().contiguous()
        z_list.append(z)
        t_lat_list.append(int(z.shape[2]))
    t_lat_min = int(min(t_lat_list)) if t_lat_list else 0
    if t_lat_min <= 0:
        raise ValueError(f"invalid batch latent lengths: {t_lat_list}")

    traj_list = []
    t_traj_list = []
    for s in samples:
        t = s["traj"]
        if isinstance(t, torch.Tensor):
            tt = t.float().contiguous()
        else:
            tt = torch.from_numpy(np.asarray(t, dtype=np.float32)).contiguous()
        traj_list.append(tt)
        t_traj_list.append(int(tt.shape[0]))
    t_traj_min = int(min(t_traj_list)) if t_traj_list else 0
    if t_traj_min <= 0:
        raise ValueError(f"invalid batch traj lengths: {t_traj_list}")

    z_ext = torch.cat([z[:, :, :t_lat_min].contiguous() for z in z_list], dim=0).contiguous()
    traj = torch.stack([t[:t_traj_min].contiguous() for t in traj_list], dim=0).contiguous()
    meta = [s.get("meta", {}) for s in samples]
    return {"z_ext": z_ext, "traj": traj, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_json", type=str, required=True)
    ap.add_argument("--items_key", type=str, default="ALL", help="Manifest key(s) to use; comma-separated, or ALL for every items_* list.")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--require_T", type=int, default=49)

    ap.add_argument("--tsformer_pretrained", type=str, required=True)
    ap.add_argument("--adapter_ckpt", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument(
        "--label_stats_json",
        type=str,
        default="",
        help="可选：TSformer 训练时的 run_config.json（或仅包含 label_stats 的 json）。用于将 GT delta 归一化到 checkpoint 输出分布。",
    )

    ap.add_argument("--infinitystar_vae_path", type=str, required=True)
    ap.add_argument("--infinitystar_vae_type", type=int, default=64)
    ap.add_argument("--infinitystar_root", type=str, default="")

    # These must match the VAE checkpoint architecture.
    ap.add_argument("--semantic_scale_dim", type=int, default=16)
    ap.add_argument("--detail_scale_dim", type=int, default=64)
    ap.add_argument("--use_learnable_dim_proj", type=int, default=0)
    ap.add_argument("--detail_scale_min_tokens", type=int, default=350)
    ap.add_argument("--use_feat_proj", type=int, default=2)
    ap.add_argument("--semantic_scales", type=int, default=11)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--global_batch_size", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--window_stride", type=int, default=1)
    ap.add_argument("--windows_chunk", type=int, default=8)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true", default=False)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--collate_mode", type=str, default="crop", choices=["crop", "per_sample"])

    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head_lr_mult", type=float, default=10.0)
    ap.add_argument("--adapter_lr_mult", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Optimizer step happens every N loader iterations (no microbatching within a sample).",
    )
    ap.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm for clipping (0 = disabled)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--tqdm", action="store_true", default=False)
    ap.add_argument("--log_file", type=str, default="train.log")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", default=False)

    # loss config
    ap.add_argument("--translation_divisor", type=float, default=1.0)
    ap.add_argument("--angles_in_degrees", action="store_true", default=True)
    ap.add_argument(
        "--traj_mode",
        type=str,
        default="abs_pose",
        choices=["abs_pose", "delta"],
        help="How to interpret traj_json_path. abs_pose: (T,6) cumulative pose -> convert to delta. "
        "delta: per-step delta (T-1,6)/(T,6) or yaw-only (T-1,4)/(T,4).",
    )
    ap.add_argument(
        "--delta_dyaw_unit",
        type=str,
        default="auto",
        choices=["auto", "deg", "rad"],
        help="When traj_mode=delta, interpret dz(dyaw) unit. auto uses magnitude heuristic.",
    )
    ap.add_argument("--rot_loss_weight_start", type=float, default=0.0)
    ap.add_argument("--rot_loss_weight_max", type=float, default=0.0)
    ap.add_argument("--rot_warmup_epochs", type=int, default=0)
    ap.add_argument("--trans_xy_weight", type=float, default=1.0)
    ap.add_argument("--trans_z_weight", type=float, default=2.0)
    ap.add_argument("--trans_z_weight_start", type=float, default=None)
    ap.add_argument("--trans_z_weight_end", type=float, default=None)
    ap.add_argument("--trans_z_decay_epochs", type=int, default=0)
    ap.add_argument("--trans_vertical_index", type=int, default=2)

    # train switches
    ap.add_argument("--train_adapter", action="store_true", default=False)
    ap.add_argument("--freeze_adapter_epochs", type=int, default=0, help="Freeze adapter for the first N epochs so TSformer can adapt to its token distribution")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--train_vae_after_epoch", type=int, default=0)
    ap.add_argument("--vae_lr_mult", type=float, default=0.1)
    ap.add_argument("--vae_disable_slicing", action="store_true", default=False)
    ap.add_argument("--vae_disable_tiling", action="store_true", default=False)
    ap.add_argument("--vae_num_sample_frames_batch_size", type=int, default=0)

    # export
    ap.add_argument("--export_combined", action="store_true", default=True)
    ap.add_argument("--export_name", type=str, default="stage2_latent2action_combined.pt")

    args = ap.parse_args()

    local_rank = ddp_setup()
    device = torch.device(str(args.device))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available() is False")
        device = torch.device(f"cuda:{local_rank}")

    seed_everything(int(args.seed) + get_rank())
    if not str(args.out_dir).strip():
        raise ValueError("--out_dir 不能为空")
    os.makedirs(str(args.out_dir), exist_ok=True)

    global _RANK0_LOG_FH
    if get_rank() == 0:
        lf = str(args.log_file).strip()
        if lf:
            _RANK0_LOG_FH = open(os.path.join(str(args.out_dir), lf), "a", encoding="utf-8")
            rank0_print(f"[log] writing rank0 logs to {os.path.join(str(args.out_dir), lf)}")
        try:
            args_dump = json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            args_dump = str(vars(args))
        rank0_print(
            "[config]"
            f" rank={get_rank()}"
            f" world_size={get_world_size()}"
            f" device={device}"
            f" cuda={torch.cuda.is_available()}"
            f" amp={bool(args.amp)}"
        )
        rank0_print("[config] args:\n" + args_dump)

    # global batch override
    if int(args.global_batch_size) > 0:
        ws = int(get_world_size())
        if int(args.global_batch_size) % ws != 0:
            raise ValueError(f"global_batch_size={args.global_batch_size} must be divisible by world_size={ws}")
        args.batch_size = int(args.global_batch_size) // ws
        if int(args.batch_size) < 1:
            raise ValueError("per-GPU batch_size computed < 1")
        rank0_print(f"[batch] global_batch_size={args.global_batch_size} world_size={ws} per_gpu_batch={args.batch_size}")

    tsformer = build_tsformer().to(device)
    load_tsformer(tsformer, str(args.tsformer_pretrained))

    label_stats_np, label_stats_src = _try_load_label_stats(
        label_stats_json=str(args.label_stats_json),
        tsformer_pretrained=str(args.tsformer_pretrained),
    )
    label_stats_t = None
    if label_stats_np is not None:
        label_stats_t = {
            k: torch.from_numpy(v).to(device=device, dtype=torch.float32)
            for k, v in label_stats_np.items()
        }
        rank0_print(f"[config] label_stats loaded from: {label_stats_src}")
    else:
        label_stats_src = ""
        rank0_print(
            "[warn] label_stats not found; will train in UN-normalized target space (rad+m). "
            "If you initialize from a TSformer checkpoint trained with normalized labels, consider passing --label_stats_json "
            "or placing run_config.json next to --tsformer_pretrained."
        )

    adapter = Vae96ToTSformerEmbedAdapter().to(device)
    adapter_sd = _load_adapter_state_dict(str(args.adapter_ckpt))
    missing, unexpected = adapter.load_state_dict(adapter_sd, strict=False)
    if len(missing) or len(unexpected):
        rank0_print(f"[warn] adapter strict=False missing={missing[:10]} unexpected={unexpected[:10]}")
    adapter.requires_grad_(bool(args.train_adapter))
    adapter.eval()

    inf_root = str(args.infinitystar_root).strip() or None
    vae_model = load_infinitystar_vae(
        vae_path=str(args.infinitystar_vae_path),
        vae_type=int(args.infinitystar_vae_type),
        device=device,
        infinitystar_root=inf_root,
        proj_root=_PROJ_ROOT,
        semantic_scale_dim=int(args.semantic_scale_dim),
        detail_scale_dim=int(args.detail_scale_dim),
        use_learnable_dim_proj=int(args.use_learnable_dim_proj),
        detail_scale_min_tokens=int(args.detail_scale_min_tokens),
        use_feat_proj=int(args.use_feat_proj),
        semantic_scales=int(args.semantic_scales),
    )

    if bool(args.vae_disable_slicing) and hasattr(vae_model, "disable_slicing"):
        try:
            vae_model.disable_slicing()
            rank0_print("[config] vae.disable_slicing()")
        except Exception as e:
            rank0_print(f"[warn] vae.disable_slicing() failed: {e}")
    if bool(args.vae_disable_tiling) and hasattr(vae_model, "disable_tiling"):
        try:
            vae_model.disable_tiling()
            rank0_print("[config] vae.disable_tiling()")
        except Exception as e:
            rank0_print(f"[warn] vae.disable_tiling() failed: {e}")
    if int(args.vae_num_sample_frames_batch_size) > 0 and hasattr(vae_model, "num_sample_frames_batch_size"):
        try:
            vae_model.num_sample_frames_batch_size = int(args.vae_num_sample_frames_batch_size)
            rank0_print(f"[config] set vae.num_sample_frames_batch_size={int(args.vae_num_sample_frames_batch_size)}")
        except Exception as e:
            rank0_print(f"[warn] setting vae.num_sample_frames_batch_size failed: {e}")

    will_train_vae = bool(int(args.train_vae_after_epoch) < int(args.epochs))

    class _VaeDecodeOnly(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, z_ext: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(z_ext, return_dict=False)[0]

    vae_decode = _VaeDecodeOnly(vae_model)

    def _vae_set_trainable(trainable: bool):
        for name in ("proj_up", "post_quant_conv", "decoder", "scale_learnable_parameters"):
            sub = getattr(vae_model, name, None)
            if sub is None:
                continue
            try:
                for p in sub.parameters():
                    p.requires_grad_(bool(trainable))
            except Exception:
                # scale_learnable_parameters might not be a module
                pass

    if is_dist() and will_train_vae:
        # DDP requires at least one trainable parameter at construction.
        _vae_set_trainable(True)
        vae_decode = DDP(vae_decode, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)
        _vae_set_trainable(False)

    # optimizer groups
    backbone_params = []
    head_params = []
    for n, p in tsformer.named_parameters():
        if n.startswith("head."):
            head_params.append(p)
        else:
            backbone_params.append(p)

    params = [{"params": backbone_params, "lr": float(args.lr)}, {"params": head_params, "lr": float(args.lr) * float(args.head_lr_mult)}]
    if bool(args.train_adapter):
        params.append(
            {"params": [p for p in adapter.parameters() if p.requires_grad], "lr": float(args.lr) * float(args.adapter_lr_mult)}
        )

    m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
    vae_params = [p for p in m_vae.vae.parameters()]
    vae_pg_idx = len(params)
    params.append({"params": vae_params, "lr": 0.0})

    optimizer = torch.optim.AdamW(params, weight_decay=float(args.weight_decay), betas=(0.9, 0.95))
    scaler = GradScaler(enabled=bool(args.amp) and torch.cuda.is_available())

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    resume_path = str(args.resume).strip()
    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(resume_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            tsformer.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "adapter_state_dict" in ckpt:
                adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
            if "vae_state_dict" in ckpt:
                try:
                    m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                    missing, unexpected = m_vae.vae.load_state_dict(ckpt["vae_state_dict"], strict=False)
                    if len(missing) or len(unexpected):
                        rank0_print(
                            f"[warn] resume vae strict=False missing={missing[:10]} unexpected={unexpected[:10]}"
                        )
                except Exception as e:
                    rank0_print(f"[warn] resume vae failed: {e}")
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                _optimizer_state_to_device(optimizer, device)
            if "scaler_state_dict" in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            best_val = float(ckpt.get("best_val", best_val))
            rank0_print(f"[resume] epoch={start_epoch} step={global_step} from {resume_path}")

    # NOTE: We intentionally do NOT wrap tsformer or adapter in DDP.
    # TSformer's forward uses .forward_features_from_patch_tokens() which would
    # bypass DDP's forward hooks, causing gradient sync to silently fail.
    # Adapter backward fires multiple times per step (once per sample in per_sample
    # mode), which would cause incorrect repeated all-reduce under DDP.
    # Instead, we manually all-reduce all trainable gradients before optimizer.step()
    # via _allreduce_grads(). This guarantees correct gradient averaging regardless
    # of how many backward() calls happen within a single optimizer step.

    # Repository root (the folder containing Worldmodel/, infer/, train/, etc.)
    vln_uav_root = os.path.abspath(os.path.join(_PROJ_ROOT, "..", "..", ".."))
    max_items = int(args.max_items) if int(args.max_items) > 0 else None
    ds = LatentTrajManifestDataset(
        manifest_json=str(args.manifest_json),
        items_key=str(args.items_key),
        workspace_root=vln_uav_root,
        transform=None,
        load_frames=False,
        max_items=max_items,
        require_T=int(args.require_T) if int(args.require_T) > 0 else None,
    )
    sampler = DistributedSampler(ds, shuffle=True, drop_last=False) if is_dist() else None
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=partial(collate_fn, mode=str(args.collate_mode)),
        persistent_workers=bool(args.persistent_workers) if int(args.num_workers) > 0 else False,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
    )
    if get_rank() == 0:
        rank0_print(
            "[config]"
            f" dataset_len={len(ds)}"
            f" batches_per_epoch={len(dl)}"
            f" batch_size_per_rank={int(args.batch_size)}"
            f" global_batch_size={int(args.batch_size) * get_world_size()}"
            f" num_workers={int(args.num_workers)}"
        )

    window_size = 4
    W_grid = 40

    for epoch in range(start_epoch, int(args.epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)

        rot_w = _linear_schedule(
            epoch=epoch + 1,
            warmup_or_decay_epochs=int(args.rot_warmup_epochs),
            start=float(args.rot_loss_weight_start),
            end=float(args.rot_loss_weight_max),
        )
        if args.trans_z_weight_start is not None and args.trans_z_weight_end is not None:
            trans_z_w = _linear_schedule(
                epoch=epoch + 1,
                warmup_or_decay_epochs=int(args.trans_z_decay_epochs),
                start=float(args.trans_z_weight_start),
                end=float(args.trans_z_weight_end),
            )
        else:
            trans_z_w = float(args.trans_z_weight)

        train_vae_now = (epoch + 1) > int(args.train_vae_after_epoch)
        _vae_set_trainable(train_vae_now)
        optimizer.param_groups[vae_pg_idx]["lr"] = float(args.lr) * float(args.vae_lr_mult) if train_vae_now else 0.0

        freeze_backbone = int(args.freeze_backbone_epochs) > 0 and (epoch < int(args.freeze_backbone_epochs))
        if freeze_backbone:
            for n, p in tsformer.named_parameters():
                p.requires_grad = n.startswith("head.")
        else:
            for p in tsformer.parameters():
                p.requires_grad = True

        freeze_adapter_now = int(args.freeze_adapter_epochs) > 0 and (epoch < int(args.freeze_adapter_epochs))
        adapter_trainable_now = bool(args.train_adapter) and (not freeze_adapter_now)
        adapter.requires_grad_(adapter_trainable_now)

        tsformer.train()
        if adapter_trainable_now:
            adapter.train()
        else:
            adapter.eval()

        if get_rank() == 0 and epoch == 0:
            rank0_print(f"[phase] freeze_adapter={freeze_adapter_now} adapter_trainable={adapter_trainable_now} freeze_backbone={freeze_backbone} train_vae={train_vae_now}")
        if get_rank() == 0 and (freeze_adapter_now != (int(args.freeze_adapter_epochs) > 0 and (epoch - 1 < int(args.freeze_adapter_epochs)))):
            rank0_print(f"[phase] epoch={epoch+1} adapter unfreezing: freeze_adapter={freeze_adapter_now} adapter_trainable={adapter_trainable_now}")

        running = torch.zeros((), device=device)
        nb = 0
        t0 = time.time()
        use_tqdm = bool(args.tqdm) and (get_rank() == 0) and (tqdm is not None)
        pbar = tqdm(total=len(dl), desc=f"stage2 epoch {epoch+1}/{int(args.epochs)}", dynamic_ncols=True, leave=True) if use_tqdm else None

        t_after_step = time.time()
        grad_accum_steps = max(1, int(args.grad_accum_steps))
        accum_i = 0
        for batch in dl:
            nb += 1
            global_step += 1
            data_s = time.time() - t_after_step
            t_step0 = time.time()
            if accum_i == 0:
                optimizer.zero_grad(set_to_none=True)
            m = tsformer
            accum_scale = 1.0 / float(grad_accum_steps)

            if isinstance(batch, dict) and "samples" in batch:
                samples = batch["samples"]
                denom = float(max(1, len(samples)))
                used = 0
                loss_sum = torch.zeros((), device=device, dtype=torch.float32)

                for s in samples:
                    z_ext = s["z_ext"].to(device, non_blocking=True)  # (1,64,T_lat,16,16)
                    traj_t = s["traj"]
                    traj_np = traj_t.detach().cpu().numpy() if isinstance(traj_t, torch.Tensor) else np.asarray(traj_t, dtype=np.float32)
                    if int(traj_np.shape[0]) < window_size:
                        continue

                    d = traj_to_delta(
                        traj_np,
                        traj_mode=str(args.traj_mode),
                        angles_in_degrees=bool(args.angles_in_degrees),
                        translation_divisor=float(args.translation_divisor),
                        delta_dyaw_unit=str(args.delta_dyaw_unit),
                    )
                    delta_bt6 = torch.from_numpy(d).unsqueeze(0).to(device, non_blocking=True)  # (1,T,6) rad+m
                    if label_stats_t is not None:
                        delta_bt6 = _normalize_delta_bt6(delta_bt6, label_stats_t)
                    T_traj = int(delta_bt6.shape[1])

                    allow_adapter_grad = adapter_trainable_now
                    tokens_btnd, T_dec = _decode_tokens_full_T(
                        vae_decode_module=vae_decode,
                        adapter=adapter,
                        z_ext=z_ext,
                        allow_adapter_grad=allow_adapter_grad,
                        allow_vae_grad=bool(train_vae_now),
                        amp_enabled=bool(args.amp),
                        expected_T=T_traj,
                    )
                    T_use = min(int(T_dec), int(T_traj))
                    if T_use < window_size:
                        continue
                    tokens_btnd = tokens_btnd[:, :T_use]
                    delta_bt6 = delta_bt6[:, :T_use]

                    starts_bk_all = _sample_all_window_starts(
                        B=1, T=T_use, window_size=window_size, stride=int(args.window_stride), device=device
                    )
                    K = int(starts_bk_all.shape[1])
                    if K <= 0:
                        continue

                    chunk = int(args.windows_chunk)
                    if chunk <= 0 or chunk >= K:
                        chunk = K

                    total_loss = torch.zeros((), device=device, dtype=torch.float32)
                    tokens_det = tokens_btnd.detach()
                    grad_tokens = torch.zeros_like(tokens_det, dtype=torch.float32)
                    done = 0
                    for s0 in range(0, K, chunk):
                        s1 = min(K, s0 + chunk)
                        starts_bk = starts_bk_all[:, s0:s1]
                        kc = int(starts_bk.shape[1])
                        Bwin = int(kc)

                        patch_tokens_val = _gather_window_tokens(tokens_det, starts_bk, window_size=window_size)
                        patch_tokens = patch_tokens_val.detach().requires_grad_(True)
                        targets = _gather_window_targets(delta_bt6, starts_bk, window_size=window_size)

                        with autocast(enabled=bool(args.amp) and torch.cuda.is_available()):
                            feat = m.forward_features_from_patch_tokens(patch_tokens, B=Bwin, T=window_size, W=W_grid)
                            pred = m.head(feat)
                            loss_chunk = compute_loss_mse(
                                pred=pred,
                                target=targets,
                                window_size=window_size,
                                rot_weight=float(rot_w),
                                trans_xy_weight=float(args.trans_xy_weight),
                                trans_z_weight=float(trans_z_w),
                                trans_vertical_index=int(args.trans_vertical_index),
                            )
                        w = float(kc) / float(K)
                        scaler.scale(((loss_chunk * w) / denom) * accum_scale).backward()
                        g_patch = patch_tokens.grad.detach()
                        g_patch = g_patch.view(1, kc, window_size, g_patch.shape[1], g_patch.shape[2])
                        t_idx = torch.arange(window_size, device=device, dtype=torch.long).view(1, 1, window_size)
                        idx = starts_bk.unsqueeze(-1) + t_idx
                        for dt in range(window_size):
                            t = idx[:, :, dt]
                            gi = g_patch[:, :, dt]
                            grad_tokens.scatter_add_(1, t[:, :, None, None].expand(1, kc, gi.shape[2], gi.shape[3]), gi.float())
                        total_loss = total_loss + loss_chunk.detach().float() * w
                        done += kc

                    if done != K:
                        raise RuntimeError(f"window coverage mismatch: done={done} K={K}")

                    if adapter_trainable_now or bool(train_vae_now):
                        tokens_btnd.backward(grad_tokens.to(dtype=tokens_btnd.dtype))
                    loss_sum = loss_sum + total_loss.detach()
                    used += 1

                if used <= 0:
                    t_after_step = time.time()
                    continue
                loss = loss_sum / float(used)
            else:
                z_ext = batch["z_ext"].to(device, non_blocking=True)  # (B,64,T_lat,16,16)
                traj_b = batch["traj"]
                traj_np = traj_b.detach().cpu().numpy() if isinstance(traj_b, torch.Tensor) else np.asarray(traj_b, dtype=np.float32)

                deltas = []
                for b in range(traj_np.shape[0]):
                    d = traj_to_delta(
                        traj_np[b],
                        traj_mode=str(args.traj_mode),
                        angles_in_degrees=bool(args.angles_in_degrees),
                        translation_divisor=float(args.translation_divisor),
                        delta_dyaw_unit=str(args.delta_dyaw_unit),
                    )
                    deltas.append(torch.from_numpy(d).unsqueeze(0))
                delta_bt6 = torch.cat(deltas, dim=0).to(device, non_blocking=True)  # (B,T,6) rad+m
                if label_stats_t is not None:
                    delta_bt6 = _normalize_delta_bt6(delta_bt6, label_stats_t)

                B = int(z_ext.shape[0])
                T_traj = int(delta_bt6.shape[1])

                allow_adapter_grad = adapter_trainable_now
                tokens_btnd, T_dec = _decode_tokens_full_T(
                    vae_decode_module=vae_decode,
                    adapter=adapter,
                    z_ext=z_ext,
                    allow_adapter_grad=allow_adapter_grad,
                    allow_vae_grad=bool(train_vae_now),
                    amp_enabled=bool(args.amp),
                    expected_T=T_traj,
                )
                T_use = min(int(T_dec), int(T_traj))
                if T_use < window_size:
                    t_after_step = time.time()
                    continue
                tokens_btnd = tokens_btnd[:, :T_use]
                delta_bt6 = delta_bt6[:, :T_use]

                starts_bk_all = _sample_all_window_starts(
                    B=B, T=T_use, window_size=window_size, stride=int(args.window_stride), device=device
                )
                K = int(starts_bk_all.shape[1])
                if K <= 0:
                    t_after_step = time.time()
                    continue

                chunk = int(args.windows_chunk)
                if chunk <= 0 or chunk >= K:
                    chunk = K

                total_loss = torch.zeros((), device=device, dtype=torch.float32)
                tokens_det = tokens_btnd.detach()
                grad_tokens = torch.zeros_like(tokens_det, dtype=torch.float32)
                done = 0
                for s0 in range(0, K, chunk):
                    s1 = min(K, s0 + chunk)
                    starts_bk = starts_bk_all[:, s0:s1]
                    kc = int(starts_bk.shape[1])
                    Bwin = int(B * kc)

                    patch_tokens_val = _gather_window_tokens(tokens_det, starts_bk, window_size=window_size)
                    patch_tokens = patch_tokens_val.detach().requires_grad_(True)
                    targets = _gather_window_targets(delta_bt6, starts_bk, window_size=window_size)

                    with autocast(enabled=bool(args.amp) and torch.cuda.is_available()):
                        feat = m.forward_features_from_patch_tokens(patch_tokens, B=Bwin, T=window_size, W=W_grid)
                        pred = m.head(feat)
                        loss_chunk = compute_loss_mse(
                            pred=pred,
                            target=targets,
                            window_size=window_size,
                            rot_weight=float(rot_w),
                            trans_xy_weight=float(args.trans_xy_weight),
                            trans_z_weight=float(trans_z_w),
                            trans_vertical_index=int(args.trans_vertical_index),
                        )
                    w = float(kc) / float(K)
                    scaler.scale((loss_chunk * w) * accum_scale).backward()
                    g_patch = patch_tokens.grad.detach()  # (Bwin*window,N,D)
                    g_patch = g_patch.view(B, kc, window_size, g_patch.shape[1], g_patch.shape[2])
                    t_idx = torch.arange(window_size, device=device, dtype=torch.long).view(1, 1, window_size)
                    idx = starts_bk.unsqueeze(-1) + t_idx  # (B,kc,window)
                    for dt in range(window_size):
                        t = idx[:, :, dt]  # (B,kc)
                        gi = g_patch[:, :, dt]  # (B,kc,N,D)
                        grad_tokens.scatter_add_(1, t[:, :, None, None].expand(B, kc, gi.shape[2], gi.shape[3]), gi.float())
                    total_loss = total_loss + loss_chunk.detach().float() * w
                    done += kc

                if done != K:
                    raise RuntimeError(f"window coverage mismatch: done={done} K={K}")

                if adapter_trainable_now or bool(train_vae_now):
                    tokens_btnd.backward(grad_tokens.to(dtype=tokens_btnd.dtype))
                loss = total_loss

            running += loss.detach()
            step_s = time.time() - t_step0
            t_after_step = time.time()

            accum_i += 1
            do_step = (accum_i >= grad_accum_steps)
            if do_step:
                _allreduce_grads(optimizer.param_groups)
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    all_params = [p for pg in optimizer.param_groups for p in pg["params"] if p.grad is not None]
                    if all_params:
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                accum_i = 0

            if pbar is not None:
                avg = (running / max(1, nb)).detach().item()
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.detach().item():.6f}",
                    avg=f"{avg:.6f}",
                    rot_w=f"{rot_w:.4f}",
                    trans_z_w=f"{trans_z_w:.4f}",
                    data_s=f"{data_s:.2f}",
                    step_s=f"{step_s:.2f}",
                    step=global_step,
                )

            if get_rank() == 0 and (global_step % int(args.log_every) == 0):
                avg = (running / max(1, nb)).detach().item()
                dt = time.time() - t0
                rank0_print(
                    f"[stage2] epoch={epoch+1}/{args.epochs} step={global_step} "
                    f"loss={loss.detach().item():.6f} avg={avg:.6f} rot_w={rot_w:.4f} trans_z_w={trans_z_w:.4f} dt={dt:.1f}s"
                )
                t0 = time.time()

        if pbar is not None:
            pbar.close()

        if accum_i > 0:
            _allreduce_grads(optimizer.param_groups)
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                all_params = [p for pg in optimizer.param_groups for p in pg["params"] if p.grad is not None]
                if all_params:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            accum_i = 0

        train_loss = (running / max(1, nb)).detach()
        train_loss = reduce_mean(train_loss).item()
        if get_rank() == 0:
            rank0_print(f"[epoch] {epoch+1} train_loss={train_loss:.6f}")
            if adapter_trainable_now and (epoch + 1) <= int(args.freeze_adapter_epochs) + 3:
                adapter_grad_norm = 0.0
                adapter_grad_count = 0
                for p in adapter.parameters():
                    if p.grad is not None:
                        adapter_grad_norm += p.grad.detach().float().norm().item() ** 2
                        adapter_grad_count += 1
                adapter_grad_norm = adapter_grad_norm ** 0.5
                rank0_print(
                    f"[grad-diag] adapter: {adapter_grad_count}/{sum(1 for _ in adapter.parameters())} params have grad, "
                    f"grad_norm={adapter_grad_norm:.6f}"
                )

            # Save the last checkpoint BEFORE entering VAE training stage.
            # train_vae_now becomes True when (epoch+1) > train_vae_after_epoch,
            # so the final "pre-full-train" epoch is exactly train_vae_after_epoch.
            if int(args.train_vae_after_epoch) < 10**9 and (epoch + 1) == int(args.train_vae_after_epoch):
                to_save = tsformer
                adp_save = adapter
                m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                vae_sd_full = m_vae.vae.state_dict()
                vae_sd = {
                    k: v.detach().cpu()
                    for k, v in vae_sd_full.items()
                    if k.startswith(("proj_up.", "post_quant_conv.", "decoder.", "scale_learnable_parameters"))
                }
                state_pre = {
                    "format": "stage2_latent2action_checkpoint_pre_fulltrain_v1",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    "adapter_state_dict": {k: v.detach().cpu() for k, v in adp_save.state_dict().items()},
                    "vae_state_dict": vae_sd,
                    "label_stats": label_stats_np,
                    "label_stats_source": label_stats_src,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                    "time": datetime.now().isoformat(),
                }
                path_pre = os.path.join(str(args.out_dir), f"checkpoint_pre_fulltrain_e{epoch+1}.pth")
                torch.save(state_pre, path_pre)
                rank0_print(f"[save] pre-fulltrain checkpoint: {path_pre}")

            if (epoch + 1) % int(args.save_every) == 0:
                to_save = tsformer
                adp_save = adapter
                m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                vae_sd_full = m_vae.vae.state_dict()
                vae_sd = {
                    k: v.detach().cpu()
                    for k, v in vae_sd_full.items()
                    if k.startswith(("proj_up.", "post_quant_conv.", "decoder.", "scale_learnable_parameters"))
                }
                state = {
                    "format": "stage2_latent2action_checkpoint_combined_v1",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    "adapter_state_dict": {k: v.detach().cpu() for k, v in adp_save.state_dict().items()},
                    "vae_state_dict": vae_sd,
                    "label_stats": label_stats_np,
                    "label_stats_source": label_stats_src,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_val": best_val,
                    "args": vars(args),
                    "time": datetime.now().isoformat(),
                }
                torch.save(state, os.path.join(str(args.out_dir), "checkpoint_last.pth"))
                torch.save(state, os.path.join(str(args.out_dir), f"checkpoint_e{epoch+1}.pth"))

            if bool(args.export_combined) and ((epoch + 1) % int(args.save_every) == 0):
                to_save = tsformer
                adp_save = adapter
                m_vae = vae_decode.module if isinstance(vae_decode, DDP) else vae_decode
                vae_sd_full = m_vae.vae.state_dict()
                vae_sd = {
                    k: v.detach().cpu()
                    for k, v in vae_sd_full.items()
                    if k.startswith(("proj_up.", "post_quant_conv.", "decoder.", "scale_learnable_parameters"))
                }
                export = {
                    "format": "stage2_latent2action_combined_v2_resumable",
                    "created_at": datetime.now().isoformat(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "infinitystar_vae_path": str(args.infinitystar_vae_path),
                    "infinitystar_vae_type": int(args.infinitystar_vae_type),
                    "tsformer_pretrained": str(args.tsformer_pretrained),
                    "adapter_ckpt": str(args.adapter_ckpt),
                    "model_config": {"num_frames": 4, "embed_dim": 384, "patch_size": 16, "W_grid": W_grid},
                    "vae_state_dict": vae_sd,
                    "label_stats": label_stats_np,
                    "label_stats_source": label_stats_src,
                    "adapter_state_dict": {k: v.detach().cpu() for k, v in adp_save.state_dict().items()},
                    "tsformer_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    # Alias for --resume compatibility (expects model_state_dict)
                    "model_state_dict": {k: v.detach().cpu() for k, v in to_save.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(args),
                }
                torch.save(export, os.path.join(str(args.out_dir), str(args.export_name)))

    if _RANK0_LOG_FH is not None and get_rank() == 0:
        _RANK0_LOG_FH.close()


if __name__ == "__main__":
    main()

