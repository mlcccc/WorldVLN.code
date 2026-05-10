"""
Stage-1: Distill an Adapter that maps InfinityStar VAE decoder `up_block_3` features to TSformer PatchEmbed tokens.

Teacher source (this version):
- Teacher tokens come from ORIGINAL PNG frames (frames_rgb from manifest images_dir),
  passed through TSformer `patch_embed`.

Student source:
- Student tokens come from InfinityStar VAE decoder `up_block_3` feature (hook) passed through Adapter.

Loss:
- Composite distillation loss (cosine + optional distribution stats + optional MSE).

DDP notes:
- InfinityStar VAE may enable batch slicing (use_slicing=True), causing the hook batch to be 1 even when B>1.
  We align by decoding per-sample when slicing is active.
- tqdm + train.log are rank0-only to avoid multi-rank spam.
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
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure local repo root is importable even when running via absolute path.
_TSFORMER_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _TSFORMER_ROOT not in sys.path:
    sys.path.insert(0, _TSFORMER_ROOT)

from datasets.latent_traj_manifest import LatentTrajManifestDataset  # noqa: E402
from models.vae96_to_tsformer_adapter import Vae96ToTSformerEmbedAdapter  # noqa: E402
from timesformer.models.vit import VisionTransformer  # noqa: E402

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


KITTI_MEAN = torch.tensor([0.34721234, 0.36705238, 0.36066107], dtype=torch.float32).view(1, 3, 1, 1, 1)
KITTI_STD = torch.tensor([0.30737526, 0.31515116, 0.32020183], dtype=torch.float32).view(1, 3, 1, 1, 1)
_KITTI_MEAN_3 = (0.34721234, 0.36705238, 0.36066107)
_KITTI_STD_3 = (0.30737526, 0.31515116, 0.32020183)


def compute_distill_loss(
    tok_s: torch.Tensor,
    tok_t: torch.Tensor,
    w_cos: float = 1.0,
    w_mean: float = 0.1,
    w_std: float = 0.1,
    w_mse: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Composite distillation loss: cosine direction + distribution statistics + optional MSE.

    Args:
        tok_s, tok_t: student/teacher tokens with identical shapes, last dim = D.
        w_cos:  weight for (1 - cosine_similarity) — aligns vector directions.
        w_mean: weight for MSE of per-dimension means — aligns distribution centres.
        w_std:  weight for MSE of per-dimension stds  — aligns distribution spreads.
        w_mse:  weight for raw MSE (magnitude-sensitive, optional).
    Returns:
        (total_loss_with_grad, {component_name: float_value})
    """
    D = tok_s.shape[-1]
    s_flat = tok_s.reshape(-1, D)
    t_flat = tok_t.reshape(-1, D)

    parts: dict = {}
    total = s_flat.new_zeros(())

    if w_cos > 0:
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
        l_cos = (1.0 - cos_sim).mean()
        parts["cos"] = float(l_cos.detach())
        total = total + w_cos * l_cos

    if w_mean > 0:
        l_mean = F.mse_loss(s_flat.mean(dim=0), t_flat.mean(dim=0))
        parts["mean"] = float(l_mean.detach())
        total = total + w_mean * l_mean

    if w_std > 0 and s_flat.shape[0] > 1:
        l_std = F.mse_loss(s_flat.std(dim=0), t_flat.std(dim=0))
        parts["std"] = float(l_std.detach())
        total = total + w_std * l_std

    if w_mse > 0:
        l_mse = F.mse_loss(s_flat, t_flat)
        parts["mse"] = float(l_mse.detach())
        total = total + w_mse * l_mse

    parts["total"] = float(total.detach())
    return total, parts


def _linear(a: float, b: float, t01: float) -> float:
    t01 = float(max(0.0, min(1.0, t01)))
    return float(a + (b - a) * t01)


def _loss_weights_for_epoch(epoch_num_1idx: int, args) -> Tuple[float, float]:
    """
    Returns (w_cos, w_mse) for the given 1-indexed epoch number.

    Schedules are defined in 1-indexed epoch numbers to match typical user expectations.
    """
    if str(getattr(args, "loss_schedule", "none")).strip().lower() in ("", "none"):
        return float(args.loss_cosine_w), float(args.loss_mse_w)

    # piecewise linear schedule (legacy experiments)
    hold = int(getattr(args, "loss_hold_epochs", 40))
    ramp_s = int(getattr(args, "loss_ramp_start_epoch", 45))
    ramp_e = int(getattr(args, "loss_ramp_end_epoch", 55))
    cos_a = float(getattr(args, "loss_cosine_w_start", 1.0))
    cos_b = float(getattr(args, "loss_cosine_w_end", 0.1))
    mse_a = float(getattr(args, "loss_mse_w_start", 0.1))
    mse_b = float(getattr(args, "loss_mse_w_end", 1.0))

    if epoch_num_1idx <= hold:
        return cos_a, mse_a
    if ramp_s <= epoch_num_1idx <= ramp_e:
        denom = max(1, (ramp_e - ramp_s))
        t01 = float(epoch_num_1idx - ramp_s) / float(denom)
        return _linear(cos_a, cos_b, t01), _linear(mse_a, mse_b, t01)
    if epoch_num_1idx > ramp_e:
        return cos_b, mse_b

    # gap between hold and ramp start (e.g., epoch 41-44): keep the start weights
    return cos_a, mse_a


_RANK0_LOG_FH = None
_RANK0_LOG_PATH = None
_RANK0_LOG_BROKEN = False


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
        global _RANK0_LOG_BROKEN
        print(*args, **kwargs, flush=True)
        if _RANK0_LOG_FH is not None:
            try:
                msg = " ".join(str(a) for a in args)
                _RANK0_LOG_FH.write((msg + "\n").encode("utf-8", errors="replace"))
            except OSError as e:
                _RANK0_LOG_BROKEN = True
                try:
                    _RANK0_LOG_FH.close()
                except Exception:
                    pass
                _RANK0_LOG_FH = None
                print(f"[warn] train.log write failed; disable file logging. err={e}", flush=True)


def rank0_log(msg: str):
    """
    Write a single line into train.log (rank0-only) without spamming stdout.
    """
    if get_rank() == 0:
        global _RANK0_LOG_FH, _RANK0_LOG_PATH, _RANK0_LOG_BROKEN
        if _RANK0_LOG_BROKEN:
            return
        if _RANK0_LOG_PATH is None:
            return

        # Lazy-open/reopen to survive flaky FS implementations.
        if _RANK0_LOG_FH is None:
            try:
                _RANK0_LOG_FH = open(str(_RANK0_LOG_PATH), "ab", buffering=0)
            except OSError as e:
                _RANK0_LOG_BROKEN = True
                print(f"[warn] train.log open failed; disable file logging. err={e}", flush=True)
                return
        try:
            _RANK0_LOG_FH.write((str(msg) + "\n").encode("utf-8", errors="replace"))
        except OSError:
            try:
                _RANK0_LOG_FH.close()
            except Exception:
                pass
            _RANK0_LOG_FH = None
            try:
                _RANK0_LOG_FH = open(str(_RANK0_LOG_PATH), "ab", buffering=0)
                _RANK0_LOG_FH.write((str(msg) + "\n").encode("utf-8", errors="replace"))
            except OSError as e:
                _RANK0_LOG_BROKEN = True
                try:
                    if _RANK0_LOG_FH is not None:
                        _RANK0_LOG_FH.close()
                except Exception:
                    pass
                _RANK0_LOG_FH = None
                print(f"[warn] train.log write failed; disable file logging. err={e}", flush=True)


def _atomic_torch_save(obj, path: str) -> None:
    """
    Write a torch checkpoint atomically to avoid leaving corrupted .pt files
    when jobs are preempted or crash during save.
    """
    path = str(path)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


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


def _preprocess_decoded_for_tsformer(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    x: (B,3,T,H,W) float. Range can be [-1,1] or [0,1]. Returns normalized (B,3,T,192,640).
    """
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError(f"decoded frames must be (B,3,T,H,W), got {tuple(x.shape)}")
    x = x.to(device=device, dtype=torch.float32)
    if float(x.min()) < -1e-3:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    B, C, T, H, W = x.shape
    xt = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (BT,3,H,W)
    xt = F.interpolate(xt, size=(192, 640), mode="bilinear", align_corners=False)
    xt = xt.view(B, T, C, 192, 640).permute(0, 2, 1, 3, 4).contiguous()

    mean = KITTI_MEAN.to(device=device)
    std = KITTI_STD.to(device=device)
    xt = (xt - mean) / std
    return xt


def _build_png_transform():
    """
    Build a per-frame transform that matches TSformer training preprocessing:
    Resize to (192,640) then ToTensor+Normalize(KITTI mean/std).
    """
    try:
        from torchvision import transforms  # type: ignore

        return transforms.Compose(
            [
                transforms.Resize((192, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_KITTI_MEAN_3, std=_KITTI_STD_3),
            ]
        )
    except Exception as e:
        raise RuntimeError(f"torchvision is required for PNG teacher transform but is not available: {e}")


def _expected_frames_from_latent_chunk(latent_chunk_len: int) -> int:
    # Wan-like temporal upsample: T_frames = 4*(T_lat-1)+1
    L = int(latent_chunk_len)
    if L < 2:
        return 1
    return 4 * (L - 1) + 1


def _decode_teacher_student_tokens(
    vae_decode_module: nn.Module,
    adapter: nn.Module,
    tsformer: VisionTransformer,
    z_sub: torch.Tensor,
    frames_teacher: torch.Tensor,
    amp_enabled: bool,
    adapter_frames_chunk: int = 0,
    adapter_use_checkpoint: bool = False,
    expected_T: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns: (tokens_student, tokens_teacher, T_frames)
      - tokens_*: (B,T,N,D)
    """
    m = vae_decode_module.module if isinstance(vae_decode_module, DDP) else vae_decode_module
    vae = m.vae
    B = int(z_sub.shape[0])

    # IMPORTANT:
    # We decode VAE under no_grad to save memory/avoid building VAE graphs.
    # Do NOT run `adapter()` inside the VAE hook under no_grad; some environments end up with
    # adapter grads not being populated, which makes AdamW optimizer state stay empty.
    # Instead, capture the feature slices and run `adapter()` afterwards with grad enabled.
    feat_slices = []
    start = 0

    def hook(_module, _inp, out):
        nonlocal start, feat_slices
        hs = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(hs, torch.Tensor) or hs.ndim != 5:
            raise RuntimeError("up_block_3 hook output is not a 5D Tensor")
        t_slice = int(hs.shape[2])
        # Save detached feature slice to be consumed later by adapter().
        feat_slices.append(hs.detach())
        start += t_slice

    handle = vae.decoder.up_blocks[-1].register_forward_hook(hook)
    try:
        grad_ctx = torch.no_grad()
        amp_ctx = autocast(enabled=bool(amp_enabled) and torch.cuda.is_available())
        try:
            with grad_ctx, amp_ctx:
                _ = vae_decode_module(z_sub)  # decode to trigger hook; output ignored (teacher comes from PNGs)
        except RuntimeError as e:
            msg = str(e)
            if "torch.cat(): expected a non-empty list of Tensors" in msg:
                rank0_print(f"[warn] VAE decode empty dec list; skip. z_sub={tuple(z_sub.shape)} err={msg}")
                empty = torch.empty((B, 0, 480, 384), device=z_sub.device, dtype=torch.float32)
                return empty, empty, 0
            raise
    finally:
        handle.remove()

    if len(feat_slices) == 0 or int(start) <= 0:
        empty = torch.empty((B, 0, 480, 384), device=z_sub.device, dtype=torch.float32)
        return empty, empty, 0

    # Run adapter with grad enabled (student tokens).
    # IMPORTANT (memory):
    # Adapter operates per-frame after reshape to (B*T,C,H,W). Feeding a full 49-frame sequence
    # at once can OOM due to large (B*T) activation tensors kept for backprop. We therefore
    # run adapter in temporal chunks; this is mathematically equivalent because there is no
    # temporal mixing inside the adapter.
    chunk = int(adapter_frames_chunk)
    if chunk <= 0:
        chunk = 8  # safe default for full decode
    tok_slices = []
    amp_ctx_s = autocast(enabled=bool(amp_enabled) and torch.cuda.is_available())
    with amp_ctx_s:
        for hs in feat_slices:
            Bh = int(hs.shape[0])
            t_slice = int(hs.shape[2])
            for t0 in range(0, t_slice, chunk):
                t1 = min(t0 + chunk, t_slice)
                hs_sub = hs[:, :, t0:t1].contiguous()
                if bool(adapter_use_checkpoint):
                    def _adp(x: torch.Tensor) -> torch.Tensor:
                        y, _t2, _w2 = adapter(x)
                        return y
                    # use_reentrant=False is required here because hs_sub is detached (requires_grad=False).
                    # With the default reentrant checkpoint, no input requires grad => no graph => loss has no grad_fn.
                    tok = checkpoint(_adp, hs_sub, use_reentrant=False)  # (Bh*(t1-t0),N,D)
                else:
                    tok, _t2, _w2 = adapter(hs_sub)  # (Bh*(t1-t0),N,D)
                tok = tok.view(Bh, (t1 - t0), tok.shape[1], tok.shape[2]).contiguous()
                tok_slices.append(tok)
    tok_s = torch.cat(tok_slices, dim=1)  # (B,T,N,D)
    T_s = int(tok_s.shape[1])
    if expected_T is not None and T_s != int(expected_T):
        rank0_print(f"[warn] decoded student T={T_s} but expected_T={expected_T}")

    # teacher from original PNG frames
    x_in = frames_teacher.to(device=z_sub.device, dtype=torch.float32, non_blocking=True)
    if x_in.ndim != 5 or int(x_in.shape[1]) != 3:
        raise ValueError(f"frames_teacher must be (B,3,T,H,W), got {tuple(x_in.shape)}")
    with torch.no_grad():
        tok_t, T_t, _W = tsformer.patch_embed(x_in)  # (B*T,N,D)
    tok_t = tok_t.view(B, int(T_t), tok_t.shape[1], tok_t.shape[2]).contiguous()
    if expected_T is not None and int(T_t) != int(expected_T):
        rank0_print(f"[warn] decoded teacher T={int(T_t)} but expected_T={expected_T}")

    # align lengths if mismatch
    T = min(int(tok_s.shape[1]), int(tok_t.shape[1]))
    if T <= 0:
        empty = torch.empty((B, 0, 480, 384), device=z_sub.device, dtype=torch.float32)
        return empty, empty, 0
    if int(tok_s.shape[1]) != T:
        tok_s = tok_s[:, :T].contiguous()
    if int(tok_t.shape[1]) != T:
        tok_t = tok_t[:, :T].contiguous()
    return tok_s, tok_t, int(T)


def collate_fn(samples, mode: str = "crop", latent_chunk_len: int = 3):
    mode = str(mode).strip().lower()
    L = int(latent_chunk_len)
    if mode == "per_sample":
        out_samples = []
        skipped_no_frames = 0
        skipped_no_latent = 0
        for s in samples:
            z = s.get("z_ext", None)
            if z is None:
                skipped_no_latent += 1
                continue
            if not isinstance(z, torch.Tensor):
                z = torch.as_tensor(z)
            fr = s.get("frames_rgb", None)
            if fr is None:
                # When LatentTrajManifestDataset is created with on_error="empty",
                # it may intentionally return partial samples without frames_rgb.
                # We skip such samples to keep training running.
                skipped_no_frames += 1
                continue
            if not isinstance(fr, torch.Tensor):
                fr = torch.as_tensor(fr)
            out_samples.append({"z_ext": z.float().contiguous(), "frames_rgb": fr.float().contiguous(), "meta": s.get("meta", {})})
        return {
            "samples": out_samples,
            "meta": [s.get("meta", {}) for s in samples],
            "skipped_no_frames": int(skipped_no_frames),
            "skipped_no_latent": int(skipped_no_latent),
            "total_in": int(len(samples)),
            "total_out": int(len(out_samples)),
        }

    z_list = []
    t_list = []
    for s in samples:
        z = s["z_ext"]
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        z = z.float().contiguous()
        z_list.append(z)
        t_list.append(int(z.shape[2]))

    t_min = min(t_list) if len(t_list) else 0
    # If any sample is too short, cropping to min would make the whole batch unusable.
    # Fall back to per-sample output so the train loop can skip only the short ones.
    if t_min < L:
        out_samples = [{"z_ext": z, "meta": s.get("meta", {})} for z, s in zip(z_list, samples)]
        return {"samples": out_samples, "meta": [s.get("meta", {}) for s in samples]}
    if t_min <= 0:
        raise ValueError(f"invalid latent lengths in batch: {t_list}")
    # Stack/crop frames to min T (requires all samples have frames_rgb).
    fr_list = []
    ft_list = []
    for s in samples:
        fr = s.get("frames_rgb", None)
        if fr is None:
            raise ValueError("crop collate requires frames_rgb (set load_frames=True in dataset)")
        if not isinstance(fr, torch.Tensor):
            fr = torch.as_tensor(fr)
        fr = fr.float().contiguous()
        fr_list.append(fr)
        ft_list.append(int(fr.shape[1]))
    t_frame_min = int(min(ft_list)) if ft_list else 0
    if t_frame_min <= 0:
        raise ValueError(f"invalid frame lengths in batch: {ft_list}")

    z_ext = torch.cat([z[:, :, :t_min].contiguous() for z in z_list], dim=0).contiguous()  # (B,64,t_min,16,16)
    frames_rgb = torch.stack([fr[:, :t_frame_min].contiguous() for fr in fr_list], dim=0).contiguous()  # (B,3,T,192,640)
    meta = [s.get("meta", {}) for s in samples]
    return {"z_ext": z_ext, "frames_rgb": frames_rgb, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_json", type=str, required=True)
    ap.add_argument("--items_key", type=str, default="ALL", help="Manifest key(s) to use; comma-separated, or ALL for every items_* list.")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--require_T", type=int, default=49)

    ap.add_argument("--tsformer_ckpt", type=str, required=True)
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
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--global_batch_size", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true", default=False)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--collate_mode", type=str, default="crop", choices=["crop", "per_sample"])

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--tqdm", action="store_true", default=False)
    ap.add_argument("--log_file", type=str, default="train.log")
    ap.add_argument("--log_dir", type=str, default="")
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm; 0 to disable")

    ap.add_argument("--loss_cosine_w", type=float, default=1.0)
    ap.add_argument("--loss_mean_w", type=float, default=0.1)
    ap.add_argument("--loss_std_w", type=float, default=0.1)
    ap.add_argument("--loss_mse_w", type=float, default=0.0)
    ap.add_argument(
        "--loss_schedule",
        type=str,
        default="none",
        choices=["none", "piecewise_linear"],
        help="Optional epoch-wise schedule for loss weights. 'piecewise_linear' uses 1-indexed epoch boundaries.",
    )
    ap.add_argument("--loss_hold_epochs", type=int, default=40, help="(1-indexed) hold start weights for first N epochs")
    ap.add_argument("--loss_ramp_start_epoch", type=int, default=45, help="(1-indexed) ramp start epoch (inclusive)")
    ap.add_argument("--loss_ramp_end_epoch", type=int, default=55, help="(1-indexed) ramp end epoch (inclusive)")
    ap.add_argument("--loss_cosine_w_start", type=float, default=1.0)
    ap.add_argument("--loss_cosine_w_end", type=float, default=0.1)
    ap.add_argument("--loss_mse_w_start", type=float, default=0.1)
    ap.add_argument("--loss_mse_w_end", type=float, default=1.0)
    ap.add_argument("--adapter_frames_chunk", type=int, default=8, help="adapter forward temporal chunk size; lower to save memory")
    ap.add_argument("--adapter_use_checkpoint", action="store_true", default=False, help="use torch.utils.checkpoint for adapter forward to save memory")

    ap.add_argument("--latent_chunk_len", type=int, default=3)
    # Compatibility alias (some launch wrappers inject this flag).
    ap.add_argument("--min_latent_t", type=int, default=None)
    # If set, choose a random window start for each sample (when not using full latent).
    ap.add_argument("--latent_chunk_random", action="store_true", default=False)
    # If set, use the full latent sequence (no windowing). This ensures all temporal parts participate in distillation.
    ap.add_argument("--latent_use_full", action="store_true", default=False)
    # If set, enumerate all windows for each latent (covers all parts without decoding the full sequence at once).
    ap.add_argument("--latent_cover_all", action="store_true", default=False)
    ap.add_argument("--latent_stride", type=int, default=1)
    ap.add_argument("--latent_max_windows", type=int, default=0, help="0 means all windows")

    ap.add_argument("--vae_disable_slicing", action="store_true", default=False)
    ap.add_argument("--vae_disable_tiling", action="store_true", default=False)
    ap.add_argument("--vae_num_sample_frames_batch_size", type=int, default=0)

    ap.add_argument("--resume", type=str, default="", help="Resume from a stage1 adapter checkpoint (stage1_adapter_last.pt)")

    ap.add_argument("--export_combined", action="store_true", default=True)
    ap.add_argument("--export_name", type=str, default="infinitystar_up3_plus_adapter_latent2tokens.pt")
    args = ap.parse_args()

    if args.min_latent_t is not None:
        args.latent_chunk_len = int(args.min_latent_t)

    if not str(args.out_dir).strip():
        raise ValueError("--out_dir is empty")
    os.makedirs(str(args.out_dir), exist_ok=True)

    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(int(args.seed) + int(get_rank()))

    # rank0 file logging
    global _RANK0_LOG_FH
    if get_rank() == 0:
        log_dir = str(args.log_dir).strip() or str(args.out_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(str(log_dir), str(args.log_file))
        global _RANK0_LOG_PATH, _RANK0_LOG_BROKEN
        _RANK0_LOG_PATH = log_path
        _RANK0_LOG_BROKEN = False
        _RANK0_LOG_FH = open(log_path, "ab", buffering=0)
        rank0_print(f"[log] writing rank0 logs to {log_path}")
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

    # teacher (frozen)
    tsformer = build_tsformer().to(device)
    load_tsformer(tsformer, str(args.tsformer_ckpt))
    tsformer.eval()
    for p in tsformer.parameters():
        p.requires_grad_(False)

    # student adapter (trainable)
    adapter = Vae96ToTSformerEmbedAdapter().to(device)
    adapter.train()
    if is_dist():
        adapter = DDP(adapter, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    # InfinityStar VAE (frozen)
    inf_root = str(args.infinitystar_root).strip() or None
    vae_model = load_infinitystar_vae(
        vae_path=str(args.infinitystar_vae_path),
        vae_type=int(args.infinitystar_vae_type),
        device=device,
        infinitystar_root=inf_root,
        proj_root=_TSFORMER_ROOT,
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

    class _VaeDecodeOnly(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, z_ext: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(z_ext, return_dict=False)[0]

    vae_decode = _VaeDecodeOnly(vae_model)

    optimizer = torch.optim.AdamW(
        (adapter.module if isinstance(adapter, DDP) else adapter).parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=bool(args.amp) and torch.cuda.is_available())

    start_epoch = 0
    _resume_global_step = 0
    resume_path = str(args.resume).strip()
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(resume_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError(f"Resume checkpoint must be a dict, got {type(ckpt)}")
        ad_sd = ckpt.get("adapter_state_dict") or ckpt.get("adapter") or ckpt.get("state_dict")
        if ad_sd is None:
            raise ValueError("Resume checkpoint missing adapter_state_dict")
        m_adapter = adapter.module if isinstance(adapter, DDP) else adapter
        missing, unexpected = m_adapter.load_state_dict(ad_sd, strict=False)
        if missing or unexpected:
            rank0_print(f"[resume] adapter strict=False missing={missing[:10]} unexpected={unexpected[:10]}")
        opt_sd = ckpt.get("optimizer_state_dict") or ckpt.get("optimizer")
        if opt_sd is not None:
            try:
                optimizer.load_state_dict(opt_sd)
                for state in optimizer.state.values():
                    if isinstance(state, dict):
                        for k, v in list(state.items()):
                            if torch.is_tensor(v):
                                state[k] = v.to(device, non_blocking=True)
            except Exception as e:
                rank0_print(f"[resume] optimizer load failed (will start fresh optimizer): {e}")
        # Override LR with the current --lr value (load_state_dict restores the old LR).
        new_lr = float(args.lr)
        for pg in optimizer.param_groups:
            old_lr = pg.get("lr")
            pg["lr"] = new_lr
            if old_lr != new_lr:
                rank0_print(f"[resume] overriding optimizer LR: {old_lr} -> {new_lr}")
        scaler_sd = ckpt.get("scaler_state_dict")
        if scaler_sd is not None and scaler.is_enabled():
            try:
                scaler.load_state_dict(scaler_sd)
            except Exception:
                pass
        start_epoch = int(ckpt.get("epoch", 0))
        _resume_global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
        rank0_print(f"[resume] loaded from {resume_path} epoch={start_epoch} global_step={_resume_global_step}")

    # dataset
    require_T = None if int(args.require_T) == 0 else int(args.require_T)
    png_tf = _build_png_transform()
    ds_kwargs = dict(
        manifest_json=str(args.manifest_json),
        items_key=str(args.items_key),
        workspace_root=os.path.abspath(os.path.join(_TSFORMER_ROOT, "..", "..")),
        transform=png_tf,
        load_frames=True,
        max_items=(int(args.max_items) if int(args.max_items) > 0 else None),
        require_T=require_T,
    )
    # PNG-teacher requires reading traj (to know T) and loading frames from images_dir.
    # Some environments may have an older LatentTrajManifestDataset without these kwargs; fall back gracefully.
    try:
        ds = LatentTrajManifestDataset(
            **ds_kwargs,
            load_traj=True,
            io_timeout_s=60.0,
            on_error="empty",
        )
    except TypeError:
        rank0_print("[warn] LatentTrajManifestDataset has no load_traj/io_timeout_s/on_error; falling back to legacy signature.")
        ds = LatentTrajManifestDataset(**ds_kwargs)
    sampler = DistributedSampler(ds, shuffle=True, drop_last=True) if is_dist() else None
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.persistent_workers) if int(args.num_workers) > 0 else False,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        drop_last=True,
        collate_fn=partial(collate_fn, mode=str(args.collate_mode), latent_chunk_len=int(args.latent_chunk_len)),
    )

    expected_T = None if bool(args.latent_use_full) else _expected_frames_from_latent_chunk(int(args.latent_chunk_len))
    rank0_print(
        f"[stage1] items={len(ds)} batch_size={int(args.batch_size)} global_batch_size={int(args.batch_size) * get_world_size()}"
        f" latent_chunk_len={int(args.latent_chunk_len)}"
        f" latent_use_full={bool(args.latent_use_full)} latent_cover_all={bool(args.latent_cover_all)}"
        f" expected_T={expected_T}"
    )

    global_step = _resume_global_step
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_num = int(epoch) + 1
        w_cos_ep, w_mse_ep = _loss_weights_for_epoch(epoch_num, args)
        if get_rank() == 0:
            rank0_log(
                f"[loss_w] epoch={epoch_num}/{int(args.epochs)} schedule={str(args.loss_schedule)} "
                f"w_cos={w_cos_ep:.6f} w_mse={w_mse_ep:.6f} w_mean={float(args.loss_mean_w):.6f} w_std={float(args.loss_std_w):.6f}"
            )

        if sampler is not None:
            sampler.set_epoch(epoch)
        (adapter.module if isinstance(adapter, DDP) else adapter).train()

        running = torch.zeros((), device=device)
        running_parts: dict = {}
        nb = 0
        t_after_step = time.time()
        use_tqdm = bool(args.tqdm) and (get_rank() == 0) and (tqdm is not None)
        pbar = tqdm(total=len(dl), desc=f"stage1 epoch {epoch+1}/{int(args.epochs)}", dynamic_ncols=True, leave=True) if use_tqdm else None

        for batch in dl:
            nb += 1
            global_step += 1
            data_s = time.time() - t_after_step
            t_step0 = time.time()

            optimizer.zero_grad(set_to_none=True)
            if global_step <= 3:
                rank0_log(f"[dl] got batch at step={global_step} data_s={data_s:.3f}")

            # Support per_sample collate to avoid shape issues on variable T_lat.
            if isinstance(batch, dict) and "samples" in batch:
                samples = batch["samples"]
                if get_rank() == 0 and int(batch.get("skipped_no_frames", 0)) > 0 and (global_step % int(args.log_every) == 0):
                    rank0_log(
                        f"[dl_skip] step={global_step} skipped_no_frames={int(batch.get('skipped_no_frames',0))} "
                        f"skipped_no_latent={int(batch.get('skipped_no_latent',0))} total_in={int(batch.get('total_in',0))} "
                        f"total_out={int(batch.get('total_out',0))}"
                    )
            else:
                z_ext = batch["z_ext"].to(device, non_blocking=True)
                fr = batch.get("frames_rgb", None)
                if fr is None:
                    raise ValueError("batch missing frames_rgb; make sure dataset load_frames=True")
                fr = fr.to(device, non_blocking=True)
                samples = [{"z_ext": z_ext[b : b + 1], "frames_rgb": fr[b : b + 1]} for b in range(int(z_ext.shape[0]))]

            loss_sum = torch.zeros((), device=device)
            step_parts: dict = {}
            valid = 0
            skip_short = 0
            skip_decode = 0
            did_backward = False
            backward_count = 0

            for i, s in enumerate(samples):
                z = s["z_ext"].to(device, non_blocking=True)
                fr_full = s.get("frames_rgb", None)
                if fr_full is None:
                    raise ValueError("sample missing frames_rgb; make sure collate_fn returns frames_rgb")
                fr_full = fr_full.to(device, non_blocking=True)
                # Normalize sample frame tensor shape to (1,3,T,H,W)
                if fr_full.ndim == 4:
                    fr_full = fr_full.unsqueeze(0)
                if fr_full.ndim != 5 or int(fr_full.shape[1]) != 3:
                    raise ValueError(f"frames_rgb must be (3,T,H,W) or (1,3,T,H,W), got {tuple(fr_full.shape)}")
                T_lat = int(z.shape[2])
                L = int(args.latent_chunk_len)
                # Keep (latent_start, z_sub) pairs so we can slice the teacher frames consistently.
                z_sub_list: List[Tuple[int, torch.Tensor]] = []
                if bool(args.latent_use_full):
                    z_sub_list = [(0, z.contiguous())]
                elif bool(args.latent_cover_all):
                    if T_lat < L:
                        skip_short += 1
                        continue
                    stride = max(1, int(args.latent_stride))
                    max_s = T_lat - L
                    starts = list(range(0, max_s + 1, stride))
                    mw = int(args.latent_max_windows)
                    if mw > 0 and len(starts) > mw:
                        starts = starts[:mw]
                    z_sub_list = [(int(st), z[:, :, st : st + L].contiguous()) for st in starts]
                else:
                    if T_lat < L:
                        skip_short += 1
                        continue
                    max_s = T_lat - L
                    if bool(args.latent_chunk_random):
                        st = random.randint(0, max_s)
                    else:
                        st = 0
                    z_sub_list = [(int(st), z[:, :, st : st + L].contiguous())]

                for j, (st_lat, z_sub) in enumerate(z_sub_list):
                    if global_step <= 3 and i == 0 and j == 0:
                        rank0_log(f"[step] pre_decode step={global_step} z_sub={tuple(z_sub.shape)}")

                    # Slice teacher frames for the latent window.
                    # Latent->frame alignment (Wan-like): T_frames = 4*(T_lat-1)+1.
                    # For a window that starts at latent index st_lat>0, we approximate the corresponding
                    # frame start as: 4*st_lat-3. This matches the cumulative frame count of the prefix.
                    if bool(args.latent_use_full):
                        fr_sub = fr_full
                    else:
                        if st_lat <= 0:
                            f0 = 0
                        else:
                            f0 = 4 * int(st_lat) - 3
                        # expected_T is frames length for a chunk of length L (default 9 when L=3)
                        t_need = int(expected_T) if expected_T is not None else int(fr_full.shape[2])
                        f1 = min(int(fr_full.shape[2]), f0 + t_need)
                        fr_sub = fr_full[:, :, f0:f1].contiguous()

                    tok_s, tok_t, T = _decode_teacher_student_tokens(
                        vae_decode_module=vae_decode,
                        adapter=adapter,
                        tsformer=tsformer,
                        z_sub=z_sub,
                        frames_teacher=fr_sub,
                        amp_enabled=bool(args.amp),
                        adapter_frames_chunk=int(args.adapter_frames_chunk),
                        adapter_use_checkpoint=bool(args.adapter_use_checkpoint),
                        expected_T=expected_T,
                    )
                    if global_step <= 3 and i == 0 and j == 0:
                        rank0_log(f"[step] post_decode step={global_step} decoded_T={int(T)}")
                    if int(T) <= 0:
                        skip_decode += 1
                        continue
                    loss_i, parts_i = compute_distill_loss(
                        tok_s, tok_t,
                        w_cos=float(w_cos_ep),
                        w_mean=float(args.loss_mean_w),
                        w_std=float(args.loss_std_w),
                        w_mse=float(w_mse_ep),
                    )
                    loss_sum = loss_sum + loss_i.detach()
                    for k, v in parts_i.items():
                        step_parts[k] = step_parts.get(k, 0.0) + v
                    valid += 1
                    # Backward immediately to avoid holding graphs for all samples/windows (major memory saver).
                    scaler.scale(loss_i).backward()
                    did_backward = True
                    backward_count += 1

            m_adapter = adapter.module if isinstance(adapter, DDP) else adapter
            if not did_backward:
                # Ensure we still run a backward pass (DDP-safe) and also initialize optimizer states.
                # This produces zero gradients but keeps the step structure consistent across ranks.
                s0 = None
                for p in m_adapter.parameters():
                    if p.requires_grad:
                        s0 = p.float().sum() if s0 is None else (s0 + p.float().sum())
                loss_batch = (s0 * 0.0) if s0 is not None else torch.zeros((), device=device)
                scaler.scale(loss_batch).backward()

            # Average gradients across multiple backward() calls within the step
            # to keep update magnitude stable even when a sample enumerates multiple windows.
            if backward_count > 1:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                for p in m_adapter.parameters():
                    if p.grad is not None:
                        p.grad.div_(float(backward_count))

            if float(args.grad_clip) > 0:
                if scaler.is_enabled() and backward_count <= 1:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(m_adapter.parameters(), max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            loss_mean = reduce_mean(loss_sum / max(1, valid))
            running += loss_mean.detach()
            if valid > 0:
                for k in step_parts:
                    step_parts[k] /= valid
            for k, v in step_parts.items():
                running_parts[k] = running_parts.get(k, 0.0) + v

            step_s = time.time() - t_step0
            parts_str = " ".join(f"{k}={v:.5f}" for k, v in sorted(step_parts.items()) if k != "total")
            if pbar is not None:
                pbar.update(1)
                try:
                    pbar.set_postfix({
                        "loss": float(loss_mean.item()),
                        "cos": step_parts.get("cos", 0.0),
                        "avg": float((running / nb).item()),
                        "data_s": data_s,
                        "step_s": step_s,
                    })
                except Exception:
                    pass
            else:
                if (get_rank() == 0) and (int(args.log_every) > 0) and (global_step % int(args.log_every) == 0):
                    rank0_print(
                        f"stage1 e{epoch+1}/{int(args.epochs)} step={global_step} "
                        f"loss={float(loss_mean.item()):.6f} {parts_str} "
                        f"avg={float((running/nb).item()):.6f} data_s={data_s:.2f} step_s={step_s:.2f}"
                    )

            rank0_log(
                f"[step] epoch={epoch+1}/{int(args.epochs)} step={global_step} "
                f"loss={float(loss_mean.item()):.6f} {parts_str} "
                f"avg={float((running/nb).item()):.6f} "
                f"valid={valid} bw={backward_count} skip_short={skip_short} skip_decode={skip_decode} "
                f"data_s={data_s:.3f} step_s={step_s:.3f}"
            )

            t_after_step = time.time()

        if pbar is not None:
            pbar.close()

        # save
        if (get_rank() == 0) and ((epoch + 1) % int(args.save_every) == 0):
            m_adapter = adapter.module if isinstance(adapter, DDP) else adapter
            opt_sd = optimizer.state_dict()
            state = {
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "adapter_state_dict": m_adapter.state_dict(),
                "optimizer_state_dict": opt_sd,
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "args": vars(args),
                # Backward-compatible aliases (older checkpoints used these keys).
                "adapter": m_adapter.state_dict(),
                "optimizer": opt_sd,
                "step": int(global_step),
            }
            path_e = os.path.join(str(args.out_dir), f"stage1_adapter_e{epoch+1}.pt")
            _atomic_torch_save(state, path_e)
            _atomic_torch_save(state, os.path.join(str(args.out_dir), "stage1_adapter_last.pt"))
            rank0_print(f"[save] {path_e}")

            if bool(args.export_combined):
                export_path = os.path.join(str(args.out_dir), str(args.export_name))
                combined = {
                    "vae_state_dict": vae_model.state_dict(),
                    "adapter_state_dict": m_adapter.state_dict(),
                    "exported_at": datetime.now().isoformat(timespec="seconds"),
                    "args": vars(args),
                }
                _atomic_torch_save(combined, export_path)
                rank0_print(f"[export] {export_path}")

    if _RANK0_LOG_FH is not None:
        try:
            _RANK0_LOG_FH.close()
        except Exception:
            pass

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

