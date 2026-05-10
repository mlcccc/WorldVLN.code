"""
Fine-tune TSformer-VO on UAVFlow simulated dataset with DDP.

Target:
- Use 4-frame window (num_frames=4) -> predict 3 relative motions -> 18-dim output
- Dataset layout: uavflowdatasim_output/<route_id>/{images/,raw_logs.json,preprocessed_logs.json,...}
- Images naming: images/frame_000000.png ...

Run (use GPU 4,5,6,7):
  CUDA_VISIBLE_DEVICES=4,5,6,7 \\
  python -m torch.distributed.launch --nproc_per_node=4 fine_tune_uavflow_sim_ddp.py \\
    --data_root /home/batchcom/dataset-link/xjc/uavflowdatasim_output \\
    --pretrained_ckpt checkpoint/checkpoint_model3_exp20.pth \\
    --out_dir checkpoints/uavflow_sim_ft_exp1

Note: For torch>=1.10, torchrun also works:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 fine_tune_uavflow_sim_ddp.py ...
"""

import os
import sys
# Ensure local repo modules take precedence over site-packages (e.g. HF `datasets`).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import random
import time
from dataclasses import asdict
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from datasets.uavflow_sim import UavflowSimDataset
from timesformer.models.vit import VisionTransformer


KITTI_MEAN = [0.34721234, 0.36705238, 0.36066107]
KITTI_STD = [0.30737526, 0.31515116, 0.32020183]


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
        print(*args, **kwargs, flush=True)


def ddp_setup():
    # Works with torch.distributed.launch / torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= get_world_size()
    return x


def build_model() -> VisionTransformer:
    # Must match checkpoint_model3_exp20.pth exactly
    model = VisionTransformer(
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
    return model


def load_pretrained(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) or len(unexpected):
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")


def set_head_only_trainable(model: nn.Module, head_only: bool):
    for p in model.parameters():
        p.requires_grad = not head_only
    if head_only:
        # keep final head trainable
        for p in model.head.parameters():
            p.requires_grad = True


def split_routes(num_routes: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    routes = list(range(num_routes))
    rng = random.Random(seed)
    rng.shuffle(routes)
    n_val = max(1, int(num_routes * val_ratio)) if val_ratio > 0 else 0
    val_routes = routes[:n_val]
    train_routes = routes[n_val:]
    return train_routes, val_routes


def build_subset_indices(dataset: UavflowSimDataset, keep_routes: List[int]) -> List[int]:
    keep = set(keep_routes)
    out = []
    for i, (route_idx, _start) in enumerate(dataset.samples):
        if route_idx in keep:
            out.append(i)
    return out


def compute_loss_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    rot_weight: float = 0.0,
    trans_xy_weight: float = 1.0,
    trans_z_weight: float = 1.0,
):
    """
    pred/target are flattened (B, (window_size-1)*6).
    Layout per step: [dz, dy, dx, tx, ty, tz] (all normalized).
    """
    # reshape to (B, window_size-1, 6)
    b = target.shape[0]
    t = window_size - 1
    pred = pred.view(b, t, 6)
    target = target.view(b, t, 6)

    pred_r, pred_t = pred[:, :, :3], pred[:, :, 3:]
    tgt_r, tgt_t = target[:, :, :3], target[:, :, 3:]

    loss = 0.0
    if rot_weight > 0:
        loss_r = torch.nn.functional.mse_loss(pred_r, tgt_r)
        loss = loss + rot_weight * loss_r

    # translation: keep xy at 1.0, lightly upweight z to fight collapse
    loss_xy = torch.nn.functional.mse_loss(pred_t[:, :, 0:2], tgt_t[:, :, 0:2])
    loss_z = torch.nn.functional.mse_loss(pred_t[:, :, 2], tgt_t[:, :, 2])
    loss = loss + trans_xy_weight * loss_xy + trans_z_weight * loss_z

    return loss


def _linear_schedule(epoch: int, warmup_or_decay_epochs: int, start: float, end: float) -> float:
    """
    Linear interpolate from start -> end over warmup_or_decay_epochs.
    - epoch: 1-based.
    - If warmup_or_decay_epochs <= 0: return end.
    - For epoch <= warmup_or_decay_epochs: interpolate.
    - For epoch > warmup_or_decay_epochs: return end.
    """
    if warmup_or_decay_epochs <= 0:
        return float(end)
    e = max(1, int(epoch))
    t = min(1.0, float(e) / float(warmup_or_decay_epochs))
    return float(start + t * (end - start))


def main():
    parser = argparse.ArgumentParser()
    # Compatibility with torch.distributed.launch (it appends --local_rank)
    # We still prefer reading LOCAL_RANK from env (torchrun style).
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--data_root", type=str, default="", help="单一数据根目录（旧参数，建议用 --data_roots）")
    parser.add_argument(
        "--data_roots",
        type=str,
        default="",
        help="多个数据根目录，用逗号分隔。例如: /path/a,/path/b （两份数据会一起算统一的 label 统计量）",
    )
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="checkpoint_model3_exp20.pth path")
    parser.add_argument("--out_dir", type=str, required=True, help="output checkpoint dir")
    parser.add_argument(
        "--translation_divisor",
        type=float,
        default=1.0,
        help="Divide translation deltas by this value (e.g. 100 for cm->m).",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32, help="per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="backbone LR")
    parser.add_argument("--head_lr_mult", type=float, default=10.0, help="head LR multiplier")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.05, help="route-level split")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stride", type=int, default=1, help="window stride within a route")
    parser.add_argument("--use_raw_for_labels", action="store_true", default=True)
    parser.add_argument("--angles_in_degrees", action="store_true", default=True)
    # Rotation curriculum:
    # - old flag kept for backward compatibility (acts like rot_max when no warmup is set)
    parser.add_argument("--rot_loss_weight", type=float, default=0.0, help="(旧) rotation MSE 权重；建议用 --rot_loss_weight_max + warmup")
    parser.add_argument("--rot_loss_weight_max", type=float, default=None, help="rotation MSE 最终权重（小一点，例如 0.05~0.2）")
    parser.add_argument("--rot_warmup_epochs", type=int, default=0, help="rotation 权重从 0 warmup 到 max 的 epoch 数（例如 10~20）")
    parser.add_argument("--trans_xy_weight", type=float, default=1.0, help="translation XY loss weight")
    # Translation-Z curriculum (keep XY stable):
    # - old flag kept for backward compatibility
    parser.add_argument("--trans_z_weight", type=float, default=2.0, help="(旧) translation Z 权重；不传新参数时固定为该值")
    parser.add_argument("--trans_z_weight_start", type=float, default=None, help="Z 权重起始值（前期稍高）")
    parser.add_argument("--trans_z_weight_end", type=float, default=None, help="Z 权重结束值（后期回落，避免伤 XY）")
    parser.add_argument("--trans_z_decay_epochs", type=int, default=0, help="Z 权重从 start -> end 线性回落的 epoch 数（例如 20~40）")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0, help="train head only for N epochs")
    parser.add_argument("--save_every", type=int, default=2)
    args = parser.parse_args()

    local_rank = ddp_setup()
    rank = get_rank()

    seed_everything(args.seed + rank)

    # Preflight: ensure current torch build supports the GPU (A100 is sm_80).
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
        cap_str = f"sm_{cap[0]}{cap[1]}"
        if cap_str not in arch_list and f"sm_{cap[0]}0" not in arch_list:
            raise RuntimeError(
                "当前 PyTorch 不支持该 GPU 架构，无法在 A100 上运行。\n"
                f"- GPU capability: {cap_str}\n"
                f"- torch.cuda.get_arch_list(): {arch_list}\n"
                "请换用支持 sm_80 的 PyTorch（建议 CUDA 11+ / 更新版 torch），"
                "或者改用系统里已支持 A100 的环境（例如 /opt/conda 的 torch 2.9+cu130）。"
            )

    os.makedirs(args.out_dir, exist_ok=True)

    preprocess = transforms.Compose(
        [
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=KITTI_MEAN, std=KITTI_STD),
        ]
    )

    # Resolve data roots
    data_roots: List[str] = []
    if args.data_roots.strip():
        data_roots = [p.strip() for p in args.data_roots.split(",") if p.strip()]
    elif args.data_root.strip():
        data_roots = [args.data_root.strip()]
    else:
        raise ValueError("必须提供 --data_roots 或 --data_root")

    dataset = UavflowSimDataset(
        root_dir=data_roots,
        window_size=4,
        stride=args.stride,
        transform=preprocess,
        use_raw_for_labels=args.use_raw_for_labels,
        angles_in_degrees=args.angles_in_degrees,
        translation_divisor=args.translation_divisor,
        img_ext=".png",
    )

    train_routes, val_routes = split_routes(len(dataset.routes), args.val_ratio, args.seed)
    train_idx = build_subset_indices(dataset, train_routes)
    val_idx = build_subset_indices(dataset, val_routes) if len(val_routes) else []

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx) if len(val_idx) else None

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if (is_dist() and val_set is not None) else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
        )
        if val_set is not None
        else None
    )

    # Save run metadata (rank0)
    if rank == 0:
        meta: Dict = {
            "data_roots": data_roots,
            "pretrained_ckpt": args.pretrained_ckpt,
            "out_dir": args.out_dir,
            "window_size": 4,
            "stride": args.stride,
            "num_routes": len(dataset.routes),
            "num_samples": len(dataset),
            "train_routes": train_routes,
            "val_routes": val_routes,
            "label_stats": dataset.get_stats(),
        }
        with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
            json.dump({**meta, **vars(args)}, f, indent=2)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model = build_model().to(device)
    load_pretrained(model, args.pretrained_ckpt)

    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimizer: different LR for head
    base_lr = args.lr
    head_lr = args.lr * args.head_lr_mult
    params = []
    named_params = model.module.named_parameters() if isinstance(model, DDP) else model.named_parameters()
    for n, p in named_params:
        # Keep requires_grad unchanged for DDP stability; control freezing via LR=0.
        is_head = n.startswith("head.")
        lr = head_lr if is_head else base_lr
        params.append({"params": [p], "lr": lr, "is_head": is_head})
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=args.weight_decay)

    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val = float("inf")

    rank0_print(f"DDP world_size={get_world_size()} local_rank={local_rank}")
    rank0_print(f"Routes: train={len(train_routes)} val={len(val_routes)}")
    rank0_print(f"Samples: train={len(train_set)} val={len(val_set) if val_set else 0}")

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Freeze backbone by setting its LR to 0 (do NOT toggle requires_grad after DDP wrap).
        head_only = args.freeze_backbone_epochs > 0 and epoch <= args.freeze_backbone_epochs
        if head_only:
            for g in optimizer.param_groups:
                g["lr"] = head_lr if g.get("is_head", False) else 0.0
        else:
            for g in optimizer.param_groups:
                g["lr"] = head_lr if g.get("is_head", False) else base_lr

        # ===== Curriculum schedules (do NOT change data / normalization) =====
        # Rotation: warmup from 0 -> rot_max
        rot_max = args.rot_loss_weight_max
        if rot_max is None:
            rot_max = float(args.rot_loss_weight)
        rot_w = _linear_schedule(epoch, int(args.rot_warmup_epochs), start=0.0, end=float(rot_max))

        # Z: decay from start -> end (keep XY fixed)
        z_start = args.trans_z_weight_start
        z_end = args.trans_z_weight_end
        if z_start is None and z_end is None:
            z_start = z_end = float(args.trans_z_weight)
        elif z_start is None:
            z_start = float(args.trans_z_weight)
        elif z_end is None:
            z_end = float(args.trans_z_weight)
        z_w = _linear_schedule(epoch, int(args.trans_z_decay_epochs), start=float(z_start), end=float(z_end))

        model.train()
        t0 = time.time()
        running = 0.0
        nb = 0

        train_iter = train_loader
        if rank == 0:
            train_iter = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", dynamic_ncols=True)

        for images, gt in train_iter:
            images = images.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                pred = model(images.float())
                loss = compute_loss_mse(
                    pred,
                    gt.float(),
                    window_size=4,
                    rot_weight=rot_w,
                    trans_xy_weight=args.trans_xy_weight,
                    trans_z_weight=z_w,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.detach()
            nb += 1
            if rank == 0:
                train_iter.set_postfix(loss=float(loss.detach()), avg=float((running / max(1, nb)).detach()))

        train_loss = (running / max(1, nb)).detach()
        train_loss = reduce_mean(train_loss).item()

        # validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            vrun = 0.0
            vnb = 0
            with torch.no_grad():
                val_iter = val_loader
                if rank == 0:
                    val_iter = tqdm(val_loader, desc=f"Val {epoch}/{args.epochs}", dynamic_ncols=True)
                for images, gt in val_iter:
                    images = images.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)
                    with autocast(enabled=torch.cuda.is_available()):
                        pred = model(images.float())
                        loss = compute_loss_mse(
                            pred,
                            gt.float(),
                            window_size=4,
                            rot_weight=rot_w,
                            trans_xy_weight=args.trans_xy_weight,
                            trans_z_weight=z_w,
                        )
                    vrun += loss.detach()
                    vnb += 1
                    if rank == 0:
                        val_iter.set_postfix(loss=float(loss.detach()))
            v = (vrun / max(1, vnb)).detach()
            v = reduce_mean(v).item()
            val_loss = v

        dt = time.time() - t0
        if rank == 0:
            msg = f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f}"
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.6f}"
            msg += f" | head_only={head_only} | rot_w={rot_w:.4f} z_w={z_w:.3f} | {dt:.1f}s"
            print(msg, flush=True)

        # save checkpoints (rank0)
        if rank == 0:
            to_save = model.module if isinstance(model, DDP) else model
            state = {
                "epoch": epoch,
                "model_state_dict": to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            }
            torch.save(state, os.path.join(args.out_dir, "checkpoint_last.pth"))
            if epoch % args.save_every == 0:
                torch.save(state, os.path.join(args.out_dir, f"checkpoint_e{epoch}.pth"))
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(args.out_dir, "checkpoint_best.pth"))

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

