import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
import math

# Import existing modules
from latent_patch_embed import LatentToPatchEmbed
from timesformer.models.vit import VisionTransformer
from functools import partial

# --- Dataset Definition ---
class P2PDataset(Dataset):
    def __init__(self, base_dir, ids, window_size=2, show_progress=True):
        self.samples = []
        self.window_size = window_size
        skipped_count = 0
        total_ids = len(ids) if hasattr(ids, "__len__") else None
        id_iter = tqdm(
            ids,
            total=total_ids,
            desc="Building P2P dataset",
            unit="traj",
            disable=not show_progress,
        )
        for idx, item in enumerate(id_iter, start=1):
            if isinstance(item, dict):
                # New format with explicit paths
                npy_path = item['latent_path']
                json_path = item['pose_path']
                vid_id = item.get('id', 'unknown')
            else:
                # Old format with ID string
                vid_id = item
                # Construct paths (Fixed folder name: reshape_actionhead_data)
                npy_path = os.path.join(base_dir, vid_id, "reshape_actionhead_data", "video_summed_codes.npy")
                json_path = os.path.join(base_dir, vid_id, "reshape_actionhead_data", "preprocessed_logs.json")
            
            if not os.path.exists(npy_path) or not os.path.exists(json_path):
                print(f"Skipping {vid_id}: Data missing")
                skipped_count += 1
                continue
                
            try:
                # Load Latents
                latents = np.load(npy_path)
                # Handle shapes: (1, T, C, H, W) or (T, C, H, W) or (B, T, C, H, W)
                if latents.ndim == 5 and latents.shape[0] == 1:
                    latents = latents[0]
                # If (C, T, H, W) -> (T, C, H, W) ? Unlikely. Usually (T, C, H, W) or (B, C, T, H, W)
                # Based on previous context: (1, 16, 25, 24, 80) -> (16, 25, 24, 80)
                # If input is (B, C, T, H, W), we need (T, C, H, W)
                if latents.ndim == 4:
                    # check if C is first. 
                    # If shape is (16, 25, 24, 80) -> C=16, T=25. 
                    # If shape is (25, 16, 24, 80) -> T=25, C=16.
                    if latents.shape[0] == 16 and latents.shape[1] != 16:
                         latents = latents.transpose(1, 0, 2, 3)
                
                num_latents = latents.shape[0]
                
                # Load Action Logs
                with open(json_path, 'r') as f:
                    logs = json.load(f)
                
                # Create Windows
                # Need W consecutive latents: i, i+1, ..., i+W-1
                # And W-1 actions: (i->i+1), ..., (i+W-2 -> i+W-1)
                
                for i in range(num_latents - window_size + 1):
                    # Check if action data exists for all steps
                    valid_window = True
                    window_actions = []
                    
                    for k in range(window_size - 1):
                        idx_curr = 4 * (i + k)
                        idx_next = 4 * (i + k + 1)
                        
                        if idx_next >= len(logs):
                            valid_window = False
                            break
                            
                        pose_curr = np.array(logs[idx_curr])
                        pose_next = np.array(logs[idx_next])
                        
                        # Calculate action
                        diff = pose_next - pose_curr
                        
                        # Unit Conversion
                        # First 3 (position): cm -> m
                        diff[0:3] = diff[0:3] / 100.0
                        # Last 3 (rotation): deg -> rad
                        diff[3:6] = diff[3:6] * (math.pi / 180.0)
                        
                        window_actions.append(diff)
                    
                    if valid_window:
                        self.samples.append({
                            'latent_seq': latents[i : i + window_size], # (W, C, H, W)
                            'target_seq': np.array(window_actions).flatten() # ((W-1)*6, )
                        })
                        
            except Exception as e:
                print(f"Error loading {vid_id}: {e}")
                skipped_count += 1

            if show_progress and idx % 50 == 0:
                id_iter.set_postfix(valid_samples=len(self.samples), skipped=skipped_count)
                
        if show_progress:
            id_iter.set_postfix(valid_samples=len(self.samples), skipped=skipped_count)
        print(
            f"Initialized P2P dataset with {len(self.samples)} samples "
            f"(processed trajectories: {total_ids if total_ids is not None else 'unknown'}, "
            f"skipped: {skipped_count}, Window Size: {window_size})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample['latent_seq']).float(),
            torch.from_numpy(sample['target_seq']).float()
        )

from torch.amp import GradScaler, autocast

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_all_parameters(module):
    return sum(p.numel() for p in module.parameters())

def get_model_ref(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def compute_target_stats(dataset, min_std=1e-4):
    targets = np.stack([sample["target_seq"] for sample in dataset.samples]).astype(np.float32)
    mean = torch.from_numpy(targets.mean(axis=0))
    std = torch.from_numpy(targets.std(axis=0))
    std = torch.clamp(std, min=min_std)
    return mean, std

def build_pos_rot_indices(target_dim):
    pos_idx = [i for i in range(target_dim) if i % 6 < 3]
    rot_idx = [i for i in range(target_dim) if i % 6 >= 3]
    return pos_idx, rot_idx

def build_axis_indices(target_dim):
    xy_idx = [i for i in range(target_dim) if i % 6 in (0, 1)]
    z_idx = [i for i in range(target_dim) if i % 6 == 2]
    rot_idx = [i for i in range(target_dim) if i % 6 >= 3]
    return xy_idx, z_idx, rot_idx

def linear_warmup_to_max(epoch_one_based, warmup_epochs, start, end):
    if warmup_epochs <= 0:
        return float(end)
    e = max(1, int(epoch_one_based))
    t = min(1.0, float(e) / float(warmup_epochs))
    return float(start + t * (end - start))

def weighted_regression_loss(
    outputs,
    targets,
    xy_idx,
    z_idx,
    rot_idx,
    xy_weight=1.0,
    z_weight=1.0,
    rot_weight=1.0,
):
    sq_err = (outputs - targets).pow(2)

    if len(xy_idx) > 0:
        xy_loss = sq_err[:, xy_idx].mean()
    else:
        xy_loss = outputs.new_tensor(0.0)

    if len(z_idx) > 0:
        z_loss = sq_err[:, z_idx].mean()
    else:
        z_loss = outputs.new_tensor(0.0)

    if len(rot_idx) > 0:
        rot_loss = sq_err[:, rot_idx].mean()
    else:
        rot_loss = outputs.new_tensor(0.0)

    loss = xy_weight * xy_loss + z_weight * z_loss + rot_weight * rot_loss
    return loss, xy_loss, z_loss, rot_loss

def build_warmup_cosine_lambda(total_epochs, warmup_epochs, min_lr_ratio):
    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(epoch_idx):
        # Linear warmup: scale from 1/warmup to 1.0
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)

        if total_epochs <= warmup_epochs:
            return 1.0

        progress = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs - 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda

# --- Model Builder ---
def build_p2p_model(args):
    # Output dim = (Window_Size - 1) * 6
    num_classes = (args.window_size - 1) * 6
    
    model = VisionTransformer(
        img_size=(192, 640),
        num_classes=num_classes, # Output all actions
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_frames=args.window_size, # Input window size
        attention_type='divided_space_time',
    )
    
    # Replace Patch Embed
    new_patch_embed = LatentToPatchEmbed(
        latent_dim=16,
        embed_dim=384,
        img_size=(192, 640),
        patch_size=16,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    model.patch_embed = new_patch_embed
    
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available() and args.safe_cuda_kernels:
        # Flash/mem-efficient SDPA kernels can occasionally be unstable with
        # large-token attention + DataParallel on some driver/runtime combos.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("Enabled safe CUDA attention kernels: flash=False, mem_efficient=False, math=True")
    
    # 1. Build Model
    model = build_p2p_model(args)
    
    if torch.cuda.device_count() > 1 and not args.force_single_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    elif args.force_single_gpu and torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("Force single-GPU mode enabled (cuda:0).")
        
    model.to(device)
    model_ref = get_model_ref(model)
    
    # Parameter groups:
    # - patch_embed/head use higher LR for faster latent-domain adaptation
    # - backbone uses lower LR to preserve pretrained temporal-spatial priors
    emb_head_params = list(model_ref.patch_embed.parameters()) + list(model_ref.head.parameters())
    emb_head_param_ids = {id(p) for p in emb_head_params}
    backbone_params = [p for p in model_ref.parameters() if id(p) not in emb_head_param_ids]

    print(f"Trainable params (all): {count_all_parameters(model_ref):,}")
    print(f"PatchEmbed params: {count_all_parameters(model_ref.patch_embed):,}")
    print(f"Head params: {count_all_parameters(model_ref.head):,}")
    print(f"Backbone params: {sum(p.numel() for p in backbone_params):,}")

    optimizer = optim.AdamW(
        [
            {"params": emb_head_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = None
    if args.scheduler == "cosine":
        lr_lambda = build_warmup_cosine_lambda(
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr_ratio=args.min_lr_ratio
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda, lr_lambda])

    scaler = GradScaler(device="cuda", enabled=torch.cuda.is_available())
    start_epoch = 0
    if args.resume_checkpoint is not None:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model_ref.load_state_dict(ckpt["model_state_dict"], strict=True)
            start_epoch = int(ckpt.get("epoch", 0))
            print(f"Resumed model weights from {args.resume_checkpoint} (epoch={start_epoch})")
            if args.resume_training_state:
                if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("Resumed optimizer state.")
                if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    print("Resumed scheduler state.")
        else:
            # Backward compatibility: old checkpoints saved pure model state_dict.
            model_ref.load_state_dict(ckpt, strict=True)
            print(f"Resumed model weights from legacy state_dict checkpoint: {args.resume_checkpoint}")
    
    # 2. Setup Data
    if getattr(args, 'jsonl_path', None) and os.path.exists(args.jsonl_path):
        print(f"Loading data from {args.jsonl_path}...")
        train_ids = []
        with open(args.jsonl_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    
                    # Check for new format first
                    if 'latent_path' in entry and 'pose_path' in entry:
                        train_ids.append(entry)
                        continue
                        
                    video_path = entry['video']
                    # Path format: .../uav-flowdata/<source_dir>/<id>/video.mp4
                    # We want to extract <source_dir>/<id> relative to uav-flowdata
                    
                    # Normalize path separators
                    video_path = video_path.replace('\\', '/')
                    parts = video_path.split('/')
                    
                    # Find 'uav-flowdata' index or similar anchor
                    if 'uavflowdatasim_output' in parts:
                        idx = parts.index('uavflowdatasim_output')
                        source_dir = parts[idx]
                        vid_id = parts[idx+1]
                        train_ids.append(os.path.join(source_dir, vid_id))
                    elif 'uavflowoutput' in parts:
                        idx = parts.index('uavflowoutput')
                        source_dir = parts[idx]
                        vid_id = parts[idx+1]
                        train_ids.append(os.path.join(source_dir, vid_id))
                    else:
                        continue
                except Exception as e:
                    print(f"Error parsing line: {line[:50]}... {e}")
                    continue
        print(f"Found {len(train_ids)} trajectories in jsonl.")
        dataset = P2PDataset(
            args.data_dir,
            train_ids,
            window_size=args.window_size,
            show_progress=args.show_data_progress
        )
    else:
        # Fallback to test IDs
        print("No jsonl path provided or file not found, using default test IDs.")
        test_ids = ["0", "1", "2"] 
        dataset = P2PDataset(
            args.data_dir,
            test_ids,
            window_size=args.window_size,
            show_progress=args.show_data_progress
        )
    
    if len(dataset) == 0:
        print("No data found.")
        return

    target_dim = (args.window_size - 1) * 6
    xy_idx, z_idx, rot_idx = build_axis_indices(target_dim)
    print(f"Loss groups -> xy dims: {len(xy_idx)}, z dims: {len(z_idx)}, rot dims: {len(rot_idx)}")

    target_mean = None
    target_std = None
    if args.target_standardize:
        target_mean, target_std = compute_target_stats(dataset, min_std=args.min_target_std)
        stats_path = os.path.join(args.save_dir, "p2p_target_stats.json")
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "window_size": args.window_size,
                    "target_dim": target_dim,
                    "mean": target_mean.tolist(),
                    "std": target_std.tolist(),
                },
                f,
                indent=2,
            )
        print(f"Saved target stats to {stats_path}")
        target_mean = target_mean.to(device)
        target_std = target_std.to(device)
        
    # Optimized DataLoader settings for high-throughput
    # pin_memory=True: faster host-to-device transfer
    # num_workers=8: more parallel data loading (adjust based on CPU cores)
    # prefetch_factor=2: reduce waiting time
    # persistent_workers=True: avoid worker respawn overhead
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    dataloader = DataLoader(**loader_kwargs)
    
    # Enable cuDNN benchmark for faster training on fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    end_epoch = start_epoch + args.epochs
    print(f"Start training: global epoch {start_epoch + 1} -> {end_epoch} (run epochs={args.epochs})")
    
    for epoch in range(start_epoch, end_epoch):
        # Stage 1: warm up latent adapter + head only
        if epoch < args.freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = False
            for p in emb_head_params:
                p.requires_grad = True
            stage_name = "embed/head warmup"
        else:
            for p in backbone_params:
                p.requires_grad = True
            for p in emb_head_params:
                p.requires_grad = True
            stage_name = "full finetune"

        model.train()
        total_loss = 0
        total_xy_loss = 0
        total_z_loss = 0
        total_rot_loss = 0
        nonfinite_skipped = 0
        epoch_one_based = epoch + 1
        rot_w = linear_warmup_to_max(
            epoch_one_based=epoch_one_based,
            warmup_epochs=args.rot_warmup_epochs,
            start=args.rot_weight_start,
            end=args.rot_weight_max,
        )
        z_w = linear_warmup_to_max(
            epoch_one_based=epoch_one_based,
            warmup_epochs=args.z_warmup_epochs,
            start=args.z_weight_start,
            end=args.z_weight_max,
        )
        xy_w = args.pos_weight
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{end_epoch} [{stage_name}]")
        for latents, targets in pbar:
            # latents: (B, 2, C, H, W)
            # targets: (B, 6)
            latents, targets = latents.to(device), targets.to(device)
            
            # TSformer expects (B, C, T, H, W) usually for video
            # But our LatentToPatchEmbed expects (B, T, C, H, W) based on my previous code?
            # Let's check LatentToPatchEmbed in latent_patch_embed.py:
            # "if x.dim() == 5: B, T, C, H, W = x.shape"
            # So we should pass (B, T, C, H, W).
            
            # TSformer forward:
            # x = self.patch_embed(x) -> needs to match
            # So we pass latents as is: (B, 2, C, H, W)
            
            optimizer.zero_grad()
            
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(latents) # (B, 6)
                if args.target_standardize:
                    targets_for_loss = (targets - target_mean) / target_std
                else:
                    targets_for_loss = targets

                loss, xy_loss, z_loss, rot_loss = weighted_regression_loss(
                    outputs=outputs,
                    targets=targets_for_loss,
                    xy_idx=xy_idx,
                    z_idx=z_idx,
                    rot_idx=rot_idx,
                    xy_weight=xy_w,
                    z_weight=z_w,
                    rot_weight=rot_w,
                )

            if args.skip_nonfinite_loss and (not torch.isfinite(loss).item()):
                nonfinite_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(skipped_nonfinite=nonfinite_skipped)
                continue

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_xy_loss += xy_loss.item()
            total_z_loss += z_loss.item()
            total_rot_loss += rot_loss.item()
            pbar.set_postfix(loss=loss.item(), xy=xy_loss.item(), z=z_loss.item(), rot=rot_loss.item())
            
        avg_loss = total_loss / len(dataloader)
        avg_xy_loss = total_xy_loss / len(dataloader)
        avg_z_loss = total_z_loss / len(dataloader)
        avg_rot_loss = total_rot_loss / len(dataloader)
        if scheduler is not None:
            scheduler.step()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(
            f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.6f} "
            f"(xy={avg_xy_loss:.6f}, z={avg_z_loss:.6f}, rot={avg_rot_loss:.6f}) | "
            f"weights(xy/z/rot)=({xy_w:.3f}/{z_w:.3f}/{rot_w:.3f}) | "
            f"LR(embed/head)={current_lrs[0]:.3e}, LR(backbone)={current_lrs[1]:.3e} | "
            f"skipped_nonfinite={nonfinite_skipped}"
        )
        
        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == end_epoch:
            save_path = os.path.join(args.save_dir, f"p2p_epoch_{epoch+1}.pth")
            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": model_ref.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "args": vars(args),
            }
            torch.save(ckpt, save_path)
            print(f"Saved full-framework checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=3e-4, help="LR for patch_embed + head")
    parser.add_argument("--backbone_lr", type=float, default=5e-5, help="LR for TSformer backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"], help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs for scheduler")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05, help="Min LR ratio for cosine scheduler")
    parser.add_argument("--target_standardize", action="store_true", dest="target_standardize", help="Standardize regression targets before loss")
    parser.add_argument("--no_target_standardize", action="store_false", dest="target_standardize", help="Disable target standardization")
    parser.set_defaults(target_standardize=True)
    parser.add_argument("--min_target_std", type=float, default=1e-4, help="Lower bound for target std in standardization")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Loss weight for translation XY components")
    parser.add_argument("--z_weight_start", type=float, default=1.0, help="Initial loss weight for translation Z")
    parser.add_argument("--z_weight_max", type=float, default=1.3, help="Max loss weight for translation Z after warmup")
    parser.add_argument("--z_warmup_epochs", type=int, default=15, help="Epochs to warmup translation Z weight")
    parser.add_argument("--rot_weight", type=float, default=1.0, help="(legacy) fixed rotation weight when warmup args are not used")
    parser.add_argument("--rot_weight_start", type=float, default=1.0, help="Initial rotation loss weight")
    parser.add_argument("--rot_weight_max", type=float, default=1.35, help="Max rotation loss weight after warmup")
    parser.add_argument("--rot_warmup_epochs", type=int, default=15, help="Epochs to warmup rotation weight")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5, help="Warmup epochs for embed/head only")
    parser.add_argument("--hidden_dim", type=int, default=96, help='Bottleneck dim for LatentEmbed')
    parser.add_argument("--num_layers", type=int, default=2, help='Num lightweight residual blocks')
    parser.add_argument("--window_size", type=int, default=3, help='Input sequence length (window size)')
    parser.add_argument("--jsonl_path", type=str, default=None, help='Path to jsonl file with data')
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to checkpoint for resuming")
    parser.add_argument("--resume_training_state", action="store_true", help="Also resume optimizer/scheduler states")
    parser.add_argument("--show_data_progress", action="store_true", dest="show_data_progress", help="Show dataset building progress")
    parser.add_argument("--no_show_data_progress", action="store_false", dest="show_data_progress", help="Hide dataset building progress")
    parser.set_defaults(show_data_progress=True)
    parser.add_argument("--safe_cuda_kernels", action="store_true", help="Disable flash/mem-efficient SDPA for stability")
    parser.add_argument("--force_single_gpu", action="store_true", help="Force single-GPU training (no DataParallel)")
    parser.add_argument("--num_workers", type=int, default=16, help='Number of data loading workers')
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Clip grad norm; <=0 disables clipping")
    parser.add_argument("--skip_nonfinite_loss", action="store_true", dest="skip_nonfinite_loss", help="Skip batches with non-finite loss")
    parser.add_argument("--no_skip_nonfinite_loss", action="store_false", dest="skip_nonfinite_loss", help="Do not skip non-finite loss batches")
    parser.set_defaults(skip_nonfinite_loss=True)
    
    args = parser.parse_args()
    if args.rot_weight_start is None:
        args.rot_weight_start = float(args.rot_weight)
    if args.rot_weight_max is None:
        args.rot_weight_max = float(args.rot_weight)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
