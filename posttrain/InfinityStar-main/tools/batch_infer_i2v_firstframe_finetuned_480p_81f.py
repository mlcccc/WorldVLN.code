#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Batch infer 81-frame videos for all routes under a dataset root.

Each route directory should contain:
  - meta.json  (with instruction/instruction_unified)
  - images/    (first frame used as conditioning)

Outputs:
  <out_dir>/<route_id>/i2v_81f.mp4

Example:
  cd /home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main
  CUDA_VISIBLE_DEVICES=0 python3 tools/batch_infer_i2v_firstframe_finetuned_480p_81f.py \
    --ckpt checkpoints/finetune_wan_480p_cap81_keep_short/global_step_5296.pth \
    --dataset_root /home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like \
    --out_dir output_i2v_finetune5296_batch \
    --skip_existing
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as tdist
from tqdm import tqdm
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
import sys

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.utils.arg_util import Args
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta,
    get_first_full_spatial_size_scale_index,
)
from infinity.schedules import get_encode_decode_func
from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_example,
    save_video,
    transform,
)


def _sorted_images(images_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(osp.join(images_dir, e)))
    return sorted(files)


def _find_routes(dataset_root: str) -> List[Tuple[str, str, str]]:
    """Return list of (route_dir, meta_json_path, images_dir)."""
    routes = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        if "meta.json" in filenames:
            meta_path = osp.join(dirpath, "meta.json")
            images_dir = osp.join(dirpath, "images")
            if osp.isdir(images_dir):
                routes.append((dirpath, meta_path, images_dir))
    # Stable order
    routes.sort(key=lambda x: x[0])
    return routes


def _read_prompt(meta_json_path: str, prompt_key: str) -> str:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt")
    if not prompt:
        raise ValueError(f"No prompt field in {meta_json_path} (tried key={prompt_key})")
    return str(prompt).strip()

def _dist_init_if_needed() -> Tuple[int, int, int]:
    """
    Returns (rank, world_size, local_rank).
    If launched without torchrun, this is a single-process run.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    if not tdist.is_available():
        return 0, 1, 0
    if not tdist.is_initialized():
        tdist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--prompt_key", type=str, default="instruction_unified")
    ap.add_argument("--seed", type=int, default=0, help="Base seed; per-route seed = seed + index")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--pn", type=str, default="0.40M")
    ap.add_argument("--h_div_w_template", type=float, default=0.562)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--max_routes", type=int, default=-1, help="For debugging; -1 = all")
    ap.add_argument("--empty_cache_every", type=int, default=10, help="torch.cuda.empty_cache() cadence")
    args_cli = ap.parse_args()

    ckpt = osp.abspath(args_cli.ckpt)
    dataset_root = osp.abspath(args_cli.dataset_root)
    out_dir = osp.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rank, world_size, local_rank = _dist_init_if_needed()
    is_rank0 = rank == 0

    routes = _find_routes(dataset_root)
    if args_cli.max_routes and args_cli.max_routes > 0:
        routes = routes[: args_cli.max_routes]
    if not routes:
        raise FileNotFoundError(f"No routes with meta.json + images/ found under {dataset_root}")

    # Build inference args (match 480p training config closely).
    a = Args()
    a.pn = args_cli.pn
    a.fps = int(args_cli.fps)
    a.video_frames = int(args_cli.num_frames)
    a.temporal_compress_rate = 4
    a.videovae = 10
    a.vae_type = 64
    a.vae_path = osp.join(REPO_ROOT, "checkpoint", "infinitystar_videovae.pth")
    a.text_encoder_ckpt = osp.join(REPO_ROOT, "checkpoint", "text_encoder", "flan-t5-xl-official")
    a.text_channels = 2048
    a.model_type = "infinity_qwen8b"
    a.model_path = ckpt
    a.checkpoint_type = "torch"

    a.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2_allpt"
    a.mask_type = "infinity_elegant_clip20frames_v2_allpt"

    # Inference knobs (borrow defaults from tools/infer_video_480p.py).
    a.use_flex_attn = True
    a.bf16 = 1
    a.use_apg = 1
    a.use_cfg = 0
    a.cfg = 34
    a.tau_image = 1.0
    a.tau_video = 0.4
    a.apg_norm_threshold = 0.05
    a.append_duration2caption = 1
    a.use_two_stage_lfq = 1
    a.detail_scale_min_tokens = 350
    a.semantic_scales = 11
    a.max_repeat_times = 10000
    a.apply_spatial_patchify = 0
    a.num_of_label_value = 2
    a.rope2d_each_sa_layer = 1
    a.rope2d_normalized_by_hw = 2
    a.pad_to_multiplier = 128

    # repetition configs (14 scales for 0.40M)
    a.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    a.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"

    # Precompute schedule & context once (fixed template for this dataset).
    dyn_res, h_div_w_templates = get_dynamic_resolution_meta(a.dynamic_scale_schedule, a.video_frames)
    htpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - float(args_cli.h_div_w_template)))]
    pt = (a.video_frames - 1) // a.temporal_compress_rate + 1
    scale_schedule = dyn_res[htpl][a.pn]["pt2scale_schedule"][pt]
    a.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    a.tower_split_index = a.first_full_spatial_size_scale_index + 1
    video_encode, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(a.dynamic_scale_schedule)
    context_info = get_scale_pack_info(scale_schedule, a.first_full_spatial_size_scale_index, a)
    tau = [a.tau_image] * a.tower_split_index + [a.tau_video] * (len(scale_schedule) - a.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    dur_s = (a.video_frames - 1) // a.fps

    # load models (once)
    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)
    vae = load_visual_tokenizer(a).float().to("cuda")
    infinity = load_transformer(vae, a)
    self_correction = SelfCorrection(vae, a)

    ok = 0
    failed = 0
    t0 = time.time()

    # Partition work by rank.
    # Use global indices for deterministic per-route seeds: seed = base_seed + global_idx.
    indexed_routes = list(enumerate(routes))
    my_routes = indexed_routes[rank::world_size]

    pbar = tqdm(my_routes, desc=f"routes(rank={rank}/{world_size}, gpu={local_rank})", disable=not is_rank0)
    for global_idx, (route_dir, meta_path, images_dir) in pbar:
        route_id = osp.basename(route_dir.rstrip("/"))
        out_route_dir = osp.join(out_dir, route_id)
        os.makedirs(out_route_dir, exist_ok=True)
        out_mp4 = osp.join(out_route_dir, "i2v_81f.mp4")
        if args_cli.skip_existing and osp.exists(out_mp4):
            continue

        try:
            prompt = _read_prompt(meta_path, args_cli.prompt_key)
            images = _sorted_images(images_dir)
            if not images:
                raise FileNotFoundError(f"no images in {images_dir}")
            first_image_path = images[0]

            pil = Image.open(first_image_path).convert("RGB")
            ref = transform(pil, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
            ref_T3HW = torch.stack([ref], dim=0)  # [1,3,H,W]
            ref_bcthw = ref_T3HW.permute(1, 0, 2, 3).unsqueeze(0).to("cuda")  # [1,3,1,H,W]
            with torch.no_grad():
                _, _, gt_ls_Bl, _, _, _ = video_encode(
                    vae=vae,
                    inp_B3HW=ref_bcthw,
                    vae_features=None,
                    self_correction=self_correction,
                    args=a,
                    infer_mode=True,
                    dynamic_resolution_h_w=dyn_res,
                )
            gt_leak = 14

            prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if a.append_duration2caption else prompt
            seed = int(args_cli.seed) + int(global_idx)

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
                video_uint8, _ = gen_one_example(
                    infinity,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt_infer,
                    negative_prompt="",
                    g_seed=seed,
                    gt_leak=gt_leak,
                    gt_ls_Bl=gt_ls_Bl,
                    cfg_list=a.cfg,
                    tau_list=tau,
                    scale_schedule=scale_schedule,
                    cfg_insertion_layer=[0],
                    vae_type=a.vae_type,
                    sampling_per_bits=1,
                    enable_positive_prompt=0,
                    low_vram_mode=True,
                    args=a,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    context_info=context_info,
                    noise_list=None,
                )

            # video_uint8: torch uint8 [1,T,H,W,3] (BGR) or [T,H,W,3]
            if isinstance(video_uint8, torch.Tensor) and video_uint8.dim() == 5:
                video_uint8 = video_uint8[0]
            save_video(video_uint8.cpu().numpy() if isinstance(video_uint8, torch.Tensor) else np.asarray(video_uint8), fps=a.fps, save_filepath=out_mp4)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL][rank{rank}] route={route_dir} err={e}")
        finally:
            if args_cli.empty_cache_every > 0 and (global_idx + 1) % int(args_cli.empty_cache_every) == 0:
                torch.cuda.empty_cache()

    # Aggregate stats across ranks.
    if world_size > 1 and tdist.is_initialized():
        t_ok = torch.tensor([ok], device="cuda", dtype=torch.int64)
        t_failed = torch.tensor([failed], device="cuda", dtype=torch.int64)
        tdist.all_reduce(t_ok, op=tdist.ReduceOp.SUM)
        tdist.all_reduce(t_failed, op=tdist.ReduceOp.SUM)
        ok_total = int(t_ok.item())
        failed_total = int(t_failed.item())
        tdist.barrier()
    else:
        ok_total, failed_total = ok, failed

    dt = time.time() - t0
    if is_rank0:
        print(f"[done] ok={ok_total} failed={failed_total} routes={len(routes)} elapsed_s={dt:.1f} out_dir={out_dir}")


if __name__ == "__main__":
    main()

