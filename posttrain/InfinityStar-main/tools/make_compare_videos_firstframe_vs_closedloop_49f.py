#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Generate comparison videos for routes already processed by closed-loop streaming inference.

For each route under `--closed_loop_out_dir/<route_id>/.../run_args.json`, this script will:
  1) Create a baseline I2V video: first frame + prompt -> predict `--num_frames` frames (default 49)
  2) Create a GT video by stitching raw images (default first `--gt_num_frames` frames; can also use all frames)
  3) Put both into the SAME folder for convenient comparison:
       <closed_loop_out_dir>/<route_id>/<compare_subdir>/
         - i2v_firstframe_prompt_49f.mp4
         - gt_images_49f.mp4
         - compare_meta.json
         - (optional) symlinks to latest closed-loop seg_*.mp4

NOTE:
- This script is intended to be run after `tools/batch_closed_loop_streaming_infer_routes.py`.
- It does NOT run closed-loop again. It only creates baseline+GT videos for comparison.
- It can be launched with torchrun for multi-GPU parallelism:
    torchrun --nproc_per_node=8 tools/make_compare_videos_firstframe_vs_closedloop_49f.py ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import time
from typing import Dict, List, Optional, Tuple

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
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_example,
    save_video,
    transform,
)

def _frames_inner_clip_from_schedule(schedule_name: str) -> int | None:
    """
    Infer args.frames_inner_clip (in *compressed* frames) from schedule name.
    This must match the schedule design; otherwise RoPE frame ranges can go out-of-bounds
    and crash in get_visual_rope_embeds() with empty f_frames slices.
    """
    s = str(schedule_name)
    if "clip4frames" in s or "clip16frames" in s:
        return 4
    if "clip20frames" in s:
        return 20
    return None


def _dist_init_if_needed() -> Tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). If not launched with torchrun, returns (0,1,0)."""
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


def _sorted_images(images_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(osp.join(images_dir, e)))
    return sorted(files)


def _take_with_pad(paths: List[str], n: int, pad_short_real: bool) -> List[str]:
    if len(paths) >= n:
        return paths[:n]
    if not paths:
        raise ValueError("no real frames found")
    if not pad_short_real:
        raise ValueError(f"真实帧不足：need={n} but only={len(paths)}")
    return paths + [paths[-1]] * (n - len(paths))


def _to_uint8_bgr_from_transform_tensor(x_3hw: torch.Tensor) -> np.ndarray:
    """
    x_3hw: torch float tensor in [-1,1], shape [3,H,W], RGB.
    returns uint8 BGR ndarray [H,W,3]
    """
    x = (x_3hw.clamp(-1, 1) + 1.0) * 0.5  # [0,1]
    x = (x * 255.0).round().to(torch.uint8)
    rgb = x.permute(1, 2, 0).cpu().numpy()
    bgr = rgb[..., ::-1].copy()
    return bgr


def _find_latest_closed_loop_run_dir(route_out_dir: str) -> Optional[str]:
    """
    Under <route_out_dir>/, closed-loop batch script stores subdirs like 2026-02-08_15-33-10/
    containing run_args.json and seg_*.mp4. Pick the latest by name sort.
    """
    if not osp.isdir(route_out_dir):
        return None
    subdirs = []
    for name in os.listdir(route_out_dir):
        p = osp.join(route_out_dir, name)
        if osp.isdir(p) and osp.exists(osp.join(p, "run_args.json")):
            subdirs.append(name)
    if not subdirs:
        return None
    subdirs.sort()
    return osp.join(route_out_dir, subdirs[-1])


def _safe_symlink(src: str, dst: str) -> None:
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if osp.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except Exception:
        # If symlink is not permitted, fallback to doing nothing (no copy to avoid huge IO).
        return


def _load_prompt_from_meta(meta_json_path: str, prompt_key: str) -> str:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt")
    if not prompt:
        raise ValueError(f"No prompt field in {meta_json_path} (tried key={prompt_key})")
    return str(prompt).strip()


def _pick_first_image(images_dir: str) -> str:
    images = _sorted_images(images_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {images_dir}")
    return images[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="finetuned global_step_*.pth (checkpoint_type=torch)")
    ap.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use original pretrained weights from --checkpoints_root (checkpoint_type=torch_shard).",
    )
    ap.add_argument(
        "--pretrained_model_subdir",
        type=str,
        default="infinitystar_8b_480p_weights",
        help="Subdir under --checkpoints_root that contains pretrained weights (torch_shard).",
    )
    ap.add_argument("--dataset_root", type=str, default=None, help="original dataset root containing <route_id>/meta.json + images/")
    ap.add_argument("--closed_loop_out_dir", type=str, default=None, help="output dir used by batch_closed_loop_streaming_infer_routes.py")
    ap.add_argument("--prompt_key", type=str, default="instruction", help="field in meta.json used as prompt")
    ap.add_argument("--dynamic_scale_schedule", type=str, default="infinity_elegant_clip16frames_v2_allpt")
    ap.add_argument("--mask_type", type=str, default=None, help="defaults to dynamic_scale_schedule when omitted")
    ap.add_argument("--checkpoints_root", type=str, default=osp.join(REPO_ROOT, "checkpoint"), help="contains infinitystar_videovae.pth and text_encoder/")
    ap.add_argument("--vae_path", type=str, default=None, help="override VAE checkpoint path")
    ap.add_argument("--text_encoder_ckpt", type=str, default=None, help="override text encoder directory path")

    # single-video i2v mode (only uses first frame + prompt, no GT stitching)
    ap.add_argument("--single_i2v_only", action="store_true")
    ap.add_argument("--image_dir", type=str, default=None, help="directory that contains frames; first frame is used as condition")
    ap.add_argument("--prompt_json", type=str, default=None, help="meta json that contains prompt text")
    ap.add_argument("--single_out_video", type=str, default=None, help="output mp4 path for single i2v mode")

    ap.add_argument("--num_frames", type=int, default=49, help="baseline i2v output frames")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--pn", type=str, default="0.40M")
    ap.add_argument("--h_div_w_template", type=float, default=0.562)
    ap.add_argument("--seed", type=int, default=0, help="base seed; per-route seed = seed + global_idx")
    ap.add_argument("--skip_existing", action="store_true", help="skip if compare outputs already exist")

    ap.add_argument("--compare_subdir", type=str, default="compare_49f", help="folder created under each route_out_dir")
    ap.add_argument("--symlink_closed_loop_segs", action="store_true", help="symlink seg_*.mp4 from latest closed-loop run into compare folder")

    ap.add_argument("--gt_all_frames", action="store_true", help="stitch ALL raw images into gt video (may be long)")
    ap.add_argument("--gt_num_frames", type=int, default=49, help="gt video frames if not --gt_all_frames")
    ap.add_argument("--pad_short_real", action="store_true", default=True, help="pad gt video to gt_num_frames by repeating last frame (default on)")
    ap.add_argument("--no_pad_short_real", action="store_false", dest="pad_short_real")

    ap.add_argument("--max_routes", type=int, default=-1, help="debug limit; -1 = all")
    ap.add_argument("--empty_cache_every", type=int, default=10)
    args_cli = ap.parse_args()

    ckpt = osp.abspath(args_cli.ckpt) if args_cli.ckpt else ""
    dataset_root = osp.abspath(args_cli.dataset_root) if args_cli.dataset_root else None
    closed_loop_out_dir = osp.abspath(args_cli.closed_loop_out_dir) if args_cli.closed_loop_out_dir else None

    if (not args_cli.use_pretrained) and (not ckpt):
        raise ValueError("Must provide --ckpt (finetuned global_step_*.pth) unless --use_pretrained is set.")

    if args_cli.single_i2v_only:
        if not args_cli.image_dir or not args_cli.prompt_json or not args_cli.single_out_video:
            raise ValueError("--single_i2v_only requires --image_dir --prompt_json --single_out_video")
    else:
        if not dataset_root or not closed_loop_out_dir:
            raise ValueError("batch compare mode requires --dataset_root and --closed_loop_out_dir")

    rank, world_size, local_rank = _dist_init_if_needed()
    is_rank0 = rank == 0

    my_routes: List[Tuple[int, str]] = []
    if not args_cli.single_i2v_only:
        # Collect route_ids from closed_loop_out_dir (ignore _summary_*.json etc)
        route_ids = []
        for name in os.listdir(closed_loop_out_dir):
            if name.startswith("_"):
                continue
            p = osp.join(closed_loop_out_dir, name)
            if osp.isdir(p):
                route_ids.append(name)
        route_ids.sort()
        if args_cli.max_routes and args_cli.max_routes > 0:
            route_ids = route_ids[: args_cli.max_routes]
        if not route_ids:
            raise FileNotFoundError(f"No route directories found under {closed_loop_out_dir}")
        # Partition by rank
        indexed = list(enumerate(route_ids))
        my_routes = indexed[rank::world_size]

    checkpoints_root = osp.abspath(str(args_cli.checkpoints_root))
    vae_path = osp.abspath(str(args_cli.vae_path)) if args_cli.vae_path else osp.join(checkpoints_root, "infinitystar_videovae.pth")
    text_encoder_ckpt = osp.abspath(str(args_cli.text_encoder_ckpt)) if args_cli.text_encoder_ckpt else osp.join(checkpoints_root, "text_encoder", "flan-t5-xl-official")

    # Build inference args (match 480p training config closely), but with num_frames=49.
    a = Args()
    a.pn = args_cli.pn
    a.fps = int(args_cli.fps)
    a.video_fps = int(args_cli.fps)
    a.video_frames = int(args_cli.num_frames)
    a.temporal_compress_rate = 4
    a.videovae = 10
    a.vae_type = 64
    a.vae_path = vae_path
    a.text_encoder_ckpt = text_encoder_ckpt
    a.text_channels = 2048
    a.model_type = "infinity_qwen8b"
    if bool(args_cli.use_pretrained):
        a.model_path = osp.join(checkpoints_root, str(args_cli.pretrained_model_subdir))
        a.checkpoint_type = "torch_shard"
    else:
        a.model_path = ckpt
        a.checkpoint_type = "torch"

    a.dynamic_scale_schedule = str(args_cli.dynamic_scale_schedule)
    a.mask_type = str(args_cli.mask_type) if args_cli.mask_type else str(args_cli.dynamic_scale_schedule)
    fic = _frames_inner_clip_from_schedule(a.dynamic_scale_schedule)
    if fic is not None:
        a.frames_inner_clip = int(fic)

    # Inference knobs (keep same defaults as our other 480p scripts).
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

    # Precompute schedule & context for this num_frames
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

    # Load models once per process
    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)
    vae = load_visual_tokenizer(a).float().to("cuda")
    infinity = load_transformer(vae, a)
    infinity.eval().requires_grad_(False)
    self_correction = SelfCorrection(vae, a)

    if args_cli.single_i2v_only:
        image_dir = osp.abspath(str(args_cli.image_dir))
        prompt_json = osp.abspath(str(args_cli.prompt_json))
        out_i2v = osp.abspath(str(args_cli.single_out_video))
        out_parent = osp.dirname(out_i2v)
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        if world_size > 1 and rank != 0:
            return

        prompt = _load_prompt_from_meta(prompt_json, args_cli.prompt_key)
        prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if a.append_duration2caption else prompt
        first_image_path = _pick_first_image(image_dir)

        pil = Image.open(first_image_path).convert("RGB")
        ref = transform(pil, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
        ref_T3HW = torch.stack([ref], dim=0)  # [1,3,H,W]
        ref_bcthw = ref_T3HW.permute(1, 0, 2, 3).unsqueeze(0).to("cuda")  # [1,3,1,H,W]
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
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            video_uint8_bgr, _ = gen_one_example(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt_infer,
                negative_prompt="",
                g_seed=int(args_cli.seed),
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
        if isinstance(video_uint8_bgr, torch.Tensor) and video_uint8_bgr.dim() == 5:
            video_uint8_bgr = video_uint8_bgr[0]
        save_video(video_uint8_bgr, fps=int(args_cli.fps), save_filepath=out_i2v, force_all_keyframes=True)
        if is_rank0:
            print(f"[done-single] saved={out_i2v}")
        return

    ok = 0
    failed = 0
    results: List[Dict[str, object]] = []

    pbar = tqdm(my_routes, desc=f"routes(rank={rank}/{world_size}, gpu={local_rank})", disable=not is_rank0)
    for global_idx, route_id in pbar:
        route_out_dir = osp.join(closed_loop_out_dir, route_id)
        latest_run_dir = _find_latest_closed_loop_run_dir(route_out_dir)
        if latest_run_dir is None:
            continue

        compare_dir = osp.join(route_out_dir, str(args_cli.compare_subdir))
        os.makedirs(compare_dir, exist_ok=True)

        out_i2v = osp.join(compare_dir, f"i2v_firstframe_prompt_{int(args_cli.num_frames):02d}f.mp4")
        gt_tag = "all" if bool(args_cli.gt_all_frames) else str(int(args_cli.gt_num_frames))
        out_gt = osp.join(compare_dir, f"gt_images_{gt_tag}f.mp4")
        out_meta = osp.join(compare_dir, "compare_meta.json")

        if args_cli.skip_existing and osp.exists(out_i2v) and osp.exists(out_gt) and osp.exists(out_meta):
            continue

        # Locate original route assets (prefer run_args.json from closed-loop to avoid mismatches)
        try:
            with open(osp.join(latest_run_dir, "run_args.json"), "r", encoding="utf-8") as f:
                closed_meta = json.load(f)
            images_dir = closed_meta.get("images_dir") or osp.join(dataset_root, route_id, "images")
            meta_path = closed_meta.get("meta_path") or osp.join(dataset_root, route_id, "meta.json")
            route_dir = closed_meta.get("route_dir") or osp.join(dataset_root, route_id)

            images_dir = osp.abspath(images_dir)
            meta_path = osp.abspath(meta_path)
            route_dir = osp.abspath(route_dir)

            prompt = _load_prompt_from_meta(meta_path, args_cli.prompt_key)
            prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if a.append_duration2caption else prompt

            # Build conditioning from first frame
            images = _sorted_images(images_dir)
            if not images:
                raise FileNotFoundError(f"no images in {images_dir}")
            first_image_path = images[0]

            pil = Image.open(first_image_path).convert("RGB")
            ref = transform(pil, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
            ref_T3HW = torch.stack([ref], dim=0)  # [1,3,H,W]
            ref_bcthw = ref_T3HW.permute(1, 0, 2, 3).unsqueeze(0).to("cuda")  # [1,3,1,H,W]
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

            # Baseline I2V generation
            seed = int(args_cli.seed) + int(global_idx)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
                video_uint8_bgr, _ = gen_one_example(
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
            if isinstance(video_uint8_bgr, torch.Tensor) and video_uint8_bgr.dim() == 5:
                video_uint8_bgr = video_uint8_bgr[0]
            save_video(video_uint8_bgr, fps=int(args_cli.fps), save_filepath=out_i2v, force_all_keyframes=True)

            # GT stitching
            if bool(args_cli.gt_all_frames):
                gt_paths = images
            else:
                gt_paths = _take_with_pad(images, int(args_cli.gt_num_frames), bool(args_cli.pad_short_real))
            gt_frames_bgr = []
            for p in gt_paths:
                pil2 = Image.open(p).convert("RGB")
                fr = transform(pil2, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
                gt_frames_bgr.append(_to_uint8_bgr_from_transform_tensor(fr))
            gt_video = np.stack(gt_frames_bgr, axis=0)  # [T,H,W,3] BGR uint8
            save_video(gt_video, fps=int(args_cli.fps), save_filepath=out_gt, force_all_keyframes=True)

            # Optional symlink seg videos into compare dir
            if args_cli.symlink_closed_loop_segs:
                for seg in sorted(glob.glob(osp.join(latest_run_dir, "seg_*.mp4"))):
                    dst = osp.join(compare_dir, osp.basename(seg))
                    _safe_symlink(seg, dst)
                _safe_symlink(osp.join(latest_run_dir, "run_args.json"), osp.join(compare_dir, "closed_loop_run_args.json"))

            # Save compare meta
            compare_meta = {
                "route_id": route_id,
                "route_dir": route_dir,
                "meta_path": meta_path,
                "images_dir": images_dir,
                "closed_loop_latest_run_dir": latest_run_dir,
                "prompt_key": args_cli.prompt_key,
                "prompt": prompt,
                "prompt_infer": prompt_infer,
                "seed": seed,
                "num_frames_i2v": int(args_cli.num_frames),
                "fps": int(args_cli.fps),
                "tgt_h": int(tgt_h),
                "tgt_w": int(tgt_w),
                "gt_all_frames": bool(args_cli.gt_all_frames),
                "gt_num_frames": int(len(gt_paths)),
                "gt_pad_short_real": bool(args_cli.pad_short_real),
                "out_i2v": out_i2v,
                "out_gt": out_gt,
            }
            with open(out_meta, "w", encoding="utf-8") as f:
                json.dump(compare_meta, f, ensure_ascii=False, indent=2)

            ok += 1
            results.append({"route_id": route_id, "compare_dir": compare_dir})
        except Exception as e:
            failed += 1
            print(f"[FAIL][rank{rank}] route_id={route_id} err={e}")
        finally:
            if args_cli.empty_cache_every > 0 and (ok + failed) % int(args_cli.empty_cache_every) == 0:
                torch.cuda.empty_cache()

    # Write summary per rank
    summary_path = osp.join(closed_loop_out_dir, f"_compare_summary_rank{rank}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rank": rank,
                "world_size": world_size,
                "ok": ok,
                "failed": failed,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if is_rank0:
        print(f"[done] ok={ok} failed={failed} out_dir={closed_loop_out_dir}")


if __name__ == "__main__":
    main()

