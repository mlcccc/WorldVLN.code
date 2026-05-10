#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Segmented I2V/V2V inference (49 frames, clip=16).

This script follows the official v2v injection style in `tools/infer_video_480p.py`:
- Encode conditioning frames with `video_encode(...)` to obtain `gt_ls_Bl`
- Use `gt_leak` + `gt_ls_Bl` in `gen_one_example(...)` to hard-constrain early scales

Requested behavior:
- seg00: condition on frame 1 (I2V), generate 1..49, save 2..17
- seg01: condition on frames 1..17 (V2V), generate 1..49, save 18..33
- seg02: condition on frames 1..33 (V2V), generate 1..49, save 34..49

Example:
  cd /home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main
  CUDA_VISIBLE_DEVICES=0 python3 tools/infer_v2v_segments_49f_clip16.py \
    --ckpt /home/batchcom/dataset-link/xjc/uavflow_cap49_20_90/train_outputs/finetune_480p_49f_clip4_crossclip/ckpts/global_step_10000.pth \
    --images_dir /home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-05-12_21-56-20/images \
    --meta_json /home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-05-12_21-56-20/meta.json \
    --prompt_key instruction_unified \
    --out_dir ./output_v2v_segments_2025-05-12_21-56-20_clip16

Batch over a dataset root (routes with meta.json + images/):
  CUDA_VISIBLE_DEVICES=0 python3 tools/infer_v2v_segments_49f_clip16.py \
    --ckpt /path/to/global_step_10000.pth \
    --dataset_root /home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like \
    --out_dir ./output_v2v_segments_batch_clip16 \
    --prompt_key instruction_unified \
    --min_images 33 \
    --skip_existing
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
import sys

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.utils.arg_util import Args
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.video_decoder import EncodedVideoOpencv
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform


def _sorted_images(images_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(osp.join(images_dir, e)))
    return sorted(files)


def _read_prompt(meta_json_path: str, prompt_key: str) -> str:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = meta.get(prompt_key) or meta.get("instruction_unified") or meta.get("instruction") or meta.get("prompt")
    if not prompt:
        raise ValueError(f"No prompt field in {meta_json_path} (tried key={prompt_key})")
    return str(prompt).strip()


def _load_obs_video_bcthw(frame_paths: List[str], tgt_h: int, tgt_w: int) -> torch.Tensor:
    frames = []
    for p in frame_paths:
        pil = Image.open(p).convert("RGB")
        frames.append(transform(pil, tgt_h, tgt_w))
    video_T3HW = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]


def _load_obs_video_from_mp4_bcthw(video_path: str, prefix_len: int, tgt_h: int, tgt_w: int) -> torch.Tensor:
    if prefix_len <= 0:
        raise ValueError(f"prefix_len must be positive, got {prefix_len}")
    video = EncodedVideoOpencv(video_path, os.path.basename(video_path), num_threads=0)
    try:
        if int(video._vlen) < int(prefix_len):
            raise ValueError(f"video has only {int(video._vlen)} frames, need prefix_len={prefix_len}: {video_path}")
        raw_video, _ = video.get_frames(list(range(prefix_len)))
    finally:
        video.close()

    frames = [transform(Image.fromarray(frame[:, :, ::-1]).convert("RGB"), tgt_h, tgt_w) for frame in raw_video]
    video_T3HW = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]

def _find_routes(dataset_root: str) -> List[Tuple[str, str, str]]:
    """Return list of (route_dir, meta_json_path, images_dir)."""
    routes: List[Tuple[str, str, str]] = []
    for dirpath, _, filenames in os.walk(dataset_root):
        if "meta.json" not in filenames:
            continue
        meta_path = osp.join(dirpath, "meta.json")
        images_dir = osp.join(dirpath, "images")
        if osp.isdir(images_dir):
            routes.append((dirpath, meta_path, images_dir))
    routes.sort(key=lambda x: x[0])
    return routes


def _manifest_prompt(obj: Dict, prompt_key: str) -> str:
    prompt = obj.get(prompt_key) or obj.get("prompt") or obj.get("instruction_unified") or obj.get("instruction") or obj.get("caption") or obj.get("text")
    if not prompt:
        raise ValueError(f"manifest item missing prompt-like field (tried key={prompt_key})")
    return str(prompt).strip()


def _manifest_video_path(obj: Dict) -> str:
    video_path = obj.get("video_path") or obj.get("video") or obj.get("path")
    if not video_path:
        raise ValueError("manifest item missing video path field")
    video_path = osp.abspath(str(video_path))
    if not osp.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    return video_path


def _route_id_from_video(video_path: str, index: int) -> str:
    stem = Path(video_path).stem or "video"
    digest = hashlib.md5(video_path.encode("utf-8")).hexdigest()[:8]
    return f"{index:06d}_{stem}_{digest}"


def _load_manifest_routes(manifest_json: str, prompt_key: str) -> List[Tuple[str, str, str]]:
    manifest_json = osp.abspath(manifest_json)
    with open(manifest_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"manifest must be a JSON array: {manifest_json}")

    routes: List[Tuple[str, str, str]] = []
    bad_cnt = 0
    bad_examples: List[str] = []
    for i, obj in enumerate(payload):
        if not isinstance(obj, dict):
            bad_cnt += 1
            continue
        try:
            video_path = _manifest_video_path(obj)
            prompt = _manifest_prompt(obj, prompt_key)
        except Exception as e:
            bad_cnt += 1
            if len(bad_examples) < 5:
                bad_examples.append(f"idx={i} err={e}")
            continue
        route_id = _route_id_from_video(video_path, i)
        routes.append((route_id, video_path, prompt))
    print(f"[manifest] loaded={len(routes)} dropped={bad_cnt} from {manifest_json}")
    for ex in bad_examples:
        print(f"[manifest][drop] {ex}")
    return routes

def _dist_info() -> Tuple[int, int, int]:
    """(rank, world_size, local_rank) without initializing NCCL."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Finetuned checkpoint (.pth)")
    ap.add_argument("--dataset_root", type=str, default="", help="Batch: dataset root containing many routes")
    ap.add_argument("--manifest_json", type=str, default="", help="Batch: JSON array with video_path/video and prompt fields")
    ap.add_argument("--images_dir", type=str, default="", help="Single route: directory of ordered frames (frame_*.png)")
    ap.add_argument("--meta_json", type=str, default="", help="Single route: meta.json containing instruction/instruction_unified")
    ap.add_argument("--video_path", type=str, default="", help="Single route: mp4 path when not using images_dir")
    ap.add_argument("--prompt", type=str, default="", help="Single route: prompt text when not using meta_json")
    ap.add_argument("--prompt_key", type=str, default="instruction_unified")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--pn", type=str, default="0.40M")
    ap.add_argument("--h_div_w_template", type=float, default=1.0, help="1.0 => 640x640 for pn=0.40M in clip4 schedule")
    ap.add_argument(
        "--require_tgt_hw",
        type=int,
        nargs=2,
        default=(640, 640),
        metavar=("H", "W"),
        help="Assert model target resolution equals (H,W). Use 0 0 to disable.",
    )
    ap.add_argument("--min_images", type=int, default=33, help="Batch: only routes with image_count > min_images")
    ap.add_argument("--skip_existing", action="store_true", help="Batch: skip route if seg02 output exists")
    ap.add_argument("--max_routes", type=int, default=-1, help="Batch: limit number of routes (debug)")
    args_cli = ap.parse_args()

    ckpt = osp.abspath(args_cli.ckpt)
    out_dir = osp.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Bind each torchrun worker to its own GPU *before* loading any model.
    rank, world_size, local_rank = _dist_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"[dist] rank={rank} world_size={world_size} local_rank={local_rank} device={device}")

    # ----------------------
    # Build inference args (clip=16 => frames_inner_clip=4 in compressed timeline)
    # ----------------------
    a = Args()
    a.pn = str(args_cli.pn)
    a.fps = int(args_cli.fps)
    a.video_fps = int(args_cli.fps)
    a.video_frames = int(args_cli.num_frames)
    a.temporal_compress_rate = 4
    a.frames_inner_clip = 4  # clip16 / compress4
    a.videovae = 10
    a.vae_type = 64

    a.dynamic_scale_schedule = "infinity_elegant_clip4frames_v2_allpt"
    a.mask_type = "infinity_elegant_clip4frames_v2_allpt"

    # Inference knobs (follow official infer_video_480p.py style)
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

    # paths
    a.model_type = "infinity_qwen8b"
    a.model_path = ckpt
    a.checkpoint_type = "torch"
    a.vae_path = osp.join(REPO_ROOT, "checkpoint", "infinitystar_videovae.pth")
    a.text_encoder_ckpt = osp.join(REPO_ROOT, "checkpoint", "text_encoder", "flan-t5-xl-official")
    a.text_channels = 2048

    # ----------------------
    # Build schedule for num_frames
    # ----------------------
    dyn_res, h_div_w_templates = get_dynamic_resolution_meta(a.dynamic_scale_schedule, a.video_frames)
    htpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - float(args_cli.h_div_w_template)))]
    pt_full = (a.video_frames - 1) // a.temporal_compress_rate + 1
    scale_schedule = dyn_res[htpl][a.pn]["pt2scale_schedule"][pt_full]
    a.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    a.tower_split_index = a.first_full_spatial_size_scale_index + 1
    video_encode, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(a.dynamic_scale_schedule)
    context_info = get_scale_pack_info(scale_schedule, a.first_full_spatial_size_scale_index, a)
    tau = [a.tau_image] * a.tower_split_index + [a.tau_video] * (len(scale_schedule) - a.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    req_h, req_w = int(args_cli.require_tgt_hw[0]), int(args_cli.require_tgt_hw[1])
    if req_h > 0 and req_w > 0 and (int(tgt_h), int(tgt_w)) != (req_h, req_w):
        raise ValueError(
            f"Target resolution mismatch: got {(int(tgt_h), int(tgt_w))} "
            f"from template={htpl}, but require_tgt_hw={(req_h, req_w)}. "
            f"Tip: for 640x640 use --h_div_w_template 1.0"
        )

    # ----------------------
    # Load models
    # ----------------------
    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)
    vae = load_visual_tokenizer(a).float().to(device)
    infinity = load_transformer(vae, a)
    self_correction = SelfCorrection(vae, a)

    def _infer_with_prefix(*, frame_paths: List[str] | None = None, video_path: str | None = None, prompt_infer: str, prefix_len: int, gt_leak: int, seed: int) -> np.ndarray:
        if frame_paths is not None:
            prefix_paths = frame_paths[:prefix_len]
            obs_bcthw = _load_obs_video_bcthw(prefix_paths, tgt_h, tgt_w).to(device, non_blocking=True)
        elif video_path is not None:
            obs_bcthw = _load_obs_video_from_mp4_bcthw(video_path, prefix_len, tgt_h, tgt_w).to(device, non_blocking=True)
        else:
            raise ValueError("Either frame_paths or video_path must be provided")
        with torch.no_grad():
            _, _, gt_ls_Bl, _, _, _ = video_encode(
                vae=vae,
                inp_B3HW=obs_bcthw,
                vae_features=None,
                self_correction=self_correction,
                args=a,
                infer_mode=True,
                dynamic_resolution_h_w=dyn_res,
            )
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            video_uint8, _ = gen_one_example(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt_infer,
                negative_prompt="",
                g_seed=int(seed),
                gt_leak=int(gt_leak),
                gt_ls_Bl=gt_ls_Bl,
                cfg_list=a.cfg,
                tau_list=tau,
                scale_schedule=scale_schedule,
                top_k=900,
                top_p=0.97,
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
        if isinstance(video_uint8, torch.Tensor) and video_uint8.dim() == 5:
            video_uint8 = video_uint8[0]
        vid_np = video_uint8.detach().cpu().numpy() if isinstance(video_uint8, torch.Tensor) else np.asarray(video_uint8)
        return vid_np

    def _auto_leak_for_prefix(prefix_len: int) -> int:
        pt_obs = (int(prefix_len) - 1) // int(a.temporal_compress_rate) + 1
        sched_obs = dyn_res[htpl][a.pn]["pt2scale_schedule"][pt_obs]
        return int(len(sched_obs))

    def _run_one(*, route_id: str, out_route_dir: str, global_idx: int, meta_json: str = "", images_dir: str = "", video_path: str = "", prompt: str = "") -> None:
        frame_paths: List[str] | None = None
        if images_dir:
            frame_paths = _sorted_images(images_dir)
            if len(frame_paths) < 33:
                raise ValueError(f"need at least 33 frames for seg02 prefix, but only got {len(frame_paths)}")
            prompt = _read_prompt(meta_json, args_cli.prompt_key)
        else:
            if not video_path:
                raise ValueError("video_path is required when images_dir is empty")
            if not prompt:
                raise ValueError("prompt is required when using video_path input")
        dur_s = (a.video_frames - 1) // int(a.fps)
        prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if int(getattr(a, "append_duration2caption", 0)) else prompt
        os.makedirs(out_route_dir, exist_ok=True)
        out_seg2 = osp.join(out_route_dir, "seg02_v2v_pred_034_049.mp4")
        if bool(args_cli.skip_existing) and osp.exists(out_seg2):
            return
        seed = int(args_cli.seed) + int(global_idx)
        run_args = {
            "ckpt": ckpt,
            "route_id": route_id,
            "images_dir": images_dir,
            "meta_json": meta_json,
            "video_path": video_path,
            "prompt_key": str(args_cli.prompt_key),
            "prompt": prompt,
            "prompt_infer": prompt_infer,
            "seed": int(seed),
            "fps": int(a.fps),
            "num_frames": int(a.video_frames),
            "dynamic_scale_schedule": str(a.dynamic_scale_schedule),
            "mask_type": str(a.mask_type),
            "frames_inner_clip": int(a.frames_inner_clip),
            "h_div_w_template": float(htpl),
            "tgt_h": int(tgt_h),
            "tgt_w": int(tgt_w),
        }
        with open(osp.join(out_route_dir, "run_args.json"), "w", encoding="utf-8") as f:
            json.dump(run_args, f, ensure_ascii=False, indent=2)

        # seg00: I2V
        vid0 = _infer_with_prefix(frame_paths=frame_paths, video_path=video_path or None, prompt_infer=prompt_infer, prefix_len=1, gt_leak=14, seed=seed)
        save_video(vid0[1:17], fps=int(a.fps), save_filepath=osp.join(out_route_dir, "seg00_i2v_pred_002_017.mp4"), force_all_keyframes=True)

        # seg01: V2V with prefix 1..17
        leak17 = _auto_leak_for_prefix(17)
        vid1 = _infer_with_prefix(frame_paths=frame_paths, video_path=video_path or None, prompt_infer=prompt_infer, prefix_len=17, gt_leak=leak17, seed=seed)
        save_video(vid1[17:33], fps=int(a.fps), save_filepath=osp.join(out_route_dir, "seg01_v2v_pred_018_033.mp4"), force_all_keyframes=True)

        # seg02: V2V with prefix 1..33
        leak33 = _auto_leak_for_prefix(33)
        vid2 = _infer_with_prefix(frame_paths=frame_paths, video_path=video_path or None, prompt_infer=prompt_infer, prefix_len=33, gt_leak=leak33, seed=seed)
        save_video(vid2[33:49], fps=int(a.fps), save_filepath=out_seg2, force_all_keyframes=True)

    t0 = time.time()

    if args_cli.dataset_root:
        dataset_root = osp.abspath(args_cli.dataset_root)
        routes_all = _find_routes(dataset_root)
        # filter by image count strictly greater than min_images
        routes = []
        for route_dir, meta_json, images_dir in routes_all:
            n = len(_sorted_images(images_dir))
            if n > int(args_cli.min_images):
                routes.append((route_dir, meta_json, images_dir))
        if int(args_cli.max_routes) > 0:
            routes = routes[: int(args_cli.max_routes)]
        my_routes = list(enumerate(routes))[rank::world_size]
        for global_idx, (route_dir, meta_json, images_dir) in my_routes:
            route_id = osp.basename(route_dir.rstrip("/"))
            out_route_dir = osp.join(out_dir, route_id)
            try:
                _run_one(route_dir, meta_json, images_dir, out_route_dir, global_idx=global_idx)
            except Exception as e:
                print(f"[FAIL] route={route_dir} err={e}")
        dt = time.time() - t0
        print(f"[done] batch out_dir={out_dir} routes_total={len(routes)} elapsed_s={dt:.1f}")
        return

    if args_cli.manifest_json:
        routes = _load_manifest_routes(args_cli.manifest_json, args_cli.prompt_key)
        if int(args_cli.max_routes) > 0:
            routes = routes[: int(args_cli.max_routes)]
        my_routes = list(enumerate(routes))[rank::world_size]
        for global_idx, (route_id, video_path, prompt) in my_routes:
            out_route_dir = osp.join(out_dir, route_id)
            try:
                _run_one(
                    route_id=route_id,
                    out_route_dir=out_route_dir,
                    global_idx=global_idx,
                    video_path=video_path,
                    prompt=prompt,
                )
            except Exception as e:
                print(f"[FAIL] route_id={route_id} video={video_path} err={e}")
        dt = time.time() - t0
        print(f"[done] manifest batch out_dir={out_dir} routes_total={len(routes)} elapsed_s={dt:.1f}")
        return

    if args_cli.video_path:
        route_id = Path(args_cli.video_path).stem or "route"
        out_route_dir = out_dir
        prompt = args_cli.prompt or _read_prompt(args_cli.meta_json, args_cli.prompt_key)
        _run_one(
            route_id=route_id,
            out_route_dir=out_route_dir,
            global_idx=0,
            video_path=osp.abspath(args_cli.video_path),
            prompt=prompt,
            meta_json=args_cli.meta_json,
        )
        dt = time.time() - t0
        print(f"[done] single-video out_dir={out_route_dir} elapsed_s={dt:.1f}")
        return

    if not args_cli.images_dir or not args_cli.meta_json:
        raise ValueError("Provide --dataset_root, or --manifest_json, or both --images_dir and --meta_json, or --video_path with --prompt/--meta_json.")

    images_dir = osp.abspath(args_cli.images_dir)
    meta_json = osp.abspath(args_cli.meta_json)
    route_id = osp.basename(osp.dirname(images_dir.rstrip("/"))) or "route"
    out_route_dir = out_dir
    _run_one(route_id=route_id, meta_json=meta_json, images_dir=images_dir, out_route_dir=out_route_dir, global_idx=0)
    dt = time.time() - t0
    print(f"[done] single out_dir={out_route_dir} elapsed_s={dt:.1f}")


if __name__ == "__main__":
    main()

