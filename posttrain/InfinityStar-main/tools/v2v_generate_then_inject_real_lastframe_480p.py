#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Two-stage V2V pipeline (480p):
1) Generate a 5s clip (81 frames @ 16fps) from (first frame + prompt)  [I2V in practice].
2) Replace the last frame of that generated clip with a real frame.
3) Use the edited 5s clip as the conditioning video to generate a 10s clip (official continuation mode),
   and save the last 5s as "continued" result.

Important limitation:
InfinityStar does NOT provide a native "force the boundary frame equals the injected real frame" constraint.
This method only makes the injected frame part of the conditioning video, which usually helps continuity but is not a hard guarantee.
"""

import argparse
import glob
import json
import os
import os.path as osp
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure repo root is importable (so `import infinity` works).
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from infinity.utils.arg_util import Args
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.video_decoder import EncodedVideoDecord
from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_example,
    save_video,
    transform,
)


def _sorted_frames(frames_dir: str) -> list[str]:
    fs = glob.glob(osp.join(frames_dir, "frame_*.png"))
    if not fs:
        raise FileNotFoundError(f"No frames found under: {frames_dir}")

    def key(p: str) -> int:
        m = re.search(r"frame_(\d+)\.png$", p)
        return int(m.group(1)) if m else 1_000_000_000

    return sorted(fs, key=key)


def _read_prompt(meta_json_path: str) -> str:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = meta.get("instruction_unified") or meta.get("instruction")
    if not prompt:
        raise ValueError(f"No instruction(_unified) in: {meta_json_path}")
    return str(prompt).strip()


class Pipe480p:
    def __init__(self, args: Args):
        self.args = args
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args).float().to("cuda")
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        self.video_encode, _, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)


def _build_scale_schedule(dynamic_scale_schedule: str, pn: str, video_frames: int):
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(dynamic_scale_schedule, video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][pn]["pt2scale_schedule"][(video_frames - 1) // 4 + 1]
    return dynamic_resolution_h_w, scale_schedule


def _gen_first_5s_clip(pipe: Pipe480p, prompt: str, first_frame_path: str, seed: int) -> np.ndarray:
    """
    Returns uint8 ndarray [T,H,W,3] in BGR order (consistent with save_video usage in this repo).
    """
    args = pipe.args
    # IMPORTANT:
    # Do NOT shrink args.video_frames here. The model's RoPE cache is precomputed at init time
    # based on args.video_frames. If we later generate longer videos, shrinking here can lead to
    # out-of-range RoPE frame indices (empty slices) and runtime errors.
    dynamic_resolution_h_w, scale_schedule = _build_scale_schedule(args.dynamic_scale_schedule, args.pn, 81)
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)

    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16

    # Encode first frame as I2V leakage (same as official infer_video_480p.py image_path branch)
    ref_bgr = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    if ref_bgr is None:
        raise ValueError(f"Failed to read first_frame_path: {first_frame_path}")
    ref_rgb = ref_bgr[:, :, ::-1]
    ref_img_T3HW = [transform(Image.fromarray(ref_rgb).convert("RGB"), tgt_h, tgt_w)]
    ref_img_T3HW = torch.stack(ref_img_T3HW, 0)  # [t,3,h,w]
    ref_img_bcthw = ref_img_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [b,c,t,h,w]
    _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(
        pipe.vae,
        ref_img_bcthw.cuda(),
        vae_features=None,
        self_correction=pipe.self_correction,
        args=args,
        infer_mode=True,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
    )
    gt_leak = 14

    # Prompt engineering (same as official scripts)
    mapped_duration = 5
    prompt2 = f"{prompt}, Close-up on big objects, emphasize scale and detail"
    if args.append_duration2caption:
        prompt2 = f"<<<t={mapped_duration}s>>>" + prompt2

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        video_uint8, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt2,
            negative_prompt="",
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )
    # video_uint8: torch uint8 [1,T,H,W,3] (BGR)
    if video_uint8.dim() == 5:
        video_uint8 = video_uint8[0]
    return video_uint8.cpu().numpy()


def _inject_real_last_frame(gen_bgr: np.ndarray, real_frame_path: str) -> np.ndarray:
    if gen_bgr.ndim != 4 or gen_bgr.shape[-1] != 3:
        raise ValueError(f"Unexpected gen video shape: {gen_bgr.shape}")
    real = cv2.imread(real_frame_path, cv2.IMREAD_COLOR)
    if real is None:
        raise ValueError(f"Failed to read real_frame_path: {real_frame_path}")
    h, w = gen_bgr.shape[1:3]
    if real.shape[0] != h or real.shape[1] != w:
        real = cv2.resize(real, (w, h), interpolation=cv2.INTER_AREA)
    out = gen_bgr.copy()
    out[-1] = real
    return out


def _continue_from_cond_5s(pipe: Pipe480p, prompt: str, cond_video_path: str, seed: int) -> np.ndarray:
    """
    Official continuation behavior: encode first 5s, generate a 10s video (161 frames).
    Returns uint8 ndarray [T,H,W,3] BGR.
    """
    args = pipe.args
    # Use the 10s schedule, but keep args.video_frames as the max value set at initialization.
    dynamic_resolution_h_w, scale_schedule = _build_scale_schedule(args.dynamic_scale_schedule, args.pn, 161)
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)

    # Encode conditioning 5s clip (81 frames)
    video = EncodedVideoDecord(cond_video_path, osp.basename(cond_video_path), num_threads=0)
    if video._duration < 5:
        raise ValueError(f"Condition video must be >=5 seconds, got {video._duration:.3f}s: {cond_video_path}")
    # Condition clip should be resized using the 5s schedule (81 frames).
    _, condition_scale_schedule = _build_scale_schedule(args.dynamic_scale_schedule, args.pn, 81)
    cond_tgt_h, cond_tgt_w = condition_scale_schedule[-1][1] * 16, condition_scale_schedule[-1][2] * 16
    raw_video, _ = video.get_clip(0, 5, 81)
    video_T3HW = [transform(Image.fromarray(frame).convert("RGB"), cond_tgt_h, cond_tgt_w) for frame in raw_video]
    video_T3HW = torch.stack(video_T3HW, 0)
    video_bcthw = video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)
    _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(
        pipe.vae,
        video_bcthw.cuda(),
        vae_features=None,
        self_correction=pipe.self_correction,
        args=args,
        infer_mode=True,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
    )
    gt_leak = 28

    mapped_duration = 10
    prompt2 = f"{prompt}, Close-up on big objects, emphasize scale and detail"
    if args.append_duration2caption:
        prompt2 = f"<<<t={mapped_duration}s>>>" + prompt2

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        video_uint8, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt2,
            negative_prompt="",
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )
    if video_uint8.dim() == 5:
        video_uint8 = video_uint8[0]
    return video_uint8.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--meta_json", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, default=osp.join(_REPO_ROOT, "checkpoint"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--inject_index", type=int, default=80, help="Use frame_{inject_index}.png as the real replacement for the generated last frame.")
    parser.add_argument("--out_dir", type=str, default=osp.join(_REPO_ROOT, "output", "v2v_inject_real_lastframe_480p"))
    parser.add_argument(
        "--only_first_5s",
        action="store_true",
        help="Only generate and save the first 5s (81 frames) clip (and injected cond clip), then exit.",
    )
    args_cli = parser.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)
    frames = _sorted_frames(args_cli.frames_dir)
    if args_cli.inject_index >= len(frames):
        raise ValueError(f"inject_index out of range: {args_cli.inject_index} >= {len(frames)}")
    first_frame = frames[0]
    real_last = frames[args_cli.inject_index]
    prompt = _read_prompt(args_cli.meta_json)

    # Build 480p inference args (match tools/infer_video_480p.py defaults)
    inf = Args()
    inf.pn = "0.40M"
    inf.fps = 16
    # CRITICAL: initialize model with the maximum video length we will ever request (10s / 161 frames),
    # so RoPE caches have enough frame capacity for continuation.
    inf.video_frames = 161
    inf.model_path = osp.join(args_cli.checkpoints_dir, "infinitystar_8b_480p_weights")
    inf.checkpoint_type = "torch_shard"
    inf.vae_path = osp.join(args_cli.checkpoints_dir, "infinitystar_videovae.pth")
    inf.text_encoder_ckpt = osp.join(args_cli.checkpoints_dir, "text_encoder", "flan-t5-xl-official")
    inf.videovae = 10
    inf.model_type = "infinity_qwen8b"
    inf.text_channels = 2048
    inf.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    inf.bf16 = 1
    inf.use_apg = 1
    inf.use_cfg = 0
    inf.cfg = 34
    inf.tau_image = 1
    inf.tau_video = 0.4
    inf.apg_norm_threshold = 0.05
    inf.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    inf.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    inf.append_duration2caption = 1
    inf.use_two_stage_lfq = 1
    inf.detail_scale_min_tokens = 350
    inf.semantic_scales = 11
    inf.max_repeat_times = 10000
    inf.enable_rewriter = 0
    inf.vae_type = 64

    pipe = Pipe480p(inf)

    # Stage 1: generate 5s clip
    gen5 = _gen_first_5s_clip(pipe, prompt=prompt, first_frame_path=first_frame, seed=args_cli.seed)
    out_gen5 = osp.join(args_cli.out_dir, "gen_first_5s.mp4")
    save_video(gen5, fps=inf.fps, save_filepath=out_gen5)

    # Stage 2: inject real last frame
    gen5_injected = _inject_real_last_frame(gen5, real_frame_path=real_last)
    out_cond = osp.join(args_cli.out_dir, f"cond_gen5_inject_real_{args_cli.inject_index:06d}.mp4")
    save_video(gen5_injected, fps=inf.fps, save_filepath=out_cond)

    if args_cli.only_first_5s:
        print(f"[ok] prompt={prompt}")
        print(f"[ok] first_frame={first_frame}")
        print(f"[ok] real_last_frame={real_last}")
        print(f"[ok] gen_first_5s={osp.abspath(out_gen5)}")
        print(f"[ok] cond_injected_5s={osp.abspath(out_cond)}")
        return

    # Stage 3: continue to 10s
    full10 = _continue_from_cond_5s(pipe, prompt=prompt, cond_video_path=out_cond, seed=args_cli.seed)
    out_full10 = osp.join(args_cli.out_dir, "gen_full_10s.mp4")
    save_video(full10, fps=inf.fps, save_filepath=out_full10)

    cont5 = full10[81:]
    out_cont5 = osp.join(args_cli.out_dir, "gen_continuation_last_5s.mp4")
    save_video(cont5, fps=inf.fps, save_filepath=out_cont5)

    print(f"[ok] prompt={prompt}")
    print(f"[ok] first_frame={first_frame}")
    print(f"[ok] real_last_frame={real_last}")
    print(f"[ok] gen_first_5s={osp.abspath(out_gen5)}")
    print(f"[ok] cond_injected_5s={osp.abspath(out_cond)}")
    print(f"[ok] gen_full_10s={osp.abspath(out_full10)}")
    print(f"[ok] gen_cont_last_5s={osp.abspath(out_cont5)}")


if __name__ == "__main__":
    main()

