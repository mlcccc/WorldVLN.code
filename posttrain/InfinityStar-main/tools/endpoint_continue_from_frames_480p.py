#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Endpoint-conditioned (approx) continuation demo using a frame sequence as "reference video".

NOTE:
- InfinityStar official inference supports:
  - I2V: condition on a single first frame
  - V2V continuation: condition on a 5s clip then generate a 10s clip (480p)
- It does NOT natively support conditioning on (first frame, target frame).

This script implements a practical approximation:
1) Take frame_000000.png and frame_{target}.png from a frame sequence.
2) Linearly interpolate them into a 5s conditioning clip (81 frames @ 16 fps).
3) Run 480p "video continuation" generation for 10s (161 frames).
4) Save only the last 5s (frames 81..160) as the "fixed-n-frames" output segment.
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

# Ensure repo root is importable (so `import infinity` works).
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from infinity.utils.arg_util import Args
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.video_decoder import EncodedVideoDecord
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video


def _sorted_frames(frames_dir: str) -> list[str]:
    fs = glob.glob(osp.join(frames_dir, "frame_*.png"))
    if not fs:
        raise FileNotFoundError(f"No frames found under: {frames_dir}")

    def key(p: str) -> int:
        m = re.search(r"frame_(\d+)\.png$", p)
        return int(m.group(1)) if m else 1_000_000_000

    fs = sorted(fs, key=key)
    return fs


def _read_prompt(meta_json_path: str) -> str:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Prefer unified English instruction if present.
    prompt = meta.get("instruction_unified") or meta.get("instruction")
    if not prompt:
        raise ValueError(f"No prompt field found in meta.json: {meta_json_path}")
    return str(prompt).strip()


def _make_interp_cond_video(
    first_frame_path: str,
    target_frame_path: str,
    out_mp4: str,
    fps: int = 16,
    num_frames: int = 81,
) -> str:
    a = cv2.imread(first_frame_path, cv2.IMREAD_COLOR)
    b = cv2.imread(target_frame_path, cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f"Failed to read: {first_frame_path}")
    if b is None:
        raise ValueError(f"Failed to read: {target_frame_path}")
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    os.makedirs(osp.dirname(out_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (a.shape[1], a.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    # num_frames == 81 means 0..80 inclusive
    denom = max(1, num_frames - 1)
    for i in range(num_frames):
        alpha = i / denom
        frame = cv2.addWeighted(a, 1 - alpha, b, alpha, 0.0)
        writer.write(frame)
    writer.release()
    return out_mp4


class InferencePipe:
    def __init__(self, args: Args):
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args).float().to("cuda")
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(
            args.dynamic_scale_schedule
        )


def _run_continuation_480p(pipe: InferencePipe, prompt: str, cond_video_path: str, seed: int, args: Args) -> np.ndarray:
    """
    Returns a uint8 numpy array shaped [T,H,W,3] for 10s (161 frames) generation.
    """
    mapped_duration = 10
    num_frames = mapped_duration * args.fps + 1  # 161 at 16 fps

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][(num_frames - 1) // 4 + 1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)

    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)

    # Condition: encode first 5 seconds (81 frames) of the input video
    video = EncodedVideoDecord(cond_video_path, osp.basename(cond_video_path), num_threads=0)
    duration = video._duration
    if duration < 5:
        raise ValueError(f"Condition video must be >=5 seconds, got {duration:.3f}s: {cond_video_path}")

    condition_scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][(81 - 1) // 4 + 1]
    cond_tgt_h, cond_tgt_w = condition_scale_schedule[-1][1] * 16, condition_scale_schedule[-1][2] * 16
    raw_video, _ = video.get_clip(0, 5, 81)
    from PIL import Image
    from tools.run_infinity import transform

    video_T3HW = [transform(Image.fromarray(frame).convert("RGB"), cond_tgt_h, cond_tgt_w) for frame in raw_video]
    video_T3HW = torch.stack(video_T3HW, 0)  # [t,3,h,w]
    video_bcthw = video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [b,c,t,h,w]
    _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(
        pipe.vae,
        video_bcthw.cuda(),
        vae_features=None,
        self_correction=pipe.self_correction,
        args=args,
        infer_mode=True,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
    )
    gt_leak = 28  # same as official infer_video_480p.py (video continuation path)

    # Prompt engineering consistent with official scripts
    prompt = f"{prompt}, Close-up on big objects, emphasize scale and detail"
    if args.append_duration2caption:
        prompt = f"<<<t={mapped_duration}s>>>" + prompt

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
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
    # generated: torch uint8 [1, T, H, W, 3]
    if generated.dim() == 5:
        generated = generated[0]
    return generated.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--meta_json", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, default=osp.join(_REPO_ROOT, "checkpoint"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--n_frames", type=int, default=80, help="Output fixed n frames (we will generate 10s and take last 5s).")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default=osp.join(_REPO_ROOT, "output", "endpoint_iter_demo_480p"))
    args_cli = parser.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)

    frames = _sorted_frames(args_cli.frames_dir)
    prompt = _read_prompt(args_cli.meta_json)

    # Choose indices: 0 and (n_frames) => "frame n+1" in 1-based terms
    target_idx = args_cli.n_frames
    if target_idx >= len(frames):
        raise ValueError(f"Need at least {target_idx+1} frames, but got {len(frames)}")

    first_frame = frames[0]
    target_frame = frames[target_idx]

    cond_mp4 = osp.join(args_cli.out_dir, f"cond_interp_000000_to_{target_idx:06d}.mp4")
    _make_interp_cond_video(first_frame, target_frame, cond_mp4, fps=args_cli.fps, num_frames=args_cli.n_frames + 1)

    # Build inference args (match tools/infer_video_480p.py defaults)
    inf = Args()
    inf.pn = "0.40M"
    inf.fps = args_cli.fps
    inf.video_frames = 161  # 10s @ 16fps + 1
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

    pipe = InferencePipe(inf)
    full10 = _run_continuation_480p(pipe, prompt=prompt, cond_video_path=cond_mp4, seed=args_cli.seed, args=inf)

    # Split: first 5s (0..80) is condition-ish, last 5s (81..160) is continuation
    cont5 = full10[81:]
    out_full = osp.join(args_cli.out_dir, "gen_full_10s.mp4")
    out_cont = osp.join(args_cli.out_dir, f"gen_continuation_{args_cli.n_frames}frames.mp4")
    save_video(full10, fps=args_cli.fps, save_filepath=out_full)
    save_video(cont5, fps=args_cli.fps, save_filepath=out_cont)

    print(f"[ok] prompt={prompt}")
    print(f"[ok] first_frame={first_frame}")
    print(f"[ok] target_frame={target_frame}")
    print(f"[ok] cond_mp4={osp.abspath(cond_mp4)}")
    print(f"[ok] out_full_10s={osp.abspath(out_full)}")
    print(f"[ok] out_cont_{args_cli.n_frames}frames={osp.abspath(out_cont)}")


if __name__ == "__main__":
    main()

