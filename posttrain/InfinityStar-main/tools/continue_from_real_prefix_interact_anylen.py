#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
任意长度真实视频前缀 -> 任意长度续写（InfinityStar-Interact 语义，480p/动态分辨率）。

重要说明（模型/脚本语义限制）：
- InfinityStar-Interact 的 `mode="second_v_clip"` 本质是“基于上一段 clip 的 latent 条件去生成下一段 clip”。
- 这里为了支持“任意长度前缀”，会自动取前缀的 **最后 81 帧像素（≈5s@16fps）** 作为条件；
  这与 Interact 设计一致，但并不能让模型“看见整个很长的历史”。若需要把长历史都注入，
  更贴近的做法是用 `tools/infinity_streaming_session.py` 逐段写 KV cache（计算更重）。
- 续写长度可以任意，但内部会以每次生成一段 clip（通常提供 80 新帧）循环生成，
  最后按 `--out_frames` 截断到你想要的续写帧数。

用法示例（帧序列）：
python tools/continue_from_real_prefix_interact_anylen.py \
  --images_dir "/path/to/images" \
  --meta_json "/path/to/meta.json" \
  --checkpoints_dir "./checkpoint" \
  --out_frames 320 \
  --out_dir "./output/uav_anylen"
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.utils.arg_util import Args
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
    transform,
)


def _sorted_frame_paths(images_dir: str) -> List[str]:
    paths = []
    for name in os.listdir(images_dir):
        if not name.endswith(".png"):
            continue
        m = re.match(r"frame_(\d+)\.png$", name)
        if not m:
            continue
        paths.append((int(m.group(1)), osp.join(images_dir, name)))
    paths.sort(key=lambda x: x[0])
    return [p for _, p in paths]


def _read_prompt(meta_json: Optional[str], prompt: Optional[str]) -> str:
    if prompt and str(prompt).strip():
        return str(prompt).strip()
    if not meta_json:
        raise ValueError("需要提供 --prompt 或 --meta_json 之一。")
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    p = meta.get("instruction_unified") or meta.get("instruction")
    if not p:
        raise ValueError(f"meta_json 里没找到 instruction(_unified): {meta_json}")
    return str(p).strip()


def _load_frames_tensor(frame_paths: List[str], tgt_h: int, tgt_w: int) -> torch.Tensor:
    # returns [1,3,T,H,W] in [-1,1]
    frames = []
    for p in frame_paths:
        pil = Image.open(p).convert("RGB")
        frame = transform(pil, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
        frames.append(frame)
    video_T3HW = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]


def _open_writer(save_path: str, fps: int, w: int, h: int) -> cv2.VideoWriter:
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {save_path}")
    return writer


def _ensure_uint8_bgr(frames: np.ndarray) -> np.ndarray:
    # Expect [T,H,W,3] uint8 BGR from repo pipelines; tolerate torch -> np conversion upstream.
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frame array shape: {frames.shape}")
    return frames


def _select_new_frames(pred_clip: np.ndarray) -> np.ndarray:
    """
    second_v_clip 常见返回：
    - T=81：第 0 帧为边界帧（用于对齐），后 80 帧为新预测
    - T=80：全是新预测
    为了稳健，统一返回“80 帧新预测”（若不足则尽量取全部）。
    """
    pred_clip = _ensure_uint8_bgr(pred_clip)
    T = int(pred_clip.shape[0])
    if T >= 81:
        return pred_clip[1:81]
    if T == 80:
        return pred_clip
    # 兜底：返回最后 80 帧或全部
    return pred_clip[-80:] if T > 80 else pred_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True, help="目录下需要包含 frame_XXXX.png")
    parser.add_argument("--meta_json", type=str, default="", help="用于读取 prompt 的 meta.json（可选）")
    parser.add_argument("--prompt", type=str, default="", help="直接指定 prompt（优先级高于 meta_json）")
    parser.add_argument("--checkpoints_dir", type=str, default=osp.join(REPO_ROOT, "checkpoint"))
    parser.add_argument("--out_dir", type=str, default=osp.join(REPO_ROOT, "output", "continue_anylen_interact"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--cond_window", type=int, default=81, help="作为条件的像素帧数（Interact 默认 81）")
    parser.add_argument("--out_frames", type=int, default=160, help="要续写的像素帧数（不含前缀）")
    parser.add_argument("--include_prefix", type=int, default=1, choices=[0, 1], help="输出视频是否包含真实前缀（1=包含）")
    args_cli = parser.parse_args()

    prompt = _read_prompt(args_cli.meta_json or None, args_cli.prompt or None)
    frame_paths = _sorted_frame_paths(args_cli.images_dir)
    if not frame_paths:
        raise FileNotFoundError(f"未找到 frame_*.png: {args_cli.images_dir}")

    # ---- Build args (align with tools/demo_interact_replace_clip_uav.py) ----
    args = Args()
    args.pn = "0.40M"
    args.fps = int(args_cli.fps)
    # 关键：Interact 的 1 个条件 clip 以 81 帧像素为基准（pt=21），我们按 clip 循环生成，不依赖更长的 RoPE 帧容量。
    args.video_frames = int(args_cli.cond_window)
    args.temporal_compress_rate = 4
    args.frames_inner_clip = 20
    args.context_interval = 2
    args.context_from_largest_no = 1
    args.context_frames = 1000
    args.steps_per_frame = 3

    args.model_type = "infinity_qwen8b"
    args.model_path = osp.join(args_cli.checkpoints_dir, "InfinityStarInteract_24K_iters")
    args.checkpoint_type = "torch_shard"

    args.vae_path = osp.join(args_cli.checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(args_cli.checkpoints_dir, "text_encoder", "flan-t5-xl-official")
    args.videovae = 10
    args.vae_type = 64
    args.vae_detail = "discrete_flow_vae"
    args.use_feat_proj = 2
    args.use_learnable_dim_proj = 0
    args.noise_input = 0
    args.input_noise = 1
    args.reduce_accumulate_error_method = "bsc"
    args.noise_apply_layers = 200
    args.noise_apply_requant = 1
    args.noise_apply_strength = 0.0

    args.dynamic_scale_schedule = "infinity_star_interact"
    args.mask_type = "infinity_star_interact"
    args.rope_type = "4d"
    args.rope2d_each_sa_layer = 1
    args.rope2d_normalized_by_hw = 2

    args.use_two_stage_lfq = 1
    args.semantic_scales = 11
    args.detail_scale_min_tokens = 350
    args.semantic_scale_dim = 16
    args.detail_scale_dim = 64
    args.num_lvl = 2
    args.num_of_label_value = args.num_lvl
    args.semantic_num_lvl = args.num_lvl
    args.detail_num_lvl = args.num_lvl

    args.use_cfg = 1
    args.use_apg = 0
    args.cfg = 3
    args.apg_norm_threshold = 0.05
    args.simple_text_proj = 1
    args.text_channels = 2048
    args.bf16 = 1
    args.append_duration2caption = 1
    args.use_clipwise_caption = 1

    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    args.video_scale_repetition = args.image_scale_repetition
    args.max_repeat_times = 1000
    args.apply_spatial_patchify = 0
    args.taui, args.tauv = 0.5, 0.5
    # tau 长度需要与 scale schedule 数量对齐（这里保持与 demo 一致）
    args.tau = [args.taui] * len(json.loads(args.image_scale_repetition)) + [args.tauv] * len(json.loads(args.video_scale_repetition))

    # ---- Build scale schedule for one clip (pt=21) ----
    dynamic_resolution_h_w, _ = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))

    # pick template by first frame aspect ratio
    first_img = np.array(Image.open(frame_paths[0]).convert("RGB"))
    h0, w0 = first_img.shape[:2]
    h_div_w_template_ = h_div_w_template_list[np.argmin(np.abs(h0 / w0 - h_div_w_template_list))]

    # condition clip pixel length -> pt
    cond_window = int(args_cli.cond_window)
    if cond_window < 2:
        raise ValueError("--cond_window 至少为 2（需要起始帧 + 后续帧）。")
    pt = (cond_window - 1) // args.temporal_compress_rate + 1
    if pt != 21:
        # 目前 Interact 的 second_v_clip 逻辑（条件用 20 帧 + 首帧）默认按 pt=21 设计。
        # 允许用户改，但需要理解会影响 schedule/条件对齐。
        raise ValueError(f"当前脚本仅支持 cond_window=81 (pt=21)，但得到 cond_window={cond_window} (pt={pt})")

    scale_schedule_full = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][pt]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule_full)
    scales_in_one_clip = args.first_full_spatial_size_scale_index + 1
    cur_scale_schedule = scale_schedule_full[scales_in_one_clip:]  # 只保留视频 scales（一个 clip）
    _, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    context_info = get_scale_pack_info(cur_scale_schedule, args.first_full_spatial_size_scale_index, args)

    tgt_h, tgt_w = scale_schedule_full[-1][1] * 16, scale_schedule_full[-1][2] * 16

    # ---- Load models ----
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to("cuda")
    infinity = load_transformer(vae, args)

    # ---- Prepare conditioning frames (last cond_window frames) ----
    if len(frame_paths) >= cond_window:
        cond_paths = frame_paths[-cond_window:]
    else:
        # pad by repeating last frame
        cond_paths = frame_paths[:]
        cond_paths += [frame_paths[-1]] * (cond_window - len(frame_paths))

    cond_bcthw = _load_frames_tensor(cond_paths, tgt_h, tgt_w).to("cuda")
    # encode: [1,64,pt=21,h,w]
    cond_latent_21, _, _ = vae.encode_for_raw_features(cond_bcthw, scale_schedule=None, slice=True)
    first_frame_features = cond_latent_21[:, :, 0:1]
    # Interact 语义：former_clip_features 期望是“后 20 帧 latent”
    former_clip_features = cond_latent_21[:, :, 1:21].contiguous()

    # ---- Writers ----
    os.makedirs(args_cli.out_dir, exist_ok=True)
    out_full = osp.join(args_cli.out_dir, "full_prefix_plus_continuation.mp4")
    out_cont = osp.join(args_cli.out_dir, "continuation_only.mp4")
    full_writer = _open_writer(out_full, fps=args.fps, w=tgt_w, h=tgt_h)
    cont_writer = _open_writer(out_cont, fps=args.fps, w=tgt_w, h=tgt_h)

    try:
        # write prefix if requested (resize to target resolution to match)
        if int(args_cli.include_prefix) == 1:
            for p in frame_paths:
                bgr = cv2.imread(p, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Failed to read: {p}")
                if bgr.shape[0] != tgt_h or bgr.shape[1] != tgt_w:
                    bgr = cv2.resize(bgr, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                full_writer.write(bgr)

        # loop generate clips until enough frames
        need = int(args_cli.out_frames)
        if need <= 0:
            print(f"[ok] out_frames={need}, no continuation generated.")
            return

        # 每段 second_v_clip 通常提供 80 帧新预测
        clips = int(np.ceil(need / 80.0))
        wrote = 0

        prompt_for_model = prompt
        if args.append_duration2caption:
            prompt_for_model = f"<<<t=5s>>>{prompt_for_model}"

        for k in range(clips):
            pred_video, updated_former = gen_one_example(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                prompt_for_model,
                negative_prompt="",
                g_seed=int(args_cli.seed) + k,
                gt_leak=-1,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=cur_scale_schedule,
                vae_type=args.vae_type,
                sampling_per_bits=getattr(args, "sampling_per_bits", 1),
                enable_positive_prompt=False,
                low_vram_mode=True,
                args=args,
                get_visual_rope_embeds=get_visual_rope_embeds,
                context_info=context_info,
                noise_list=None,
                mode="second_v_clip",
                former_clip_features=former_clip_features,
                first_frame_features=first_frame_features,
            )

            # update condition for next clip
            former_clip_features = updated_former

            if isinstance(pred_video, torch.Tensor):
                pred_video = pred_video.detach().cpu().numpy()
            pred_video = _ensure_uint8_bgr(np.asarray(pred_video))
            new_frames = _select_new_frames(pred_video)

            # trim to required length
            remaining = need - wrote
            if remaining <= 0:
                break
            if new_frames.shape[0] > remaining:
                new_frames = new_frames[:remaining]

            for fr in new_frames:
                cont_writer.write(fr)
                full_writer.write(fr)
            wrote += int(new_frames.shape[0])

        print(f"[ok] prompt={prompt}")
        print(f"[ok] prefix_frames={len(frame_paths)} (condition uses last {cond_window})")
        print(f"[ok] wrote_continuation_frames={wrote} -> requested {need}")
        print(f"[ok] out_full={osp.abspath(out_full)}")
        print(f"[ok] out_cont={osp.abspath(out_cont)}")
    finally:
        full_writer.release()
        cont_writer.release()


if __name__ == "__main__":
    main()

