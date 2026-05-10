#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
UAV rollout demo (InfinityStar-Interact, modified granularity: 20 predicted pixel frames per clip).

你要求的流程（总共到 81 帧像素）：
1) 输入第 0 帧真实图 + prompt -> 预测 20 帧，保存为 1.mp4
2) 以 (第 1~20 帧) 的上下文继续预测后 20 帧，保存为 2.mp4
3) 把第 20~40 帧替换为真实帧，再继续预测后 20 帧，保存为 3.mp4
4) 把第 40~60 帧替换为真实帧，再继续预测后 20 帧，保存为 4.mp4
=> 得到 1 + 4*20 = 81 帧（frame_000000 ~ frame_000080）

注意：
- 本脚本依赖你已经把 Interact 结构改成 frames_inner_clip=5（即每次输出 20 帧像素新预测）这一套代码。
- 第二步“输入第 1~20 帧”在实现上等价于直接使用上一次推理返回的 latent（更准确、更快），不需要再把像素 encode 一遍。
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import sys
from typing import List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.utils.arg_util import Args
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, transform, save_video


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


def _load_frames_tensor(frame_paths: List[str], tgt_h: int, tgt_w: int) -> torch.Tensor:
    # returns [1,3,T,H,W] in [-1,1]
    frames = []
    for p in frame_paths:
        pil = Image.open(p).convert("RGB")
        frame = transform(pil, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
        frames.append(frame)
    video_T3HW = torch.stack(frames, dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]


def main():
    # ---- dataset (user requested) ----
    data_root = "/home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_12-04-05"
    images_dir = osp.join(data_root, "images")
    meta_path = osp.join(data_root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    prompt = meta.get("instruction", "Ascend to an altitude of 8.0 meters")

    frame_paths = _sorted_frame_paths(images_dir)
    # This dataset only has ~40 frames; we will run fewer clips accordingly.
    if len(frame_paths) < 21:
        raise ValueError(f"need at least 21 frames (0..20) to form the first observation clip, got {len(frame_paths)}")

    # ---- args (Interact, clip=20 predicted pixel frames) ----
    args = Args()
    ckpt_dir = osp.join(REPO_ROOT, "checkpoint")

    args.pn = "0.40M"
    args.fps = 16
    args.video_frames = 81  # provide enough RoPE capacity; rollout length is 81 frames total
    args.temporal_compress_rate = 4
    args.frames_inner_clip = 5  # 5 compressed frames -> 20 pixel predicted frames
    args.context_interval = 2
    args.context_from_largest_no = 1
    args.context_frames = 1000
    args.steps_per_frame = 3

    args.model_type = "infinity_qwen8b"
    args.model_path = osp.join(ckpt_dir, "InfinityStarInteract_24K_iters")
    args.checkpoint_type = "torch_shard"

    args.vae_path = osp.join(ckpt_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(ckpt_dir, "text_encoder", "flan-t5-xl-official")
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
    args.tau = [args.taui] * len(json.loads(args.image_scale_repetition)) + [args.tauv] * len(json.loads(args.video_scale_repetition))

    # ---- build one-clip schedule (pt = 1 + frames_inner_clip = 6) ----
    dynamic_resolution_h_w, _ = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))

    first_img = np.array(Image.open(frame_paths[0]).convert("RGB"))
    h0, w0 = first_img.shape[:2]
    h_div_w_template_ = h_div_w_template_list[np.argmin(np.abs(h0 / w0 - h_div_w_template_list))]

    pt_one_clip = 1 + args.frames_inner_clip  # 6
    scale_schedule_full = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][pt_one_clip]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule_full)
    scales_in_one_clip = args.first_full_spatial_size_scale_index + 1
    cur_scale_schedule = scale_schedule_full[scales_in_one_clip:]  # drop image scales, keep video scales for one clip

    _, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    context_info = get_scale_pack_info(cur_scale_schedule, args.first_full_spatial_size_scale_index, args)

    tgt_h, tgt_w = scale_schedule_full[-1][1] * 16, scale_schedule_full[-1][2] * 16

    # ---- load models ----
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to("cuda")
    infinity = load_transformer(vae, args)

    out_dir = osp.join(REPO_ROOT, "output", "uav_rollout_20f_clips_2025-03-30_12-04-05")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Step 0: build initial conditioning from REAL observation clip (0..20) ----
    # This avoids the "static video" failure mode caused by repeating the first frame.
    obs0_paths = frame_paths[0:21]
    obs0_bcthw = _load_frames_tensor(obs0_paths, tgt_h, tgt_w).to("cuda")
    former_clip_features, _, _ = vae.encode_for_raw_features(obs0_bcthw, scale_schedule=None, slice=True)  # [1,64,6,h,w]
    first_frame_features = former_clip_features[:, :, 0:1]

    prompt_for_model = prompt
    if args.append_duration2caption:
        # The prompt tag is still "t=5s" in released scripts; keep it unchanged to avoid surprising side-effects.
        prompt_for_model = f"<<<t=5s>>>{prompt_for_model}"

    def run_one_clip(clip_idx_1based: int, seed: int, former_features: torch.Tensor) -> torch.Tensor:
        """Return updated former features (summed_codes) for chaining."""
        pred_video, updated_former = gen_one_example(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt_for_model,
            negative_prompt="",
            g_seed=seed,
            gt_leak=-1,
            gt_ls_Bl=None,
            cfg_list=args.cfg,
            tau_list=args.tau,
            scale_schedule=cur_scale_schedule,
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits if hasattr(args, "sampling_per_bits") else 1,
            enable_positive_prompt=False,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
            mode="second_v_clip",
            former_clip_features=former_features,
            first_frame_features=first_frame_features,
        )
        if isinstance(pred_video, torch.Tensor):
            pred_video_np = pred_video.detach().cpu().numpy()
        else:
            pred_video_np = np.asarray(pred_video)
        # pred_video is expected to be exactly 20 frames after our model-side cropping
        save_video(pred_video_np, fps=args.fps, save_filepath=osp.join(out_dir, f"{clip_idx_1based}.mp4"))
        return updated_former

    # Clip 1: predict frames 21..40 (20 frames) conditioned on REAL 0..20
    former_clip_features = run_one_clip(clip_idx_1based=1, seed=0, former_features=former_clip_features)

    # Clip 2: continue from predicted clip1 (use returned latent directly)
    former_clip_features = run_one_clip(clip_idx_1based=2, seed=1, former_features=former_clip_features)

    # Clip 3: replace 20..40 with REAL frames (if available; pad if not) then predict next 20 frames
    ss, ee = 20, 40
    real_paths = frame_paths[ss : min(ee + 1, len(frame_paths))]  # may be shorter than 21 if dataset ends early
    if not real_paths:
        print("[warn] no real frames available for replacement 20..40; skipping clip 3 replacement.")
    else:
        # pad to 21 frames by repeating the last available frame
        if len(real_paths) < 21:
            real_paths = real_paths + [real_paths[-1]] * (21 - len(real_paths))
        real_bcthw = _load_frames_tensor(real_paths, tgt_h, tgt_w).to("cuda")
        former_clip_features, _, _ = vae.encode_for_raw_features(real_bcthw, scale_schedule=None, slice=True)  # [1,64,6,h,w]
        former_clip_features = run_one_clip(clip_idx_1based=3, seed=2, former_features=former_clip_features)

    print(f"[ok] prompt={prompt}")
    print(f"[ok] saved clips to: {osp.abspath(out_dir)}")


if __name__ == "__main__":
    main()

