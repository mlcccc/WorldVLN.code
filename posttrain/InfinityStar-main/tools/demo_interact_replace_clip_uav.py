# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Demo: "Replace clip with real observation" for InfinityStar-Interact.

目标（对齐你描述的流程 & 参考 tools/infer_interact_480p.py）：
- 先生成/执行动作得到第一个 clip 的真实观测（这里直接从 UAV PNG 序列读出来）
- 用 VAE encoder 把该 clip 编码为 latent features（former_clip_features）
- 将这个 clip “换进去”（作为条件）去推理下一段 clip（mode='second_v_clip'）

关键点（来自 InfinityStar-Interact 的 clip 设计）：
- temporal_compress_rate=4
- frames_inner_clip=20 指的是“压缩帧”长度（latent 时间维），对应像素帧约 20*4=80 帧
- 因此一个 clip 在像素空间通常用 81 帧（包含起始帧），latent 时间长度为 21（1 + 20）

输出：
- 保存下一段预测视频到 output/uav_orbit_interact_replace_clip/
"""

import os
import os.path as osp
import re
import sys
import json
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
    # ---- dataset (user provided) ----
    data_root = "/home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-50-40"
    images_dir = osp.join(data_root, "images")
    meta_path = osp.join(data_root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    prompt = meta.get("instruction", "Orbit the person at a 3.0-meter radius clockwise.")

    frame_paths = _sorted_frame_paths(images_dir)
    # Need at least 21+20 frames to demonstrate first clip obs + second clip prediction window.
    assert len(frame_paths) >= 41, f"need >=41 frames (0..40) for 2 clips, got {len(frame_paths)}"

    # ---- args (mirrors tools/infer_interact_480p.py) ----
    args = Args()
    checkpoints_dir = osp.join(REPO_ROOT, "checkpoint")

    args.pn = "0.40M"
    args.fps = 16
    # After changing InfinityStar-Interact clip granularity to 20 predicted pixel frames per clip:
    # total_frames = 1 + 20 + 20 = 41 for 2 clips.
    args.video_frames = 41
    args.max_duration = 10
    args.temporal_compress_rate = 4
    # 5 compressed frames per clip -> 5*4 = 20 predicted pixel frames per clip
    args.frames_inner_clip = 5
    args.context_interval = 2
    args.context_from_largest_no = 1
    args.context_frames = 1000
    args.steps_per_frame = 3

    args.model_type = "infinity_qwen8b"
    args.model_path = osp.join(checkpoints_dir, "InfinityStarInteract_24K_iters")
    args.checkpoint_type = "torch_shard"

    args.vae_path = osp.join(checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(checkpoints_dir, "text_encoder", "flan-t5-xl-official")
    args.videovae = 10
    args.vae_type = 64
    # match infer_interact_480p.py
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

    # two-stage LFQ settings (same as released configs)
    args.use_two_stage_lfq = 1
    args.semantic_scales = 11
    args.detail_scale_min_tokens = 350
    args.semantic_scale_dim = 16
    args.detail_scale_dim = 64
    args.num_lvl = 2
    args.num_of_label_value = args.num_lvl
    args.semantic_num_lvl = args.num_lvl
    args.detail_num_lvl = args.num_lvl

    # guidance (use CFG, consistent with repo defaults)
    args.use_cfg = 1
    args.use_apg = 0
    args.cfg = 3
    args.apg_norm_threshold = 0.05
    args.simple_text_proj = 1
    args.text_channels = 2048
    args.bf16 = 1
    args.append_duration2caption = 1
    args.use_clipwise_caption = 1

    # repetition / misc
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    args.video_scale_repetition = args.image_scale_repetition
    args.max_repeat_times = 1000
    args.apply_spatial_patchify = 0
    args.taui, args.tauv = 0.5, 0.5
    args.tau = [args.taui] * len(json.loads(args.image_scale_repetition)) + [args.tauv] * len(json.loads(args.video_scale_repetition))

    # ---- build schedule (same idea as infer_interact_480p.py) ----
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_list = np.array(list(dynamic_resolution_h_w.keys()))

    # We'll use first frame size to pick template.
    first_img = np.array(Image.open(frame_paths[0]).convert("RGB"))
    h, w = first_img.shape[:2]
    h_div_w_template_ = h_div_w_template_list[np.argmin(np.abs(h / w - h_div_w_template_list))]

    # First clip: boundary + 20 predicted pixel frames => 21 pixel frames total.
    # With temporal_compress_rate=4, pt = (21-1)//4 + 1 = 6 (boundary + 5 compressed frames).
    first_clip_pixel = 21
    pt_first_clip = (first_clip_pixel - 1) // args.temporal_compress_rate + 1  # 6
    scale_schedule_full = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][pt_first_clip]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule_full)
    scales_in_one_clip = args.first_full_spatial_size_scale_index + 1
    cur_scale_schedule = scale_schedule_full[scales_in_one_clip:]  # drop image scales, keep video scales for one clip

    video_encode, video_decode, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    context_info = get_scale_pack_info(cur_scale_schedule, args.first_full_spatial_size_scale_index, args)

    # Target resolution from schedule
    tgt_h, tgt_w = scale_schedule_full[-1][1] * 16, scale_schedule_full[-1][2] * 16

    # ---- load models ----
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to("cuda")
    infinity = load_transformer(vae, args)

    # ---- Step A: "action executed, got real observation clip" ----
    # Here we directly load real frames as the observed first clip.
    obs_clip0 = _load_frames_tensor(frame_paths[0:first_clip_pixel], tgt_h, tgt_w).to("cuda")  # [1,3,81,H,W]
    former_clip_features, _, _ = vae.encode_for_raw_features(obs_clip0, scale_schedule=None, slice=True)
    first_frame_features = former_clip_features[:, :, 0:1]

    # ---- Step B: "replace clip and infer next clip" ----
    out_dir = osp.join(REPO_ROOT, "output", "uav_orbit_interact_replace_clip")
    os.makedirs(out_dir, exist_ok=True)

    # Save reconstruction of the observed clip to verify VAE compatibility with your data domain.
    recons = vae.decode(former_clip_features, slice=True)  # [1,3,T,H,W] in [-1,1]
    # Convert to uint8 [T,H,W,3] for saving (same as Infinity.summed_codes2images)
    recons = (recons + 1) / 2
    recons = torch.clamp(recons, 0, 1)
    recons = recons.permute(0, 2, 3, 4, 1)  # [1,T,H,W,3]
    recons = recons.mul(255).to(torch.uint8).cpu().numpy()[0]
    save_video(recons, fps=args.fps, save_filepath=osp.join(out_dir, "obs_clip0_recon.mp4"))

    # mode='second_v_clip' makes the model condition on:
    # - semantic_condition: resized former clip (20 frames)
    # - detail_condition: first frame + last frames
    prompt_for_model = prompt
    if args.append_duration2caption:
        prompt_for_model = f"<<<t=5s>>>{prompt_for_model}"
    pred_video, updated_former_clip_features = gen_one_example(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt_for_model,
        negative_prompt="",
        g_seed=0,
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
        former_clip_features=former_clip_features,
        first_frame_features=first_frame_features,
    )

    # pred_video may be a torch.Tensor depending on the decode path; ensure numpy for save_video.
    if isinstance(pred_video, torch.Tensor):
        pred_video = pred_video.detach().cpu().numpy()
    save_path = osp.join(out_dir, "pred_next_clip.mp4")
    save_video(pred_video, fps=args.fps, save_filepath=save_path)
    print(f"Saved predicted next clip to: {osp.abspath(save_path)}")


if __name__ == "__main__":
    main()

