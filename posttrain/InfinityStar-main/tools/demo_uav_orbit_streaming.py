# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Streaming demo using REAL UAV observations (image sequence) and saving predicted 16-frame videos.

Dataset path (user provided):
/home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-50-40

Workflow per cycle:
1) Step1: cache GT obs for current start frame (t=1)
2) Step2: infer -> decode to 17 frames, save LAST 16 frames as mp4
3) Step4: clear_pred_cache; cache GT obs for real frames (next 16 frames)
4) repeat from Step2

Note:
- This demo enables CFG (bs=2) to improve generation quality.
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
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, transform, save_video
from tools.infinity_streaming_session import InfinityStreamingSession
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta


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
    data_root = "/home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-50-40"
    images_dir = osp.join(data_root, "images")
    meta_path = osp.join(data_root, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    prompt = meta.get("instruction", "Orbit the person at a 3.0-meter radius clockwise.")

    frame_paths = _sorted_frame_paths(images_dir)
    assert len(frame_paths) >= 33, f"need at least 33 frames for 2 cycles, got {len(frame_paths)}"

    # Model config (same constraints as previous demo)
    ckpt_dir = osp.join(REPO_ROOT, "checkpoint")
    args = Args()
    args.pn = "0.40M"
    args.fps = 16
    args.video_frames = 81  # keep schedule valid; we will infer chunk_frames=17 (pt=5)
    args.temporal_compress_rate = 4
    args.videovae = 10
    args.vae_type = 64
    args.vae_path = osp.join(ckpt_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(ckpt_dir, "text_encoder", "flan-t5-xl-official")
    args.model_type = "infinity_qwen8b"
    args.model_path = osp.join(ckpt_dir, "infinitystar_8b_480p_weights")
    args.checkpoint_type = "torch_shard"
    args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    args.mask_type = "infinity_elegant_clip20frames_v2"
    args.text_channels = 2048
    args.bf16 = 1
    # Enable CFG for better quality (bs=2 path).
    args.use_apg = 0
    args.use_cfg = 1
    args.cfg = 34
    args.apg_norm_threshold = 0.05
    args.simple_text_proj = 1
    args.apply_spatial_patchify = 0
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 350
    args.semantic_scales = 11
    args.max_repeat_times = 10000
    args.tau_image = 1.0
    args.tau_video = 0.4
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    # self-correction settings (needed by video_encode to get GT bit labels)
    args.reduce_accumulate_error_method = "bsc"
    args.noise_apply_layers = 200
    args.noise_apply_requant = 1
    args.noise_apply_strength = [0.0 for _ in range(2000)]

    # load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to("cuda")
    infinity = load_transformer(vae, args)
    infinity.eval().requires_grad_(False)
    self_correction = SelfCorrection(vae, args)

    session = InfinityStreamingSession(
        args=args,
        infinity_model=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        h_div_w_template=0.571,
    )

    chunk_frames = 17  # 1 + 16 predicted frames
    sched = session.build_schedule_for_num_frames(num_frames=chunk_frames)
    tgt_h, tgt_w = sched.tgt_h, sched.tgt_w

    out_dir = osp.join("output", "uav_orbit_streaming")
    os.makedirs(out_dir, exist_ok=True)

    # dynamic resolution meta needed for video_encode (gt_leak)
    dynamic_resolution_h_w, _ = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)

    # Init: cache text GT
    session.reset(prompt, cfg_scale=args.cfg)
    print("after reset(text GT):", session.cache_stats())

    # Cycle 0:
    # Step1: frame0 as GT
    obs0 = _load_frames_tensor([frame_paths[0]], tgt_h, tgt_w).to("cuda")
    session.compute_kv_cache_gt(obs0)
    print("after Step1(frame0 GT):", session.cache_stats())

    # Build GT bit labels for the first frame and enable GT leak for image scales.
    # This is crucial; otherwise the model behaves like near-unconditional sampling and decodes as noise.
    _, _, gt_ls_Bl0, _, _, _ = session.video_encode(
        vae,
        obs0,
        vae_features=None,
        self_correction=self_correction,
        args=args,
        infer_mode=True,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
    )
    gt_leak = args.first_full_spatial_size_scale_index + 1

    # Step2: infer -> save predicted 16 frames (frames 1..16)
    tau_list = [args.tau_image] * sched.tower_split_index + [args.tau_video] * (len(sched.scale_schedule) - sched.tower_split_index)
    cfg_list = [float(args.cfg)] * len(sched.scale_schedule)
    _, img0 = session.infer_chunk(
        num_frames=chunk_frames,
        cfg_list=cfg_list,
        tau_list=tau_list,
        seed=0,
        low_vram_mode=True,
        gt_leak=gt_leak,
        gt_ls_Bl=gt_ls_Bl0,
    )
    pred0 = img0[0]  # [T,H,W,3] uint8
    pred0_16 = pred0[1:17].cpu().numpy()  # take last 16 frames, to numpy for save_video
    save_video(pred0_16, fps=args.fps, save_filepath=osp.join(out_dir, "pred_cycle0_16f.mp4"))
    print("after Step2(pred cached):", session.cache_stats())

    # Step4: clear pred + write real frames 1..16 as GT (对应真实第2~17帧)
    session.correction_clear_pred()
    real1_16 = _load_frames_tensor(frame_paths[1:17], tgt_h, tgt_w).to("cuda")  # 16 frames
    session.compute_kv_cache_gt(real1_16)
    print("after Step4(clear pred + write real1_16):", session.cache_stats())

    # Cycle 1:
    # STRICT MODE (align with wan_va "compute_kv_cache uses the whole observed chunk"):
    # We must make sure frames 1..16 ALL participate in the conditioning for the next prediction.
    #
    # However, InfinityElegant's pt2scale_schedule keys are 1,5,9,... (compressed frames),
    # while 16 input frames become latent T=4 (pt=4) after temporal compression, which is not a valid key.
    # To keep *all* frames 1..16 and align to pt=5, we pad by repeating the last frame once
    # (frames 1..16 + frame16 duplicated => 17 frames => latent T=5).
    real1_16_plus = torch.cat([real1_16, real1_16[:, :, -1:]], dim=2)  # [1,3,17,H,W]
    _, _, gt_ls_Bl1, _, _, _ = session.video_encode(
        vae,
        real1_16_plus,
        vae_features=None,
        self_correction=self_correction,
        args=args,
        infer_mode=True,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
    )

    # Step2: infer next chunk -> save predicted 16 frames (frames 17..32)
    _, img1 = session.infer_chunk(
        num_frames=chunk_frames,
        cfg_list=cfg_list,
        tau_list=tau_list,
        seed=1,
        low_vram_mode=True,
        gt_leak=gt_leak,
        gt_ls_Bl=gt_ls_Bl1,
    )
    pred1 = img1[0]
    pred1_16 = pred1[1:17].cpu().numpy()
    save_video(pred1_16, fps=args.fps, save_filepath=osp.join(out_dir, "pred_cycle1_16f.mp4"))
    print("after Step5(pred again cached):", session.cache_stats())

    # Optional: Step4 for next cycle (write real frames 17..32)
    session.correction_clear_pred()
    real17_32 = _load_frames_tensor(frame_paths[17:33], tgt_h, tgt_w).to("cuda")
    session.compute_kv_cache_gt(real17_32)
    print("after Step4(next clear pred + write real17_32):", session.cache_stats())


if __name__ == "__main__":
    main()

