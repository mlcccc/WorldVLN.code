#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Infer 81-frame video from a single first-frame image + prompt, using a finetuned Infinity checkpoint
saved as `global_step_*.pth` (checkpoint_type=torch).

Example (your case):
  python3 tools/infer_i2v_firstframe_finetuned_480p_81f.py \
    --ckpt /home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main/checkpoints/finetune_wan_480p_cap81_keep_short/global_step_5296.pth \
    --image_dir /home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-49-14/images \
    --prompt_json /home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-49-14/meta.json \
    --out_dir /home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main/output_i2v_finetune5296
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import time

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
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, gen_one_example, save_video, transform


def _pick_first_image(image_dir: str) -> str:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(osp.join(image_dir, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"no images found in {image_dir}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="global_step_*.pth")
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--prompt_json", type=str, required=True)
    ap.add_argument("--prompt_key", type=str, default="instruction_unified", help="field to read from prompt_json")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--pn", type=str, default="0.40M")
    ap.add_argument("--h_div_w_template", type=float, default=0.562)
    args_cli = ap.parse_args()

    ckpt = osp.abspath(args_cli.ckpt)
    image_dir = osp.abspath(args_cli.image_dir)
    prompt_json = osp.abspath(args_cli.prompt_json)
    out_dir = osp.abspath(args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(prompt_json, "r", encoding="utf-8") as f:
        pj = json.load(f)
    prompt = pj.get(args_cli.prompt_key) or pj.get("instruction") or pj.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"cannot find prompt in {prompt_json} (key={args_cli.prompt_key})")
    prompt = prompt.strip()

    image_path = _pick_first_image(image_dir)
    print(f"[input] image_path={image_path}")
    print(f"[input] prompt={prompt}")
    print(f"[input] ckpt={ckpt}")

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

    # Use the same schedule family used in finetune (all-pt).
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
    a.seed = int(args_cli.seed)

    # repetition configs (14 scales for 0.40M)
    a.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    a.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"

    # load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=a.text_encoder_ckpt)
    vae = load_visual_tokenizer(a).float().to("cuda")
    infinity = load_transformer(vae, a)
    self_correction = SelfCorrection(vae, a)
    video_encode, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(a.dynamic_scale_schedule)

    # schedule determines target resolution
    dyn_res, h_div_w_templates = get_dynamic_resolution_meta(a.dynamic_scale_schedule, a.video_frames)
    htpl = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - float(args_cli.h_div_w_template)))]
    pt = (a.video_frames - 1) // a.temporal_compress_rate + 1
    scale_schedule = dyn_res[htpl][a.pn]["pt2scale_schedule"][pt]
    a.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    a.tower_split_index = a.first_full_spatial_size_scale_index + 1
    context_info = get_scale_pack_info(scale_schedule, a.first_full_spatial_size_scale_index, a)
    tau = [a.tau_image] * a.tower_split_index + [a.tau_video] * (len(scale_schedule) - a.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16

    # encode first frame to get gt_leak conditioning
    pil = Image.open(image_path).convert("RGB")
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

    # prompt with duration tag (match training prompt format)
    dur_s = (a.video_frames - 1) // a.fps
    prompt_infer = f"<<<t={dur_s}s>>>{prompt}" if a.append_duration2caption else prompt

    start = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        video, _ = gen_one_example(
            infinity,
            vae,
            text_tokenizer,
            text_encoder,
            prompt_infer,
            negative_prompt="",
            g_seed=a.seed,
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
    elapsed = time.time() - start

    # save
    # NOTE: repo inference path returns uint8 frames in BGR order:
    #   torch uint8 [1,T,H,W,3] or [T,H,W,3]
    if isinstance(video, torch.Tensor) and video.dim() == 5:
        video = video[0]
    save_path = osp.join(out_dir, "i2v_81f.mp4")
    save_video(video.cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video), fps=a.fps, save_filepath=save_path)
    print(f"[done] saved={save_path} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()

