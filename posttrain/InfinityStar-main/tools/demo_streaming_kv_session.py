# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Minimal demo for the streaming KV-cache loop:

1) reset(prompt) -> cache text as GT
2) compute_kv_cache_gt(frame1) -> cache GT obs
3) infer_chunk(...) -> writes Pred KV cache
4) correction_clear_pred() -> removes only Pred entries
5) compute_kv_cache_gt(frames2_17) -> refresh GT obs
6) infer_chunk(...) again

Run with the provided conda env python, e.g.
`/home/batchcom/dataset-link/xjc/infinitystar/bin/python tools/demo_streaming_kv_session.py`
"""

import os
import os.path as osp
import sys
import json

import numpy as np
import torch
from PIL import Image

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.utils.arg_util import Args
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, transform
from tools.infinity_streaming_session import InfinityStreamingSession


def _make_obs_from_image(image_path: str, tgt_h: int, tgt_w: int, t: int) -> torch.Tensor:
    """Return obs video tensor [1,3,T,H,W] in [-1,1]."""
    pil = Image.open(image_path).convert("RGB")
    frame = transform(pil, tgt_h, tgt_w)  # [3,H,W] in [-1,1]
    video_T3HW = torch.stack([frame for _ in range(t)], dim=0)  # [T,3,H,W]
    return video_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]


def main():
    ckpt_dir = osp.join(REPO_ROOT, "checkpoint")
    args = Args()
    # Keep close to released 480p setup
    args.pn = "0.40M"
    args.fps = 16
    # IMPORTANT:
    # - dynamic_scale_schedule 'infinity_elegant_clip20frames_v2' requires model video_frames such that
    #   (compressed_frames - 1) % 20 == 0 where compressed_frames = video_frames//4 + 1.
    # - so we keep model configured with 5s (81 frames), but run streaming chunks with num_frames=17 (pt=5).
    args.video_frames = 81
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
    args.use_apg = 1
    args.use_cfg = 0
    # Streaming demo uses cfg=1 to keep batch size = 1 (GT caches are written with bs=1).
    # If you need cfg/apg>1, extend InfinityStreamingSession to write GT caches with bs=2.
    args.cfg = 1
    args.apg_norm_threshold = 0.05
    args.simple_text_proj = 1
    args.apply_spatial_patchify = 0
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 350
    args.semantic_scales = 11
    args.max_repeat_times = 10000
    # repetition configs (14 scales for 0.40M)
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"

    # load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to("cuda")
    infinity = load_transformer(vae, args)
    infinity.eval().requires_grad_(False)

    session = InfinityStreamingSession(
        args=args,
        infinity_model=infinity,
        vae=vae,
        text_tokenizer=text_tokenizer,
        text_encoder=text_encoder,
        h_div_w_template=0.571,
    )

    # schedule determines target resolution
    chunk_frames = 17
    sched = session.build_schedule_for_num_frames(num_frames=chunk_frames)
    tgt_h, tgt_w = sched.tgt_h, sched.tgt_w

    prompt = "A drone is hovering. The camera is stable."
    session.reset(prompt)
    print("after reset(text GT):", session.cache_stats())

    # Step 1: first frame as GT obs
    img_path = osp.join(REPO_ROOT, "assets", "reference_image.webp")
    obs1 = _make_obs_from_image(img_path, tgt_h, tgt_w, t=1).to("cuda")
    session.compute_kv_cache_gt(obs1)
    print("after Step1(GT obs cache):", session.cache_stats())

    # Step 2: infer chunk -> pred cache grows
    out1 = session.infer_chunk(
        num_frames=chunk_frames,
        cfg_list=[1.0] * len(sched.scale_schedule),
        tau_list=[1.0] * len(sched.scale_schedule),
        seed=0,
        low_vram_mode=True,
    )
    print("after Step2(infer -> Pred cache):", session.cache_stats())

    # Step 4: clear pred cache
    session.correction_clear_pred()
    print("after Step4(clear_pred_cache):", session.cache_stats())

    # Step 4 (continued): write back GT obs for frames 2~17 (demo uses repeated image)
    obs2_17 = _make_obs_from_image(img_path, tgt_h, tgt_w, t=17).to("cuda")
    session.compute_kv_cache_gt(obs2_17)
    print("after Step4(write GT obs 2~17):", session.cache_stats())

    # Step 5: infer again
    out2 = session.infer_chunk(
        num_frames=chunk_frames,
        cfg_list=[1.0] * len(sched.scale_schedule),
        tau_list=[1.0] * len(sched.scale_schedule),
        seed=1,
        low_vram_mode=True,
    )
    print("after Step5(infer again):", session.cache_stats())


if __name__ == "__main__":
    main()

