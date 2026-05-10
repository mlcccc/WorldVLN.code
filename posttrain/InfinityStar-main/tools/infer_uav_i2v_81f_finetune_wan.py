#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
"""
I2V inference (81 frames @ 480p schedule) using a finetuned InfinityStar checkpoint (.pth).

User request:
- Use finetuned weights: global_step_5296.pth
- Input: first frame image + prompt
- Output: 81-frame video

This script reuses the official I2V codepath from `tools/infer_video_480p.py` (image_path mode),
but swaps model weights to your finetuned .pth checkpoint (checkpoint_type='torch').
"""

import argparse
import json
import os
import os.path as osp
import sys

# Ensure repo root is importable (so `import infinity` / `import tools.*` works).
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from infinity.utils.arg_util import Args
from tools.infer_video_480p import InferencePipe, perform_inference
from tools.run_infinity import save_video


def _read_prompt(meta_json_path: str) -> str:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    prompt = meta.get("instruction_unified") or meta.get("instruction")
    if not prompt:
        raise ValueError(f"meta.json missing instruction fields: {meta_json_path}")
    return str(prompt).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default="/home/batchcom/dataset-link/xjc/Infinity/InfinityStar-main/checkpoints/finetune_wan_480p_cap81_keep_short/global_step_5296.pth",
        help="Finetuned checkpoint path (.pth).",
    )
    p.add_argument(
        "--frames_dir",
        type=str,
        default="/home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-49-14/images",
        help="Directory containing frame_000000.png ...",
    )
    p.add_argument(
        "--meta_json",
        type=str,
        default="/home/batchcom/dataset-link/xjc/actionhead/TSformer-VO-main/TSformer-VO-main/data/reference_train_uavflow_like/2025-03-30_11-49-14/meta.json",
        help="meta.json containing instruction/instruction_unified.",
    )
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--duration", type=int, default=5, help="seconds. 5 -> 81 frames at 16 fps.")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=osp.join(_REPO_ROOT, "checkpoint"),
        help="Directory containing InfinityStar base assets (text encoder, VAE, etc).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=osp.join(_REPO_ROOT, "output", "uav_i2v_finetune_wan_81f.mp4"),
    )
    args_cli = p.parse_args()

    first_frame = osp.join(args_cli.frames_dir, "frame_000000.png")
    if not osp.isfile(first_frame):
        raise FileNotFoundError(f"First frame not found: {first_frame}")

    if not osp.isfile(args_cli.ckpt):
        raise FileNotFoundError(
            f"Finetuned checkpoint not found: {args_cli.ckpt}\n"
            f"If you only have logs under `InfinityStar-main/checkpoints/`, copy the .pth here first."
        )

    # Build inference args (match training schedule as much as possible)
    inf = Args()
    inf.pn = "0.40M"
    inf.fps = int(args_cli.fps)
    inf.video_frames = int(args_cli.duration) * int(args_cli.fps) + 1  # 81 for 5s@16fps

    # Load finetuned weights
    inf.model_path = osp.abspath(args_cli.ckpt)
    inf.checkpoint_type = "torch"

    # Required base components
    inf.vae_path = osp.join(args_cli.checkpoint_dir, "infinitystar_videovae.pth")
    inf.text_encoder_ckpt = osp.join(args_cli.checkpoint_dir, "text_encoder", "flan-t5-xl-official")

    if not osp.isfile(inf.vae_path):
        raise FileNotFoundError(
            f"Missing VAE weights: {inf.vae_path}\n"
            f"Please run: `python {osp.join(_REPO_ROOT, 'download_infinity.py')}` to download required files."
        )

    # Model + schedule settings (aligned with finetune_480p_81f_wan.sh)
    inf.videovae = 10
    inf.model_type = "infinity_qwen8b"
    inf.text_channels = 2048
    inf.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2_allpt"
    inf.mask_type = "infinity_elegant_clip20frames_v2_allpt"
    inf.bf16 = 1

    # Inference hyperparams (reuse official defaults)
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
    prompt = _read_prompt(args_cli.meta_json)

    data = {
        "seed": int(args_cli.seed),
        "image_path": first_frame,
        "prompt": prompt,
        "duration": int(args_cli.duration),  # seconds
    }

    out_dict = perform_inference(pipe, data, inf)
    os.makedirs(osp.dirname(args_cli.out) or ".", exist_ok=True)
    save_video(out_dict["output"], fps=inf.fps, save_filepath=args_cli.out)
    print(f"[ok] prompt={prompt}")
    print(f"[ok] first_frame={osp.abspath(first_frame)}")
    print(f"[ok] out={osp.abspath(args_cli.out)}")


if __name__ == "__main__":
    main()

