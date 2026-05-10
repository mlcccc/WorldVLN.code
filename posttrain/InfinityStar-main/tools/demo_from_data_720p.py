#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Demo: use a video first frame + prompt from ./data as I2V input.

Default source:
  data/interactive_toy_videos/<story_id>/{0000_refine_720p.mp4,prompt.txt}

Output:
  ./output/demo_from_data/demo.mp4
"""

import os
import os.path as osp
import argparse

import sys
import cv2

# Ensure repo root is importable (so `import infinity` works).
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from infinity.utils.arg_util import Args
from tools.run_infinity import (
    load_tokenizer,
    load_transformer,
    load_visual_tokenizer,
    gen_one_example,
    save_video,
    transform,
)
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func


def _read_first_prompt_line(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"prompt.txt is empty: {prompt_path}")


def _extract_first_frame(video_path: str, save_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Unable to read first frame: {video_path}")
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    ok = cv2.imwrite(save_path, frame_bgr)
    if not ok:
        raise ValueError(f"Failed to write first frame to: {save_path}")
    return save_path


class InferencePipe:
    def __init__(self, args: Args):
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        self.vae = load_visual_tokenizer(args).float().to("cuda")
        # load infinity
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(
            args.dynamic_scale_schedule
        )


def perform_inference(pipe: InferencePipe, prompt: str, image_path: str, seed: int, args: Args):
    mapped_duration = 5
    num_frames = 81

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[(abs(h_div_w_templates - 0.571)).argmin()]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][(num_frames - 1) // 4 + 1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)

    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16

    # Encode reference image -> gt leak (I2V)
    ref_bgr = cv2.imread(image_path)
    if ref_bgr is None:
        raise ValueError(f"Failed to read image_path: {image_path}")
    ref_rgb = ref_bgr[:, :, ::-1]
    from PIL import Image
    ref_img_T3HW = [transform(Image.fromarray(ref_rgb).convert("RGB"), tgt_h, tgt_w)]
    import torch

    ref_img_T3HW = torch.stack(ref_img_T3HW, 0)  # [t,3,h,w]
    ref_img_bcthw = ref_img_T3HW.permute(1, 0, 2, 3).unsqueeze(0)  # [c,t,h,w] -> [b,c,t,h,w]
    _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(
        pipe.vae,
        ref_img_bcthw.cuda(),
        vae_features=None,
        self_correction=pipe.self_correction,
        args=args,
        infer_mode=True,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
    )
    gt_leak = len(scale_schedule) // 2

    # Prompt engineering consistent with official scripts
    prompt = f"{prompt}, Close-up on big objects, emphasize scale and detail"
    if args.append_duration2caption:
        prompt = f"<<<t={mapped_duration}s>>>" + prompt

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated_image, _ = gen_one_example(
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
    return generated_image


def main():
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=osp.join(repo_root, "data"),
        help="Path to repo data/ directory",
    )
    parser.add_argument(
        "--story_id",
        type=str,
        default="00a79efb495c29e082c246e9ca9a7e8f",
        help="Subdir name under data/interactive_toy_videos/",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default=osp.join(repo_root, "checkpoint"),
        help="Path to checkpoint/ directory",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", type=str, default=osp.join(repo_root, "output", "demo_from_data", "demo.mp4"))
    args_cli = parser.parse_args()

    story_dir = osp.join(args_cli.data_root, "interactive_toy_videos", args_cli.story_id)
    prompt_path = osp.join(story_dir, "prompt.txt")
    video_path = osp.join(story_dir, "0000_refine_720p.mp4")
    if not osp.exists(prompt_path):
        raise FileNotFoundError(prompt_path)
    if not osp.exists(video_path):
        raise FileNotFoundError(video_path)

    prompt = _read_first_prompt_line(prompt_path)
    ref_img_path = _extract_first_frame(
        video_path,
        save_path=osp.join(repo_root, "output", "demo_from_data", "ref_first_frame.jpg"),
    )

    # Build inference args (match tools/infer_video_720p.py defaults)
    inf_args = Args()
    inf_args.pn = "0.90M"
    inf_args.fps = 16
    inf_args.video_frames = 81
    inf_args.model_path = osp.join(args_cli.checkpoints_dir, "infinitystar_8b_720p_weights")
    inf_args.checkpoint_type = "torch_shard"
    inf_args.vae_path = osp.join(args_cli.checkpoints_dir, "infinitystar_videovae.pth")
    inf_args.text_encoder_ckpt = osp.join(args_cli.checkpoints_dir, "text_encoder", "flan-t5-xl-official")
    inf_args.model_type = "infinity_qwen8b"
    inf_args.text_channels = 2048
    inf_args.dynamic_scale_schedule = "infinity_elegant_clip20frames_v2"
    inf_args.bf16 = 1
    inf_args.use_apg = 1
    inf_args.use_cfg = 0
    inf_args.cfg = 34
    inf_args.tau_image = 1
    inf_args.tau_video = 0.4
    inf_args.apg_norm_threshold = 0.05
    inf_args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    inf_args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]"
    inf_args.append_duration2caption = 1
    inf_args.use_two_stage_lfq = 1
    inf_args.detail_scale_min_tokens = 750
    inf_args.semantic_scales = 12
    inf_args.max_repeat_times = 10000
    inf_args.enable_rewriter = 0
    inf_args.videovae = 10
    inf_args.vae_type = 64

    pipe = InferencePipe(inf_args)
    video_uint8 = perform_inference(pipe, prompt=prompt, image_path=ref_img_path, seed=args_cli.seed, args=inf_args)
    save_video(video_uint8.cpu().numpy(), fps=inf_args.fps, save_filepath=args_cli.out)
    print(f"[demo] prompt_path={prompt_path}")
    print(f"[demo] video_path={video_path}")
    print(f"[demo] ref_img_path={ref_img_path}")
    print(f"[demo] out={osp.abspath(args_cli.out)}")


if __name__ == "__main__":
    main()

