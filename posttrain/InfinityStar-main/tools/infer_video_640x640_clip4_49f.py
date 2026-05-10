# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
#
# Inference script for finetuned 49f clip4 cross-clip model, forcing square 640x640 output.
#
# Differences vs tools/infer_video_480p_clip4_49f.py:
# - Force h_div_w_template = 1.0 (square), so target pixel size becomes 640x640 under
#   dynamic_scale_schedule=infinity_elegant_clip4frames_v2_allpt, pn=0.40M, video_frames=49.

import sys
import os
import os.path as osp
import time
import numpy as np
import torch
import cv2
import argparse
from PIL import Image

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from infinity.utils.arg_util import Args


class InferencePipe:
    def __init__(self, args):
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args)
        self.vae = self.vae.float().to("cuda")
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        self._models = [self.text_tokenizer, self.text_encoder, self.vae, self.infinity, self.self_correction]
        (
            self.video_encode,
            self.video_decode,
            self.get_visual_rope_embeds,
            self.get_scale_pack_info,
        ) = get_encode_decode_func(args.dynamic_scale_schedule)


def perform_inference(pipe, data, args, force_h_div_w_template: float = 1.0):
    prompt = data["prompt"]
    seed = data["seed"]
    num_frames = args.video_frames  # 49
    image_path = data.get("image_path", None)

    # Build scale schedule for 49 frames with clip4
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    # Force square template => 640x640 (for pn=0.40M under this schedule).
    h_div_w_template_ = float(h_div_w_templates[np.argmin(np.abs(h_div_w_templates - float(force_h_div_w_template)))])

    pt = (num_frames - 1) // 4 + 1  # 13
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][pt]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)

    # tau: image scales use tau_image, video scales use tau_video
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    # I2V: encode reference image as condition (leak first clip = 14 scales)
    if image_path is not None:
        ref_image = [cv2.imread(image_path)[:, :, ::-1]]
        ref_img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in ref_image]
        ref_img_T3HW = torch.stack(ref_img_T3HW, 0)
        ref_img_bcthw = ref_img_T3HW.permute(1, 0, 2, 3).unsqueeze(0)
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

    mapped_duration = (num_frames - 1) / args.fps  # 48/16 = 3.0
    negative_prompt = ""
    if args.append_duration2caption:
        prompt = f"<<<t={mapped_duration}s>>>" + prompt

    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        generated_image, _ = gen_one_example(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
            negative_prompt=negative_prompt,
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
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        print(f"Generated shape: {generated_image.shape} (tgt_h={tgt_h}, tgt_w={tgt_w}, h_div_w_template={h_div_w_template_})")

    end_time = time.time()
    return {
        "output": generated_image.cpu().numpy(),
        "elapsed_time": end_time - start_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference 49f clip4 cross-clip finetuned model (force 640x640)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/finetune_480p_49f_clip4_crossclip/",
        help="Path to finetuned model weights (e.g. global_step_8000.pth)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./",
        help="Root dir for VAE and text encoder (contains infinitystar_videovae.pth and text_encoder/...)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--image_path", type=str, default=None, help="Reference image for I2V (omit for T2V)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output_clip4_49f_640x640")
    parser.add_argument("--output_name", type=str, default="demo_640x640.mp4")
    parser.add_argument("--tau_video", type=float, default=0.4)
    parser.add_argument("--cfg", type=float, default=34)
    parser.add_argument(
        "--force_h_div_w_template",
        type=float,
        default=1.0,
        help="Force nearest h/w template (default 1.0 => square 640x640 for this schedule)",
    )
    cli_args = parser.parse_args()

    args = Args()
    args.pn = "0.40M"
    args.fps = 16
    args.video_frames = 49
    args.model_path = cli_args.model_path
    args.checkpoint_type = "torch"
    args.vae_path = osp.join(cli_args.checkpoints_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(cli_args.checkpoints_dir, "text_encoder/flan-t5-xl-official/")
    args.videovae = 10
    args.model_type = "infinity_qwen8b"
    args.text_channels = 2048
    args.bf16 = 1

    # clip4 schedule (must match training)
    args.dynamic_scale_schedule = "infinity_elegant_clip4frames_v2_allpt"
    args.frames_inner_clip = 4
    args.context_from_largest_no = 0
    args.context_frames = 10000
    args.context_interval = 2

    # Generation params
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = cli_args.cfg
    args.tau_image = 1
    args.tau_video = cli_args.tau_video
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    args.video_scale_repetition = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.semantic_scale_dim = 16
    args.detail_scale_dim = 64
    args.detail_scale_min_tokens = 350
    args.semantic_scales = 11
    args.max_repeat_times = 10000
    args.enable_rewriter = 0

    print(f"Loading models from {args.model_path} ...")
    pipe = InferencePipe(args)

    data = {
        "seed": cli_args.seed,
        "prompt": cli_args.prompt,
    }
    if cli_args.image_path:
        data["image_path"] = cli_args.image_path

    output_dict = perform_inference(pipe, data, args, force_h_div_w_template=cli_args.force_h_div_w_template)

    os.makedirs(osp.join(cli_args.output_dir, "gen_videos"), exist_ok=True)
    gen_video_path = osp.join(cli_args.output_dir, "gen_videos", cli_args.output_name)
    save_video(output_dict["output"], fps=args.fps, save_filepath=gen_video_path)
    print(f"Done! {gen_video_path} ({output_dict['elapsed_time']:.1f}s)")

