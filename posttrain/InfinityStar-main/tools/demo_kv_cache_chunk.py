#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
KV-cache chunk demo for InfinityStar.

This is a correctness demo for the new cache semantics:
- write GT cache (is_pred=False)
- write Pred cache (is_pred=True)
- clear_pred_cache() removes only pred entries

It does NOT implement full streaming generation yet. It validates that cache bookkeeping
matches the LingBot-VA style workflow.
"""

import os
import os.path as osp
import sys

import torch

# Ensure repo root is importable
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from infinity.utils.arg_util import Args
from tools.run_infinity import load_tokenizer, load_transformer, load_visual_tokenizer, encode_prompt


def _count_cache_entries(infinity_model):
    """Return (total_entries, pred_entries, gt_entries) summed over blocks."""
    total = pred = gt = 0
    for blk in infinity_model.unregistered_blocks:
        meta = getattr(blk.attn, "cached_is_pred", {})
        total += len(meta)
        pred += sum(1 for v in meta.values() if v)
        gt += sum(1 for v in meta.values() if not v)
    return total, pred, gt


def main():
    ckpt_dir = osp.join(REPO_ROOT, "checkpoint")
    args = Args()
    args.pn = "0.40M"
    args.video_frames = 161  # max to avoid RoPE frame issues
    args.videovae = 10
    args.vae_type = 64
    args.vae_path = osp.join(ckpt_dir, "infinitystar_videovae.pth")
    args.text_encoder_ckpt = osp.join(ckpt_dir, "text_encoder", "flan-t5-xl-official")
    args.model_type = "infinity_qwen8b"
    args.model_path = osp.join(ckpt_dir, "infinitystar_8b_480p_weights")
    args.checkpoint_type = "torch_shard"
    args.dynamic_scale_schedule = "infinity_star_interact"
    args.mask_type = "infinity_star_interact"
    args.text_channels = 2048
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 3
    args.apg_norm_threshold = 0.05
    args.simple_text_proj = 1
    args.apply_spatial_patchify = 0

    # Load models
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args).float().to("cuda")
    infinity = load_transformer(vae, args)
    infinity.eval().requires_grad_(False)

    prompt = "A drone is hovering. The camera is stable."
    label = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False)

    # Turn on caching (reset)
    for blk in infinity.unregistered_blocks:
        blk.attn.kv_caching(True, reset=True)

    # 1) Write GT cache entry: run a tiny forward that stores under scale key 't0_gt'
    infinity.set_cache_write_is_pred(False)
    # Keep dtype consistent with model (often bf16 in this repo).
    model_dtype = next(iter(infinity.parameters())).dtype
    x = torch.zeros((1, 1, infinity.vae_embed_dim), device="cuda", dtype=model_dtype)  # dummy visual token
    # We call model.forward() path is too heavy; instead directly exercise attn caching by calling one block.
    # Build a minimal rope cache (zeros) that matches expected shape: (2,1,1,1,L,dim_div2)
    # For Infinity's 4D RoPE, last dim should be head_dim/2.
    head_dim = infinity.C // infinity.num_heads
    rope_cache = torch.zeros(
        (2, 1, 1, 1, 1, head_dim // 2),
        device="cuda",
        dtype=model_dtype,
    )
    # Prepare a minimal hidden state [B,L,C]
    h = torch.zeros((1, 1, infinity.C), device="cuda", dtype=torch.float32)
    blk0 = infinity.unregistered_blocks[0]
    with torch.amp.autocast("cuda", dtype=model_dtype):
        _ = blk0(x=h, cond_BD=None, ca_kv=None, attn_bias_or_two_vector=None, attn_fn=None, rope2d_freqs_grid=rope_cache, scale_schedule=None, scale_ind="t0_gt", context_info=None, last_repetition_step=True, ref_text_scale_inds=[])

    print("after GT write:", _count_cache_entries(infinity))

    # 2) Write Pred cache entry: another key 'pred_step0'
    infinity.set_cache_write_is_pred(True)
    with torch.amp.autocast("cuda", dtype=model_dtype):
        _ = blk0(x=h, cond_BD=None, ca_kv=None, attn_bias_or_two_vector=None, attn_fn=None, rope2d_freqs_grid=rope_cache, scale_schedule=None, scale_ind="pred_step0", context_info=None, last_repetition_step=True, ref_text_scale_inds=[])

    print("after Pred write:", _count_cache_entries(infinity))

    # 3) Clear pred cache
    infinity.clear_pred_cache()
    print("after clear_pred_cache:", _count_cache_entries(infinity))


if __name__ == "__main__":
    main()

