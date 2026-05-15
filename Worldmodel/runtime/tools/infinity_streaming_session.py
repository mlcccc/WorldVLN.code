# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Streaming/session wrapper for the InfinityStar KV-cache workflow.

Goal: match the lingbot-va semantics of `Compute KV -> Infer chunk -> Correction`.

Key conventions:
- Use 't0' as the text-prefix cache key and write it as GT (is_pred=False) so it won't be removed by clear_pred_cache().
- Use 'gt_obs' as the observation-frame cache key and write it as GT (is_pred=False).
- KV-cache entries written during inference are treated as Pred (is_pred=True) and are cleared in one shot during correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from tools.run_infinity import encode_prompt


@dataclass
class StreamingSchedule:
    scale_schedule: List[Tuple[int, int, int]]
    context_info: Dict[int, Dict[str, Any]]
    tgt_h: int
    tgt_w: int
    tower_split_index: int
    first_full_spatial_size_scale_index: int


def _count_cache_entries(infinity_model) -> Tuple[int, int, int]:
    """Return (total_entries, pred_entries, gt_entries) summed over blocks."""
    total = pred = gt = 0
    for blk in infinity_model.unregistered_blocks:
        meta = getattr(blk.attn, "cached_is_pred", {})
        total += len(meta)
        pred += sum(1 for v in meta.values() if v)
        gt += sum(1 for v in meta.values() if not v)
    return total, pred, gt


class InfinityStreamingSession:
    def __init__(
        self,
        *,
        args,
        infinity_model,
        vae,
        text_tokenizer,
        text_encoder,
        h_div_w_template: float = 0.571,
        gt_obs_cache_key: str = "gt_obs",
        gt_obs_rope_real_sid: int = 850,
    ):
        self.args = args
        self.infinity = infinity_model
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder

        self.h_div_w_template = float(h_div_w_template)
        self.gt_obs_cache_key = gt_obs_cache_key
        self.gt_obs_rope_real_sid = int(gt_obs_rope_real_sid)

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(
            args.dynamic_scale_schedule
        )

        self._text_cond_tuple = None
        self.bs = 1  # batch size for caching (1 for no-CFG, 2 for CFG)

    def build_schedule_for_num_frames(self, num_frames: int) -> StreamingSchedule:
        """Build scale_schedule / context_info for the current chunk (aligned with the official inference script)."""
        args = self.args
        dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - self.h_div_w_template))]

        # Video token timeline is temporally compressed: pt = (num_frames-1)//temporal_compress_rate + 1
        pt = (num_frames - 1) // args.temporal_compress_rate + 1
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"][pt]

        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        args.first_full_spatial_size_scale_index = first_full_spatial_size_scale_index
        args.tower_split_index = first_full_spatial_size_scale_index + 1
        context_info = self.get_scale_pack_info(scale_schedule, first_full_spatial_size_scale_index, args)

        tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
        return StreamingSchedule(
            scale_schedule=scale_schedule,
            context_info=context_info,
            tgt_h=tgt_h,
            tgt_w=tgt_w,
            tower_split_index=args.tower_split_index,
            first_full_spatial_size_scale_index=first_full_spatial_size_scale_index,
        )

    @torch.no_grad()
    def reset(self, prompt: str, negative_prompt: str = "", cfg_scale: float = 1.0):
        """Clear all KV caches, then write the text-prefix cache ('t0') as GT."""
        args = self.args
        model_dtype = next(iter(self.infinity.parameters())).dtype
        self.bs = 2 if float(cfg_scale) != 1.0 else 1

        # 1) reset all blocks' caches
        for blk in self.infinity.unregistered_blocks:
            blk.attn.kv_caching(True, reset=True)

        # 2) encode prompt (cond/uncond if CFG is used)
        text_cond_tuple = encode_prompt(args.text_encoder_ckpt, self.text_tokenizer, self.text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False)
        if negative_prompt:
            neg_tuple = encode_prompt(args.text_encoder_ckpt, self.text_tokenizer, self.text_encoder, negative_prompt, enable_positive_prompt=False, low_vram_mode=False)
        else:
            neg_tuple = None
        self._text_cond_tuple = (text_cond_tuple, neg_tuple)

        # 3) write text cache as GT (important: prevent being cleared by clear_pred_cache)
        self.infinity.set_cache_write_is_pred(False)

        # We re-use model helper to build prefix tokens, then forward once with scale_ind='t0'
        # This mirrors `ar_infer_infinity_*` "text tokens forward" block.
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_cond_tuple
        text_maxlen_this_iter = max_seqlen_k
        prefix_tokens, _ = self.infinity.prepare_text_conditions(
            label_B_or_BLT=text_cond_tuple,
            cfg_list=[float(cfg_scale)],
            B=1,
            # IMPORTANT: keep consistent with future inference (skip_text_forward=True).
            # If negative_prompt is provided and cfg_scale != 1, unconditional branch
            # should use negative prompt tokens instead of cfg_uncond.
            negative_label_B_or_BLT=neg_tuple,
            vae_scale_schedule=None,
            text_token_only=False,
            text_maxlen_this_iter=text_maxlen_this_iter,
        )

        device = prefix_tokens.device
        self.infinity.rope2d_freqs_grid["freqs_text"] = self.infinity.rope2d_freqs_grid["freqs_text"].to(device)
        rope_cache = self.infinity.rope2d_freqs_grid["freqs_text"][:, :, :, :, :text_maxlen_this_iter]

        block_chunks = self.infinity.block_chunks if getattr(self.infinity, "num_block_chunks", 1) > 1 else self.infinity.blocks
        last_stage = prefix_tokens.to(dtype=model_dtype)
        with torch.amp.autocast("cuda", dtype=model_dtype):
            for b in block_chunks:
                last_stage = b(
                    x=last_stage,
                    cond_BD=None,
                    ca_kv=None,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=None,
                    rope2d_freqs_grid=rope_cache.to(dtype=model_dtype),
                    scale_ind="t0",
                    context_info=None,
                    last_repetition_step=True,
                    ref_text_scale_inds=[],
                )

        self.infinity.set_cache_write_is_pred(True)

    @torch.no_grad()
    def compute_kv_cache_gt(self, obs_video_bcthw: torch.Tensor):
        """
        Step 1 / Step 4: encode observed frames into latent tokens and write them to the cache
        (key=gt_obs_cache_key, is_pred=False).
        """
        assert obs_video_bcthw.ndim == 5 and obs_video_bcthw.shape[1] == 3, "expect [B,3,T,H,W]"
        device = next(iter(self.infinity.parameters())).device
        dtype = next(iter(self.infinity.parameters())).dtype

        obs_video_bcthw = obs_video_bcthw.to(device=device, dtype=torch.float32)
        # VAE expects float in [-1,1]
        features, _, _ = self.vae.encode_for_raw_features(obs_video_bcthw, scale_schedule=None, slice=True)  # [B,d,t,h,w]

        pt, ph, pw = features.shape[-3:]
        scale_schedule = [(pt, ph, pw)]
        mini_scale_pack_info = {0: {"frame_ss": 0, "frame_ee": pt}}

        # Pick a valid RoPE scale index within precomputed range.
        max_scales = int(self.infinity.rope2d_freqs_grid["freqs_scales"].shape[1])
        real_sid = min(self.gt_obs_rope_real_sid, max_scales - 1)

        rope_cache = self.get_visual_rope_embeds(
            self.infinity.rope2d_freqs_grid,
            scale_schedule,
            0,  # sid
            real_sid,
            device,
            self.args,
            mini_scale_pack_info,
            0,  # first_full_spatial_size_scale_index
        )

        # write GT cache
        self.infinity.set_cache_write_is_pred(False)
        # repeat to match bs (CFG uses bs=2)
        last_stage = self.infinity.embeds_codes2input(features.to(dtype=dtype), repeat=self.bs)
        block_chunks = self.infinity.block_chunks if getattr(self.infinity, "num_block_chunks", 1) > 1 else self.infinity.blocks
        with torch.amp.autocast("cuda", dtype=dtype):
            for b in block_chunks:
                last_stage = b(
                    x=last_stage,
                    cond_BD=None,
                    ca_kv=None,
                    attn_bias_or_two_vector=None,
                    attn_fn=None,
                    scale_schedule=None,
                    rope2d_freqs_grid=rope_cache.to(dtype=dtype),
                    scale_ind=self.gt_obs_cache_key,
                    context_info=None,
                    last_repetition_step=True,
                    ref_text_scale_inds=[],
                )
        self.infinity.set_cache_write_is_pred(True)

    @torch.no_grad()
    def infer_chunk(
        self,
        *,
        num_frames: int,
        cfg_list: List[float],
        tau_list: List[float],
        top_k: int = 0,
        top_p: float = 0.0,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        low_vram_mode: bool = True,
        gt_leak: int = -1,
        gt_ls_Bl=None,
    ):
        """
        Step 2: infer a future chunk based on the current cache, and mark any cache entries written during inference
        as Pred (is_pred=True).

        Note: this uses the official `autoregressive_infer`, with:
        - kv_cache_reset=False: keep historical caches (including GT 't0' / 'gt_obs')
        - skip_text_forward=True: avoid re-writing text cache
        - extra_ref_text_scale_inds=['gt_obs']: allow all visual scales to attend to the observation cache
        """
        assert self._text_cond_tuple is not None, "call reset() first"
        text_cond_tuple, neg_tuple = self._text_cond_tuple
        if negative_prompt and neg_tuple is None:
            neg_tuple = encode_prompt(self.args.text_encoder_ckpt, self.text_tokenizer, self.text_encoder, negative_prompt, enable_positive_prompt=False, low_vram_mode=False)

        sched = self.build_schedule_for_num_frames(num_frames)

        # Ensure lists length
        if not isinstance(cfg_list, list):
            cfg_list = [cfg_list] * len(sched.scale_schedule)
        if not isinstance(tau_list, list):
            tau_list = [tau_list] * len(sched.scale_schedule)

        # pred writes
        self.infinity.set_cache_write_is_pred(True)

        # Only reference gt_obs cache when it has been written.
        # This avoids KeyError on step-0 direct-like inference where we intentionally
        # do not write gt_obs cache to prevent double-conditioning artifacts.
        has_gt_obs_cache = False
        for blk in self.infinity.unregistered_blocks:
            cached_k = getattr(blk.attn, "cached_k", {})
            if self.gt_obs_cache_key in cached_k:
                has_gt_obs_cache = True
                break
        extra_ref_text_scale_inds = [self.gt_obs_cache_key] if has_gt_obs_cache else []

        model_dtype = next(iter(self.infinity.parameters())).dtype
        with torch.amp.autocast("cuda", dtype=model_dtype):
            return self.infinity.autoregressive_infer(
                vae=self.vae,
                scale_schedule=sched.scale_schedule,
                label_B_or_BLT=text_cond_tuple,
                negative_label_B_or_BLT=neg_tuple,
                B=1,
                g_seed=seed,
                cfg_list=cfg_list,
                tau_list=tau_list,
                top_k=int(top_k or 0),
                top_p=float(top_p or 0.0),
                trunk_scale=1000,
                gt_leak=gt_leak,
                gt_ls_Bl=gt_ls_Bl,
                low_vram_mode=low_vram_mode,
                args=self.args,
                get_visual_rope_embeds=self.get_visual_rope_embeds,
                context_info=sched.context_info,
                kv_cache_reset=False,
                skip_text_forward=True,
                cache_text_as_gt=False,
                extra_ref_text_scale_inds=extra_ref_text_scale_inds,
            )

    def correction_clear_pred(self):
        """Step 4: clear Pred KV cache in one shot (GT caches are kept)."""
        self.infinity.clear_pred_cache()

    def cache_stats(self) -> Tuple[int, int, int]:
        return _count_cache_entries(self.infinity)

