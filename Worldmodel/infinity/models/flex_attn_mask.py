# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT
from functools import partial
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
try:
    # Present in newer torch versions.
    from torch._dynamo.exc import CacheLimitExceeded  # type: ignore
except Exception:
    # Older torch may not expose this symbol.
    CacheLimitExceeded = None  # type: ignore

from infinity.schedules.dynamic_resolution import get_full_spatial_size_scale_indices, get_first_full_spatial_size_scale_index


def _length_to_offsets(lengths, device):
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    visual = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), counts)
    return visual

def _generate_video_tower_mask(offsets, context_frames, full_resolution_scales, prefix_lens):
    document_id = _offsets_to_doc_ids_tensor(offsets)
    visual_tokens = offsets[-2]
    def _mask_prefix_valid(b, h, q_idx, kv_idx):
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx >= visual_tokens) & (q_idx < text_token_ends) & (kv_idx >= visual_tokens) & (kv_idx < text_token_ends)
    def _mask_visual(b, h, q_idx, kv_idx):
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx < visual_tokens) & (
                        (document_id[q_idx] == document_id[kv_idx]) | 
                        ((kv_idx >= visual_tokens) & (kv_idx < text_token_ends)) | 
                        (
                            (document_id[q_idx] > document_id[kv_idx]) & (document_id[q_idx] - document_id[kv_idx] < context_frames) & (document_id[kv_idx] in full_resolution_scales)
                        )
                    )
    def video_tower_mask(b, h, q_idx, kv_idx):
        mask_prefix_valid = _mask_prefix_valid(b, h, q_idx, kv_idx)
        mask_visual = _mask_visual(b, h, q_idx, kv_idx)
        return mask_prefix_valid | mask_visual
    return video_tower_mask

def _generate_two_pyramid_mask(offsets, first_full_spatial_size_scale_index, prefix_lens):
    document_id = _offsets_to_doc_ids_tensor(offsets)
    visual_tokens = offsets[-2]
    def _mask_prefix_valid(b, h, q_idx, kv_idx):
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx >= visual_tokens) & (q_idx < text_token_ends) & (kv_idx >= visual_tokens) & (kv_idx < text_token_ends)
    def _mask_visual(b, h, q_idx, kv_idx):
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx < visual_tokens) & (
                        (document_id[q_idx] == document_id[kv_idx]) | 
                        ((kv_idx >= visual_tokens) & (kv_idx < text_token_ends)) | 
                        (document_id[q_idx] > document_id[kv_idx]) & (document_id[kv_idx] == first_full_spatial_size_scale_index)
                    )
    def video_two_pyramid_mask(b, h, q_idx, kv_idx):
        mask_prefix_valid = _mask_prefix_valid(b, h, q_idx, kv_idx)
        mask_visual = _mask_visual(b, h, q_idx, kv_idx)
        return mask_prefix_valid | mask_visual
    return video_two_pyramid_mask

def _generate_inner_scale_only_mask(offsets, prefix_lens):
    document_id = _offsets_to_doc_ids_tensor(offsets)
    visual_tokens = offsets[-2]
    def _mask_prefix_valid(b, h, q_idx, kv_idx):
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx >= visual_tokens) & (q_idx < text_token_ends) & (kv_idx >= visual_tokens) & (kv_idx < text_token_ends)
    def _mask_visual(b, h, q_idx, kv_idx):
        text_token_ends = visual_tokens + prefix_lens[b]
        return (q_idx < visual_tokens) & (
                        (document_id[q_idx] == document_id[kv_idx]) | 
                        ((kv_idx >= visual_tokens) & (kv_idx < text_token_ends))
                    )
    def overall_mask(b, h, q_idx, kv_idx):
        mask_prefix_valid = _mask_prefix_valid(b, h, q_idx, kv_idx)
        mask_visual = _mask_visual(b, h, q_idx, kv_idx)
        return mask_prefix_valid | mask_visual
    return overall_mask

def _generate_infinity_pack(offsets, querysid_refsid):
    document_id = _offsets_to_doc_ids_tensor(offsets) # to scale_ind
    def overall_mask(b, h, q_idx, kv_idx):
        querysid = document_id[q_idx]
        kv_sid = document_id[kv_idx]
        return querysid_refsid[querysid][kv_sid]
    return overall_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _is_cache_limit_exceeded(err: Exception) -> bool:
    """Compat check across torch versions where CacheLimitExceeded may not exist."""
    if (CacheLimitExceeded is not None) and isinstance(err, CacheLimitExceeded):
        return True
    err_name = type(err).__name__.lower()
    err_msg = str(err).lower()
    return ("cachelimitexceeded" in err_name) or ("cache limit" in err_msg)

def build_flex_attn_func(
        flex_attention,
        seq_l,
        prefix_lens,
        args,
        device,
        batch_size,
        heads,
        pad_seq_len,
        sequece_packing_scales,
        super_scale_lengths,
        super_querysid_super_refsid,
):
    """
    Build a flex attn function for a given scale schedule.
    Args:
        flex_attention: compiled flex attention
        seq_l: seq length
        prefix_lens: valid text prefix lens, [bs]
        args: arguments
        device: device
        batch_size: batch size
        heads: heads
        pad_seq_len: pad_seq_len
        sequece_packing_scales: list of scale schedule
        querysid_refsid: list of scale_pack_info
    Returns:
        attn_fn: flex attn function
    """
    assert sum(super_scale_lengths) == seq_l, f'{sum(super_scale_lengths)}!= {seq_l}'
    offsets = _length_to_offsets(super_scale_lengths, device=device)
    mask_mod = _generate_infinity_pack(offsets, super_querysid_super_refsid)
    # TorchDynamo compilation of block masks can hit CacheLimitExceeded or trigger
    # device-side asserts in Triton/Inductor on some stacks. Allow opt-out via env:
    #   INFINITY_COMPILE_BLOCK_MASK=0
    if not hasattr(build_flex_attn_func, "_compile_block_mask"):
        env = os.environ.get("INFINITY_COMPILE_BLOCK_MASK", "1").strip().lower()
        build_flex_attn_func._compile_block_mask = env not in ("0", "false", "no", "off")
    # Some torch versions/stacks can trigger CUDA device-side asserts inside flex_attention's
    # mask-to-blockmask conversion. Allow building block_mask on CPU and moving it to CUDA:
    #   INFINITY_BLOCK_MASK_DEVICE=cpu
    bm_dev = os.environ.get("INFINITY_BLOCK_MASK_DEVICE", "cuda").strip().lower()
    block_mask_device = torch.device("cpu") if bm_dev in ("cpu", "host") else device
    try:
        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=heads,
            Q_LEN=seq_l,
            KV_LEN=seq_l,
            device=block_mask_device,
            _compile=bool(build_flex_attn_func._compile_block_mask),
        )
    except Exception as e:
        if not _is_cache_limit_exceeded(e):
            raise
        build_flex_attn_func._compile_block_mask = False
        block_mask = create_block_mask(
            mask_mod,
            B=batch_size,
            H=heads,
            Q_LEN=seq_l,
            KV_LEN=seq_l,
            device=block_mask_device,
            _compile=False,
        )
    if block_mask_device.type == "cpu" and device.type == "cuda":
        # BlockMask supports .to(...) in newer torch; fall back to identity if unavailable.
        try:
            block_mask = block_mask.to(device)
        except Exception:
            pass
    attn_fn = partial(flex_attention, block_mask=block_mask)
    return attn_fn
