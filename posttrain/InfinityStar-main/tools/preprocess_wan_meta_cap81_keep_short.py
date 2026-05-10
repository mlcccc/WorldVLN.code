#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Preprocess `/path/to/wan_meta.jsonl` into InfinityStar training meta shards, with:
- frame index list (frame_idxs) to enforce:
  - if video has > cap_frames: keep first/last, uniformly subsample middle frames
  - else: keep all frames (no dropping), or uniformly pad to cap_frames when enabled
- optional forced aspect-ratio template for 480p-like center crop during training

This script does NOT encode / cache VAE tokens. Training can extract features online.

Example:
  python3 tools/preprocess_wan_meta_cap81_keep_short.py \
    --input_jsonl /home/batchcom/dataset-link/xjc/wan_meta.jsonl \
    --video_root /home/batchcom/dataset-link/xjc \
    --out_dir data/wan_split_jsonls_cap81_keep_short \
    --chunk_size 1000 \
    --cap_frames 81 \
    --force_h_div_w_template 0.562
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any, Dict, Iterator, List, Tuple, Optional

import cv2
import numpy as np


def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _probe_video(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 1e-6:
        fps = 25.0
    if frame_count <= 0:
        frame_count = 1
    return frame_count, fps, width, height


def _make_frame_idxs(frame_count: int, cap_frames: int, pad_short_to_cap: bool = False) -> List[int]:
    """Return frame indices with boundary-preserving sampling (and optional short-video padding)."""
    frame_count = int(frame_count)
    cap_frames = int(cap_frames)
    if frame_count <= 0:
        return [0]
    if cap_frames <= 0:
        return list(range(frame_count))
    if frame_count <= cap_frames:
        if not pad_short_to_cap or frame_count == cap_frames:
            return list(range(frame_count))
        # Uniformly pad short videos to cap_frames while preserving first/last.
        if frame_count == 1:
            return [0] * cap_frames
        if frame_count == 2:
            xs = np.linspace(0, 1, num=cap_frames, dtype=np.float64)
            idxs = np.rint(xs).astype(np.int64).tolist()
            idxs[0], idxs[-1] = 0, 1
            return idxs
        k = cap_frames - 2
        lo, hi = 1, frame_count - 2
        mid = np.linspace(lo, hi, num=k, dtype=np.float64)
        mid = np.rint(mid).astype(np.int64)
        mid = np.clip(mid, lo, hi)
        out = [0] + mid.astype(int).tolist() + [frame_count - 1]
        assert len(out) == cap_frames
        return out

    # Keep boundaries, uniformly choose internal frames.
    k = cap_frames - 2
    assert k >= 0
    if k == 0:
        return [0, frame_count - 1]

    # Candidate internal range [1, frame_count-2]
    lo, hi = 1, frame_count - 2
    xs = np.linspace(lo, hi, num=k, dtype=np.float64)
    idxs = np.rint(xs).astype(np.int64)
    idxs = np.clip(idxs, lo, hi)

    # Enforce strictly increasing (avoid duplicates from rounding)
    for i in range(1, len(idxs)):
        if idxs[i] <= idxs[i - 1]:
            idxs[i] = idxs[i - 1] + 1
    # If we overflowed hi, shift back.
    if idxs[-1] > hi:
        overflow = int(idxs[-1] - hi)
        idxs = idxs - overflow
        idxs = np.clip(idxs, lo, hi)
        for i in range(1, len(idxs)):
            if idxs[i] <= idxs[i - 1]:
                idxs[i] = idxs[i - 1] + 1
        idxs = np.clip(idxs, lo, hi)

    # Final safety: if still not strictly increasing due to extreme corner cases, fallback to simple range.
    if not np.all(idxs[1:] > idxs[:-1]):
        idxs = np.arange(lo, lo + k, dtype=np.int64)
        idxs = np.clip(idxs, lo, hi)

    out = [0] + idxs.astype(int).tolist() + [frame_count - 1]
    assert len(out) == cap_frames
    return out


def _write_chunk(out_file: str, metas: List[Dict[str, Any]]) -> None:
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def _rewrite_video_path(src_path: str, rewrite_prefix: str, strip_videos_prefix: bool) -> str:
    if not rewrite_prefix:
        return src_path
    rel = src_path.replace("\\", "/").lstrip("./")
    if strip_videos_prefix and rel.startswith("videos/"):
        rel = rel[len("videos/") :]
    return osp.join(rewrite_prefix, rel)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--video_root", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--cap_frames", type=int, default=81)
    ap.add_argument(
        "--pad_short_to_cap",
        type=int,
        default=0,
        help="If 1, videos shorter than cap_frames are uniformly padded (repeat indices) to cap_frames.",
    )
    ap.add_argument("--min_frames", type=int, default=1)
    ap.add_argument(
        "--force_h_div_w_template",
        type=float,
        default=0.562,
        help="Force aspect ratio template (e.g. 0.562 ~ 9/16) for 480p-like center crop. "
             "Set to 0 to disable.",
    )
    ap.add_argument(
        "--num_shards",
        type=int,
        default=0,
        help="If > 0, write exactly num_shards jsonl files for multi-GPU parallel input.",
    )
    ap.add_argument(
        "--rewrite_video_prefix",
        type=str,
        default="",
        help="If set, rewrite input video relative path to this absolute prefix.",
    )
    ap.add_argument(
        "--strip_videos_prefix",
        type=int,
        default=1,
        help="When rewriting video prefix, strip leading 'videos/' from the input path.",
    )
    args = ap.parse_args()

    in_path = args.input_jsonl
    out_dir = args.out_dir
    chunk_size = max(1, int(args.chunk_size))
    cap_frames = int(args.cap_frames)
    pad_short_to_cap = int(args.pad_short_to_cap) == 1
    min_frames = max(1, int(args.min_frames))
    force_tpl = float(args.force_h_div_w_template)
    num_shards = max(0, int(args.num_shards))
    rewrite_prefix = str(args.rewrite_video_prefix).strip()
    strip_videos_prefix = int(args.strip_videos_prefix) == 1
    if abs(force_tpl) < 1e-9:
        force_tpl = 0.0

    total = kept = bad = 0
    shard_rows: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)] if num_shards > 0 else []

    def flush(chunk_id: int, chunk: List[Dict[str, Any]]) -> None:
        if not chunk:
            return
        bucket = (chunk_id - 1) // 1000 + 1
        out_file = osp.join(out_dir, f"{bucket:06d}", f"{chunk_id:04d}_XXXX_000000000.jsonl")
        _write_chunk(out_file, chunk)

    chunk_id = 1
    chunk: List[Dict[str, Any]] = []

    for obj in _iter_jsonl(in_path):
        total += 1
        rel = obj.get("video") or obj.get("video_path") or obj.get("path")
        prompt = obj.get("prompt") or obj.get("caption") or obj.get("text") or ""
        if not rel or not isinstance(rel, str):
            bad += 1
            continue
        if not isinstance(prompt, str):
            prompt = str(prompt)

        video_path = rel
        video_path = _rewrite_video_path(video_path, rewrite_prefix, strip_videos_prefix)
        if args.video_root and not osp.isabs(video_path):
            video_path = osp.join(args.video_root, video_path)
        video_path = osp.abspath(video_path)
        if not osp.exists(video_path):
            bad += 1
            continue

        try:
            frame_count, fps, width, height = _probe_video(video_path)
        except Exception:
            bad += 1
            continue

        if frame_count < min_frames:
            continue

        frame_idxs = _make_frame_idxs(frame_count, cap_frames=cap_frames, pad_short_to_cap=pad_short_to_cap)
        meta: Dict[str, Any] = {
            "video_path": video_path,
            "begin_frame_id": 0,
            "end_frame_id": frame_count - 1,  # inclusive
            "fps": fps,
            "tarsier2_caption": prompt.strip(),
            "width": width,
            "height": height,
            "cap_frames": cap_frames,
            "frame_idxs": frame_idxs,  # absolute indices in the container
            "sample_frames": len(frame_idxs),
        }
        if force_tpl > 0:
            meta["force_h_div_w_template"] = force_tpl

        if num_shards > 0:
            shard_rows[(kept % num_shards)].append(meta)
        else:
            chunk.append(meta)
        kept += 1
        if num_shards <= 0 and len(chunk) >= chunk_size:
            flush(chunk_id, chunk)
            chunk_id += 1
            chunk = []

    if num_shards > 0:
        for sid, rows in enumerate(shard_rows):
            shard_file = osp.join(out_dir, f"part_{sid:02d}.jsonl")
            _write_chunk(shard_file, rows)
    else:
        flush(chunk_id, chunk)
    print(
        json.dumps(
            {
                "total": total,
                "kept": kept,
                "bad": bad,
                "out_dir": osp.abspath(out_dir),
                "num_shards": num_shards,
                "shard_sizes": [len(x) for x in shard_rows] if num_shards > 0 else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

