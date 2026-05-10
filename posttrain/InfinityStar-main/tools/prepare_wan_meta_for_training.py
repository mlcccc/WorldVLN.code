#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Convert a simple jsonl like:
  {"video": "rel/path.mp4", "prompt": "..."}
into InfinityStar training meta jsonl shards:
  {"video_path": "...", "begin_frame_id": 0, "end_frame_id": N-1, "fps": fps, "tarsier2_caption": "..."}

Output layout matches JointViIterableDataset expectation:
  <out_dir>/<bucket_id>/<chunk_id>_... .jsonl

Example:
  python3 tools/prepare_wan_meta_for_training.py \
    --input_jsonl /home/batchcom/dataset-link/xjc/wan_meta.jsonl \
    --video_root /home/batchcom/dataset-link/xjc \
    --out_dir data/wan_split_jsonls \
    --chunk_size 1000
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Iterator, Dict, Any, List, Tuple

import cv2


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
    # fallbacks
    if fps <= 1e-6:
        fps = 16.0
    if frame_count <= 0:
        # last resort: treat as 1 frame to avoid crashes; caller may filter by min frames later
        frame_count = 1
    return frame_count, fps, width, height


def _write_chunk(out_file: str, metas: List[Dict[str, Any]]) -> None:
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--video_root", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--min_frames", type=int, default=1, help="drop videos with fewer than this many frames")
    args = ap.parse_args()

    in_path = args.input_jsonl
    video_root = args.video_root
    out_dir = args.out_dir
    chunk_size = max(1, int(args.chunk_size))

    # materialize and shard
    metas: List[Dict[str, Any]] = []
    total = 0
    kept = 0
    bad = 0

    def flush(chunk_id: int, chunk: List[Dict[str, Any]]) -> None:
        if not chunk:
            return
        # bucket every 1000 chunks to match existing convention (000001/000002/...)
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
        if video_root and not osp.isabs(video_path):
            video_path = osp.join(video_root, video_path)
        video_path = osp.abspath(video_path)
        if not osp.exists(video_path):
            bad += 1
            continue

        try:
            frame_count, fps, width, height = _probe_video(video_path)
        except Exception:
            bad += 1
            continue

        if frame_count < args.min_frames:
            continue

        meta = {
            "video_path": video_path,
            "begin_frame_id": 0,
            # Use inclusive end frame index (best-effort).
            "end_frame_id": frame_count - 1,
            "fps": fps,
            "tarsier2_caption": prompt.strip(),
            "width": width,
            "height": height,
        }
        chunk.append(meta)
        kept += 1
        if len(chunk) >= chunk_size:
            flush(chunk_id, chunk)
            chunk_id += 1
            chunk = []

    # flush remainder
    flush(chunk_id, chunk)

    print(
        json.dumps(
            {"total": total, "kept": kept, "bad": bad, "out_dir": osp.abspath(out_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

