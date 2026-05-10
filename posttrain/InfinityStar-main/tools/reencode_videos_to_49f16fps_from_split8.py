#!/usr/bin/env python3
# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Re-encode videos referenced by split8 jsonl shards into fixed-length clips:
- Always output exactly N frames (default 49)
- Always output a fixed container FPS (default 16)
- For videos with <N frames: uniformly *pad* by repeating frames (endpoints kept)
- For videos with >N frames: uniformly *downsample* (endpoints kept)

Also writes a new split8 jsonl directory pointing to the re-encoded videos, with:
  begin_frame_id=0, end_frame_id=N-1, fps=fps_out, cap_frames=N,
  frame_idxs=[0..N-1], sample_frames=N

This makes training with `video_frames=N` and `video_fps=fps_out` robust even
when the original videos are too short in duration.

Example:
  conda run -p /home/batchcom/dataset-link/xjc/infinitystar --no-capture-output \
    python tools/reencode_videos_to_49f16fps_from_split8.py \
      --input_split_dir /home/batchcom/dataset-link/xjc/wan_meta_selected_targets_and_distance_split8_jsonl \
      --out_split_dir /home/batchcom/dataset-link/xjc/wan_meta_selected_targets_and_distance_fixed49f16fps_split8_jsonl \
      --out_video_root /home/batchcom/dataset-link/xjc/wan_meta_selected_targets_and_distance_fixed49f16fps_videos \
      --num_workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_relpath_from_prefix(abs_path: str, prefix: str) -> Optional[str]:
    abs_path = osp.abspath(abs_path)
    prefix = osp.abspath(prefix)
    try:
        rel = osp.relpath(abs_path, prefix)
    except Exception:
        return None
    if rel.startswith(".."):
        return None
    return rel.replace("\\", "/")


def _out_path_for_video(video_path: str, out_video_root: str, common_prefix: Optional[str]) -> str:
    vp = osp.abspath(video_path)
    rel: Optional[str] = None
    if common_prefix:
        rel = _safe_relpath_from_prefix(vp, common_prefix)
    if not rel:
        # fallback: flatten absolute path into a filename-safe relative path
        rel = vp.lstrip("/").replace("/", "__")
    out_path = osp.join(osp.abspath(out_video_root), rel)
    # force .mp4 extension (some inputs might be .webm etc.)
    base, ext = osp.splitext(out_path)
    if ext.lower() != ".mp4":
        out_path = base + ".mp4"
    return out_path


def _read_all_frames_bgr(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot_open: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"no_frames: {video_path}")
    return frames


def _select_indices_uniform(n_src: int, n_tgt: int) -> np.ndarray:
    if n_tgt <= 0:
        raise ValueError(f"{n_tgt=}")
    if n_src <= 0:
        raise ValueError(f"{n_src=}")
    if n_src == 1:
        return np.zeros((n_tgt,), dtype=np.int64)
    # uniform mapping, keep endpoints; duplicates => padding, skips => downsample
    idx = np.rint(np.linspace(0, n_src - 1, n_tgt)).astype(np.int64)
    idx[0] = 0
    idx[-1] = n_src - 1
    idx = np.clip(idx, 0, n_src - 1)
    return idx


def _verify_video(video_path: str, n_frames: int, fps_out: float) -> Tuple[bool, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False, "cannot_open"
    n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if n != int(n_frames):
        return False, f"frame_count={n}"
    # fps in containers can be slightly off; allow tiny tolerance
    if fps_out > 0 and fps > 0 and abs(fps - fps_out) > 0.2:
        return False, f"fps={fps}"
    return True, "ok"


def _write_video_mp4(frames_bgr: List[np.ndarray], out_path: str, fps_out: float) -> None:
    h, w = frames_bgr[0].shape[:2]
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    # Keep a real container extension so ffmpeg can infer muxer.
    tmp_path = out_path + ".tmp.mp4"
    if osp.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    for fr in frames_bgr:
        if fr.shape[0] != h or fr.shape[1] != w:
            raise RuntimeError("inconsistent_frame_size")
        if fr.dtype != np.uint8:
            raise RuntimeError(f"bad_dtype: {fr.dtype}")

    # Use ffmpeg for encoding to avoid OpenCV builds without mp4 encoders.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(float(fps_out)),
        "-i",
        "pipe:0",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        tmp_path,
    ]
    raw = b"".join([fr.tobytes() for fr in frames_bgr])
    p = subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", errors="ignore")[:2000]
        raise RuntimeError(f"ffmpeg_encode_failed: {err}")
    os.replace(tmp_path, out_path)


@dataclass(frozen=True)
class Job:
    src: str
    dst: str


def _process_one(job: Job, n_frames: int, fps_out: float, overwrite: bool) -> Tuple[str, str]:
    if (not overwrite) and osp.exists(job.dst):
        ok, msg = _verify_video(job.dst, n_frames=n_frames, fps_out=fps_out)
        if ok:
            return job.src, "skip_exists_ok"
        # re-generate if existing is wrong
    frames = _read_all_frames_bgr(job.src)
    idx = _select_indices_uniform(len(frames), int(n_frames))
    out_frames = [frames[int(i)] for i in idx.tolist()]
    _write_video_mp4(out_frames, job.dst, fps_out=fps_out)
    ok, msg = _verify_video(job.dst, n_frames=n_frames, fps_out=fps_out)
    if not ok:
        raise RuntimeError(f"verify_failed: {msg}")
    return job.src, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_split_dir", type=str, default="")
    src.add_argument("--input_jsonl", type=str, default="")
    ap.add_argument(
        "--video_prefix",
        type=str,
        default="",
        help="When using --input_jsonl with relative `video`/`video_path`, join with this prefix to form absolute video_path.",
    )
    ap.add_argument("--out_split_dir", type=str, required=True)
    ap.add_argument("--out_video_root", type=str, required=True)
    ap.add_argument("--num_frames", type=int, default=49)
    ap.add_argument("--fps_out", type=float, default=16.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_shards", type=int, default=8, help="Only used with --input_jsonl")
    ap.add_argument(
        "--common_prefix",
        type=str,
        default="/home/batchcom/dataset-link/xjc",
        help="When src video_path is under this prefix, preserve relative path under out_video_root.",
    )
    args = ap.parse_args()

    input_split_dir = Path(osp.abspath(args.input_split_dir)) if args.input_split_dir else None
    input_jsonl = osp.abspath(args.input_jsonl) if args.input_jsonl else None
    out_split_dir = Path(osp.abspath(args.out_split_dir))
    out_video_root = osp.abspath(args.out_video_root)
    n_frames = int(args.num_frames)
    fps_out = float(args.fps_out)
    num_workers = max(1, int(args.num_workers))
    overwrite = bool(args.overwrite)
    common_prefix = str(args.common_prefix or "").strip() or None
    num_shards = max(1, int(args.num_shards))
    video_prefix = osp.abspath(args.video_prefix) if args.video_prefix else ""

    def normalize_to_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
        raw_video = obj.get("video") or obj.get("video_path") or obj.get("path") or ""
        prompt = obj.get("prompt") or obj.get("caption") or obj.get("text") or ""
        if not isinstance(raw_video, str):
            raw_video = str(raw_video)
        if not isinstance(prompt, str):
            prompt = str(prompt)

        rel = raw_video.strip().replace("\\", "/").lstrip("./")
        # common forms: "videos/xxx.mp4" or "uavflowdatasim_output/.."
        if rel.startswith("videos/"):
            rel = rel[len("videos/") :]

        if osp.isabs(rel):
            video_path = rel
        else:
            if not video_prefix:
                raise RuntimeError("input_jsonl has relative video path but --video_prefix not provided")
            video_path = osp.join(video_prefix, rel)
            # Fallback for datasets that store `data/...` paths under TSformer repo.
            if (not osp.exists(video_path)) and rel.startswith("data/"):
                alt_root = osp.join(video_prefix, "actionhead/TSformer-VO-main/TSformer-VO-main")
                alt_path = osp.join(alt_root, rel)
                if osp.exists(alt_path):
                    video_path = alt_path

        return {
            "video_path": osp.abspath(video_path),
            "begin_frame_id": 0,
            "end_frame_id": n_frames - 1,
            "fps": float(fps_out),
            "tarsier2_caption": prompt.strip(),
            "cap_frames": n_frames,
            "frame_idxs": list(range(n_frames)),
            "sample_frames": n_frames,
        }

    parts: List[Path] = []
    all_rows_by_part: List[Tuple[Path, List[Dict[str, Any]]]] = []
    total_rows = 0
    if input_split_dir is not None:
        parts = sorted(input_split_dir.glob("part_*.jsonl"))
        if not parts:
            raise SystemExit(f"no parts found under {input_split_dir}")
        for p in parts:
            rows = list(_iter_jsonl(str(p)))
            total_rows += len(rows)
            all_rows_by_part.append((p, rows))
    else:
        assert input_jsonl is not None
        shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                meta = normalize_to_meta(obj)
                shards[i % num_shards].append(meta)
                total_rows += 1
        # Use conventional part naming; only used for writing out_split_dir.
        for shard_id, rows in enumerate(shards):
            p = Path(f"part_{shard_id:02d}.jsonl")
            all_rows_by_part.append((p, rows))

    # Load metas, collect unique video paths
    uniq: Dict[str, str] = {}
    for p, rows in all_rows_by_part:
        for r in rows:
            vp = r.get("video_path", "")
            if not vp:
                continue
            if vp not in uniq:
                uniq[vp] = _out_path_for_video(vp, out_video_root=out_video_root, common_prefix=common_prefix)

    jobs = [Job(src=k, dst=v) for k, v in uniq.items()]
    print(json.dumps({"total_rows": total_rows, "unique_videos": len(jobs)}, ensure_ascii=False, indent=2))

    ok_cnt = skip_cnt = fail_cnt = 0
    fail_list: List[Tuple[str, str]] = []
    failed_src: set[str] = set()

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        fut2job = {ex.submit(_process_one, j, n_frames, fps_out, overwrite): j for j in jobs}
        for fut in as_completed(fut2job):
            try:
                _src, status = fut.result()
                if status == "ok":
                    ok_cnt += 1
                else:
                    skip_cnt += 1
            except Exception as e:
                j = fut2job[fut]
                fail_cnt += 1
                failed_src.add(j.src)
                fail_list.append((j.src, str(e)))

    # Write new split8 jsonl
    kept = 0
    bad_rows: List[Dict[str, Any]] = []
    for p, rows in all_rows_by_part:
        out_rows: List[Dict[str, Any]] = []
        for r in rows:
            vp = r.get("video_path", "")
            if (not vp) or (vp not in uniq) or (vp in failed_src):
                bad_rows.append(r)
                continue
            new_vp = uniq[vp]
            r2 = dict(r)
            r2["video_path"] = new_vp
            r2["begin_frame_id"] = 0
            r2["end_frame_id"] = n_frames - 1
            r2["fps"] = float(fps_out)
            r2["cap_frames"] = n_frames
            r2["frame_idxs"] = list(range(n_frames))
            r2["sample_frames"] = n_frames
            out_rows.append(r2)
        out_file = out_split_dir / p.name
        _write_jsonl(str(out_file), out_rows)
        kept += len(out_rows)

    if bad_rows:
        _write_jsonl(str(out_split_dir / "bad_rows.jsonl"), bad_rows)

    if fail_list:
        _write_jsonl(
            str(out_split_dir / "failed_jobs.jsonl"),
            [{"src": s, "error": e} for s, e in fail_list],
        )

    print(
        json.dumps(
            {
                "input_split_dir": str(input_split_dir) if input_split_dir is not None else "",
                "input_jsonl": str(input_jsonl) if input_jsonl is not None else "",
                "video_prefix": video_prefix,
                "out_split_dir": str(out_split_dir),
                "out_video_root": out_video_root,
                "num_frames": n_frames,
                "fps_out": fps_out,
                "num_workers": num_workers,
                "overwrite": overwrite,
                "num_shards": num_shards,
                "common_prefix": common_prefix,
                "unique_videos": len(jobs),
                "video_ok": ok_cnt,
                "video_skip": skip_cnt,
                "video_fail": fail_cnt,
                "rows_total": total_rows,
                "rows_kept": kept,
                "rows_bad": len(bad_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

