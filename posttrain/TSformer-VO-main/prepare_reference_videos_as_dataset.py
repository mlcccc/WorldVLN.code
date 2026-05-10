"""
Prepare training data by matching:
  - data/reference_videos_all_v2/<id>/reference.mp4
  - data/test_jsons/<id>.json

into an "uavflowdatasim_output-like" folder that our existing training pipeline can use:

out_root/<id>/
  images/frame_000000.png ...
  raw_logs.json              # (T,6) absolute [x,y,z,roll,yaw,pitch] (typically cm + degrees)
  preprocessed_logs.json     # (T,6) relative-to-start (if available)
  meta.json                  # minimal metadata

Notes
- We sample video frames to match the length of reference_path_raw (or reference_path_preprocessed).
- We do NOT change units here. Keep whatever units test_json uses (often cm for xyz and degrees for angles).
  During training, use:
    --translation_divisor 100   (cm->m)
    --angles_in_degrees         (deg->rad)

Example:
  /opt/conda/bin/python3 prepare_reference_videos_as_dataset.py \\
    --videos_root data/reference_videos_all_v2 \\
    --json_root data/test_jsons \\
    --out_root data/reference_train_uavflow_like \\
    --ids 2025-03-30_11-49-14
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _read_all_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)  # keep BGR for imwrite
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Empty video: {video_path}")
    return frames


def _sample_indices(n_frames: int, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.arange(n_frames, dtype=np.int32)
    if target_len >= n_frames:
        return np.arange(n_frames, dtype=np.int32)
    return np.round(np.linspace(0, n_frames - 1, target_len)).astype(np.int32)


def _write_frames(frames_bgr: List[np.ndarray], idxs: np.ndarray, images_dir: str):
    _ensure_dir(images_dir)
    for out_i, src_i in enumerate(idxs.tolist()):
        frame = frames_bgr[int(src_i)]
        out_path = os.path.join(images_dir, f"frame_{out_i:06d}.png")
        cv2.imwrite(out_path, frame)


def _pick_reference_paths(task: Dict):
    """
    Prefer reference_path_raw (absolute), fall back to initial_pos only.
    """
    raw = task.get("reference_path_raw")
    pre = task.get("reference_path_preprocessed")
    if isinstance(raw, list) and len(raw) > 0:
        raw = raw
    else:
        raw = [task.get("initial_pos")] if task.get("initial_pos") else None
    if isinstance(pre, list) and len(pre) > 0:
        pre = pre
    else:
        pre = None
    return raw, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=str, required=True)
    ap.add_argument("--json_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--ids", type=str, default="all", help='Comma-separated ids or "all"')
    ap.add_argument("--max_ids", type=int, default=0, help="If >0, limit number of ids")
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    wanted = None
    if args.ids.strip().lower() not in ("all", "*", ""):
        wanted = set([p.strip() for p in args.ids.split(",") if p.strip()])

    ids = [d for d in os.listdir(args.videos_root) if os.path.isdir(os.path.join(args.videos_root, d))]
    ids.sort()
    if wanted is not None:
        ids = [i for i in ids if i in wanted]
    if args.max_ids and args.max_ids > 0:
        ids = ids[: int(args.max_ids)]

    _ensure_dir(args.out_root)

    kept = 0
    skipped = 0

    for vid_id in ids:
        video_path = os.path.join(args.videos_root, vid_id, "reference.mp4")
        json_path = os.path.join(args.json_root, f"{vid_id}.json")
        if not os.path.exists(video_path) or not os.path.exists(json_path):
            skipped += 1
            continue

        out_dir = os.path.join(args.out_root, vid_id)
        images_dir = os.path.join(out_dir, "images")
        raw_out = os.path.join(out_dir, "raw_logs.json")
        pre_out = os.path.join(out_dir, "preprocessed_logs.json")
        meta_out = os.path.join(out_dir, "meta.json")

        if os.path.exists(out_dir) and os.listdir(out_dir) and not args.overwrite:
            skipped += 1
            continue

        task = _load_json(json_path)
        ref_raw, ref_pre = _pick_reference_paths(task)
        if not ref_raw or not isinstance(ref_raw, list) or len(ref_raw) < 4:
            skipped += 1
            continue

        # Use reference length to sample frames
        frames_bgr = _read_all_frames(video_path)
        idxs = _sample_indices(len(frames_bgr), target_len=len(ref_raw))

        # If still too short for window=4, skip
        if len(idxs) < 4:
            skipped += 1
            continue

        _ensure_dir(out_dir)
        _write_frames(frames_bgr, idxs, images_dir=images_dir)

        # Write raw_logs.json (absolute pose sequence)
        # Expect each entry: [x,y,z,roll,yaw,pitch] in original units (often cm + deg)
        with open(raw_out, "w") as f:
            json.dump(ref_raw, f)

        # Write preprocessed_logs.json if available; otherwise build relative-to-start cumulative
        if ref_pre is None:
            arr = np.asarray(ref_raw, dtype=np.float32)
            start = arr[0]
            pre_arr = (arr - start).tolist()
        else:
            pre_arr = ref_pre
        with open(pre_out, "w") as f:
            json.dump(pre_arr, f)

        meta = {
            "id": vid_id,
            "instruction": task.get("instruction"),
            "instruction_unified": task.get("instruction_unified"),
            "source_video": video_path,
            "source_json": json_path,
            "length": int(len(ref_raw)),
            "frames_in_video": int(len(frames_bgr)),
            "frames_written": int(len(idxs)),
        }
        with open(meta_out, "w") as f:
            json.dump(meta, f, indent=2)

        kept += 1

    print(f"Done. kept={kept} skipped={skipped} out_root={args.out_root}", flush=True)


if __name__ == "__main__":
    main()

