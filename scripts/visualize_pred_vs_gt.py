#!/usr/bin/env python3
"""Create side-by-side comparison videos: predicted frames vs GT frames.

For each trajectory:
- Left: GT frames from eval_samples_sim/{traj_id}/images/
- Right: stitched prediction from infer/cache/:
  GT frame 1 + seg00_new4_16f.mp4 + seg01_new4_16f.mp4 + seg02_new4_16f.mp4
- Overlay: instruction text
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def find_latent_dir(cache_dir: Path, traj_id: str, run_id: str) -> Path:
    """Find the Infinity latent cache directory for a trajectory."""
    # Cache dir naming: {traj_id}__{run_id}__{run_id}_{hash}_infinity_latnet
    pattern = f"{traj_id}__{run_id}__*_infinity_latnet"
    matches = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return matches[0]


def find_pred_video(latent_dir: Path, source: str) -> Path:
    """Find a single full-horizon predicted video."""
    if source == "full_final":
        preferred = ["seg02_pred_full_049f.mp4", "seg01_pred_full_049f.mp4", "seg00_pred_full_049f.mp4"]
    elif source == "full_seg00":
        preferred = ["seg00_pred_full_049f.mp4", "seg00_pred_full_033f.mp4", "seg00_pred_full_017f.mp4"]
    else:
        preferred = []

    for name in preferred:
        p = latent_dir / name
        if p.exists():
            return p

    for p in sorted(latent_dir.glob("seg*_pred_full_*.mp4"), reverse=True):
        return p
    return None


def get_gt_frames(gt_dir: Path) -> list:
    """Get sorted list of GT frame paths."""
    img_dir = gt_dir / "images"
    if not img_dir.exists():
        return []
    frames = sorted(img_dir.glob("frame_*.jpg")) + sorted(img_dir.glob("frame_*.png"))
    return frames


def load_instruction(gt_dir: Path) -> str:
    """Load instruction from meta.json."""
    meta_path = gt_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            d = json.load(f)
        return d.get("instruction_unified", d.get("instruction", ""))
    return ""


def read_video_frames(video_path: Path) -> list:
    """Read all frames from an mp4 as BGR arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def build_stitched_pred_frames(latent_dir: Path, gt_frames: list, max_frames: int = 49) -> list:
    """Build closed-loop-style video: GT first frame + each segment's new 16 frames."""
    pred_frames = []

    if gt_frames:
        first = cv2.imread(str(gt_frames[0]))
        if first is not None:
            pred_frames.append(first)

    seg_videos = sorted(latent_dir.glob("seg*_new4_16f.mp4"))
    for video_path in seg_videos:
        pred_frames.extend(read_video_frames(video_path))

    if max_frames > 0:
        pred_frames = pred_frames[:max_frames]
    return pred_frames


def make_comparison_video(
    gt_frames: list,
    pred_frames: list,
    out_path: Path,
    instruction: str,
    pred_label: str,
    target_h: int = 360,
    fps: int = 10,
):
    """Create side-by-side comparison video."""
    if not pred_frames:
        print("  Warning: no predicted frames")
        return False

    # Read GT frames
    gt_imgs = []
    for fp in gt_frames:
        img = cv2.imread(str(fp))
        if img is not None:
            gt_imgs.append(img)

    if not gt_imgs:
        print(f"  Warning: no GT frames")
        return False

    # Determine output dimensions
    # Resize both to same height
    def resize_to_h(img, h):
        scale = h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    gt_sample = resize_to_h(gt_imgs[0], target_h)
    pred_sample = resize_to_h(pred_frames[0], target_h)

    # Use the wider width for both
    panel_w = max(gt_sample.shape[1], pred_sample.shape[1])
    total_w = panel_w * 2 + 10  # 10px gap

    # Text overlay height
    text_h = 40
    total_h = target_h + text_h

    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        print(f"  Warning: cannot open video writer: {out_path}")
        return False

    max_len = min(len(gt_imgs), len(pred_frames))

    for i in range(max_len):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # GT panel (left)
        gt_resized = resize_to_h(gt_imgs[i], target_h)
        # Center in panel
        x_off = (panel_w - gt_resized.shape[1]) // 2
        canvas[text_h:text_h + target_h, x_off:x_off + gt_resized.shape[1]] = gt_resized

        # Pred panel (right)
        pred_resized = resize_to_h(pred_frames[i], target_h)
        x_off = panel_w + 10 + (panel_w - pred_resized.shape[1]) // 2
        canvas[text_h:text_h + target_h, x_off:x_off + pred_resized.shape[1]] = pred_resized

        # Labels
        cv2.putText(canvas, "GT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, pred_label, (panel_w + 20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        # Instruction text
        if instruction:
            cv2.putText(canvas, instruction[:80], (10, text_h + target_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Frame counter
        cv2.putText(canvas, f"Frame {i+1}/{max_len}", (total_w - 180, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(canvas)

    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare predicted vs GT frames")
    parser.add_argument("--samples_root", type=str, default="eval_samples_sim",
                        help="GT samples directory")
    parser.add_argument("--cache_dir", type=str, default="infer/cache",
                        help="Server cache directory with predicted videos")
    parser.add_argument("--run_id", type=str, default="",
                        help="Run ID to match (auto-detect if empty)")
    parser.add_argument("--out_dir", type=str, default="eval_vis_sim_pred",
                        help="Output directory for comparison videos")
    parser.add_argument("--pred_source", type=str, default="stitched_new4",
                        choices=["stitched_new4", "full_final", "full_seg00"],
                        help="Prediction source: stitched_new4 uses GT first frame + seg*_new4_16f; full_final uses latest seg*_pred_full_049f; full_seg00 uses seg00_pred_full")
    parser.add_argument("--target_h", type=int, default=360,
                        help="Target height for each panel")
    parser.add_argument("--fps", type=int, default=10,
                        help="Output video FPS")
    parser.add_argument("--max_traj", type=int, default=0,
                        help="Max trajectories to process (0=all)")
    args = parser.parse_args()

    samples_root = Path(args.samples_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not samples_root.exists():
        print(f"Error: samples_root not found: {samples_root}")
        return 1

    # Auto-detect run_id from results
    run_id = args.run_id
    if not run_id:
        # Find latest run_id from cache dirs
        all_dirs = list(cache_dir.glob("*_infinity_latnet"))
        run_ids = set()
        for d in all_dirs:
            parts = d.name.split("__")
            if len(parts) >= 2:
                run_ids.add(parts[1])
        if run_ids:
            run_id = sorted(run_ids)[-1]
            print(f"Auto-detected run_id: {run_id}")
        else:
            print("Error: no cached predictions found. Run inference first.")
            return 1

    # Process each trajectory
    traj_dirs = sorted([d for d in samples_root.iterdir() if d.is_dir()])
    if args.max_traj > 0:
        traj_dirs = traj_dirs[:args.max_traj]

    print(f"Processing {len(traj_dirs)} trajectories (run_id={run_id})")

    success = 0
    skipped = 0
    for i, traj_dir in enumerate(traj_dirs):
        traj_id = traj_dir.name
        instruction = load_instruction(traj_dir)

        # Find latent cache dir
        latent_dir = find_latent_dir(cache_dir, traj_id, run_id)
        if latent_dir is None:
            skipped += 1
            continue

        # Get GT frames
        gt_frames = get_gt_frames(traj_dir)
        if not gt_frames:
            skipped += 1
            continue

        if args.pred_source == "stitched_new4":
            pred_frames = build_stitched_pred_frames(latent_dir, gt_frames, max_frames=len(gt_frames))
            pred_label = "Pred stitched"
        else:
            pred_path = find_pred_video(latent_dir, args.pred_source)
            if pred_path is None:
                skipped += 1
                continue
            pred_frames = read_video_frames(pred_path)
            pred_label = "Pred full"

        # Generate comparison video
        out_path = out_dir / f"{traj_id}_compare.mp4"
        ok = make_comparison_video(
            gt_frames,
            pred_frames,
            out_path,
            instruction,
            pred_label=pred_label,
            target_h=args.target_h,
            fps=args.fps,
        )
        if ok:
            success += 1
            print(f"[{i+1}/{len(traj_dirs)}] {traj_id} -> {out_path.name}")
        else:
            skipped += 1

    print(f"\nDone: {success} videos generated, {skipped} skipped")
    print(f"Output: {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
