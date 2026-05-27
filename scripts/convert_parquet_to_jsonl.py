#!/usr/bin/env python3
"""Convert UAV-Flow parquet data to JSONL format for WorldVLN training.

Each parquet file contains rows with: id, frame_idx, image (bytes), log (JSON).
This script:
1. Groups rows by trajectory ID
2. Extracts JPEG images and creates MP4 videos
3. Writes JSONL entries with video_path, frame info, and captions
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import tqdm


def extract_video_from_frames(frame_rows, video_path, fps=16):
    """Create MP4 video from a sequence of image bytes (PNG or JPEG)."""
    from PIL import Image
    import io

    tmpdir = tempfile.mkdtemp(prefix="uavflow_")
    try:
        # Write frames as numbered JPEG files (convert from PNG if needed)
        for i, (_, row) in enumerate(frame_rows.iterrows()):
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_path = os.path.join(tmpdir, f"frame_{i+1:06d}.jpg")
            img.save(img_path, "JPEG", quality=95)

        # Create video with ffmpeg (use mpeg4 encoder, no libx264 preset)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.jpg"),
            "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
            "-q:v", "2", "-loglevel", "error",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                print(f"ffmpeg error for {os.path.basename(video_path)}: {stderr[:300]}")
            return False
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 100:
            print(f"Warning: output video missing or too small")
            return False
        return True
    finally:
        # Cleanup temp files
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)


def process_parquet_file(parquet_path, output_dir, fps=16, max_trajectories=None):
    """Process a single parquet file and return JSONL entries."""
    df = pd.read_parquet(parquet_path)
    video_dir = os.path.join(output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    jsonl_entries = []
    unique_ids = df["id"].unique()
    if max_trajectories:
        unique_ids = unique_ids[:max_trajectories]

    for traj_id in tqdm.tqdm(unique_ids, desc=f"Processing {os.path.basename(parquet_path)}"):
        traj_df = df[df["id"] == traj_id].sort_values("frame_idx")
        num_frames = len(traj_df)

        # Need at least 2 frames for a video
        if num_frames < 2:
            continue

        # Get instruction from the first frame's log
        log_data = json.loads(traj_df.iloc[0]["log"])
        instruction = log_data.get("instruction_unified", log_data.get("instruction", ""))
        if not instruction:
            continue

        # Create video
        safe_id = traj_id.replace("/", "_").replace(" ", "_")
        video_filename = f"{safe_id}.mp4"
        video_path = os.path.join(video_dir, video_filename)

        if not os.path.exists(video_path):
            success = extract_video_from_frames(traj_df, video_path, fps=fps)
            if not success:
                continue

        # Build JSONL entry
        # Cap frames to max_frames (49 by default, matching video_frames in training)
        max_frames = 49
        if num_frames > max_frames:
            # Uniformly subsample max_frames from the trajectory
            indices = [int(i * (num_frames - 1) / (max_frames - 1)) for i in range(max_frames)]
            frame_idxs = indices
            capped_frames = max_frames
        else:
            frame_idxs = list(range(num_frames))
            capped_frames = num_frames

        entry = {
            "video_path": os.path.abspath(video_path),
            "begin_frame_id": 0,
            "end_frame_id": num_frames - 1,
            "fps": float(fps),
            "tarsier2_caption": instruction,
            "frame_idxs": frame_idxs,
            "sample_frames": capped_frames,
        }
        jsonl_entries.append(entry)

    return jsonl_entries


def main():
    parser = argparse.ArgumentParser(description="Convert UAV-Flow parquet to JSONL")
    parser.add_argument("--parquet_dir", type=str, default="uav-flow",
                        help="Directory containing parquet files")
    parser.add_argument("--output_dir", type=str, default="train/data/uavflow_jsonl",
                        help="Output directory for JSONL and videos")
    parser.add_argument("--fps", type=int, default=16,
                        help="Video FPS for created videos")
    parser.add_argument("--max_per_shard", type=int, default=None,
                        help="Max trajectories per parquet file (for testing)")
    parser.add_argument("--num_shards", type=int, default=8,
                        help="Number of JSONL output shards")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    # Process all parquet files
    all_entries = []
    for pq_file in parquet_files:
        entries = process_parquet_file(
            str(pq_file), args.output_dir,
            fps=args.fps, max_trajectories=args.max_per_shard,
        )
        all_entries.extend(entries)
        print(f"  {pq_file.name}: {len(entries)} trajectories")

    print(f"\nTotal trajectories: {len(all_entries)}")

    # Split into shards
    shard_size = max(1, len(all_entries) // args.num_shards)
    for shard_idx in range(args.num_shards):
        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < args.num_shards - 1 else len(all_entries)
        shard_entries = all_entries[start:end]
        if not shard_entries:
            continue

        shard_path = os.path.join(args.output_dir, f"part_{shard_idx:02d}.jsonl")
        with open(shard_path, "w") as f:
            for entry in shard_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"  Written {shard_path}: {len(shard_entries)} entries")

    print("Done!")


if __name__ == "__main__":
    main()
