#!/usr/bin/env python3
"""Run full evaluation on UAV-Flow dataset with periodic visualization."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, desc=""):
    """Run shell command and print output."""
    print(f"\n{'='*60}")
    if desc:
        print(f"[{desc}]")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: command returned {result.returncode}")
    return result.returncode


def extract_trajectories(parquet_dir, out_dir, num_trajectories, min_frames=40):
    """Extract trajectories from parquet files."""
    cmd = f"""python3 scripts/extract_samples.py \
        --parquet "{parquet_dir}/train-00000-of-00054.parquet" \
        --out_dir "{out_dir}" \
        --num_trajectories {num_trajectories} \
        --min_frames {min_frames}"""
    return run_cmd(cmd, "Extract trajectories")


def run_inference(dataset_root, server_url, out_dir, num_frames=49, step=16):
    """Run inference on dataset."""
    cmd = f"""python3 infer/client.py \
        --mode dataset \
        --dataset_root "{dataset_root}" \
        --server_url {server_url} \
        --out_dir "{out_dir}" \
        --num_frames {num_frames} \
        --step {step} \
        --prefix_mode 1 \
        --allow_future_last_seg 1"""
    return run_cmd(cmd, "Run inference")


def convert_to_eval_format(results_root, gt_root, out_root):
    """Convert results to evaluation format."""
    cmd = f"""python3 scripts/convert_to_eval_format.py \
        --results_root "{results_root}" \
        --gt_root "{gt_root}" \
        --out_root "{out_root}" \
        --run_id test_run_002"""
    return run_cmd(cmd, "Convert to eval format")


def run_evaluation(pred_root, gt_root, out_root):
    """Run official evaluation."""
    cmd = f"""python3 train/action_decoder/tools/eval_endpoints.py \
        --pred_root "{pred_root}" \
        --gt_root "{gt_root}" \
        --out_root "{out_root}" \
        --gt_pose_file preprocessed_logs.json \
        --translation_divisor 1.0 \
        --angles_in_degrees \
        --dist_thr_m 3.0 \
        --ang_thr_deg 10.0"""
    return run_cmd(cmd, "Run evaluation")


def visualize_selected(samples_root, results_root, out_dir, interval=10):
    """Visualize trajectories at fixed intervals."""
    # Get list of trajectories
    traj_dirs = sorted([d for d in Path(samples_root).iterdir() if d.is_dir()])

    # Select at fixed intervals
    selected = traj_dirs[::interval]
    print(f"\nVisualizing {len(selected)} trajectories (every {interval}th)")

    for traj_dir in selected:
        traj_id = traj_dir.name
        result_dir = os.path.join(results_root, traj_id)
        if os.path.exists(result_dir):
            cmd = f"""python3 scripts/visualize_results.py \
                --sample_dir "{traj_dir}" \
                --results_root "{results_root}" \
                --run_id test_run_002 \
                --out_dir "{out_dir}" """
            subprocess.run(cmd, shell=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Run full evaluation on UAV-Flow")
    parser.add_argument("--parquet_dir", type=str, default="uav-flow", help="Parquet files directory")
    parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to evaluate")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8001", help="Inference server URL")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames per trajectory")
    parser.add_argument("--step", type=int, default=16, help="Step size for segments")
    parser.add_argument("--vis_interval", type=int, default=10, help="Visualization interval")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference (use existing results)")
    args = parser.parse_args()

    # Directory setup
    eval_samples_dir = "eval_samples_full"
    eval_results_dir = "eval_results_full/client_run_test_run_002"
    eval_converted_dir = "eval_results_full_converted"
    eval_output_dir = "eval_output_full"
    eval_vis_dir = "eval_vis_full"

    if not args.skip_inference:
        # Step 1: Extract trajectories
        print("\n" + "="*80)
        print("STEP 1: Extracting trajectories from parquet")
        print("="*80)
        extract_trajectories(args.parquet_dir, eval_samples_dir, args.num_trajectories)

        # Step 2: Run inference
        print("\n" + "="*80)
        print("STEP 2: Running inference")
        print("="*80)
        run_inference(eval_samples_dir, args.server_url, eval_results_dir, args.num_frames, args.step)

    # Step 3: Convert to evaluation format
    print("\n" + "="*80)
    print("STEP 3: Converting to evaluation format")
    print("="*80)
    convert_to_eval_format(eval_results_dir, eval_samples_dir, eval_converted_dir)

    # Step 4: Run evaluation
    print("\n" + "="*80)
    print("STEP 4: Running evaluation")
    print("="*80)
    run_evaluation(eval_converted_dir, eval_samples_dir, eval_output_dir)

    # Step 5: Visualize selected trajectories
    print("\n" + "="*80)
    print("STEP 5: Visualizing selected trajectories")
    print("="*80)
    visualize_selected(eval_samples_dir, eval_converted_dir, eval_vis_dir, args.vis_interval)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Results: {eval_output_dir}/summary.txt")
    print(f"Visualizations: {eval_vis_dir}/")


if __name__ == "__main__":
    main()
