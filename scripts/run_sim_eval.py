#!/usr/bin/env python3
"""Run offline evaluation on UAV-Flow-Sim trajectories.

Pipeline:
1. Extract trajectories from uav-flow-sim parquet
2. Run inference via server
3. Convert to eval format
4. Run official eval (SR = qualified rate)
5. Visualize selected trajectories
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_run_dir(results_base):
    """Auto-detect the run directory (client_run_*) inside results_base."""
    p = Path(results_base)
    if not p.exists():
        return results_base
    candidates = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("client_run_")]
    if candidates:
        return str(candidates[0])
    return results_base


def run(cmd, desc=""):
    print(f"\n{'='*60}")
    if desc:
        print(f"[{desc}]")
    print(f"Running: {cmd}")
    print('='*60)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"Warning: returned {r.returncode}")
    return r.returncode


def main():
    parser = argparse.ArgumentParser(description="UAV-Flow-Sim offline evaluation")
    parser.add_argument("--parquet", type=str, default="uav-flow-sim/train-00000-of-00021.parquet")
    parser.add_argument("--num_trajectories", type=int, default=273)
    parser.add_argument("--min_frames", type=int, default=5)
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    args = parser.parse_args()

    samples_dir = "eval_samples_sim"
    results_dir = "eval_results_sim/client_run_sim_test"
    converted_dir = "eval_results_sim_converted"
    output_dir = "eval_output_sim"
    vis_dir = "eval_vis_sim"

    # Step 1: Extract
    if not args.skip_extract:
        run(f'python3 scripts/extract_samples.py '
            f'--parquet "{args.parquet}" '
            f'--out_dir "{samples_dir}" '
            f'--num_trajectories {args.num_trajectories} '
            f'--min_frames {args.min_frames}',
            "Extract trajectories")

    # Step 2: Inference
    if not args.skip_inference:
        run(f'python3 infer/client.py '
            f'--mode dataset '
            f'--dataset_root "{samples_dir}" '
            f'--server_url {args.server_url} '
            f'--out_dir "{results_dir}" '
            f'--num_frames {args.num_frames} '
            f'--step {args.step} '
            f'--prefix_mode 1 '
            f'--allow_future_last_seg 1 '
            f'--pad_short_real 1',
            "Run inference")

    # Step 3: Convert (auto-detect run directory)
    actual_results = find_run_dir(results_dir)
    print(f"Detected results directory: {actual_results}")
    run(f'python3 scripts/convert_to_eval_format.py '
        f'--results_root "{actual_results}" '
        f'--gt_root "{samples_dir}" '
        f'--out_root "{converted_dir}"',
        "Convert to eval format")

    # Step 4: Evaluate
    run(f'python3 train/action_decoder/tools/eval_endpoints.py '
        f'--pred_root "{converted_dir}" '
        f'--gt_root "{samples_dir}" '
        f'--out_root "{output_dir}" '
        f'--gt_pose_file preprocessed_logs.json '
        f'--translation_divisor 100.0 '
        f'--angles_in_degrees '
        f'--dist_thr_m 3.0 '
        f'--ang_thr_deg 10.0',
        "Run evaluation")

    # Step 5: Visualize
    run(f'python3 scripts/visualize_results.py '
        f'--samples_root "{samples_dir}" '
        f'--results_root "{converted_dir}" '
        f'--out_dir "{vis_dir}"',
        "Visualize")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {output_dir}/summary.txt")
    print(f"Visualizations: {vis_dir}/")


if __name__ == "__main__":
    main()
