import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import torch
import math

# Add current directory to path to import inference_p2p
sys.path.append(os.getcwd())
try:
    from inference_p2p import run_inference
except ImportError:
    print("Error: Could not import inference_p2p. Make sure inference_p2p.py is in the current directory.")
    sys.exit(1)

def find_latent_files(root_dir):
    latent_files = []
    for root, dirs, files in os.walk(root_dir):
        if "video_summed_codes.npy" in files:
            latent_files.append(os.path.join(root, "video_summed_codes.npy"))
    return latent_files

def calculate_endpoint_error(traj_pred, traj_gt):
    if traj_pred is None or traj_gt is None:
        return None
    
    # Check if lengths match or take the minimum length
    # Usually we compare the last available point
    
    # Assuming both start at (0,0,0)
    end_pred = traj_pred[-1, :3] # x, y, z
    end_gt = traj_gt[-1, :3]
    
    # Calculate Euclidean distance
    error = np.linalg.norm(end_pred - end_gt)
    return error


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "nan"
    return f"{float(v):.6f}"

def main():
    parser = argparse.ArgumentParser(description="Batch Inference and Error Analysis")
    parser.add_argument("--test_root", type=str, default="/home/dataset-assist-0/xjc/TSformer-VO-main/test_data_latent", help="Root directory of test data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--stats_path", type=str, default=None, help="Path to stats file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save summary and histogram")
    
    # Args needed for run_inference but fixed for this batch run
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print(f"Searching for latent files in {args.test_root}...")
    latent_files = find_latent_files(args.test_root)
    print(f"Found {len(latent_files)} files.")
    
    endpoint_errors = []
    attitude_endpoint_errors = []
    attitude_mean_errors = []
    results = []  # store dict rows
    
    for latent_path in tqdm(latent_files):
        # Prepare args for run_inference
        # We need to construct a namespace-like object or modify args
        
        # Set specific paths for this iteration
        args.latent_path = latent_path
        args.output_plot = os.path.join(os.path.dirname(latent_path), "inference_result.png")
        args.output_rotation_plot = os.path.join(os.path.dirname(latent_path), "rotation_plot.png")
        
        # Run inference
        # Suppress print output from run_inference to keep tqdm clean? 
        # For now, let it print.
        
        out = run_inference(args)
        if isinstance(out, tuple) and len(out) == 3:
            traj_pred, traj_gt, metrics = out
        elif isinstance(out, tuple) and len(out) == 2:
            traj_pred, traj_gt = out
            metrics = {}
        else:
            traj_pred, traj_gt, metrics = None, None, {}

        if traj_pred is not None and traj_gt is not None:
            endpoint_error = metrics.get("endpoint_error_m", None)
            if endpoint_error is None:
                endpoint_error = calculate_endpoint_error(traj_pred, traj_gt)
            attitude_endpoint_error = metrics.get("attitude_endpoint_error_deg", None)
            if attitude_endpoint_error is None:
                # backward compatibility with previous metric key
                attitude_endpoint_error = metrics.get("yaw_endpoint_error_deg", None)
            attitude_mean_error = metrics.get("attitude_mean_error_deg", None)
            if endpoint_error is not None:
                endpoint_errors.append(float(endpoint_error))
            if attitude_endpoint_error is not None:
                attitude_endpoint_errors.append(float(attitude_endpoint_error))
            if attitude_mean_error is not None:
                attitude_mean_errors.append(float(attitude_mean_error))
            results.append(
                {
                    "latent_path": latent_path,
                    "endpoint_error_m": endpoint_error,
                    "attitude_endpoint_error_deg": attitude_endpoint_error,
                    "attitude_mean_error_deg": attitude_mean_error,
                    "pred_len": metrics.get("pred_len", len(traj_pred)),
                    "gt_len": metrics.get("gt_len", len(traj_gt)),
                    "rotation_plot_path": metrics.get("rotation_plot_path", ""),
                    "status": "ok",
                }
            )
        else:
            results.append(
                {
                    "latent_path": latent_path,
                    "endpoint_error_m": None,
                    "attitude_endpoint_error_deg": None,
                    "attitude_mean_error_deg": None,
                    "pred_len": 0,
                    "gt_len": 0,
                    "rotation_plot_path": "",
                    "status": "Inference or GT loading failed",
                }
            )

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "accuracy_report.txt")
    print(f"Saving summary to {summary_path}...")
    with open(summary_path, 'w') as f:
        processed_count = sum(1 for r in results if r["status"] == "ok")
        skipped_count = len(results) - processed_count
        f.write("Test Accuracy Report\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"processed_count: {processed_count}\n")
        f.write(f"skipped_count: {skipped_count}\n\n")

        if endpoint_errors:
            ep = np.asarray(endpoint_errors, dtype=np.float32)
            f.write("Summary\n")
            f.write(f"endpoint_error_mean_m: {_fmt(float(ep.mean()))}\n")
            f.write(f"endpoint_error_median_m: {_fmt(float(np.median(ep)))}\n")
            f.write(f"endpoint_error_p90_m: {_fmt(float(np.percentile(ep, 90)))}\n")
            f.write(f"endpoint_error_min_m: {_fmt(float(ep.min()))}\n")
            f.write(f"endpoint_error_max_m: {_fmt(float(ep.max()))}\n")
            f.write("\n")
        else:
            f.write("Summary\n")
            f.write("endpoint_error_*: N/A\n\n")

        if attitude_endpoint_errors:
            ae = np.asarray(attitude_endpoint_errors, dtype=np.float32)
            f.write(f"attitude_endpoint_error_mean_deg: {_fmt(float(ae.mean()))}\n")
            f.write(f"attitude_endpoint_error_median_deg: {_fmt(float(np.median(ae)))}\n")
            f.write(f"attitude_endpoint_error_p90_deg: {_fmt(float(np.percentile(ae, 90)))}\n")
            f.write(f"attitude_endpoint_error_min_deg: {_fmt(float(ae.min()))}\n")
            f.write(f"attitude_endpoint_error_max_deg: {_fmt(float(ae.max()))}\n")
            if attitude_mean_errors:
                ame = np.asarray(attitude_mean_errors, dtype=np.float32)
                f.write(f"attitude_framewise_mean_error_deg: {_fmt(float(ame.mean()))}\n")
            f.write("\n")
        else:
            f.write("attitude_error_*: N/A\n\n")

        # Top-20 worst endpoint
        ok_rows = [r for r in results if r["status"] == "ok" and r["endpoint_error_m"] is not None]
        worst = sorted(ok_rows, key=lambda x: float(x["endpoint_error_m"]), reverse=True)[:20]
        f.write("Top-20 Worst Endpoint Errors\n")
        for r in worst:
            f.write(
                f"{r['latent_path']} | endpoint={_fmt(r['endpoint_error_m'])} m | "
                f"attitude_end={_fmt(r['attitude_endpoint_error_deg'])} deg | "
                f"pred_len={r['pred_len']} gt_len={r['gt_len']} | "
                f"rot_plot={r['rotation_plot_path']}\n"
            )
        f.write("\n")

        f.write("Per-trajectory Metrics\n")
        f.write(
            "latent_path\tendpoint_error_m\tattitude_endpoint_error_deg\tattitude_mean_error_deg\t"
            "pred_len\tgt_len\trotation_plot\tstatus\n"
        )
        for r in results:
            f.write(
                f"{r['latent_path']}\t{_fmt(r['endpoint_error_m'])}\t"
                f"{_fmt(r['attitude_endpoint_error_deg'])}\t{_fmt(r['attitude_mean_error_deg'])}\t"
                f"{r['pred_len']}\t{r['gt_len']}\t{r['rotation_plot_path']}\t{r['status']}\n"
            )

    # Plot endpoint histogram
    endpoint_hist_path = os.path.join(args.output_dir, "endpoint_error_hist.png")
    if len(endpoint_errors) > 0:
        print(f"Plotting endpoint histogram to {endpoint_hist_path}...")
        plt.figure(figsize=(10, 6))
        plt.hist(endpoint_errors, bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Endpoint Euclidean Distance Errors')
        plt.xlabel('Error (m)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add mean/median text
        mean_err = np.mean(endpoint_errors)
        median_err = np.median(endpoint_errors)
        plt.axvline(mean_err, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_err:.2f}m')
        plt.axvline(median_err, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_err:.2f}m')
        plt.legend()
        
        plt.savefig(endpoint_hist_path)
        plt.close()
        print("Endpoint histogram saved.")
        
        print(f"Mean Error: {mean_err:.4f} m")
        print(f"Median Error: {median_err:.4f} m")
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No valid endpoint error data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(endpoint_hist_path)
        plt.close()
        print("No valid endpoint errors calculated to plot.")

    # Plot attitude endpoint error histogram
    yaw_hist_path = os.path.join(args.output_dir, "angle_error_hist.png")
    if len(attitude_endpoint_errors) > 0:
        print(f"Plotting angle histogram to {yaw_hist_path}...")
        plt.figure(figsize=(10, 6))
        plt.hist(attitude_endpoint_errors, bins=20, color='mediumpurple', edgecolor='black')
        plt.title('Distribution of Attitude Endpoint Errors')
        plt.xlabel('Attitude Error (deg)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)

        mean_yaw = np.mean(attitude_endpoint_errors)
        median_yaw = np.median(attitude_endpoint_errors)
        plt.axvline(mean_yaw, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_yaw:.2f}deg')
        plt.axvline(median_yaw, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_yaw:.2f}deg')
        plt.legend()
        plt.savefig(yaw_hist_path)
        plt.close()
        print("Angle histogram saved.")
    else:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No valid angle error data", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(yaw_hist_path)
        plt.close()
        print("No valid angle errors calculated to plot.")

if __name__ == "__main__":
    main()
