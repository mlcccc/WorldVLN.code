#!/usr/bin/env python3
"""Visualize WorldVLN converted predictions against ground-truth trajectories."""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_gt_poses(preprocessed_logs_path):
    """Load ground truth poses from preprocessed_logs.json: [x, y, z, roll, yaw, pitch].

    Returns poses in centimeters (cm) to match predicted poses units.
    """
    logs = load_json(preprocessed_logs_path)
    poses = np.array([row[:6] for row in logs], dtype=np.float32)
    max_abs_xyz = float(np.max(np.abs(poses[:, :3]))) if poses.size else 0.0
    # UAV-Flow uses meters, while UAV-Flow-Sim samples here use centimeters.
    if max_abs_xyz <= 100.0:
        poses[:, :3] = poses[:, :3] * 100.0
    return poses


def load_converted_pred_poses(result_dir):
    """Load converted predicted poses from pred_path.json.

    Converted poses use [roll_rad, yaw_rad, pitch_rad, x_m, y_m, z_m].
    Return [x, y, z, roll, yaw, pitch] in cm/degrees to match GT loading.
    """
    pred_path = Path(result_dir) / "pred_path.json"
    if not pred_path.exists():
        return np.array([])

    data = load_json(pred_path)
    poses = np.array(data.get("poses", []), dtype=np.float32)
    if poses.size == 0:
        return np.array([])

    out = np.zeros((len(poses), 6), dtype=np.float32)
    out[:, :3] = poses[:, 3:6] * 100.0
    out[:, 3:6] = poses[:, 0:3] * (180.0 / np.pi)
    return out


def load_raw_pred_poses(result_dir, prefix):
    """Load predicted poses from raw segment pose files."""
    all_points = []
    seg_files = sorted(Path(result_dir).glob(f"{prefix}_seg*_poses.json"))
    for sf in seg_files:
        data = load_json(sf)
        pts = data.get("points", [])
        all_points.extend(pts)
    return np.array(all_points) if all_points else np.array([])


def load_converted_actions(result_dir):
    """Load converted pred_actions.json and map to plot order.

    pred_actions.json stores rotations first and translations in meters:
    [dyaw_rad, dpitch_rad, droll_rad, tx_m, ty_m, tz_m].
    The action plot expects [dx, dy, dz, droll, dyaw, dpitch] in cm/degrees.
    """
    actions_path = Path(result_dir) / "pred_actions.json"
    if not actions_path.exists():
        return []

    data = load_json(actions_path)
    actions = np.array(data.get("actions6", []), dtype=np.float32)
    if actions.size == 0:
        return []

    mapped = np.zeros((len(actions), 6), dtype=np.float32)
    mapped[:, 0:3] = actions[:, 3:6] * 100.0
    mapped[:, 3] = actions[:, 2] * (180.0 / np.pi)
    mapped[:, 4] = actions[:, 0] * (180.0 / np.pi)
    mapped[:, 5] = actions[:, 1] * (180.0 / np.pi)
    return [{"actions_server_order": mapped.tolist()}]


def normalize_trajectory(poses):
    """Center trajectory at origin (relative to first point)."""
    if len(poses) == 0:
        return poses
    origin = poses[0].copy()
    centered = poses - origin
    return centered


def plot_trajectory_comparison(gt_poses, pred_poses, instruction, out_path, title=""):
    """Plot GT vs predicted trajectory in XY plane."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Normalize both to start from origin
    gt_norm = normalize_trajectory(gt_poses[:, :3]) if len(gt_poses) > 0 else np.array([])
    pred_norm = normalize_trajectory(pred_poses[:, :3]) if len(pred_poses) > 0 else np.array([])

    # XY plane (top-down view)
    ax = axes[0]
    if len(gt_norm) > 0:
        ax.plot(gt_norm[:, 0], gt_norm[:, 1], "b-o", markersize=3, linewidth=1.5, label="Ground Truth", alpha=0.8)
        ax.plot(gt_norm[0, 0], gt_norm[0, 1], "bs", markersize=10, label="Start")
        ax.plot(gt_norm[-1, 0], gt_norm[-1, 1], "b^", markersize=10, label="GT End")
    if len(pred_norm) > 0:
        ax.plot(pred_norm[:, 0], pred_norm[:, 1], "r-s", markersize=3, linewidth=1.5, label="Predicted", alpha=0.8)
        ax.plot(pred_norm[-1, 0], pred_norm[-1, 1], "r^", markersize=10, label="Pred End")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("XY Plane (Top-Down)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Altitude (Z) over time
    ax2 = axes[1]
    if len(gt_norm) > 0:
        ax2.plot(gt_norm[:, 2], "b-o", markersize=2, label="GT Z", alpha=0.8)
    if len(pred_norm) > 0:
        ax2.plot(pred_norm[:, 2], "r-s", markersize=2, label="Pred Z", alpha=0.8)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Z (cm)")
    ax2.set_title("Altitude over Time")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add instruction as prominent text box
    instruction_text = f"Instruction: {instruction}"
    fig.text(0.5, 0.02, instruction_text, ha='center', va='bottom',
             fontsize=12, fontweight='bold', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for instruction text
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_actions(actions_data, out_path, title=""):
    """Plot action distributions per segment."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    labels = ["dx", "dy", "dz", "droll", "dyaw", "dpitch"]
    units = ["cm", "cm", "cm", "deg", "deg", "deg"]

    all_actions = []
    for seg_data in actions_data:
        acts = seg_data.get("actions_server_order", [])
        all_actions.extend(acts)

    if not all_actions:
        plt.close()
        return

    all_actions = np.array(all_actions)
    for i, (label, unit) in enumerate(zip(labels, units)):
        ax = axes[i // 3][i % 3]
        ax.hist(all_actions[:, i], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(f"{label} ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} distribution")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)

    fig.suptitle(f"Action Distributions\n{title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_frame_strip(sample_dir, out_path, n_frames=6):
    """Create a strip of key input frames."""
    images_dir = os.path.join(sample_dir, "images")
    if not os.path.isdir(images_dir):
        return

    frame_files = sorted(Path(images_dir).glob("frame_*.jpg"))
    if len(frame_files) < n_frames:
        n_frames = len(frame_files)

    indices = np.linspace(0, len(frame_files) - 1, n_frames, dtype=int)
    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4))

    for i, idx in enumerate(indices):
        img = Image.open(frame_files[idx])
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {idx + 1}", fontsize=9)
        axes[i].axis("off")

    fig.suptitle("Input Frames", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_3d_trajectory(gt_poses, pred_poses, instruction, out_path, title=""):
    """Plot 3D trajectory comparison."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    gt_norm = normalize_trajectory(gt_poses[:, :3]) if len(gt_poses) > 0 else np.array([])
    pred_norm = normalize_trajectory(pred_poses[:, :3]) if len(pred_poses) > 0 else np.array([])

    if len(gt_norm) > 0:
        ax.plot(gt_norm[:, 0], gt_norm[:, 1], gt_norm[:, 2], "b-o", markersize=2, linewidth=1.5, label="Ground Truth")
        ax.scatter(*gt_norm[0], c="blue", s=100, marker="s", label="Start")
        ax.scatter(*gt_norm[-1], c="blue", s=100, marker="^", label="GT End")
    if len(pred_norm) > 0:
        ax.plot(pred_norm[:, 0], pred_norm[:, 1], pred_norm[:, 2], "r-s", markersize=2, linewidth=1.5, label="Predicted")
        ax.scatter(*pred_norm[-1], c="red", s=100, marker="^", label="Pred End")

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title("3D Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # Add instruction as prominent text box
    instruction_text = f"Instruction: {instruction}"
    fig.text(0.5, 0.02, instruction_text, ha='center', va='bottom',
             fontsize=11, fontweight='bold', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def find_run_id(result_dir, traj_id):
    """Auto-detect run_id from result directory."""
    # Try to find summary file
    summary_files = list(Path(result_dir).glob("*_summary.json"))
    if summary_files:
        summary = load_json(summary_files[0])
        session_id = summary.get("session_id", "")
        if "__" in session_id:
            return session_id.split("__")[-1]

    # Try to find poses file and extract run_id
    poses_files = list(Path(result_dir).glob("*_seg*_poses.json"))
    if poses_files:
        fname = poses_files[0].stem
        # Pattern: {traj_id}__{run_id}_seg{XX}_poses
        if "__" in fname:
            parts = fname.split("__")
            if len(parts) >= 2:
                run_part = parts[1].split("_seg")[0]
                return run_part

    return None


def resolve_result_dir(results_root, traj_id):
    """Resolve either results_root/traj_id or an already trajectory-specific dir."""
    root = Path(results_root)
    if (root / "pred_path.json").exists() or list(root.glob("*_seg*_poses.json")):
        return str(root)
    return str(root / traj_id)


def process_trajectory(sample_dir, result_dir, run_id, out_dir):
    """Process one trajectory and generate all visualizations."""
    traj_id = os.path.basename(sample_dir)

    # Load instruction
    meta = load_json(os.path.join(sample_dir, "meta.json"))
    instruction = meta.get("instruction", meta.get("instruction_unified", ""))

    # Load GT poses (use preprocessed_logs.json which is in relative coordinates)
    preprocessed_logs_path = os.path.join(sample_dir, "preprocessed_logs.json")
    gt_poses = load_gt_poses(preprocessed_logs_path) if os.path.exists(preprocessed_logs_path) else np.array([])

    # Prefer converted predictions. Fall back to raw client segment files.
    pred_poses = load_converted_pred_poses(result_dir)
    actions_data = load_converted_actions(result_dir)

    if len(pred_poses) == 0:
        actual_run_id = run_id or find_run_id(result_dir, traj_id)
        if not actual_run_id:
            print(f"Skipping {traj_id}: no pred_path.json and cannot detect raw run_id")
            return

        prefix = f"{traj_id}__{actual_run_id}"
        pred_poses = load_raw_pred_poses(result_dir, prefix)
        action_files = sorted(Path(result_dir).glob(f"{prefix}_seg*_actions.json"))
        actions_data = [load_json(f) for f in action_files]

    os.makedirs(out_dir, exist_ok=True)

    # Generate plots
    plot_trajectory_comparison(
        gt_poses, pred_poses, instruction,
        os.path.join(out_dir, f"{traj_id}_trajectory.png"),
        title=f"Trajectory: {traj_id}"
    )

    if len(gt_poses) > 0 and len(pred_poses) > 0:
        plot_3d_trajectory(
            gt_poses, pred_poses, instruction,
            os.path.join(out_dir, f"{traj_id}_trajectory_3d.png"),
            title=traj_id
        )

    if actions_data:
        plot_actions(
            actions_data,
            os.path.join(out_dir, f"{traj_id}_actions.png"),
            title=traj_id
        )

    plot_frame_strip(sample_dir, os.path.join(out_dir, f"{traj_id}_frames.png"))

    # Print summary and compute metrics
    if len(gt_poses) > 0 and len(pred_poses) > 0:
        gt_end = gt_poses[-1, :3]
        pred_end = pred_poses[-1, :3] if len(pred_poses) > 0 else np.zeros(3)
        gt_norm = normalize_trajectory(gt_poses[:, :3])
        pred_norm = normalize_trajectory(pred_poses[:, :3])

        # Compute displacement error at each step
        min_len = min(len(gt_norm), len(pred_norm))
        if min_len > 0:
            ade = np.mean(np.linalg.norm(gt_norm[:min_len] - pred_norm[:min_len], axis=1))
            fde = np.linalg.norm(gt_norm[-1] - pred_norm[-1])

            # RMSE metrics (similar to predict_pose.py)
            rmse_xyz = np.sqrt(np.mean((gt_norm[:min_len] - pred_norm[:min_len]) ** 2))

            print(f"\n  [{traj_id}]")
            print(f"    Instruction: {instruction}")
            print(f"    GT frames: {len(gt_poses)}, Pred points: {len(pred_poses)}")
            print(f"    ADE (Avg Displacement Error): {ade:.2f} cm")
            print(f"    FDE (Final Displacement Error): {fde:.2f} cm")
            print(f"    RMSE XYZ: {rmse_xyz:.2f} cm")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT samples against converted predictions."
    )
    parser.add_argument("--sample_dir", type=str, default="", help="Single sample dir (with images/ and meta.json)")
    parser.add_argument("--samples_root", type=str, default="", help="Root of all sample dirs, e.g. eval_samples_sim")
    parser.add_argument("--results_root", type=str, default="", help="Converted prediction root, e.g. eval_results_sim_converted")
    parser.add_argument("--run_id", type=str, default="", help="Optional raw client run_id fallback")
    parser.add_argument("--out_dir", type=str, default="eval_vis", help="Output visualization dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.sample_dir:
        traj_id = os.path.basename(args.sample_dir.rstrip(os.sep))
        result_dir = resolve_result_dir(args.results_root, traj_id)
        if not os.path.isdir(result_dir):
            print(f"Skipping {traj_id}: no results found at {result_dir}")
            return
        process_trajectory(args.sample_dir, result_dir, args.run_id, args.out_dir)
    else:
        for name in sorted(os.listdir(args.samples_root)):
            sample_dir = os.path.join(args.samples_root, name)
            if not os.path.isdir(sample_dir):
                continue
            result_dir = resolve_result_dir(args.results_root, name)
            if not os.path.isdir(result_dir):
                print(f"Skipping {name}: no results found")
                continue
            process_trajectory(sample_dir, result_dir, args.run_id, args.out_dir)

    print(f"\nVisualizations saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
