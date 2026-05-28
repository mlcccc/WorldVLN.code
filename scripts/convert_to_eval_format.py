#!/usr/bin/env python3
"""Convert WorldVLN inference results to UAV-Flow evaluation format.

Converts from:
  - *_poses.json: [x,y,z,roll,yaw,pitch] in cm/deg (client world coordinates)
To:
  - pred_path.json: [roll,yaw,pitch,x,y,z] in radians/meters (GT-local coordinates)
  - pred_actions.json: [dz,dy,dx,tx,ty,tz] in radians/meters
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_segment_poses(result_dir, prefix):
    """Load all segment poses and concatenate."""
    all_points = []
    seg_files = sorted(Path(result_dir).glob(f"{prefix}_seg*_poses.json"))
    for sf in seg_files:
        data = _read_json(sf)
        pts = data.get("points", [])
        all_points.extend(pts)
    return np.array(all_points) if all_points else np.array([])


def _angle_delta_deg(values, start):
    """Shortest signed angle delta in degrees."""
    return (values - start + 180.0) % 360.0 - 180.0


def _infer_gt_translation_divisor(gt_arr):
    """Infer whether GT xyz is stored in meters or centimeters."""
    if gt_arr.size == 0:
        return 1.0
    max_abs_xyz = float(np.max(np.abs(gt_arr[:, :3])))
    return 100.0 if max_abs_xyz > 100.0 else 1.0


def convert_poses_to_pred_path(poses_cm_deg, raw_start_pose_cm_deg):
    """Convert client world poses to GT-local meters/radians.

    Args:
        poses_cm_deg: (N,6) [x,y,z,roll,yaw,pitch] in client world coordinates
        raw_start_pose_cm_deg: (6,) first raw pose, same world frame as client

    Returns:
        (N,6) [roll,yaw,pitch,x,y,z] in meters/radians, relative to route start
    """
    if len(poses_cm_deg) == 0:
        return np.array([])

    start = np.asarray(raw_start_pose_cm_deg[:6], dtype=np.float32)
    delta_xyz = poses_cm_deg[:, :3].astype(np.float32) - start[:3]

    # preprocessed_logs uses the route-start local frame:
    # local_xy = R(-yaw0) * (world_xy - world_xy0).
    yaw0 = float(start[4]) * (math.pi / 180.0)
    cos_y = math.cos(-yaw0)
    sin_y = math.sin(-yaw0)
    local_x = cos_y * delta_xyz[:, 0] - sin_y * delta_xyz[:, 1]
    local_y = sin_y * delta_xyz[:, 0] + cos_y * delta_xyz[:, 1]
    local_z = delta_xyz[:, 2]

    # Client actions are in centimeters, so pose deltas are centimeter-scale.
    xyz_m = np.stack([local_x, local_y, local_z], axis=1) / 100.0

    rpy_delta_deg = np.zeros((len(poses_cm_deg), 3), dtype=np.float32)
    for i in range(3):
        rpy_delta_deg[:, i] = _angle_delta_deg(poses_cm_deg[:, 3 + i], start[3 + i])
    rpy_rad = rpy_delta_deg * (math.pi / 180.0)

    # Reorder to [roll,yaw,pitch,x,y,z]
    result = np.zeros((len(poses_cm_deg), 6), dtype=np.float32)
    result[:, 0:3] = rpy_rad
    result[:, 3:6] = xyz_m

    return result


def compute_actions_from_trajectory(traj):
    """Compute delta actions from trajectory.

    Args:
        traj: (N,6) [roll,yaw,pitch,x,y,z] in radians/meters

    Returns:
        (N-1,6) actions [dz,dy,dx,tx,ty,tz] in radians/meters
    """
    if len(traj) < 2:
        return np.array([])

    N = len(traj)
    actions = np.zeros((N - 1, 6), dtype=np.float32)

    for i in range(1, N):
        # Rotation deltas (simplified - using difference instead of proper SO3)
        dr = traj[i, 0] - traj[i - 1, 0]  # droll
        dy = traj[i, 1] - traj[i - 1, 1]  # dyaw
        dp = traj[i, 2] - traj[i - 1, 2]  # dpitch

        # Translation in previous frame coords (simplified)
        dx = traj[i, 3] - traj[i - 1, 3]
        dy_t = traj[i, 4] - traj[i - 1, 4]
        dz = traj[i, 5] - traj[i - 1, 5]

        # Store as [dz,dy,dx,tx,ty,tz] (yaw,pitch,roll order for rotations)
        actions[i - 1] = [dy, dp, dr, dx, dy_t, dz]

    return actions


def process_trajectory(traj_dir, gt_dir, output_dir):
    """Process one trajectory and convert to evaluation format."""
    traj_id = os.path.basename(traj_dir)

    # Find the run prefix
    summary_files = list(Path(traj_dir).glob("*_summary.json"))
    if not summary_files:
        print(f"Skipping {traj_id}: no summary file found")
        return

    summary = _read_json(summary_files[0])
    run_id = summary.get("session_id", "").split("__")[-1] if "__" in summary.get("session_id", "") else "run"
    prefix = f"{traj_id}__{run_id}"

    # Load predicted poses
    pred_poses = load_segment_poses(traj_dir, prefix)
    if len(pred_poses) == 0:
        print(f"Skipping {traj_id}: no predicted poses found")
        return

    # Load GT pose data.
    gt_preprocessed = os.path.join(gt_dir, "preprocessed_logs.json")
    if not os.path.exists(gt_preprocessed):
        print(f"Skipping {traj_id}: no GT preprocessed_logs.json found")
        return

    gt_data = _read_json(gt_preprocessed)
    gt_arr = np.array(gt_data, dtype=np.float32)
    gt_translation_divisor = _infer_gt_translation_divisor(gt_arr)

    raw_logs = os.path.join(gt_dir, "raw_logs.json")
    if os.path.exists(raw_logs):
        raw_arr = np.array(_read_json(raw_logs), dtype=np.float32)
        raw_start_pose = raw_arr[0, :6]
    else:
        raw_start_pose = pred_poses[0, :6]

    # Convert predicted poses into the same start-local coordinate frame as GT.
    pred_traj = convert_poses_to_pred_path(pred_poses, raw_start_pose)

    # Compute actions
    actions = compute_actions_from_trajectory(pred_traj)

    # Save pred_path.json
    out_dir = os.path.join(output_dir, traj_id)
    os.makedirs(out_dir, exist_ok=True)

    pred_path = {
        "route": traj_id,
        "start_pose_abs": {
            "x": float(gt_arr[0, 0] / gt_translation_divisor),
            "y": float(gt_arr[0, 1] / gt_translation_divisor),
            "z": float(gt_arr[0, 2] / gt_translation_divisor),
            "roll_rad": float(gt_arr[0, 3] * math.pi / 180),
            "yaw_rad": float(gt_arr[0, 4] * math.pi / 180),
            "pitch_rad": float(gt_arr[0, 5] * math.pi / 180),
        },
        "coordinate_frame": "start_local",
        "gt_translation_divisor_inferred": gt_translation_divisor,
        "poses_layout": ["roll_rad", "yaw_rad", "pitch_rad", "x_m", "y_m", "z_m"],
        "poses": pred_traj.tolist(),
    }
    _write_json(os.path.join(out_dir, "pred_path.json"), pred_path)

    # Save pred_actions.json
    if len(actions) > 0:
        pred_actions = {
            "route": traj_id,
            "actions6_layout": ["dz_rad", "dy_rad", "dx_rad", "tx_m", "ty_m", "tz_m"],
            "actions6": actions.tolist(),
        }
        _write_json(os.path.join(out_dir, "pred_actions.json"), pred_actions)

    print(f"Converted {traj_id}: {len(pred_poses)} poses -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert inference results to evaluation format")
    parser.add_argument("--results_root", type=str, required=True, help="Root of inference results")
    parser.add_argument("--gt_root", type=str, required=True, help="Root of GT samples")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory for converted results")
    parser.add_argument("--run_id", type=str, default="test_run_002", help="Run identifier")
    args = parser.parse_args()

    # Find all trajectory directories
    results_root = Path(args.results_root)
    if not results_root.exists():
        print(f"Error: results_root not found: {results_root}")
        return

    # Process each trajectory
    for traj_dir in sorted(results_root.iterdir()):
        if not traj_dir.is_dir():
            continue
        traj_id = traj_dir.name
        gt_dir = os.path.join(args.gt_root, traj_id)
        if not os.path.exists(gt_dir):
            print(f"Skipping {traj_id}: GT directory not found")
            continue
        process_trajectory(str(traj_dir), gt_dir, args.out_root)

    print(f"\nDone. Converted results saved to: {args.out_root}")


if __name__ == "__main__":
    main()
