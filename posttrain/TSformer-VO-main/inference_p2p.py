import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import json
import math
from mpl_toolkits.mplot3d import Axes3D

# Try to import build_p2p_model from pretrain_latent_p2p
try:
    from pretrain_latent_p2p import build_p2p_model
except ImportError:
    # If import fails, we might need to add current dir to path or copy the function
    import sys
    sys.path.append(os.getcwd())
    from pretrain_latent_p2p import build_p2p_model

def load_latents(npy_path):
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    latents = np.load(npy_path)
    
    # Handle shapes based on P2PDataset logic
    if latents.ndim == 5 and latents.shape[0] == 1:
        latents = latents[0]
    
    # (C, T, H, W) -> (T, C, H, W) correction if needed
    if latents.ndim == 4:
        if latents.shape[0] == 16 and latents.shape[1] != 16:
             latents = latents.transpose(1, 0, 2, 3)
             
    return latents

def create_windows(latents, window_size=2):
    # latents: (T, C, H, W)
    num_latents = latents.shape[0]
    windows = []
    
    for i in range(num_latents - window_size + 1):
        window = latents[i : i + window_size]
        windows.append(window)
        
    return np.array(windows)

def reconstruct_trajectory(predictions, start_pose=None):
    # predictions: (N, 6) -> [dx, dy, dz, droll, dpitch, dyaw]
    # We assume predictions are differences in global frame as per training logic
    
    if start_pose is None:
        start_pose = np.zeros(6)
        
    trajectory = [start_pose]
    current_pose = start_pose.copy()
    
    for pred in predictions:
        # Simple integration
        current_pose = current_pose + pred
        trajectory.append(current_pose.copy())
        
    return np.array(trajectory)


def _rpy_to_R_zyx(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _rotation_geodesic_deg(r1, r2):
    # geodesic distance on SO(3): theta = acos((trace(R_rel)-1)/2)
    r_rel = r1.T @ r2
    cos_theta = (np.trace(r_rel) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _compute_attitude_errors_deg(traj_pred, traj_gt):
    if traj_pred is None or traj_gt is None:
        return None, None, None
    n = min(len(traj_pred), len(traj_gt))
    if n < 2:
        return None, None, None

    errs = []
    for i in range(n):
        pr = traj_pred[i, 3:6]  # [roll, pitch, yaw] in rad
        gt = traj_gt[i, 3:6]
        r_pred = _rpy_to_R_zyx(pr[0], pr[1], pr[2])
        r_gt = _rpy_to_R_zyx(gt[0], gt[1], gt[2])
        errs.append(_rotation_geodesic_deg(r_pred, r_gt))
    errs = np.asarray(errs, dtype=np.float64)
    return float(errs[-1]), float(errs.mean()), errs


def visualize_rotation(trajectory, ground_truth=None, save_path=None, title_prefix="Rotation Analysis"):
    # Keep yaw trend visualization; metric in title is 3D attitude geodesic error.
    if trajectory is None or len(trajectory) == 0:
        return
    pred_yaw_deg = np.degrees(np.unwrap(trajectory[:, 5]))
    pred_yaw_rel = pred_yaw_deg - pred_yaw_deg[0]

    gt_yaw_rel = None
    attitude_endpoint_error = None
    attitude_mean_error = None
    if ground_truth is not None and len(ground_truth) > 0:
        n = min(len(pred_yaw_rel), len(ground_truth))
        gt_yaw_deg = np.degrees(np.unwrap(ground_truth[:n, 5]))
        gt_yaw_rel = gt_yaw_deg - gt_yaw_deg[0]
        pred_yaw_rel = pred_yaw_rel[:n]
        attitude_endpoint_error, attitude_mean_error, _ = _compute_attitude_errors_deg(trajectory[:n], ground_truth[:n])

    plt.figure(figsize=(10, 6))
    if gt_yaw_rel is not None:
        plt.plot(gt_yaw_rel, label=f"Ground Truth (Total: {gt_yaw_rel[-1]:.1f}deg)", color="red", linewidth=2)
    plt.plot(
        pred_yaw_rel,
        label=f"Predicted (Total: {pred_yaw_rel[-1]:.1f}deg)",
        color="blue",
        linestyle="--",
        linewidth=2,
    )

    if attitude_endpoint_error is not None:
        plt.title(
            f"{title_prefix}\nAttitude Endpoint Error: {attitude_endpoint_error:.2f}deg | "
            f"Mean: {attitude_mean_error:.2f}deg"
        )
    else:
        plt.title(f"{title_prefix}\nAttitude Error: N/A")
    plt.xlabel("Frame Index")
    plt.ylabel("Relative Yaw (degrees)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Rotation plot saved to {save_path}")
    else:
        plt.savefig("rotation_plot.png")
    plt.close()

def visualize_trajectory(trajectory, ground_truth=None, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot predicted
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Predicted', marker='.', alpha=0.6)
    
    # Plot start
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='x', s=100, label='End')

    if ground_truth is not None:
         ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', linestyle='--', alpha=0.6, color='orange')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Reconstructed Trajectory')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        print("Saving plot to 'trajectory_plot.png' (default)")
        plt.savefig('trajectory_plot.png')
    plt.close(fig)

def run_inference(args):
    # 1. Load Model
    print(f"Loading model from {args.checkpoint}...")
    try:
        model = build_p2p_model(args)
    except Exception as e:
        print(f"Error building model: {e}")
        return None, None, None

    model.to(args.device)
    model.eval()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return None, None, None

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        
    # Handle DataParallel keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("Model weights loaded (strict=False).")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return None, None, None
    
    # 2. Load Stats if available
    target_mean = None
    target_std = None
    
    # Try to find stats automatically if not provided
    if args.stats_path is None:
        # Check same dir as checkpoint
        potential_path = os.path.join(os.path.dirname(args.checkpoint), "p2p_target_stats.json")
        if os.path.exists(potential_path):
            args.stats_path = potential_path
            print(f"Found stats file automatically: {args.stats_path}")
            
    if args.stats_path and os.path.exists(args.stats_path):
        print(f"Loading stats from {args.stats_path}")
        with open(args.stats_path, 'r') as f:
            stats = json.load(f)
            target_mean = torch.tensor(stats["mean"]).to(args.device)
            target_std = torch.tensor(stats["std"]).to(args.device)
    else:
        print("Warning: No stats file found or provided. Predictions might be unscaled if model was trained with standardization.")
    
    # 3. Load Data
    print(f"Loading latents from {args.latent_path}...")
    try:
        latents = load_latents(args.latent_path) # (T, C, H, W)
    except Exception as e:
        print(f"Error loading latents: {e}")
        return None, None, None

    # Create windows
    windows = create_windows(latents, args.window_size) # (N, W, C, H, W)
    print(f"Created {len(windows)} windows from {len(latents)} latents.")
    
    if len(windows) == 0:
        print("Not enough frames to create windows.")
        return None, None, None

    # 4. Inference
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i : i + batch_size]
            batch_tensor = torch.from_numpy(batch_windows).float().to(args.device)
            
            # Forward
            outputs = model(batch_tensor) # (B, 6)
            
            # Un-standardize
            if target_mean is not None and target_std is not None:
                outputs = outputs * target_std + target_mean
                
            predictions.append(outputs.cpu().numpy())
            
    predictions = np.concatenate(predictions, axis=0)
    
    # 5. Reconstruct
    trajectory = reconstruct_trajectory(predictions)
    
    # 6. Load GT if possible (assuming standard directory structure)
    gt_path = os.path.join(os.path.dirname(args.latent_path), "preprocessed_logs.json")
    gt_traj = None
    if os.path.exists(gt_path):
        print(f"Found Ground Truth at {gt_path}")
        try:
            with open(gt_path, 'r') as f:
                logs = json.load(f)
            
            gt_points = []
            num_latents = latents.shape[0]
            
            start_pose_idx = 0
            if len(logs) > 0:
                start_pose = np.array(logs[0])
                start_pose[0:3] /= 100.0
                start_pose[3:6] *= (math.pi / 180.0)
            else:
                start_pose = np.zeros(6)

            # We can just extract all sampled poses
            for i in range(num_latents):
                idx = 4 * i
                if idx < len(logs):
                    pose = np.array(logs[idx])
                    pose[0:3] = pose[0:3] / 100.0
                    pose[3:6] = pose[3:6] * (math.pi / 180.0)
                    gt_points.append(pose)
            
            gt_traj = np.array(gt_points)
            
            # Align GT start to 0
            if len(gt_traj) > 0:
                 gt_traj = gt_traj - gt_traj[0]
                 
        except Exception as e:
            print(f"Error loading GT: {e}")
            
    # 7. Visualize
    visualize_trajectory(trajectory, ground_truth=gt_traj, save_path=args.output_plot)

    rotation_plot_path = getattr(args, "output_rotation_plot", None)
    if rotation_plot_path is None:
        rotation_plot_path = os.path.join(os.path.dirname(args.latent_path), "rotation_plot.png")
    visualize_rotation(trajectory, ground_truth=gt_traj, save_path=rotation_plot_path)

    endpoint_error_m = None
    attitude_endpoint_error_deg = None
    attitude_mean_error_deg = None
    if gt_traj is not None and len(gt_traj) > 0:
        n = min(len(trajectory), len(gt_traj))
        if n >= 1:
            endpoint_error_m = float(np.linalg.norm(trajectory[n - 1, :3] - gt_traj[n - 1, :3]))
        if n >= 2:
            attitude_endpoint_error_deg, attitude_mean_error_deg, _ = _compute_attitude_errors_deg(
                trajectory[:n], gt_traj[:n]
            )

    metrics = {
        "endpoint_error_m": endpoint_error_m,
        "attitude_endpoint_error_deg": attitude_endpoint_error_deg,
        "attitude_mean_error_deg": attitude_mean_error_deg,
        "rotation_plot_path": rotation_plot_path,
        "pred_len": int(len(trajectory)) if trajectory is not None else 0,
        "gt_len": int(len(gt_traj)) if gt_traj is not None else 0,
    }
    return trajectory, gt_traj, metrics

def main():
    parser = argparse.ArgumentParser(description="Inference for Latent P2P VO")
    parser.add_argument("--latent_path", type=str, required=True, help="Path to input .npy latent file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--stats_path", type=str, default=None, help="Path to target stats json (optional, for un-standardizing)")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_plot", type=str, default=None, help="Path to save plot. Defaults to inference_result.png in latent dir")
    
    args = parser.parse_args()

    if args.output_plot is None:
        args.output_plot = os.path.join(os.path.dirname(args.latent_path), "inference_result.png")
    
    run_inference(args)
    print("Done.")

if __name__ == "__main__":
    main()
