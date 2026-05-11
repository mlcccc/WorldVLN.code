#!/usr/bin/env python3
"""
Evaluate latent-to-action predictions on UAV-Flow-style route folders.

Expected ground-truth route layout:
  <gt_root>/<route>/
    preprocessed_logs.json   # preferred, (T,6) [x,y,z,roll,yaw,pitch]
    raw_logs.json            # optional fallback, same layout

Expected prediction route layout:
  <pred_root>/<route>/
    pred_actions.json        # preferred, contains actions6 = [dz,dy,dx,tx,ty,tz]
    pred_path.json           # fallback, contains poses = [roll,yaw,pitch,x,y,z]

Units:
  - Prediction actions are always interpreted as radians + meters.
  - GT angles are degrees by default (--angles_in_degrees).
  - GT translation is divided by --translation_divisor before comparison.
    For UAV-Flow data this is usually 1.0.

Outputs:
  <out_root>/summary.txt
  <out_root>/images/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_TOOLS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.utils import euler_to_rotation, rotation_to_euler  # noqa: E402


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    _mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _unwrap_angles_rad(rpy: np.ndarray) -> np.ndarray:
    out = np.empty_like(rpy, dtype=np.float32)
    for i in range(3):
        out[:, i] = np.unwrap(rpy[:, i])
    return out


def _R_from_rpy_zyx(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """Build R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)


def _rpy_from_R_zyx(R: np.ndarray) -> np.ndarray:
    zyx = rotation_to_euler(R, seq="zyx")  # [yaw, pitch, roll]
    yaw, pitch, roll = float(zyx[0]), float(zyx[1]), float(zyx[2])
    return np.asarray([roll, yaw, pitch], dtype=np.float32)


def _geodesic_angle_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_rel = (R_gt.T @ R_pred).astype(np.float32)
    c = (float(np.trace(R_rel)) - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c) * 180.0 / math.pi)


def _yaw_error_deg(pred_yaw_rad: float, gt_yaw_rad: float) -> float:
    dy = (float(pred_yaw_rad - gt_yaw_rad) * 180.0 / math.pi) % 360.0
    if dy > 180.0:
        dy = 360.0 - dy
    return float(abs(dy))


def integrate_actions_to_traj(
    actions6: np.ndarray,
    *,
    start_xyz_m: np.ndarray,
    start_rpy_rad: np.ndarray,
) -> np.ndarray:
    """
    actions6: (T-1,6), each [dz,dy,dx,tx,ty,tz].
      - dz/dy/dx are relative ZYX Euler rotations in radians.
      - tx/ty/tz is translation in the previous-frame coordinate system, meters.
    returns: (T,6) absolute [roll,yaw,pitch,x,y,z] in radians + meters.
    """
    actions6 = np.asarray(actions6, dtype=np.float32)
    T = int(actions6.shape[0]) + 1
    traj = np.zeros((T, 6), dtype=np.float32)

    roll0, yaw0, pitch0 = [float(x) for x in start_rpy_rad.reshape(3)]
    R = _R_from_rpy_zyx(roll0, yaw0, pitch0)
    p = start_xyz_m.astype(np.float32).reshape(3).copy()
    traj[0, 0:3] = np.asarray([roll0, yaw0, pitch0], dtype=np.float32)
    traj[0, 3:6] = p

    for i in range(1, T):
        dz, dy, dx = [float(x) for x in actions6[i - 1, 0:3]]
        t_rel = actions6[i - 1, 3:6].astype(np.float32)
        R_rel = _R_from_rpy_zyx(roll=dx, yaw=dz, pitch=dy)
        p = p + (R @ t_rel)
        R = (R @ R_rel).astype(np.float32)
        traj[i, 0:3] = _rpy_from_R_zyx(R)
        traj[i, 3:6] = p

    traj[:, 0:3] = _unwrap_angles_rad(traj[:, 0:3])
    return traj


def _load_gt_traj(gt_dir: str, *, pose_file: str, translation_divisor: float, angles_in_degrees: bool) -> np.ndarray:
    candidates = [os.path.join(gt_dir, pose_file)]
    if pose_file != "preprocessed_logs.json":
        candidates.append(os.path.join(gt_dir, "preprocessed_logs.json"))
    candidates.append(os.path.join(gt_dir, "raw_logs.json"))

    path = ""
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if not path:
        raise FileNotFoundError(f"missing GT pose json under {gt_dir}; tried {candidates}")

    arr = np.asarray(_read_json(path), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"bad GT shape {arr.shape} at {path}")
    xyz_m = arr[:, 0:3].astype(np.float32) / float(translation_divisor)
    rpy = arr[:, 3:6].astype(np.float32)
    if bool(angles_in_degrees):
        rpy = rpy * (np.pi / 180.0)
    rpy = _unwrap_angles_rad(rpy)
    return np.concatenate([rpy, xyz_m], axis=1).astype(np.float32)  # [rpy,xyz]


def _load_pred_actions(path: str) -> np.ndarray:
    obj = _read_json(path)
    acts = obj.get("actions6")
    if not isinstance(acts, list):
        raise ValueError("pred_actions.json missing actions6")
    a = np.asarray(acts, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] < 6:
        raise ValueError(f"bad actions6 shape {a.shape} at {path}")
    return a[:, :6]


def _load_pred_path(path: str) -> np.ndarray:
    obj = _read_json(path)
    poses = obj.get("poses")
    if not isinstance(poses, list):
        raise ValueError("pred_path.json missing poses")
    p = np.asarray(poses, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 6:
        raise ValueError(f"bad poses shape {p.shape} at {path}")
    return p[:, :6].astype(np.float32)  # [roll,yaw,pitch,x,y,z], rad+m


def _discover_routes(pred_root: str) -> List[str]:
    routes: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(pred_root):
        if "pred_actions.json" in filenames or "pred_path.json" in filenames:
            rel = os.path.relpath(dirpath, pred_root)
            if rel != ".":
                routes.append(rel.replace(os.sep, "/"))
    routes.sort()
    return routes


@dataclass
class RouteEval:
    route: str
    T_use: int
    dist_m: float
    dist_cm: float
    ang_deg: float
    yaw_err_deg: float
    qualified: bool


def _eval_one(
    route: str,
    pred_dir: str,
    gt_dir: str,
    *,
    pose_file: str,
    translation_divisor: float,
    angles_in_degrees: bool,
    dist_thr_m: float,
    ang_thr_deg: float,
) -> RouteEval:
    gt_traj = _load_gt_traj(
        gt_dir,
        pose_file=pose_file,
        translation_divisor=translation_divisor,
        angles_in_degrees=angles_in_degrees,
    )
    gt_rpy = gt_traj[:, 0:3].astype(np.float32)
    gt_xyz = gt_traj[:, 3:6].astype(np.float32)

    pred_actions_json = os.path.join(pred_dir, "pred_actions.json")
    pred_path_json = os.path.join(pred_dir, "pred_path.json")
    if os.path.exists(pred_actions_json):
        actions6 = _load_pred_actions(pred_actions_json)
        T_use = min(int(gt_traj.shape[0]), int(actions6.shape[0]) + 1)
        pred_traj = integrate_actions_to_traj(
            actions6[: max(0, T_use - 1)],
            start_xyz_m=gt_xyz[0],
            start_rpy_rad=gt_rpy[0],
        )
    elif os.path.exists(pred_path_json):
        pred_traj = _load_pred_path(pred_path_json)
        T_use = min(int(gt_traj.shape[0]), int(pred_traj.shape[0]))
        pred_traj = pred_traj[:T_use]
    else:
        raise FileNotFoundError(f"missing pred_actions.json or pred_path.json under {pred_dir}")

    gt_rpy = gt_rpy[:T_use]
    gt_xyz = gt_xyz[:T_use]
    pred_rpy = pred_traj[:T_use, 0:3].astype(np.float32)
    pred_xyz = pred_traj[:T_use, 3:6].astype(np.float32)

    dp = (pred_xyz[-1] - gt_xyz[-1]).astype(np.float32)
    dist_m = float(np.linalg.norm(dp))
    R_pred = _R_from_rpy_zyx(float(pred_rpy[-1, 0]), float(pred_rpy[-1, 1]), float(pred_rpy[-1, 2]))
    R_gt = _R_from_rpy_zyx(float(gt_rpy[-1, 0]), float(gt_rpy[-1, 1]), float(gt_rpy[-1, 2]))
    ang_deg = _geodesic_angle_deg(R_pred, R_gt)
    yaw_err_deg = _yaw_error_deg(float(pred_rpy[-1, 1]), float(gt_rpy[-1, 1]))

    qualified = dist_m <= float(dist_thr_m) and ang_deg <= float(ang_thr_deg)
    return RouteEval(
        route=route,
        T_use=int(T_use),
        dist_m=dist_m,
        dist_cm=dist_m * 100.0,
        ang_deg=ang_deg,
        yaw_err_deg=yaw_err_deg,
        qualified=bool(qualified),
    )


def _plot_distribution(vals: List[float], *, title: str, xlabel: str, out_path: str, bins: int = 50) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(vals, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    _mkdir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 6))
    plt.hist(v, bins=int(bins), color="skyblue", edgecolor="black", alpha=0.9)
    plt.axvline(float(np.mean(v)), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(v):.2f}")
    plt.axvline(float(np.median(v)), color="green", linestyle="--", linewidth=1.5, label=f"Median: {np.median(v):.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_overlay(routes: List[str], *, pred_root: str, gt_root: str, pose_file: str, translation_divisor: float, angles_in_degrees: bool, out_path: str, max_routes: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _mkdir(os.path.dirname(out_path))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    n = 0
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for r in routes[: int(max_routes)]:
        try:
            gt = _load_gt_traj(
                os.path.join(gt_root, r),
                pose_file=pose_file,
                translation_divisor=translation_divisor,
                angles_in_degrees=angles_in_degrees,
            )
            pred_path = os.path.join(pred_root, r, "pred_path.json")
            if not os.path.exists(pred_path):
                continue
            pred = _load_pred_path(pred_path)
        except Exception:
            continue
        gt_xyz = gt[:, 3:6].astype(np.float32)
        pred_xyz = pred[:, 3:6].astype(np.float32)
        T = min(int(gt_xyz.shape[0]), int(pred_xyz.shape[0]))
        if T <= 1:
            continue
        gt_xyz = gt_xyz[:T]
        pred_xyz = pred_xyz[:T]
        ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="#1f77b4", alpha=0.15, linewidth=1.0)
        ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color="#d62728", alpha=0.15, linewidth=1.0)
        mins = np.minimum(mins, np.min(gt_xyz, axis=0))
        mins = np.minimum(mins, np.min(pred_xyz, axis=0))
        maxs = np.maximum(maxs, np.max(gt_xyz, axis=0))
        maxs = np.maximum(maxs, np.max(pred_xyz, axis=0))
        n += 1

    ax.set_title(f"UAV-Flow trajectories overlay (GT blue / Pred red), n={n}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs)) and np.all(maxs > mins):
        center = (mins + maxs) / 2.0
        span = float(np.max(maxs - mins))
        lo = center - span / 2.0
        hi = center + span / 2.0
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Evaluate UAV-Flow latent-to-action endpoint errors.")
    ap.add_argument("--pred_root", type=str, default="./outputs/stage2_latent2action", help="Prediction root containing route subdirectories.")
    ap.add_argument("--gt_root", type=str, default="./data/uavflow", help="Ground-truth UAV-Flow route root.")
    ap.add_argument("--out_root", type=str, default="./outputs/eval_uavflow", help="Evaluation output directory.")
    ap.add_argument("--gt_pose_file", type=str, default="preprocessed_logs.json", help="GT pose json filename inside each route.")
    ap.add_argument("--translation_divisor", type=float, default=1.0, help="Divide GT xyz by this value before comparison.")
    ap.add_argument("--angles_in_degrees", action="store_true", default=True, help="Interpret GT rpy values as degrees.")
    ap.add_argument("--dist_thr_m", type=float, default=3.0, help="Qualified endpoint distance threshold in meters.")
    ap.add_argument("--ang_thr_deg", type=float, default=10.0, help="Qualified endpoint rotation threshold in degrees.")
    ap.add_argument("--expect_routes", type=int, default=0, help="Expected number of routes; 0 disables the hint.")
    ap.add_argument("--overlay_max_routes", type=int, default=300, help="Max routes in the 3D overlay plot.")
    ap.add_argument("--tqdm", action="store_true", default=True, help="Show tqdm progress bar when available.")
    args = ap.parse_args()

    pred_root = os.path.abspath(str(args.pred_root))
    gt_root = os.path.abspath(str(args.gt_root))
    out_root = os.path.abspath(str(args.out_root))
    img_root = os.path.join(out_root, "images")
    _mkdir(img_root)

    routes = _discover_routes(pred_root)
    results: List[RouteEval] = []
    skipped: List[Tuple[str, str]] = []
    iterator = tqdm(routes, desc="eval routes", dynamic_ncols=True) if bool(args.tqdm) and tqdm is not None else routes
    for r in iterator:
        try:
            results.append(
                _eval_one(
                    r,
                    os.path.join(pred_root, r),
                    os.path.join(gt_root, r),
                    pose_file=str(args.gt_pose_file),
                    translation_divisor=float(args.translation_divisor),
                    angles_in_degrees=bool(args.angles_in_degrees),
                    dist_thr_m=float(args.dist_thr_m),
                    ang_thr_deg=float(args.ang_thr_deg),
                )
            )
        except Exception as e:
            skipped.append((r, str(e)))

    ok = len(results)
    qual = sum(1 for x in results if x.qualified)
    qual_rate = float(qual) / float(max(1, ok))
    dist_cm = [x.dist_cm for x in results]
    ang_deg = [x.ang_deg for x in results]
    yaw_err_deg = [x.yaw_err_deg for x in results]
    results_sorted = sorted(results, key=lambda x: (x.dist_m, x.ang_deg))

    expected = f"expected~{int(args.expect_routes)}" if int(args.expect_routes) > 0 else "expected=unspecified"
    lines: List[str] = []
    lines.append("dataset=uav-flow")
    lines.append(f"pred_root={pred_root}")
    lines.append(f"gt_root={gt_root}")
    lines.append(f"gt_pose_file={args.gt_pose_file}")
    lines.append(f"translation_divisor={float(args.translation_divisor)}")
    lines.append(f"evaluated_routes={ok} ({expected}) skipped={len(skipped)}")
    lines.append(f"qualified(dist<={args.dist_thr_m}m & ang<={args.ang_thr_deg}deg) = {qual}/{ok} = {qual_rate*100.0:.2f}%")
    if ok > 0:
        lines.append(f"distance_cm: mean={np.mean(dist_cm):.2f} p50={np.percentile(dist_cm,50):.2f} p90={np.percentile(dist_cm,90):.2f} max={np.max(dist_cm):.2f}")
        lines.append(f"angle_deg:   mean={np.mean(ang_deg):.2f} p50={np.percentile(ang_deg,50):.2f} p90={np.percentile(ang_deg,90):.2f} max={np.max(ang_deg):.2f}")
        lines.append(f"yaw_err_deg: mean={np.mean(yaw_err_deg):.2f} p50={np.percentile(yaw_err_deg,50):.2f} p90={np.percentile(yaw_err_deg,90):.2f} max={np.max(yaw_err_deg):.2f}")
    lines.append("")
    lines.append("Per-route endpoint errors (sorted by dist then angle):")
    lines.append("route\tT_use\tdist_cm\tang_deg(geo)\tyaw_err_deg\tqualified")
    for x in results_sorted:
        lines.append(f"{x.route}\t{x.T_use}\t{x.dist_cm:.2f}\t{x.ang_deg:.2f}\t{x.yaw_err_deg:.2f}\t{int(x.qualified)}")
    if skipped:
        lines.append("")
        lines.append("Skipped routes:")
        for r, e in skipped[:100]:
            lines.append(f"{r}\t{e}")
        if len(skipped) > 100:
            lines.append(f"... ({len(skipped) - 100} more)")
    _write_text(os.path.join(out_root, "summary.txt"), "\n".join(lines) + "\n")

    if ok > 0:
        _plot_distribution(dist_cm, title="Endpoint Distance Error", xlabel="distance (cm)", out_path=os.path.join(img_root, "distance_error_distribution.png"))
        _plot_distribution(ang_deg, title="Endpoint Rotation Error", xlabel="angle (deg)", out_path=os.path.join(img_root, "rotation_error_distribution.png"))
        _plot_distribution(yaw_err_deg, title="Endpoint Yaw Error", xlabel="yaw error (deg)", out_path=os.path.join(img_root, "yaw_error_distribution.png"))
        _plot_overlay(
            routes,
            pred_root=pred_root,
            gt_root=gt_root,
            pose_file=str(args.gt_pose_file),
            translation_divisor=float(args.translation_divisor),
            angles_in_degrees=bool(args.angles_in_degrees),
            out_path=os.path.join(img_root, "trajectories_3d_overlay.png"),
            max_routes=int(args.overlay_max_routes),
        )

    print(f"Done. summary={os.path.join(out_root, 'summary.txt')} images_dir={img_root}")


if __name__ == "__main__":
    main()
