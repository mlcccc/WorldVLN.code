import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import euler_to_rotation, rotation_to_euler

@dataclass(frozen=True)
class UavflowRoute:
    route_dir: str
    images_dir: str
    raw_logs_path: str
    preprocessed_logs_path: str
    length: int


def _is_route_dir(d: str) -> bool:
    """A valid route dir contains images/ + raw/preprocessed logs."""
    return (
        os.path.isdir(os.path.join(d, "images"))
        and os.path.isfile(os.path.join(d, "raw_logs.json"))
        and os.path.isfile(os.path.join(d, "preprocessed_logs.json"))
    )


def _load_json_array(path: str) -> np.ndarray:
    with open(path, "r") as f:
        arr = json.load(f)
    return np.asarray(arr, dtype=np.float32)


def _unwrap_angles_radians(angles_rad: np.ndarray) -> np.ndarray:
    """
    Unwrap angles along time for each channel.
    angles_rad: (T, 3)
    """
    out = np.empty_like(angles_rad)
    for i in range(3):
        out[:, i] = np.unwrap(angles_rad[:, i])
    return out


def _rpy_to_R_zyx(roll: float, yaw: float, pitch: float) -> np.ndarray:
    """
    Build rotation matrix from roll/yaw/pitch (radians) using ZYX order:
      R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    return np.asarray(euler_to_rotation(z=yaw, y=pitch, x=roll, isRadian=True, seq="zyx"), dtype=np.float32)


class UavflowSimDataset(Dataset):
    """
    Dataset for uavflowdatasim_output:
      - route_dir/
          images/frame_000000.png ...
          raw_logs.json            # (T,6): [p0,p1,p2,a0,a1,a2] absolute
          preprocessed_logs.json   # (T,6): relative-to-start, cumulative

    We train TSformer-VO on *frame-to-frame* relative motion, similarly to KITTI:
      T_rel = inv(T_prev) @ T_curr
    where translation is expressed in the previous frame coordinate system:
      t_rel = R_prev^T (p_curr - p_prev)

    Angle convention:
    - raw_logs angles are assumed to be [roll, yaw, pitch] (same order as test_json initial_pos)
    - absolute rotation uses ZYX (yaw->pitch->roll)
    - relative rotation is converted with seq='zyx' giving [z, y, x] in radians

    Each sample is a window of `window_size` frames:
      X: (C,T,H,W)
      y: flattened ((window_size-1) * 6) with per-step [angles(3), translation(3)]
    """

    def __init__(
        self,
        root_dir: Union[str, List[str]],
        window_size: int = 4,
        stride: int = 1,
        transform: Optional[Callable] = None,
        use_raw_for_labels: bool = True,
        angles_in_degrees: bool = True,
        translation_divisor: float = 1.0,
        img_ext: str = ".png",
        max_routes: Optional[int] = None,
        stats_sample_routes: int = 500,
    ):
        # Support training on multiple roots at once (e.g. uavflowdatasim_output + reference_train_uavflow_like).
        if isinstance(root_dir, (list, tuple)):
            roots = [str(r) for r in root_dir]
        else:
            roots = [str(root_dir)]
        roots = [r for r in roots if r]
        if len(roots) == 0:
            raise ValueError("root_dir is empty")
        self.root_dirs = roots
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.transform = transform
        self.use_raw_for_labels = bool(use_raw_for_labels)
        self.angles_in_degrees = bool(angles_in_degrees)
        self.translation_divisor = float(translation_divisor)
        self.img_ext = img_ext

        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if not np.isfinite(self.translation_divisor) or self.translation_divisor <= 0:
            raise ValueError("translation_divisor must be a finite number > 0")

        # 1) Discover routes
        route_dirs = []
        for root in self.root_dirs:
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                p = os.path.join(root, name)
                if os.path.isdir(p) and _is_route_dir(p):
                    route_dirs.append(p)
        if max_routes is not None:
            route_dirs = route_dirs[: int(max_routes)]
        if len(route_dirs) == 0:
            raise FileNotFoundError(f"No route folders found in roots={self.root_dirs}")

        self.routes: List[UavflowRoute] = []
        for rd in route_dirs:
            raw_p = os.path.join(rd, "raw_logs.json")
            pre_p = os.path.join(rd, "preprocessed_logs.json")
            raw = _load_json_array(raw_p)
            pre = _load_json_array(pre_p)
            length = int(min(len(raw), len(pre)))
            if length >= self.window_size:
                self.routes.append(
                    UavflowRoute(
                        route_dir=rd,
                        images_dir=os.path.join(rd, "images"),
                        raw_logs_path=raw_p,
                        preprocessed_logs_path=pre_p,
                        length=length,
                    )
                )
        if len(self.routes) == 0:
            raise FileNotFoundError(
                f"Found route dirs but none have length >= window_size={self.window_size}"
            )

        # 2) Precompute per-route labels (delta pose) and build global sample index
        self._delta_by_route: List[np.ndarray] = []  # each: (T,6) with delta[0]=0
        self.samples: List[Tuple[int, int]] = []  # (route_idx, start_frame)

        for r_idx, r in enumerate(self.routes):
            raw = _load_json_array(r.raw_logs_path)[: r.length]
            pre = _load_json_array(r.preprocessed_logs_path)[: r.length]

            if self.use_raw_for_labels:
                # raw: [x, y, z, roll, yaw, pitch] absolute (angles often in degrees)
                pos = raw[:, 0:3].astype(np.float32) / self.translation_divisor  # cm->m if divisor=100
                rpy = raw[:, 3:6].astype(np.float32)
                if self.angles_in_degrees:
                    rpy = rpy * (np.pi / 180.0)
                rpy = _unwrap_angles_radians(rpy)

                # Build per-step relative motion: inv(T_prev) @ T_curr
                delta = np.zeros((r.length, 6), dtype=np.float32)
                R_prev = _rpy_to_R_zyx(float(rpy[0, 0]), float(rpy[0, 1]), float(rpy[0, 2]))
                p_prev = pos[0]

                for t in range(1, r.length):
                    R_curr = _rpy_to_R_zyx(float(rpy[t, 0]), float(rpy[t, 1]), float(rpy[t, 2]))
                    p_curr = pos[t]

                    R_rel = R_prev.T @ R_curr
                    t_rel = R_prev.T @ (p_curr - p_prev)

                    zyx = rotation_to_euler(R_rel, seq="zyx")  # [z, y, x] (rad)
                    delta[t, 0:3] = np.asarray(zyx, dtype=np.float32)
                    delta[t, 3:6] = t_rel.astype(np.float32)

                    R_prev = R_curr
                    p_prev = p_curr
            else:
                # preprocessed is cumulative relative-to-start; diff gives per-step increments
                # pre: [q0,q1,q2,r0,r1,r2]
                q = pre[:, 0:3].astype(np.float32)
                r_ang = pre[:, 3:6].astype(np.float32)
                if self.angles_in_degrees:
                    r_ang = r_ang * (np.pi / 180.0)
                r_ang = _unwrap_angles_radians(r_ang)
                dq = np.zeros_like(q)
                dr = np.zeros_like(r_ang)
                dq[1:] = q[1:] - q[:-1]
                dr[1:] = r_ang[1:] - r_ang[:-1]
                if self.translation_divisor != 1.0:
                    dq = dq / self.translation_divisor
                delta = np.concatenate([dr, dq], axis=1)

            self._delta_by_route.append(delta.astype(np.float32))

            # windows
            for s in range(0, r.length - self.window_size + 1, self.stride):
                self.samples.append((r_idx, s))

        # 3) Compute normalization stats on a subset of routes (for speed)
        sample_every = max(1, len(self.routes) // max(1, int(stats_sample_routes)))
        all_angles = []
        all_trans = []
        for ridx in range(0, len(self.routes), sample_every):
            d = self._delta_by_route[ridx]
            # skip delta[0] which is all-zero
            if len(d) > 1:
                all_angles.append(d[1:, 0:3])
                all_trans.append(d[1:, 3:6])
        if len(all_angles) > 0:
            all_angles = np.concatenate(all_angles, axis=0)
            all_trans = np.concatenate(all_trans, axis=0)
            self.mean_angles = np.mean(all_angles, axis=0)
            self.std_angles = np.std(all_angles, axis=0) + 1e-6
            self.mean_t = np.mean(all_trans, axis=0)
            self.std_t = np.std(all_trans, axis=0) + 1e-6
        else:
            self.mean_angles = np.zeros(3, dtype=np.float32)
            self.std_angles = np.ones(3, dtype=np.float32)
            self.mean_t = np.zeros(3, dtype=np.float32)
            self.std_t = np.ones(3, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _frame_path(self, images_dir: str, frame_idx: int) -> str:
        # Expected naming: frame_000000.png
        p = os.path.join(images_dir, f"frame_{frame_idx:06d}{self.img_ext}")
        if os.path.exists(p):
            return p
        # Fallbacks (just in case user has other exports)
        p2 = os.path.join(images_dir, f"{frame_idx:06d}{self.img_ext}")
        if os.path.exists(p2):
            return p2
        p3 = os.path.join(images_dir, f"{frame_idx}{self.img_ext}")
        if os.path.exists(p3):
            return p3
        raise FileNotFoundError(f"Frame {frame_idx} not found under {images_dir}")

    def __getitem__(self, idx: int):
        route_idx, start = self.samples[idx]
        r = self.routes[route_idx]
        delta = self._delta_by_route[route_idx]

        # 1) images -> (C,T,H,W)
        imgs = []
        for i in range(self.window_size):
            frame_idx = start + i
            img_path = self._frame_path(r.images_dir, frame_idx)
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img.unsqueeze(0))  # (1,C,H,W)
        imgs = torch.cat(imgs, dim=0)  # (T,C,H,W)
        imgs = imgs.transpose(0, 1)  # (C,T,H,W)

        # 2) labels: use deltas for frames (start+1..start+window_size-1)
        y = []
        for i in range(1, self.window_size):
            pose_idx = start + i
            d = delta[pose_idx]  # [angles(3), trans(3)]
            ang = (d[0:3] - self.mean_angles) / self.std_angles
            t = (d[3:6] - self.mean_t) / self.std_t
            y.extend(list(ang) + list(t))
        y = np.asarray(y, dtype=np.float32)
        return imgs, y

    def get_stats(self) -> Dict[str, List[float]]:
        return {
            "mean_angles": self.mean_angles.astype(float).tolist(),
            "std_angles": self.std_angles.astype(float).tolist(),
            "mean_t": self.mean_t.astype(float).tolist(),
            "std_t": self.std_t.astype(float).tolist(),
        }

