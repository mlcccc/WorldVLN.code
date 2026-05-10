import json
import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import signal


@dataclass(frozen=True)
class LatentTrajManifestItem:
    latent_path: str
    traj_json_path: str
    images_dir: str


def _resolve_path(p: str, workspace_root: str) -> str:
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(workspace_root, p))


class LatentTrajManifestDataset(Dataset):
    """
    Dataset backed by a manifest JSON file containing one or more `items_*` lists.

    Each item provides:
    - z_ext: (1,64,T_lat,16,16) float32
    - frames_rgb: (3,T,H,W) float32 (optional, controlled by load_frames)
    - traj: (T,6) float32
    """

    def __init__(
        self,
        manifest_json: str,
        items_key: str = "ALL",
        workspace_root: Optional[str] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        load_frames: bool = False,
        load_traj: bool = True,
        max_items: Optional[int] = None,
        require_T: Optional[int] = 49,
        io_timeout_s: float = 0.0,
        on_error: str = "raise",
    ):
        self.manifest_json = str(manifest_json)
        self.items_key = str(items_key)
        self.workspace_root = str(workspace_root).strip() if workspace_root else os.getcwd()
        self.transform = transform
        self.load_frames = bool(load_frames)
        self.load_traj = bool(load_traj)
        self.require_T = int(require_T) if require_T is not None else None
        self.io_timeout_s = float(io_timeout_s) if io_timeout_s is not None else 0.0
        self.on_error = str(on_error).strip().lower() if on_error is not None else "raise"
        if self.on_error not in ("raise", "empty"):
            raise ValueError(f"on_error must be 'raise' or 'empty', got {on_error!r}")

        with open(self.manifest_json, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("manifest must be a JSON dict")

        key_s = str(self.items_key).strip()
        if key_s.lower() == "all":
            keys = [k for k, v in data.items() if isinstance(k, str) and k.startswith("items_") and isinstance(v, list)]
            if len(keys) == 0:
                raise ValueError("manifest has no keys like items_* with list values")
        else:
            keys = [k.strip() for k in key_s.split(",") if k.strip()]
            for k in keys:
                if k not in data:
                    raise ValueError(f"manifest missing key={k}")
                if not isinstance(data.get(k), list):
                    raise ValueError(f"manifest[{k}] must be a list")

        raw_items: List[Dict[str, Any]] = []
        for k in keys:
            raw_items.extend(data.get(k, []))

        items: List[LatentTrajManifestItem] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            lp = it.get("latent_path")
            tp = it.get("traj_json_path")
            im = it.get("images_dir")
            if not (lp and tp and im):
                continue
            items.append(
                LatentTrajManifestItem(
                    latent_path=_resolve_path(lp, self.workspace_root),
                    traj_json_path=_resolve_path(tp, self.workspace_root),
                    images_dir=_resolve_path(im, self.workspace_root),
                )
            )

        if max_items is not None:
            items = items[: int(max_items)]
        if len(items) == 0:
            raise FileNotFoundError(f"No valid items found in manifest={self.manifest_json} key={self.items_key}")

        self.items = items

    @contextmanager
    def _io_timeout(self):
        """
        Best-effort wall-time timeout for slow network filesystems.
        Only works on Unix when signals are available; otherwise it's a no-op.
        """
        sec = float(self.io_timeout_s)
        if sec <= 0:
            yield
            return
        if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handler(_signum, _frame):
            raise TimeoutError(f"I/O timeout after {sec:.1f}s")

        old = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, sec)
            yield
        finally:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except Exception:
                pass
            try:
                signal.signal(signal.SIGALRM, old)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.items)

    def _load_latents(self, path: str) -> torch.Tensor:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "latents" in obj:
            z = obj["latents"]
        else:
            z = obj
        if not isinstance(z, torch.Tensor):
            raise ValueError(f"latents at {path} is not a Tensor")
        if z.ndim != 5:
            raise ValueError(f"expected latents ndim=5 at {path}, got shape={tuple(z.shape)}")
        return z.float().contiguous()

    def _load_traj(self, path: str) -> np.ndarray:
        with open(path, "r") as f:
            arr = json.load(f)

        # Common formats:
        # 1) List[[x,y,z,roll,yaw,pitch], ...]  -> (T,6)
        # 2) Dict wrapping the list under a known key (preprocessed_logs / processed_logs / traj / poses / logs)
        # 3) IndoorUAV "processed_log.json" format: dict with action6 of length (T-1,6) in layout:
        #    [dz(dyaw), dy, dx, tx, ty, tz] where dz may be in rad or deg depending on generator version.
        if isinstance(arr, dict):
            for k in ("preprocessed_logs", "processed_logs", "traj", "poses", "logs"):
                v = arr.get(k, None)
                if isinstance(v, list):
                    arr = v
                    break
            else:
                a6 = arr.get("action6", None)
                if isinstance(a6, list) and (len(a6) == 0 or isinstance(a6[0], (list, tuple))):
                    action6 = np.asarray(a6, dtype=np.float32)
                    if action6.ndim != 2 or action6.shape[1] < 6:
                        raise ValueError(f"action6 must be (T-1,6+) at {path}, got shape={action6.shape}")

                    # Ensure dz(dyaw) is in radians when possible (best-effort, matches IndoorUavF49Dataset).
                    unit = "auto"
                    meta = arr.get("meta") if isinstance(arr.get("meta"), dict) else {}
                    layout = meta.get("action6_layout")
                    layout_s = " ".join([str(x) for x in layout]).lower() if isinstance(layout, list) else str(layout or "").lower()
                    if "rad" in layout_s:
                        unit = "rad"
                    elif "deg" in layout_s:
                        unit = "deg"
                    else:
                        dz = action6[:, 0]
                        p95 = float(np.nanpercentile(np.abs(dz), 95)) if dz.size else 0.0
                        unit = "deg" if p95 > 1.0 else "rad"
                    if unit == "deg":
                        action6 = action6.copy()
                        action6[:, 0] = action6[:, 0] * (np.pi / 180.0)

                    # Convert per-step action6 (T-1,6) to a (T,6) sequence with delta[0]=0.
                    T = int(action6.shape[0]) + 1
                    delta = np.zeros((T, 6), dtype=np.float32)
                    delta[1:, :] = action6[:, :6].astype(np.float32)
                    return delta

        traj = np.asarray(arr, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] < 6:
            raise ValueError(f"traj must be (T,6+) at {path}, got shape={traj.shape}")
        return traj[:, :6]

    def _frame_path(self, images_dir: str, idx: int) -> str:
        p = os.path.join(images_dir, f"frame_{idx:06d}.png")
        if os.path.exists(p):
            return p
        for ext in (".jpg", ".jpeg", ".webp"):
            p2 = os.path.join(images_dir, f"frame_{idx:06d}{ext}")
            if os.path.exists(p2):
                return p2
        raise FileNotFoundError(f"Frame not found: {images_dir} idx={idx}")

    def _load_frames(self, images_dir: str, T: int) -> torch.Tensor:
        if self.transform is None:
            raise ValueError("transform must be provided when load_frames=True")
        frames: List[torch.Tensor] = []
        for i in range(int(T)):
            img = Image.open(self._frame_path(images_dir, i)).convert("RGB")
            x = self.transform(img)
            if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] != 3:
                raise ValueError("transform(img) must return a Tensor shaped (3,H,W)")
            frames.append(x.unsqueeze(0))
        x = torch.cat(frames, dim=0)  # (T,3,H,W)
        return x.transpose(0, 1).contiguous()  # (3,T,H,W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[int(idx)]
        try:
            with self._io_timeout():
                z_ext = self._load_latents(it.latent_path)  # (1,64,T_lat,16,16)
            traj = None
            T = None
            if self.load_traj:
                with self._io_timeout():
                    traj = self._load_traj(it.traj_json_path)  # (T,6)
                T = int(traj.shape[0])
                if self.require_T is not None and T != int(self.require_T):
                    raise ValueError(f"require_T={self.require_T} but got T={T} for {it.traj_json_path}")
        except Exception as e:
            if self.on_error == "empty":
                # Return an empty latent so training code can safely skip this sample
                z_ext = torch.empty((1, 64, 0, 16, 16), dtype=torch.float32)
                traj = np.zeros((0, 6), dtype=np.float32) if self.load_traj else None
                T = 0
                err_s = f"{type(e).__name__}: {e}"
                out: Dict[str, Any] = {
                    "z_ext": z_ext,
                    "meta": {
                        "latent_path": it.latent_path,
                        "traj_json_path": it.traj_json_path,
                        "images_dir": it.images_dir,
                        "error": err_s[:500],
                    },
                }
                if self.load_traj:
                    out["traj"] = traj
                return out
            raise

        out: Dict[str, Any] = {
            "z_ext": z_ext,
            "meta": {
                "latent_path": it.latent_path,
                "traj_json_path": it.traj_json_path,
                "images_dir": it.images_dir,
            },
        }
        if self.load_traj:
            out["traj"] = traj
        if self.load_frames:
            if T is None:
                raise ValueError("load_frames=True requires load_traj=True (to get T)")
            out["frames_rgb"] = self._load_frames(it.images_dir, T=T)
        return out

