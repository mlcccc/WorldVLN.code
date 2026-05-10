# Action Decoder Module

This repository contains the action-decoder portion of the project: a TSformer-based latent-to-action pipeline that uses an external InfinityStar video VAE to decode latents, an adapter to map VAE decoder features into TSformer token space, and a Stage-2 model to predict 6-DoF motion deltas.

The current public package is organized around the UAV-Flow workflow:

1. **Stage 1**: distill an adapter from InfinityStar VAE decoder features to TSformer patch tokens.
2. **Stage 2**: train a latent-to-action model on top of the frozen or partially trainable adapter / TSformer / VAE stack.
3. **Inference**: run batch latent-to-action inference on route folders containing `latents.pt` and pose logs.
4. **Evaluation**: compute endpoint distance / rotation errors on UAV-Flow-style trajectories.

## What Is Included

The main supported entry points are:

- `tools/train_stage1_ddp.py`: Stage-1 adapter distillation.
- `tools/train_stage2_ddp.py`: Stage-2 latent-to-action training.
- `tools/predict_pose.py`: batch inference for latent route folders.
- `tools/eval_endpoints.py`: endpoint evaluation for UAV-Flow-style predictions.
- `scripts/train_stage1_ddp.sh`: thin DDP launcher for Stage 1.
- `scripts/train_stage2_ddp.sh`: thin DDP launcher for Stage 2.

Core code lives in:

- `datasets/`: manifest-backed dataset loading and pose utilities.
- `models/`: adapter modules and model helpers.
- `timesformer/`: TSformer backbone and supporting utilities.

## External Dependencies

This repository does **not** vendor the InfinityStar source tree or VAE weights.

You need:

- an installed InfinityStar codebase, discoverable by one of:
  - `--infinitystar_root`
  - environment variable `INFINITYSTAR_ROOT`
  - environment variable `INFINITYSTAR_HOME`
  - `third_party/InfinityStar-main` under this repo
- a valid InfinityStar VAE checkpoint path
- PyTorch / torchvision built for your target CUDA or CPU environment

`requirements.txt` in this repository includes the Python packages commonly
needed by `action_module` itself and by the InfinityStar VAE import path used
in `tools/train_stage1_ddp.py`, `tools/train_stage2_ddp.py`, and
`tools/predict_pose.py`. If you also plan to run other scripts from the
InfinityStar repository directly, install any additional dependencies required
by that checkout as well.

## Supported Data Layouts

### 1. Training manifest

Stage-1 and Stage-2 training use `datasets/latent_traj_manifest.py`.

The manifest is a JSON object containing one or more `items_*` lists. Each item is expected to contain:

```json
{
  "latent_path": "path/to/latents.pt",
  "traj_json_path": "path/to/preprocessed_logs.json",
  "images_dir": "path/to/images"
}
```

The training scripts accept:

- `--manifest_json`: manifest file path
- `--items_key`: one key, multiple comma-separated keys, or `ALL`

### 2. Inference route folders

`tools/predict_pose.py` expects each route directory under `--data_root` to contain:

```text
<route>/
  latents.pt
  preprocessed_logs.json
```

`preprocessed_logs.json` is expected to contain absolute poses with layout:

```text
[x, y, z, roll, yaw, pitch]
```

By default:

- translations are interpreted with `--translation_divisor 1.0`
- angles are interpreted as degrees (`--angles_in_degrees`)

### 3. Evaluation route folders

`tools/eval_endpoints.py` expects:

- ground truth under `--gt_root/<route>/`
- predictions under `--pred_root/<route>/`

Ground truth:

```text
preprocessed_logs.json   # preferred
raw_logs.json            # fallback
```

Prediction:

```text
pred_actions.json        # preferred
pred_path.json           # fallback
```

The evaluator uses:

- `pred_actions.json["actions6"]` when available
- SE(3) integration from the GT start pose
- endpoint distance in meters / centimeters
- geodesic rotation error in degrees
- yaw-only endpoint error in degrees

## Installation

Recommended Python version: **3.10+**

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

or use Conda:

```bash
conda create -n action-decoder python=3.10
conda activate action-decoder
```

### 2. Install PyTorch

Install a PyTorch build that matches your CUDA / CPU setup by following the official instructions:

- [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

If you are binding this repository to a separate InfinityStar checkout, make
sure the two environments are version-compatible, especially for PyTorch,
torchvision, and `timm`.

## Training

The bash launchers are intentionally thin wrappers. Required data / checkpoint paths are passed through environment variables, and any additional training arguments can be appended with `EXTRA_ARGS`.

### Stage 1: Adapter distillation

Required environment variables:

- `MANIFEST_JSON`
- `TSFORMER_CKPT`
- `INF_VAE_PATH`

Common optional variables:

- `ITEMS_KEY` (default: `ALL`)
- `CONDA_ENV_PREFIX`
- `OUT_DIR`
- `LOG_DIR`
- `EXTRA_ARGS`

Example:

```bash
export MANIFEST_JSON=/path/to/latent_traj_manifest.json
export TSFORMER_CKPT=/path/to/tsformer_checkpoint.pth
export INF_VAE_PATH=/path/to/infinitystar_videovae.pth
export OUT_DIR=./outputs/stage1_adapter
export EXTRA_ARGS="--epochs 20 --global_batch_size 32 --amp --latent_use_full --collate_mode per_sample"

bash scripts/train_stage1_ddp.sh
```

Main script:

```bash
python tools/train_stage1_ddp.py --help
```

### Stage 2: Latent-to-action training

Required environment variables:

- `MANIFEST_JSON`
- `TSFORMER_PRETRAINED`
- `ADAPTER_CKPT`
- `INFINITYSTAR_VAE_PATH`

Common optional variables:

- `ITEMS_KEY` (default: `ALL`)
- `LABEL_STATS_JSON`
- `RESUME`
- `OUT_DIR`
- `LOG_DIR`
- `EXTRA_ARGS`

Example:

```bash
export MANIFEST_JSON=/path/to/latent_traj_manifest.json
export TSFORMER_PRETRAINED=/path/to/tsformer_pretrained.pth
export ADAPTER_CKPT=/path/to/stage1_adapter_last.pt
export INFINITYSTAR_VAE_PATH=/path/to/infinitystar_videovae.pth
export OUT_DIR=./outputs/stage2_latent2action
export EXTRA_ARGS="--epochs 40 --global_batch_size 8 --grad_accum_steps 2 --amp --train_adapter --angles_in_degrees --translation_divisor 1.0"

bash scripts/train_stage2_ddp.sh
```

Main script:

```bash
python tools/train_stage2_ddp.py --help
```

## Inference

Run latent-to-action batch inference with:

```bash
python tools/predict_pose.py \
  --ckpt /path/to/stage2_latent2action_combined.pt \
  --data_root /path/to/uavflow_routes \
  --out_dir ./outputs/infer \
  --infinitystar_vae_path /path/to/infinitystar_videovae.pth \
  --angles_in_degrees \
  --translation_divisor 1.0 \
  --tqdm
```

Useful arguments:

- `--routes route_a,route_b`
- `--first_n N`
- `--device cuda:0`
- `--compute_metrics`
- `--infinitystar_root /path/to/InfinityStar-main`

Outputs per route typically include:

- `pred_actions.json`
- `pred_path.json`
- `deltas.npy`
- `window_deltas.npy`
- `trajectory.npy`
- `trajectory.json`
- `trajectory_m_deg.npy`
- `trajectory_m_deg.json`
- `actions6_m_deg.npy`
- `actions6_m_deg.json`
- `metrics.json`

## Evaluation

Evaluate endpoint errors with:

```bash
python tools/eval_endpoints.py \
  --pred_root ./outputs/infer \
  --gt_root /path/to/uavflow_routes \
  --out_root ./outputs/eval \
  --gt_pose_file preprocessed_logs.json \
  --translation_divisor 1.0 \
  --angles_in_degrees \
  --dist_thr_m 3.0 \
  --ang_thr_deg 10.0 \
  --tqdm
```

Evaluation outputs:

- `summary.txt`
- `images/distance_error_distribution.png`
- `images/rotation_error_distribution.png`
- `images/yaw_error_distribution.png`
- `images/trajectories_3d_overlay.png`

## Deployment Notes

- Keep the repo root on `PYTHONPATH` or run scripts from the repository root.
- Install InfinityStar separately and point the scripts to it with `--infinitystar_root` or environment variables.
- Provide your own data manifests, route folders, and checkpoints. This repository does not ship datasets or pretrained weights.
- If you publish checkpoints, review checkpoint metadata first: some training exports include paths such as VAE checkpoint paths or pretrained TSformer paths.

## Repository Layout

```text
action_module/
  datasets/
  models/
  scripts/
  timesformer/
  tools/
  build_model.py
  train.py
  requirements.txt
```

`train.py` and `build_model.py` are kept for the original TSformer-VO baseline code path. The main latent-to-action workflow is driven by the scripts under `tools/` and `scripts/`.
