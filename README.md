# WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation


This is the official code repository for WorldVLN. The repository includes the main code paths used for backbone training, action decoding, inference serving, and action-aware GRPO workflows.

## Installation

We recommend using a single Python 3.10 environment for the released workflows. In our validated launch scripts, the Python interpreter is passed explicitly through `PYTHON_BIN`, so after activating your environment it is recommended to export:

```bash
export PYTHON_BIN=$(which python)
```

### Recommended Environment

1. Create a Python 3.10 environment.

```bash
conda create -n worldvln python=3.10
conda activate worldvln
```

2. Install a PyTorch build that matches your CUDA environment. For the released training and action-aware GRPO workflows, a PyTorch 2.5.1 environment is the recommended baseline.

3. Install the shared dependencies used by the released workflows.

```bash
pip install -r requirements.txt
```

## Setup

### Model Weights

Official WorldVLN backbone weights are available on Hugging Face:

- [WorldVLN backbone weights](https://huggingface.co/anonymous-WorldVLN/WorldVLN/tree/main/WorldVLN_backbone)

Download the weights to your preferred checkpoint directory and configure the relevant training or inference scripts to point to them.

## Inference

The repository currently provides two main inference surfaces.

![WorldVLN model](./assets/model.png)

### Online Inference Service

The online service lives under [infer/](./infer) and is intended for deployment-oriented usage.

- Entry points: [infer/run_server.sh](./infer/run_server.sh), [infer/infinity_tsformer_api_server.py](./infer/infinity_tsformer_api_server.py)
- Configuration: [infer/config.json](./infer/config.json)
- Typical usage: serve the model behind an HTTP API for online prediction or system integration

At a high level, this service consumes the current observation context and model inputs, then returns action predictions through the API server.

#### Quick start

From the repository root:

```bash
export PYTHON_BIN=$(which python)
export INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth
export STAGE2_LATENT2ACTION_CKPT=/path/to/stage2_latent2action_combined.pt

bash infer/run_server.sh
```

Common environment variables:

- `INFINITY_CKPT`: main InfinityStar / WorldVLN checkpoint used by the service
- `STAGE2_LATENT2ACTION_CKPT`: Stage-2 latent-to-action checkpoint for action prediction
- `INFINITY_SERVER_CONFIG`: optional override for `infer/config.json`
- `INFINITY_REPO_ROOT`: optional override for the default `Worldmodel/runtime/`
- `INFINITY_LATENT_CACHE_ROOT`: runtime cache directory used by the service
- `HOST`, `PORT`: bind address for Uvicorn

### Batch Latent-to-Action Inference

This path is intended for **offline** inference and evaluation on route-level data (as opposed to online serving).

The batch inference entrypoints live under `train/action_decoder/tools/`:

- Inference: [train/action_decoder/tools/predict_pose.py](./train/action_decoder/tools/predict_pose.py)
- Evaluation: [train/action_decoder/tools/eval_endpoints.py](./train/action_decoder/tools/eval_endpoints.py)

#### 1) Prepare route folders

Each route directory under `--data_root` should contain:

```text
<route>/
  latents.pt
  preprocessed_logs.json
```

`preprocessed_logs.json` is a list of poses with layout `[x, y, z, roll, yaw, pitch]` (angles are treated as degrees by default).

#### 2) Run batch inference

From the repository root:

```bash
python train/action_decoder/tools/predict_pose.py \
  --ckpt <path/to/stage2_checkpoint>.pth \
  --data_root <route_root_dir> \
  --out_dir <output_root_dir> \
  --infinitystar_root <path/to/Worldmodel/runtime> \
  --infinitystar_vae_path <path/to/infinitystar_videovae.pth>
```

Outputs are written per-route under `--out_dir/<route>/` and include `pred_actions.json` and `pred_path.json` for downstream evaluation or integration.

#### 3) Evaluate endpoints (optional)

```bash
python train/action_decoder/tools/eval_endpoints.py \
  --gt_root <gt_route_root_dir> \
  --pred_root <output_root_dir> \
  --out_root <eval_out_dir>
```

## Training

This repository is organized into two stages:

- **Stage 1 (supervised)**: backbone finetuning + action decoder training.
- **Stage 2 (action-aware GRPO)**: rollout collection + GRPO training.

![WorldVLN framework](./assets/framework.png)

### Stage 1: Supervised Training

#### Backbone Training

The backbone finetuning workflow is located under [train/](./train).

- Entry point: [train/scripts/train_from_base.sh](./train/scripts/train_from_base.sh)
- Main trainer: [train/train.py](./train/train.py)
- Detailed guide: [train/TRAINING.md](./train/TRAINING.md)

Use this workflow when you want to fine-tune the WorldVLN backbone from base checkpoints.

#### Quick start

```bash
bash train/scripts/train_from_base.sh
```

#### Action Decoder Training

The action decoder workflow is located under [Worldmodel/action_decoder/src/](./Worldmodel/action_decoder/src) and is organized into two stages.

The action decoder training entrypoints live under [train/action_decoder/](./train/action_decoder) and are organized into two stages.

- Stage 1 adapter distillation: [train/action_decoder/scripts/train_stage1_ddp.sh](./train/action_decoder/scripts/train_stage1_ddp.sh)
- Stage 2 latent-to-action training: [train/action_decoder/scripts/train_stage2_ddp.sh](./train/action_decoder/scripts/train_stage2_ddp.sh)
- Main scripts: [train/action_decoder/tools/train_stage1_ddp.py](./train/action_decoder/tools/train_stage1_ddp.py), [train/action_decoder/tools/train_stage2_ddp.py](./train/action_decoder/tools/train_stage2_ddp.py)

This workflow trains the mapping from visual latent features to 6-DoF motion outputs.

Data contract (training manifest):

```json
{
  "items_train": [
    {
      "latent_path": "path/to/latents.pt",
      "traj_json_path": "path/to/preprocessed_logs.json",
      "images_dir": "path/to/images"
    }
  ]
}
```

Stage 1 required environment variables:

- `MANIFEST_JSON`
- `TSFORMER_CKPT`
- `INF_VAE_PATH`

Run Stage 1:

```bash
bash train/action_decoder/scripts/train_stage1_ddp.sh
```

Stage 2 required environment variables:

- `MANIFEST_JSON`
- `TSFORMER_PRETRAINED`
- `ADAPTER_CKPT`
- `INFINITYSTAR_VAE_PATH`

Run Stage 2:

```bash
bash train/action_decoder/scripts/train_stage2_ddp.sh
```

### Stage 2: Action-aware GRPO

The action-aware GRPO workflow is located under [action_aware_grpo/](./action_aware_grpo) and is organized into two steps: **rollout** and **train**.

- Rollout collection: [action_aware_grpo/scripts/run_stagea_collect.sh](./action_aware_grpo/scripts/run_stagea_collect.sh)
- Train (partial-freeze optimization): [action_aware_grpo/scripts/run_stageb_partialfreeze.sh](./action_aware_grpo/scripts/run_stageb_partialfreeze.sh)
- Remote simulator service wrapper: [action_aware_grpo/scripts/run_remote_sim_service.sh](./action_aware_grpo/scripts/run_remote_sim_service.sh)
- Local inference launcher used by rollout: [action_aware_grpo/run_infer_server.sh](./action_aware_grpo/run_infer_server.sh)

At a high level:

- Rollout consumes rollout sources and model assets, then generates rollout caches and replay metadata.
- Train consumes replay metadata and runs optimization to produce updated checkpoints and logs.

#### Quick start (rollout + train)

Start the local inference service used by rollout:

```bash
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
bash action_aware_grpo/run_infer_server.sh
```

Run rollout collection:

```bash
unset ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json \
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
CUDA_VISIBLE_DEVICES=0 \
GRPO_LOCAL_GPU_IDS=0 \
NPROC_PER_NODE=1 \
NNODES=1 \
NODE_RANK=0 \
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim \
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765 \
UAVFLOW_SIMULATOR_TIMEOUT_S=120 \
UAVFLOW_TASK_JSON_ROOT=/path/to/UAV-Flow-Eval/test_jsons \
bash action_aware_grpo/scripts/run_stagea_collect.sh RUN_ID=remote_sim_smoke TOP_N=1 K_CAND=1 STAGEA_NPROC=1 STAGEA_PROGRESS_EVERY_N=1
```

Run train (partial-freeze optimization):

```bash
CHECKPOINTS_DIR=/path/to/checkpointsinf \
RUSH_RESUME=/path/to/infinity/global_step_xxx.pth \
REPLAY_META_DIR=/path/to/replay_meta_rollout_smoke \
bash action_aware_grpo/scripts/run_stageb_partialfreeze.sh PARTIAL_FREEZE_MODE=smoke RUN_ID=stageb_smoke
```

For simulator-backed rollout details, see [action_aware_grpo/docs/remote_sim.md](./action_aware_grpo/docs/remote_sim.md).

## License

This project is released under the MIT License. See `LICENSE`.