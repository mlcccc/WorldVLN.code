# Training Guide

This repository keeps a single training entrypoint:

- `scripts/train_from_base.sh`

The script starts finetuning from the original sharded base weights and uses repository-relative defaults for data, checkpoints, and outputs.

## Open-source Readiness

`scripts/train_from_base.sh` no longer contains machine-specific hard-coded paths such as `/home/...`, `/manifold-obs/...`, or old workspace locations.

Default paths are resolved relative to the repository root:

- checkpoints: `checkpoints/`
- data: `data/`
- outputs: `outputs/`

The script still converts the repository root into an absolute runtime path for `PYTHONPATH`, but that path is derived from the local clone location at launch time and is portable across machines.

## Required Layout

Expected repository layout:

```text
Train_WAM/
|-- train.py
|-- scripts/
|   `-- train_from_base.sh
|-- checkpoints/
|   |-- text_encoder/
|   |   `-- flan-t5-xl-official/
|   |-- infinitystar_videovae.pth
|   `-- infinitystar_8b_480p_weights/
|-- data/
|   `-- <your jsonl shard directory>
`-- outputs/
```

## Required Weights

Before launching training, prepare these files under `checkpoints/` or override them with environment variables:

1. T5 text encoder directory

   Default path:

   - `checkpoints/text_encoder/flan-t5-xl-official`

   Environment override:

   - `T5_PATH`

2. Video VAE checkpoint

   Default path:

   - `checkpoints/infinitystar_videovae.pth`

   Environment override:

   - `VAE_PATH`

3. Base model sharded weights

   Default path:

   - `checkpoints/infinitystar_8b_480p_weights`

   Environment override:

   - `TORCHSHARD_RESUME_PATH`

The script validates that all three paths exist before calling `train.py`.

## Required Training Data

The script expects `VIDEO_DATA_PATH` to point to a directory containing JSONL shards.

Default search order:

1. `data/uavflow_49f_from_40_60_split8_jsonl`
2. `data/uavflow_40_60_split8_jsonl`
3. `data/split8_jsonl`

Environment override:

- `VIDEO_DATA_PATH`

Supported shard layouts:

1. Flat layout

```text
data/split8_jsonl/
|-- part_00.jsonl
|-- part_01.jsonl
|-- ...
`-- part_07.jsonl
```

2. Bucketed layout

```text
data/split8_jsonl/
|-- bucket_0/
|   |-- part_00.jsonl
|   `-- part_01.jsonl
`-- bucket_1/
    |-- part_02.jsonl
    `-- part_03.jsonl
```

## Minimum JSONL Schema

Each line must be a JSON object. For video training, the loader expects at least:

```json
{
  "video_path": "relative/or/absolute/path/to/video.mp4",
  "begin_frame_id": 0,
  "end_frame_id": 48,
  "fps": 16.0,
  "tarsier2_caption": "A UAV flies over a road."
}
```

Recommended optional fields:

```json
{
  "frame_idxs": [0, 1, 2, 3, 4],
  "sample_frames": 49,
  "quality_prompt": "high quality, detailed",
  "MiniCPM_V_2_6_caption": "Alternative caption text"
}
```

Notes:

1. `video_path` may be absolute or relative, but it must resolve on the training machine.
2. `end_frame_id` is treated as an inclusive frame index by the loader.
3. If `frame_idxs` is provided, the loader uses those explicit frame indices.
4. If `frame_idxs` is not provided, sampling is inferred from the annotated segment and the training configuration.

## Launch Behavior

`scripts/train_from_base.sh` uses the following defaults:

- model: `infinity_qwen8b`
- resolution preset: `0.40M`
- video frames: `49`
- video fps: `16`
- mask schedule: `infinity_elegant_clip4frames_v2_allpt`
- optimizer learning rate: `1e-5`
- total epochs: `10`
- save frequency: `1000` iterations

Outputs are written to:

- logs: `outputs/run_logs/<EXP_NAME>`
- checkpoints: `outputs/checkpoints/<EXP_NAME>`
- token cache: `outputs/cache/<EXP_NAME>`

## Quick Start

Run from the repository root:

```bash
bash scripts/train_from_base.sh
```

Or with explicit overrides:

```bash
CHECKPOINTS_DIR=./checkpoints \
VIDEO_DATA_PATH=./data/split8_jsonl \
OUTPUT_ROOT=./outputs \
ARNOLD_WORKER_GPU=8 \
EXP_NAME=my_train_run \
bash scripts/train_from_base.sh
```

## Environment Variables

Common overrides:

- `PYTHON_BIN`: Python executable to use
- `CHECKPOINTS_DIR`: base directory for all default weight paths
- `T5_PATH`: explicit T5 path
- `VAE_PATH`: explicit VAE checkpoint path
- `TORCHSHARD_RESUME_PATH`: explicit base model shard path
- `DATA_ROOT`: base data directory
- `VIDEO_DATA_PATH`: explicit JSONL shard directory
- `OUTPUT_ROOT`: base output directory
- `LOCAL_OUT_PATH`: explicit run log directory
- `BED_PATH`: explicit checkpoint save directory
- `TOKEN_CACHE_DIR`: explicit token cache directory
- `EXP_NAME`: experiment name
- `TRAIN_EPOCHS`: total epochs
- `SAVE_FREQ_ITERS`: checkpoint save interval
- `TLR`: learning rate
- `ARNOLD_WORKER_GPU`: number of GPUs per node
- `ARNOLD_WORKER_NUM`: number of nodes
- `ARNOLD_ID`: node rank
- `ARNOLD_WORKER_0_HOST`: master host
- `ARNOLD_WORKER_0_PORT`: master port

## Pre-flight Checklist

Before training, verify:

1. `train.py` exists in the repository root.
2. `checkpoints/text_encoder/flan-t5-xl-official` exists.
3. `checkpoints/infinitystar_videovae.pth` exists.
4. `checkpoints/infinitystar_8b_480p_weights` exists.
5. `VIDEO_DATA_PATH` points to a directory containing JSONL files.
6. Every JSONL entry points to a readable local video file.
7. Required Python dependencies from `../requirements.txt` (repository root) are installed.

If any required path is missing, the script exits immediately with an error.
