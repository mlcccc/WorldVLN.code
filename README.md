# vln_uav Open Source Release

This repository contains the public-facing code packages extracted from the UAV-Flow research codebase. Instead of shipping a single monolithic project, the release is organized as a small set of focused components for post-training, inference, action decoding, and backbone training.

The goal of this layout is to keep each workflow self-contained while preserving the pieces that are most useful for reproduction, deployment, and follow-up research.

## Overview

The repository is split into four top-level packages:

| Directory | Purpose | Typical Use Case |
| --- | --- | --- |
| `posttrain/` | UAV-Flow RL and post-training package | Run StageA rollout collection, build replay data, and launch StageB training |
| `infer/` | Lightweight online inference service | Deploy an `InfinityStar -> latent2action` API service |
| `action_module/` | Action decoder training and evaluation package | Train or evaluate latent-to-action models |
| `backbone/` | Trimmed InfinityStar backbone training package | Fine-tune the visual backbone from base checkpoints |

## Repository Layout

```text
opensource/
|-- README.md
|-- action_module/
|-- backbone/
|-- infer/
`-- posttrain/
```

## Getting Started

This repository does not provide a single root-level installation script. Each package is intended to be used independently, with its own dependencies, runtime assets, and entrypoints.

If you are new to the project, the recommended starting points are:

1. Start with `posttrain/` if your goal is to reproduce the UAV-Flow StageA/StageB workflow.
2. Start with `infer/` if you only need the online inference server.
3. Start with `action_module/` if you want to train or evaluate the latent-to-action component.
4. Start with `backbone/` if you want a minimal InfinityStar backbone fine-tuning package.

## Packages

### `posttrain/`

`posttrain/` is the main entry point for the UAV-Flow post-training pipeline. It contains the publishable subset of the original RL training tree, repackaged to avoid dependence on the private experimental repository layout.

This package includes:

- StageA rollout collection
- replay meta generation
- StageB partial-freeze training
- local inference server integration
- remote simulator service integration for online rollout

Primary entrypoints:

- `posttrain/README.md`
- `posttrain/scripts/run_stagea_collect.sh`
- `posttrain/scripts/run_stageb_partialfreeze.sh`
- `posttrain/scripts/run_remote_sim_service.sh`
- `posttrain/run_infer_server.sh`

Use this package if you want the most complete public workflow for UAV-Flow training.

### `infer/`

`infer/` is a lightweight deployment-oriented package centered on the online inference server. It is designed for users who only need to run the `InfinityStar -> latent2action` serving path and do not need the full post-training workflow.

Primary entrypoints:

- `infer/run_server.sh`
- `infer/infinity_tsformer_api_server.py`
- `infer/config.json`
- `infer/OPEN_SOURCE_CLEANUP.md`

Use this package when you want a smaller operational surface for serving or integration.

### `action_module/`

`action_module/` contains the action-decoder portion of the project. It supports adapter distillation, latent-to-action training, batch inference, and endpoint evaluation for UAV-Flow-style trajectories.

Primary entrypoints:

- `action_module/README.md`
- `action_module/scripts/train_stage1_ddp.sh`
- `action_module/scripts/train_stage2_ddp.sh`
- `action_module/tools/predict_pose.py`
- `action_module/tools/eval_endpoints.py`

Use this package if your main interest is the motion prediction model rather than the full RL pipeline.

### `backbone/`

`backbone/` is a trimmed training release for InfinityStar backbone fine-tuning. It keeps a single supported training path and omits unrelated demo or application code.

Primary entrypoints:

- `backbone/README.md`
- `backbone/TRAINING.md`
- `backbone/scripts/train_from_base.sh`

Use this package if you need a compact backbone training setup built around original sharded base weights.

## Dependency Model

The four packages are related, but they are not intended to behave like one tightly integrated monorepo.

At a high level:

- `posttrain/` and `infer/` each vendor local copies of `InfinityStar-main/` and `TSformer-VO-main/` so that they can run as self-contained packages.
- `action_module/` depends on an externally available InfinityStar codebase and VAE assets.
- `backbone/` contains its own training code, but still expects user-provided checkpoints, data, and configuration.

In practice, you should treat each directory as an independently documented package.

## External Assets

This repository does not ship private checkpoints, datasets, or internal experiment caches. Depending on the package you use, you will typically need to provide some combination of the following:

- InfinityStar checkpoints
- shared T5 and VAE assets
- action-head checkpoints and matching config files
- UAV-Flow task JSON files or rollout source manifests
- replay metadata or training JSONL files

The strongest external dependency surface is in `posttrain/`, where runtime assets are intentionally injected through environment variables or explicit arguments rather than hard-coded private paths.

## Recommended Reading Order

For a first pass through the release, the most useful reading order is:

1. `posttrain/README.md`
2. `posttrain/docs/remote_sim.md`
3. `infer/OPEN_SOURCE_CLEANUP.md`
4. `action_module/README.md`
5. `backbone/TRAINING.md`

## Notes

- There is no single root-level one-command setup for the entire repository.
- `posttrain/` is the closest package to the original UAV-Flow training workflow and should usually be your default entrypoint.
- `infer/` is better suited for standalone deployment.
- `action_module/` and `backbone/` are model-component packages rather than end-to-end application pipelines.

## License

Each subdirectory may include its own license file or upstream attribution details. Please review the `LICENSE`, `README.md`, and training documentation in the package you intend to use.