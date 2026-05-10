# UAV-Flow Post-Training Package

This directory is the publishable UAV-Flow post-training subset extracted from the experimental rl_partialfreeze tree. All cleanup and packaging changes live here; the original training repo is left untouched.

## Layout

- InfinityStar-main: copied training core used by StageA and StageB.
- TSformer-VO-main: copied action-head dependency used by the local inference service.
- infinity_tsformer_api_server.py: local inference service used by StageA.
- runtime/infinity_tsformer_client.py: validated simulator-side client and remote_sim service.
- scripts/run_stagea_collect.sh: public StageA wrapper.
- scripts/run_stageb_partialfreeze.sh: public StageB wrapper.
- scripts/run_remote_sim_service.sh: public simulator-service wrapper.
- run_infer_server.sh: public local inference server launcher.
- outputs: default local output root for rollout caches, logs, token caches, and checkpoints.

## External Assets

This package does not ship private checkpoints or dataset payloads. You must provide these paths via environment variables or command arguments.

- SRC_JSON: UAV-Flow rollout source manifest JSON.
- INFINITY_CKPT: InfinityStar checkpoint used for StageA/local inference.
- CHECKPOINTS_DIR: shared InfinityStar assets used by StageA/local inference and StageB, including T5 and VAE files.
- ACTIONHEAD_CKPT: action-head checkpoint for actionhead_ref_vit mode.
- ACTIONHEAD_RUN_CONFIG: matching run_config.json for the action head.
- RUSH_RESUME: optional StageB starting checkpoint.
- UAVFLOW_TASK_JSON_ROOT: required for remote_sim StageA; directory of UAV-Flow-Eval task json files.

## Quick Start

Start the local inference service:

```bash
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
bash run_infer_server.sh
```

Run StageA with offline reference videos:

```bash
SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json \
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
bash scripts/run_stagea_collect.sh RUN_ID=stagea_smoke TOP_N=1 K_CAND=1
```

Run StageA against a remote simulator service:

```bash
SRC_JSON=/path/to/reference_video_full_49f_trajectory_prompts.json \
INFINITY_CKPT=/path/to/infinity/global_step_xxx.pth \
CHECKPOINTS_DIR=/path/to/checkpointsinf \
ACTIONHEAD_CKPT=/path/to/actionhead/checkpoint_last.pth \
ACTIONHEAD_RUN_CONFIG=/path/to/actionhead/run_config.json \
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim \
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765 \
UAVFLOW_TASK_JSON_ROOT=/path/to/UAV-Flow-Eval/test_jsons \
bash scripts/run_stagea_collect.sh RUN_ID=remote_sim_smoke TOP_N=1 K_CAND=1 STAGEA_NPROC=1
```

Run StageB partial-freeze training on a replay_meta directory:

```bash
CHECKPOINTS_DIR=/path/to/checkpointsinf \
RUSH_RESUME=/path/to/infinity/global_step_xxx.pth \
REPLAY_META_DIR=/path/to/replay_meta_stagea_smoke \
bash scripts/run_stageb_partialfreeze.sh PARTIAL_FREEZE_MODE=smoke RUN_ID=stageb_smoke
```

## Remote Simulator

Use the simulator service wrapper on the machine that can access UnrealCV/UAV-Flow:

```bash
python runtime/infinity_tsformer_client.py --mode service --host 0.0.0.0 --port 8765 --task_json_root /path/to/UAV-Flow-Eval/test_jsons
```

If the training host cannot directly reach the simulator host, use reverse SSH port forwarding and point UAVFLOW_SIMULATOR_BASE_URL at the forwarded localhost port. See docs/remote_sim.md for the expected request flow.

## Notes

- Default outputs go under outputs/. Override OUTPUT_ROOT, FAST_OUT_DIR, RL_CACHE_ROOT, TRAIN_CKPT_ROOT, TRAIN_LOG_ROOT, or TRAIN_TOKEN_CACHE_ROOT if needed.
- config.json is publish-safe by default: checkpoint fields are intentionally empty and should be provided through environment variables.
- StageB includes the small-smoke prof-frequency guard from the validated training tree, so tiny replay_meta smoke runs no longer divide by zero.
