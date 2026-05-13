# WorldVLN Inference Packaging Notes

This document is intended for maintainers who want to keep the `infer/` directory publishable as a clean WorldVLN inference package.

## Primary Entry Points

The current public inference surface is built around:

- `infinity_tsformer_api_server.py`
- `run_server.sh`
- `config.json`

## Files Required for the Service Path

If you want to preserve only the `InfinityStar -> latent2action` online service path, the following code should remain available:

- `infinity_tsformer_api_server.py`
- `config.json`
- `run_server.sh`
- `../Worldmodel/runtime/infinity/`
- `../Worldmodel/runtime/tools/closed_loop_streaming_infer_480p_81f.py`
- `../Worldmodel/runtime/tools/infinity_streaming_session.py`
- `../Worldmodel/runtime/tools/run_infinity.py`
- `../Worldmodel/action_decoder/actionhead_runtime/timesformer/`
- `../Worldmodel/action_decoder/actionhead_runtime/models/vae96_to_tsformer_adapter.py`

## Optional Files

The following files are not required for the main online serving path, but may still be useful for local debugging or experimentation:

- `config.local_bestrecord.json`
- `../train/action_decoder/actionhead_training/pretrain_latent_p2p.py`
- `../train/action_decoder/actionhead_training/latent_patch_embed.py`

## Files That Should Not Be Published

To keep the package clean for open-source release, avoid committing local runtime artifacts, private assets, and unnecessary experimental files.

Examples include:

- `__pycache__/`
- local cache directories such as `cache/`
- private checkpoints
- logs, archives, temporary files, and local experiment artifacts

When pruning vendored trees, retain only the source files needed by the published service workflow and avoid carrying unrelated training or legacy experiment code unless it is still required by the runtime path.

## Notes

- The inference package now reuses the top-level `Worldmodel/` and `action_decoder/` folders instead of vendoring separate copies.
- Default paths have been converted to repository-relative behavior where possible.
- Model weights should continue to be supplied through environment variables or mounted local paths rather than committed into the repository.

