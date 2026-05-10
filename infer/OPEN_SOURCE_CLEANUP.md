# `infer2` 开源整理说明

这个目录当前已经整理成“以 `infinity_tsformer_api_server.py` 为主入口”的在线推理包。

## 当前主入口

- 服务入口：`infinity_tsformer_api_server.py`
- 启动脚本：`run_server.sh`
- 默认配置：`config.json`

## 运行时必须保留的代码

如果你只保留在线服务 `InfinityStar -> latent2action` 这条链路，下面这些代码应保留：

- `infinity_tsformer_api_server.py`
- `config.json`
- `run_server.sh`
- `InfinityStar-main/infinity/`
- `InfinityStar-main/tools/closed_loop_streaming_infer_480p_81f.py`
- `InfinityStar-main/tools/infinity_streaming_session.py`
- `InfinityStar-main/tools/run_infinity.py`
- `TSformer-VO-main/timesformer/`
- `TSformer-VO-main/models/vae96_to_tsformer_adapter.py`

## 可选保留

这些不是服务运行必需，但可能对调试或示例有帮助：

- `infinity_tsformer_client.py`
- `config.local_bestrecord.json`
- `TSformer-VO-main/pretrain_latent_p2p.py`
- `TSformer-VO-main/latent_patch_embed.py`

## 建议删除或不要提交到 GitHub

如果目标是发布一个干净的在线推理仓库，下面这些内容可以删掉，或者至少不要提交：

- 顶层目录：
  - `__pycache__/`
  - `OFFLINE_GRPO_INFINITY_UAV_PLAN.md`
  - 任何本地缓存目录，例如 `cache/`
  - 任何权重、日志、压缩包、临时文件
- `InfinityStar-main/` 下：
  - `infinity_tsformer_api_server.py`
  - `infinity_tsformer_api_server copy.py`
  - `train.py`
  - `scripts/`
  - `tools/` 中除以下 3 个以外的大多数脚本：
    - `closed_loop_streaming_infer_480p_81f.py`
    - `infinity_streaming_session.py`
    - `run_infinity.py`
- `TSformer-VO-main/` 下：
  - `__pycache__/`
  - `adapter_only_embedding/`
  - `checkpoint/`
  - `checkpoints/`
  - `datasets/`
  - `train.py`
  - `train_custom.py`
  - `train_multiseq.py`
  - `fine_tune_uavflow_sim_ddp.py`
  - `run_pretrain_latent_p2p_two_stage.sh`
  - `batch_inference.py`
  - `batch_process_tokens.py`
  - `extract_patch_tokens.py`
  - `prepare_test_data.py`
  - `pretrain_latent_embed.py`
  - `predict_custom_batch.py`
  - `predict_from_video_or_images.py`
  - `predict_poses.py`
  - `predict_reference_videos_batch.py`
  - `predict_uavflow_sim.py`
  - `prepare_reference_videos_as_dataset.py`
  - `plot_results.py`
  - `tsformer-vo.jpg`

## 备注

- 这次整理已经把原先依赖外部 `Actiondecoder/TSformer-VO-main` 的 stage2 代码收回到了当前目录的 `TSformer-VO-main/` 中。
- 当前默认路径全部改成相对路径，适合后续迁移到 GitHub。
- 权重文件仍然建议通过环境变量或本地目录挂载，不要直接提交到仓库。

