# WorldVLN 推理、评测与可视化指南

## 概述

本文档介绍 WorldVLN 的完整评测流程：数据提取、推理、格式转换、官方评测、可视化。

**流程步骤：**
1. 从 UAV-Flow parquet 文件中提取轨迹
2. 通过自回归推理服务器运行推理
3. 将结果转换为官方评测格式
4. 运行官方端点评测
5. 生成可视化

## 1. 从 UAV-Flow 提取数据

### 1.1 提取轨迹

```bash
python3 scripts/extract_samples.py \
    --parquet "uav-flow/train-00000-of-00054.parquet" \
    --out_dir "eval_samples_full" \
    --num_trajectories 100 \
    --min_frames 40
```

**参数说明：**
- `--parquet`：UAV-Flow 数据集中的 parquet 文件路径
- `--out_dir`：提取后的轨迹输出目录
- `--num_trajectories`：要提取的轨迹数量
- `--min_frames`：每条轨迹的最少帧数（不足则跳过）

### 1.2 提取后的数据结构

```
eval_samples_full/
└── {trajectory_id}/            # 例如 2025-04-03_14-55-09
    ├── images/                 # 帧图像 (frame_000000.jpg, ...)
    ├── meta.json               # 元数据（指令、轨迹ID等）
    ├── raw_logs.json           # 原始 UTM 坐标（全局，单位：米）
    └── preprocessed_logs.json  # 起点局部坐标，起点为原点（UAV-Flow 为米，UAV-Flow-Sim 为厘米）
```

**坐标格式：**
- `raw_logs.json`：UTM 全局坐标，单位米，格式 `[x, y, z, roll, yaw, pitch]`
- `preprocessed_logs.json`：起点局部坐标，格式 `[x, y, z, roll, yaw, pitch]`，起点为原点；UAV-Flow 样本通常为米，UAV-Flow-Sim 样本为厘米

## 2. 推理

### 2.1 启动推理服务器

```bash
# 设置模型路径
export INFINITY_CKPT=./checkpoints/infinity/global_step_xxx.pth
export STAGE2_CKPT=./checkpoints/stage2_latent2action/checkpoint_last.pth

# 启动服务器（默认端口 8001）
python3 infer/server.py \
    --infinity_ckpt "$INFINITY_CKPT" \
    --stage2_ckpt "$STAGE2_CKPT" \
    --port 8001
```

服务器在 `http://127.0.0.1:8001` 提供 REST API，使用基于 `session_id` 的自回归闭环协议进行多段推理。

### 2.2 对数据集运行推理

```bash
python3 infer/client.py \
    --mode dataset \
    --dataset_root eval_samples_full \
    --server_url http://127.0.0.1:8001 \
    --out_dir eval_results_full/client_run_test_run_002 \
    --num_frames 49 \
    --step 16 \
    --prefix_mode 1 \
    --allow_future_last_seg 1
```

**参数说明：**
- `--mode dataset`：对数据集中的轨迹批量推理
- `--dataset_root`：轨迹数据目录（步骤1的输出）
- `--server_url`：推理服务器地址
- `--out_dir`：结果输出目录
- `--num_frames`：每条轨迹的帧数（默认49，最大50）
- `--step`：分段步长（默认16）
- `--prefix_mode`：文件名前缀模式（0或1）
- `--allow_future_last_seg`：最后一段是否允许使用未来帧（0或1）

### 2.3 对单条轨迹推理

```bash
python3 infer/client.py \
    --mode single \
    --images_dir /path/to/trajectory/images \
    --server_url http://127.0.0.1:8001 \
    --out_dir /path/to/output \
    --num_frames 49
```

### 2.4 推理输出结构

```
eval_results_full/
└── client_run_{run_id}/        # run_id 为自动生成的时间戳（如 2026-05-27_23-40-11）
    └── {trajectory_id}/
        ├── {traj_id}__{run_id}_seg00_actions.json  # 第0段动作
        ├── {traj_id}__{run_id}_seg00_poses.json    # 第0段位姿
        ├── {traj_id}__{run_id}_seg01_actions.json  # 第1段动作
        ├── {traj_id}__{run_id}_seg01_poses.json    # 第1段位姿
        ├── {traj_id}__{run_id}_summary.json        # 会话摘要
        └── ...
```

**输出位姿格式：**
- 单位：平移 **厘米**，角度 **度**
- 顺序：`[x, y, z, roll, yaw, pitch]`
- 坐标：客户端积分出的 raw/world 绝对坐标；转换脚本会转为起点局部坐标

**输出动作格式：**
- 增量动作，单位 cm/deg
- 顺序：`[dx, dy, dz, droll, dyaw, dpitch]`

## 3. 格式转换

官方评测脚本要求的格式与客户端输出不同，需要进行转换。

### 3.1 转换为评测格式

```bash
python3 scripts/convert_to_eval_format.py \
    --results_root eval_results_full/client_run_test_run_002 \
    --gt_root eval_samples_full \
    --out_root eval_results_full_converted
```

**参数说明：**
- `--results_root`：推理结果目录（步骤2的输出）
- `--gt_root`：GT 数据目录（步骤1的输出）
- `--out_root`：转换后的输出目录
- `--run_id`：运行标识符（默认 test_run_002）

### 3.2 转换对照

| 属性 | 客户端输出 | 评测格式 |
|------|-----------|---------|
| 位置单位 | 厘米 | 米 |
| 角度单位 | 度 | 弧度 |
| 位姿顺序 | `[x,y,z,roll,yaw,pitch]` | `[roll,yaw,pitch,x,y,z]` |
| 坐标类型 | raw/world 绝对坐标 | 起点局部坐标 |

转换脚本的处理流程：
1. 读取各段的 `*_poses.json` 文件
2. 减去起始 raw/world 位姿，并按起始 yaw 旋转到 `preprocessed_logs.json` 使用的起点局部坐标系
3. 位置从厘米转为米，角度从度转为弧度
4. 重排序为 `[roll, yaw, pitch, x, y, z]`
5. 计算增量动作：`[dz_rad, dy_rad, dx_rad, tx_m, ty_m, tz_m]`

### 3.3 转换后的输出结构

```
eval_results_full_converted/
└── {trajectory_id}/
    ├── pred_path.json      # 预测轨迹（起点局部坐标，弧度/米）
    └── pred_actions.json   # 预测动作（弧度/米）
```

## 4. 官方评测

### 4.1 运行评测

```bash
python3 train/action_decoder/tools/eval_endpoints.py \
    --pred_root eval_results_full_converted \
    --gt_root eval_samples_full \
    --out_root eval_output_full \
    --gt_pose_file preprocessed_logs.json \
    --translation_divisor 1.0 \
    --angles_in_degrees \
    --dist_thr_m 3.0 \
    --ang_thr_deg 10.0
```

**参数说明：**
- `--pred_root`：转换后的预测结果目录（步骤3的输出）
- `--gt_root`：GT 数据目录
- `--out_root`：评测结果输出目录
- `--gt_pose_file`：GT 位姿文件名（preprocessed_logs.json）
- `--translation_divisor`：平移单位除数（1.0 表示米）
- `--angles_in_degrees`：GT 角度单位为度
- `--dist_thr_m`：距离合格阈值（米）
- `--ang_thr_deg`：角度合格阈值（度）

### 4.2 评测指标

- **端点距离 (Endpoint Distance, cm)**：预测与 GT 最终位置的欧氏距离
- **端点旋转 (Endpoint Rotation, deg)**：预测与 GT 最终朝向的旋转误差
- **偏航角误差 (Yaw Error, deg)**：偏航角绝对差值
- **合格率 (Qualified Rate)**：距离 ≤ 3.0m 且角度 ≤ 10.0deg 的轨迹占比

### 4.3 评测输出

```
eval_output_full/
├── summary.txt                 # 整体指标和逐条结果
├── endpoint_errors.json        # 详细逐条误差
└── plots/
    ├── distance_distribution.png
    ├── angle_distribution.png
    └── 3d_overlay.png
```

**summary.txt 示例：**
```
dataset=uav-flow
evaluated_routes=100
qualified(dist<=3.0m & ang<=10.0deg) = 51/100 = 51.00%
distance_cm: mean=251.89 p50=186.49 p90=481.93 max=1080.68
angle_deg:   mean=21.70 p50=3.24 p90=95.05 max=157.99
yaw_err_deg: mean=19.80 p50=0.70 p90=95.05 max=158.01
```

## 5. 可视化

### 5.1 可视化所有轨迹

```bash
python3 scripts/visualize_results.py \
    --samples_root eval_samples_sim \
    --results_root eval_results_sim_converted \
    --out_dir eval_vis_sim
```

**注意：** 可视化阶段应使用 `convert_to_eval_format.py` 生成的转换后预测目录。也就是用 `eval_samples_sim` 作为 GT 输入，用 `eval_results_sim_converted` 作为预测输入。`--run_id` 只用于兼容旧的客户端原始结果目录，正常可视化转换结果时不需要传。

```bash
# 推荐：GT 样本 + 转换后的预测
python3 scripts/visualize_results.py \
    --samples_root eval_samples_sim \
    --results_root eval_results_sim_converted \
    --out_dir eval_vis_sim
```

### 5.2 可视化单条轨迹

```bash
python3 scripts/visualize_results.py \
    --sample_dir eval_samples_sim/2025-03-14_14-46-22 \
    --results_root eval_results_sim_converted \
    --out_dir eval_vis_sim
```

### 5.3 可视化输出

每条轨迹生成 4 个文件：

| 文件 | 说明 |
|------|------|
| `{traj_id}_trajectory.png` | 2D 对比图：XY 平面（俯视）+ 高度随时间变化 |
| `{traj_id}_trajectory_3d.png` | 3D 轨迹可视化 |
| `{traj_id}_actions.png` | 动作分布直方图（dx, dy, dz, droll, dyaw, dpitch） |
| `{traj_id}_frames.png` | 关键输入帧展示 |

每张图底部都有导航指令的文本框显示。

### 5.4 可视化指标

脚本会计算并显示每条轨迹的：

- **ADE（平均位移误差）**：所有时间步预测与 GT 位置的平均欧氏距离（cm）
- **FDE（最终位移误差）**：预测与 GT 最终位置的欧氏距离（cm）
- **RMSE XYZ**：位置坐标的均方根误差（cm）

## 6. 一键运行完整流程

### 6.1 运行完整评测

```bash
python3 scripts/run_full_eval.py \
    --parquet_dir uav-flow \
    --num_trajectories 100 \
    --server_url http://127.0.0.1:8001 \
    --num_frames 49 \
    --step 16 \
    --vis_interval 10
```

自动执行全部 5 个步骤：
1. 从 parquet 提取轨迹
2. 运行推理
3. 转换为评测格式
4. 运行官方评测
5. 每隔 10 条轨迹生成可视化

### 6.2 跳过推理（使用已有结果）

```bash
python3 scripts/run_full_eval.py --skip_inference
```

## 7. 常见问题

### 7.1 Run ID 不匹配

客户端会自动生成基于时间戳的 run_id（如 `2026-05-27_23-40-11`），忽略传入的 `--run_id` 参数。结果文件命名为 `{traj_id}__{timestamp}_segXX_poses.json`。

**解决方法：** 可视化脚本使用 `--run_id ""` 启用自动检测，或手动指定结果目录中的实际 run_id。

### 7.2 坐标单位对照

| 数据来源 | 位置单位 | 角度单位 |
|---------|---------|---------|
| `raw_logs.json` | 米（UTM 全局） | 度 |
| `preprocessed_logs.json` | UAV-Flow 为米，UAV-Flow-Sim 为厘米 | 度 |
| 客户端输出 (`*_poses.json`) | **厘米**（raw/world 绝对坐标） | **度** |
| 评测格式 (`pred_path.json`) | **米**（起点局部） | **弧度** |

可视化脚本会自动检测 GT 是米还是厘米并统一到厘米显示。转换脚本会把客户端 raw/world 预测转成起点局部坐标，再处理厘米到米、度到弧度的转换。

### 7.3 可视化中预测位姿为空

通常是 run_id 不匹配导致。检查：
1. 结果目录是否存在：`ls eval_results_full/client_run_*/`
2. 文件名中的 run_id 是否匹配：`{traj_id}__{run_id}_seg*_poses.json`
3. 使用自动检测：`--run_id ""`

### 7.4 服务器无响应

```bash
# 检查服务器是否运行
curl http://127.0.0.1:8001/health

# 查看服务器日志，应输出: "UAV WorldModel inference server ready"
```

### 7.5 缺少帧图像

确保轨迹的 `images/` 目录下有按 `frame_000000.jpg`、`frame_000001.jpg` 等命名的帧图像。

## 8. UAV-Flow-Sim 离线评测

论文在 **UAV-Flow-Sim 测试集**（273 条轨迹）上报告 SR (Success Rate)。由于测试集图像需要从 UnrealZoo 模拟器实时采集，以下提供离线近似评测方案。

### 8.1 数据说明

| 数据集 | 来源 | 轨迹数 | 说明 |
|--------|------|--------|------|
| `uav-flow/` | HuggingFace UAV-Flow | ~27,000 | 真实世界轨迹（训练集） |
| `uav-flow-sim/` | HuggingFace UAV-Flow-Sim | ~10,109 | 仿真轨迹（训练集） |
| `UAV-Flow-Eval/test_jsons/` | UAV-Flow 仓库 | 273 | 官方测试集（仅含 GT 参考路径） |

**注意：** 官方测试集（273 条）不在 parquet 文件中，其 ID 与 parquet 数据零重叠。测试集轨迹来自不同日期的模拟器录制。

### 8.2 测试集分类

官方测试集按指令类型分为 10 类：

| 类别 | 数量 | 说明 |
|------|------|------|
| Turn | 15 | 转向 |
| Move | 15 | 移动 |
| Shift | 49 | 平移 |
| Rotate | 15 | 旋转 |
| Surround | 12 | 环绕 |
| Ascend/Descend | 19 | 升降 |
| Approach | 42 | 接近 |
| Retreat | 12 | 后退 |
| Pass | 40 | 经过 |
| Land | 54 | 降落 |

### 8.3 离线评测流程

从 `uav-flow-sim` parquet 中提取轨迹，用现有推理流程跑评测：

```bash
# 一键运行（完整流程）
python3 scripts/run_sim_eval.py \
    --parquet uav-flow-sim/train-00000-of-00021.parquet \
    --num_trajectories 273 \
    --min_frames 5 \
    --server_url http://127.0.0.1:8001

# 或分步执行：

# 1. 提取轨迹
python3 scripts/extract_samples.py \
    --parquet uav-flow-sim/train-00000-of-00021.parquet \
    --out_dir eval_samples_sim \
    --num_trajectories 273 \
    --min_frames 5

# 2. 推理（需先启动服务器）
python3 infer/client.py \
    --mode dataset \
    --dataset_root eval_samples_sim \
    --server_url http://127.0.0.1:8001 \
    --out_dir eval_results_sim/client_run_sim_test \
    --num_frames 49 --step 16 --prefix_mode 1 --allow_future_last_seg 1

# 3. 转换格式（注意：results_root 要指向实际的 run 目录）
python3 scripts/convert_to_eval_format.py \
    --results_root eval_results_sim/client_run_sim_test/client_run_XXXXXXXX_XXXXXX \
    --gt_root eval_samples_sim \
    --out_root eval_results_sim_converted

# 4. 评测（SR = qualified rate）
python3 train/action_decoder/tools/eval_endpoints.py \
    --pred_root eval_results_sim_converted \
    --gt_root eval_samples_sim \
    --out_root eval_output_sim \
    --gt_pose_file preprocessed_logs.json \
    --translation_divisor 100.0 \
    --angles_in_degrees \
    --dist_thr_m 3.0 \
    --ang_thr_deg 10.0
```

### 8.4 评测指标

- **SR (Success Rate)** = `eval_output_sim/summary.txt` 中的 `qualified rate`
  - 判定条件：端点距离 ≤ 3.0m 且端点角度 ≤ 10.0°
- **nDTW (normalized Dynamic Time Warping)**：论文的另一个指标，需要在线仿真环境计算

**单位注意：** sim 数据的 `preprocessed_logs.json` 中 xyz 是**厘米**（需 `--translation_divisor 100.0`），real-world 数据的 xyz 是**米**（需 `--translation_divisor 1.0`）。

### 8.5 预测帧 vs GT 帧对比视频

推理服务器会自动将预测的视觉帧保存到 `infer/cache/` 目录（`seg*_pred_full_*.mp4`）。可以用对比脚本生成左右并排视频：

```bash
# 生成对比视频
python3 scripts/visualize_pred_vs_gt.py \
    --samples_root eval_samples_sim \
    --cache_dir infer/cache \
    --run_id "" \
    --out_dir eval_vis_sim_pred

# 限制数量
python3 scripts/visualize_pred_vs_gt.py --max_traj 10
```

输出格式：
- 左侧：GT 帧（来自 parquet）
- 右侧：WorldModel 预测帧（VAE 解码）
- 顶部：标签 + 帧计数
- 底部：指令文本

### 8.6 在线仿真评测（官方方式）

严格的 SR 和 nDTW 评测需要 UnrealZoo 模拟器环境：

1. Windows 系统 + 下载 [UnrealZoo 环境](https://modelscope.cn/datasets/UnrealZoo/UnrealZoo-UE4/file/view/master/Collection_WinNoEditor_0424_25.zip)
2. 配置 `UAV-Flow-Eval/gym_unrealcv/envs/setting/Track/DowntownWest.json` 中的 `env_bin_win` 路径
3. 启动推理服务器
4. 运行 `python UAV-Flow-Eval/batch_run_act_all.py`
5. 计算指标：`python UAV-Flow-Eval/metric.py`

## 9. 文件索引

| 文件 | 用途 |
|------|------|
| `infer/server.py` | FastAPI 推理服务器 |
| `infer/client.py` | 推理客户端（单条/数据集模式） |
| `scripts/extract_samples.py` | 从 UAV-Flow parquet 提取轨迹 |
| `scripts/convert_to_eval_format.py` | 将客户端输出转换为官方评测格式 |
| `scripts/visualize_results.py` | 生成轨迹可视化 |
| `scripts/visualize_pred_vs_gt.py` | 预测帧 vs GT 帧对比视频 |
| `scripts/run_full_eval.py` | 真实数据一键评测流程 |
| `scripts/run_sim_eval.py` | UAV-Flow-Sim 离线评测流程 |
| `train/action_decoder/tools/eval_endpoints.py` | 官方端点评测脚本 |
| `train/action_decoder/tools/predict_pose.py` | 批量推理及 RMSE 指标计算 |
| `UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py` | 在线仿真评测（需模拟器） |
| `UAV-Flow/UAV-Flow-Eval/metric.py` | nDTW 指标计算 |
