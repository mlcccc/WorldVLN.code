#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash run_pretrain_latent_p2p_two_stage.sh /path/to/data /path/to/save [jsonl_path]

DATA_DIR="${1:-/path/to/uavflowoutput}"
SAVE_DIR="${2:-./adapter_p2p}"
JSONL_PATH="${3:-}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/batchcom/.conda/envs/tsformer}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-80}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# New adapter/training defaults
LR="${LR:-3e-4}"               # patch_embed + head
BACKBONE_LR="${BACKBONE_LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SCHEDULER="${SCHEDULER:-none}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.05}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
NUM_LAYERS="${NUM_LAYERS:-2}"
TARGET_STANDARDIZE="${TARGET_STANDARDIZE:-1}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
ROT_WEIGHT="${ROT_WEIGHT:-1.0}"
Z_WEIGHT_START="${Z_WEIGHT_START:-1.0}"
Z_WEIGHT_MAX="${Z_WEIGHT_MAX:-1.3}"
Z_WARMUP_EPOCHS="${Z_WARMUP_EPOCHS:-15}"
ROT_WEIGHT_START="${ROT_WEIGHT_START:-1.0}"
ROT_WEIGHT_MAX="${ROT_WEIGHT_MAX:-1.35}"
ROT_WARMUP_EPOCHS="${ROT_WARMUP_EPOCHS:-15}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_TRAINING_STATE="${RESUME_TRAINING_STATE:-0}"
SHOW_DATA_PROGRESS="${SHOW_DATA_PROGRESS:-1}"
SAFE_CUDA_KERNELS="${SAFE_CUDA_KERNELS:-1}"
FORCE_SINGLE_GPU="${FORCE_SINGLE_GPU:-0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
SKIP_NONFINITE_LOSS="${SKIP_NONFINITE_LOSS:-1}"

mkdir -p "${SAVE_DIR}"
LOG_FILE="${SAVE_DIR}/log.txt"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: Python not found or not executable: ${PYTHON_BIN}"
  echo "Hint: set PYTHON_BIN=/path/to/python or CONDA_ENV_PATH=/path/to/conda_env"
  exit 1
fi

# Default to 4-GPU training unless overridden.
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "[Launcher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Environment preflight: verify torch/cuda compatibility before long training.
"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
except Exception as e:
    print(f"[Preflight] Failed to import torch: {e}")
    sys.exit(1)

print(f"[Preflight] torch={torch.__version__}, cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    print("[Preflight] CUDA is not available. This training script expects CUDA.")
    sys.exit(1)

dev_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print(f"[Preflight] device={dev_name}, capability=sm_{capability[0]}{capability[1]}")
print(f"[Preflight] supported_arch={arch_list}")

# A100 is sm_80. If the running GPU is Ampere or newer, ensure arch is supported.
required_arch = f"sm_{capability[0]}{capability[1]}"
if required_arch not in arch_list:
    print(
        f"[Preflight] Incompatible PyTorch build: device requires {required_arch}, "
        f"but this torch supports {arch_list}."
    )
    sys.exit(1)
PY

CMD=(
  "${PYTHON_BIN}" pretrain_latent_p2p.py
  --data_dir "${DATA_DIR}"
  --save_dir "${SAVE_DIR}"
  --lr "${LR}"
  --backbone_lr "${BACKBONE_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --scheduler "${SCHEDULER}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --min_lr_ratio "${MIN_LR_RATIO}"
  --pos_weight "${POS_WEIGHT}"
  --rot_weight "${ROT_WEIGHT}"
  --z_weight_start "${Z_WEIGHT_START}"
  --z_weight_max "${Z_WEIGHT_MAX}"
  --z_warmup_epochs "${Z_WARMUP_EPOCHS}"
  --rot_warmup_epochs "${ROT_WARMUP_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --save_every_epochs "${SAVE_EVERY_EPOCHS}"
  --freeze_backbone_epochs "${FREEZE_EPOCHS}"
  --hidden_dim "${HIDDEN_DIM}"
  --num_layers "${NUM_LAYERS}"
  --window_size "${WINDOW_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
)

if [[ -n "${JSONL_PATH}" ]]; then
  CMD+=(--jsonl_path "${JSONL_PATH}")
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

if [[ -n "${ROT_WEIGHT_START}" ]]; then
  CMD+=(--rot_weight_start "${ROT_WEIGHT_START}")
fi

if [[ -n "${ROT_WEIGHT_MAX}" ]]; then
  CMD+=(--rot_weight_max "${ROT_WEIGHT_MAX}")
fi

if [[ "${RESUME_TRAINING_STATE}" == "1" ]]; then
  CMD+=(--resume_training_state)
fi

if [[ "${SHOW_DATA_PROGRESS}" == "1" ]]; then
  CMD+=(--show_data_progress)
else
  CMD+=(--no_show_data_progress)
fi

if [[ "${SAFE_CUDA_KERNELS}" == "1" ]]; then
  CMD+=(--safe_cuda_kernels)
fi

if [[ "${FORCE_SINGLE_GPU}" == "1" ]]; then
  CMD+=(--force_single_gpu)
fi

if [[ "${SKIP_NONFINITE_LOSS}" == "1" ]]; then
  CMD+=(--skip_nonfinite_loss)
else
  CMD+=(--no_skip_nonfinite_loss)
fi

if [[ "${TARGET_STANDARDIZE}" == "1" ]]; then
  CMD+=(--target_standardize)
else
  CMD+=(--no_target_standardize)
fi

echo "Running: ${CMD[*]}"
{
  echo "========== Training Launch =========="
  echo "launch_time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "script: $0"
  echo "data_dir: ${DATA_DIR}"
  echo "save_dir: ${SAVE_DIR}"
  echo "jsonl_path: ${JSONL_PATH:-<none>}"
  echo "python_bin: ${PYTHON_BIN}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
  echo "env_overrides:"
  echo "  EPOCHS=${EPOCHS}"
  echo "  FREEZE_EPOCHS=${FREEZE_EPOCHS}"
  echo "  ROT_WEIGHT=${ROT_WEIGHT}"
  echo "  Z_WEIGHT_START=${Z_WEIGHT_START}"
  echo "  Z_WEIGHT_MAX=${Z_WEIGHT_MAX}"
  echo "  Z_WARMUP_EPOCHS=${Z_WARMUP_EPOCHS}"
  echo "  ROT_WEIGHT_START=${ROT_WEIGHT_START:-<auto-from-ROT_WEIGHT>}"
  echo "  ROT_WEIGHT_MAX=${ROT_WEIGHT_MAX:-<auto-from-ROT_WEIGHT>}"
  echo "  ROT_WARMUP_EPOCHS=${ROT_WARMUP_EPOCHS}"
  echo "  POS_WEIGHT=${POS_WEIGHT}"
  echo "  BATCH_SIZE=${BATCH_SIZE}"
  echo "  NUM_WORKERS=${NUM_WORKERS}"
  echo "  LR=${LR}"
  echo "  BACKBONE_LR=${BACKBONE_LR}"
  echo "  WEIGHT_DECAY=${WEIGHT_DECAY}"
  echo "  SCHEDULER=${SCHEDULER}"
  echo "  WARMUP_EPOCHS=${WARMUP_EPOCHS}"
  echo "  MIN_LR_RATIO=${MIN_LR_RATIO}"
  echo "  WINDOW_SIZE=${WINDOW_SIZE}"
  echo "  HIDDEN_DIM=${HIDDEN_DIM}"
  echo "  NUM_LAYERS=${NUM_LAYERS}"
  echo "  TARGET_STANDARDIZE=${TARGET_STANDARDIZE}"
  echo "  SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS}"
  echo "  SAFE_CUDA_KERNELS=${SAFE_CUDA_KERNELS}"
  echo "  FORCE_SINGLE_GPU=${FORCE_SINGLE_GPU}"
  echo "  GPU_IDS=${GPU_IDS}"
  echo "  GRAD_CLIP_NORM=${GRAD_CLIP_NORM}"
  echo "  SKIP_NONFINITE_LOSS=${SKIP_NONFINITE_LOSS}"
  echo "  RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-<none>}"
  echo "  RESUME_TRAINING_STATE=${RESUME_TRAINING_STATE}"
  echo "resolved_command: ${CMD[*]}"
  echo "log_file: ${LOG_FILE}"
  echo "====================================="
  "${CMD[@]}"
} 2>&1 | tee -a "${LOG_FILE}"
