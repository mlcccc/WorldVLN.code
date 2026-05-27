#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== WorldVLN Training Setup ==="
echo "Repository root: ${REPO_ROOT}"

# 1. Check Python environment
echo ""
echo "[1/5] Checking Python environment..."
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
echo "Python: ${PYTHON_BIN}"
$PYTHON_BIN --version

# 2. Check CUDA
echo ""
echo "[2/5] Checking CUDA..."
$PYTHON_BIN -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 3. Verify checkpoints
echo ""
echo "[3/5] Verifying checkpoints..."
CHECKPOINTS_DIR="${REPO_ROOT}/train/checkpoints"

# Backbone
BACKBONE_DIR="${CHECKPOINTS_DIR}/infinitystar_8b_480p_weights"
if [ -d "${BACKBONE_DIR}" ]; then
    echo "  Backbone: OK (${BACKBONE_DIR})"
else
    echo "  Backbone: MISSING (${BACKBONE_DIR})"
    echo "  Expected: symlink to worldvln/WorldVLN_backbone/backbone"
fi

# VAE
VAE_PATH="${CHECKPOINTS_DIR}/infinitystar_videovae.pth"
if [ -f "${VAE_PATH}" ]; then
    echo "  VAE: OK (${VAE_PATH})"
else
    echo "  VAE: MISSING (${VAE_PATH})"
    echo "  Convert from worldvln/WorldVLN_backbone/vae/model.safetensors"
fi

# T5
T5_PATH="${CHECKPOINTS_DIR}/text_encoder/flan-t5-xl-official"
if [ -d "${T5_PATH}" ] && [ -f "${T5_PATH}/config.json" ] || [ -f "${T5_PATH}/google/flan-t5-xl/config.json" ]; then
    echo "  T5: OK (${T5_PATH})"
else
    echo "  T5: MISSING or incomplete (${T5_PATH})"
    echo "  Download from HuggingFace or ModelScope"
fi

# 4. Verify data
echo ""
echo "[4/5] Verifying data..."
DATA_DIR="${REPO_ROOT}/train/data/uavflow_jsonl"
if [ -d "${DATA_DIR}" ]; then
    JSONL_COUNT=$(find "${DATA_DIR}" -name "*.jsonl" | wc -l)
    VIDEO_COUNT=$(find "${DATA_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
    echo "  JSONL files: ${JSONL_COUNT}"
    echo "  Video files: ${VIDEO_COUNT}"
else
    echo "  Data directory missing: ${DATA_DIR}"
    echo "  Run: python3 scripts/convert_parquet_to_jsonl.py"
fi

# 5. Setup PYTHONPATH
echo ""
echo "[5/5] Setting up PYTHONPATH..."
export PYTHONPATH="${REPO_ROOT}/Worldmodel/runtime${PYTHONPATH:+:${PYTHONPATH}}"
echo "  PYTHONPATH includes: ${REPO_ROOT}/Worldmodel/runtime"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training:"
echo "  cd ${REPO_ROOT}/train"
echo "  bash scripts/train_from_base.sh"
