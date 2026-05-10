#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Stage 2 — latent→action training (DDP)
#
# Required paths are provided through environment variables below.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
echo "[cwd] $(pwd)"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Optional conda env. If unset, use the current `python` on PATH.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-}"
CONDA_ROOT_HINT="${CONDA_ROOT_HINT:-}"

activate_conda_prefix() {
  local prefix="$1"
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$prefix"; return 0
  fi
  if [[ -f "${CONDA_ROOT_HINT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT_HINT}/etc/profile.d/conda.sh"
    conda activate "$prefix"; return 0
  fi
  if [[ -f "${CONDA_ROOT_HINT}/bin/activate" ]]; then
    source "${CONDA_ROOT_HINT}/bin/activate" "$prefix"; return 0
  fi
  if [[ -f "${prefix}/bin/activate" ]]; then
    source "${prefix}/bin/activate"; return 0
  fi
  echo "[error] Failed to activate conda env prefix: ${prefix}" >&2
  exit 1
}
if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  activate_conda_prefix "${CONDA_ENV_PREFIX}"
  echo "[env] CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX}"
fi
echo "[env] python=$(command -v python)"
python -V

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MANIFEST_JSON="${MANIFEST_JSON:-}"
TSFORMER_PRETRAINED="${TSFORMER_PRETRAINED:-}"
ADAPTER_CKPT="${ADAPTER_CKPT:-}"
INFINITYSTAR_VAE_PATH="${INFINITYSTAR_VAE_PATH:-}"
OUT_DIR="${OUT_DIR:-${PROJ_ROOT}/outputs/stage2_latent2action}"

ITEMS_KEY="${ITEMS_KEY:-ALL}"
INFINITYSTAR_VAE_TYPE="${INFINITYSTAR_VAE_TYPE:-64}"
LABEL_STATS_JSON="${LABEL_STATS_JSON:-}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

require_env() {
  local name="$1"
  local val="${!name:-}"
  if [[ -z "${val}" ]]; then
    echo "[error] Missing required env var: ${name}" >&2
    exit 2
  fi
}

require_env MANIFEST_JSON
require_env TSFORMER_PRETRAINED
require_env ADAPTER_CKPT
require_env INFINITYSTAR_VAE_PATH

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
STDOUT_STDERR_LOG="${STDOUT_STDERR_LOG:-${LOG_DIR}/stdout_stderr.log}"
exec > >(tee -a "${STDOUT_STDERR_LOG}") 2>&1
echo "[log] stdout/stderr -> ${STDOUT_STDERR_LOG}"

# ---- DDP ----
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(python - <<'PY'
import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()
PY
)}"
echo "[ddp] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

# ---- optional args ----
RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
  RESUME_ARGS+=(--resume "${RESUME}")
fi

LABEL_STATS_ARGS=()
if [[ -n "${LABEL_STATS_JSON}" ]]; then
  LABEL_STATS_ARGS+=(--label_stats_json "${LABEL_STATS_JSON}")
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${PROJ_ROOT}/tools/train_stage2_latent2action_ddp.py" \
  --manifest_json "${MANIFEST_JSON}" \
  --items_key "${ITEMS_KEY}" \
  --tsformer_pretrained "${TSFORMER_PRETRAINED}" \
  --adapter_ckpt "${ADAPTER_CKPT}" \
  "${LABEL_STATS_ARGS[@]}" \
  --infinitystar_vae_path "${INFINITYSTAR_VAE_PATH}" \
  --infinitystar_vae_type "${INFINITYSTAR_VAE_TYPE}" \
  --out_dir "${OUT_DIR}" \
  --tqdm \
  "${RESUME_ARGS[@]}" \
  --log_file "${LOG_DIR}/train.log" \
  ${EXTRA_ARGS}
