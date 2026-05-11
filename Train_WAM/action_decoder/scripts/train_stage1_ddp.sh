#!/usr/bin/env bash
set -euo pipefail

# Make this script runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
echo "[cwd] $(pwd)"

# Avoid writing .pyc on quota-limited filesystems; flush logs immediately.
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Stage-1 (DDP): distill an Adapter mapping InfinityStar VAE up_block_3 features
# to TSformer PatchEmbed tokens.
# Required paths are provided through environment variables below.

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
  echo "[error] Failed to activate conda env prefix: ${prefix}" >&2; exit 1
}

if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  activate_conda_prefix "${CONDA_ENV_PREFIX}"
  echo "[env] CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX}"
fi
echo "[env] python=$(command -v python)"
python -V

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# DDP rendezvous (avoid port conflicts on shared machines)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(
python - <<'PY'
import socket
s=socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
echo "[ddp] master_addr=${MASTER_ADDR} master_port=${MASTER_PORT}"

ITEMS_KEY="${ITEMS_KEY:-ALL}"

TSFORMER_CKPT="${TSFORMER_CKPT:-}"
INF_VAE_PATH="${INF_VAE_PATH:-}"
MANIFEST_JSON="${MANIFEST_JSON:-}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/stage1_adapter}"
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
require_env TSFORMER_CKPT
require_env INF_VAE_PATH

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
STDOUT_STDERR_LOG="${STDOUT_STDERR_LOG:-${LOG_DIR}/stdout_stderr.log}"
exec > >(tee -a "${STDOUT_STDERR_LOG}") 2>&1
echo "[log] stdout/stderr -> ${STDOUT_STDERR_LOG}"

${TORCHRUN_BIN} --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/tools/train_stage1_infinitystar_up3_adapter_distill_ddp.py" \
  --out_dir "${OUT_DIR}" \
  --tqdm --log_file "train.log" --log_dir "${LOG_DIR}" \
  --manifest_json "${MANIFEST_JSON}" \
  --items_key "${ITEMS_KEY}" \
  --tsformer_ckpt "${TSFORMER_CKPT}" \
  --infinitystar_vae_path "${INF_VAE_PATH}" \
  ${EXTRA_ARGS}

