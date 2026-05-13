#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"

export INFINITY_SERVER_CONFIG="${INFINITY_SERVER_CONFIG:-${SCRIPT_DIR}/config.json}"
export INFINITY_REPO_ROOT="${INFINITY_REPO_ROOT:-${REPO_ROOT}/Worldmodel/runtime}"
export INFINITY_LATENT_CACHE_ROOT="${INFINITY_LATENT_CACHE_ROOT:-${SCRIPT_DIR}/outputs/latent_cache}"
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${SCRIPT_DIR}/models/infinity}"
export ACTION_HEAD_MODE="${ACTION_HEAD_MODE:-actionhead_ref_vit}"
export ACTIONHEAD_REPO_ROOT="${ACTIONHEAD_REPO_ROOT:-${REPO_ROOT}/Worldmodel/action_decoder/actionhead_runtime}"

mkdir -p "${INFINITY_LATENT_CACHE_ROOT}"

if [[ ! -f "${INFINITY_SERVER_CONFIG}" ]]; then
  echo "Config file not found: ${INFINITY_SERVER_CONFIG}" >&2
  exit 1
fi

if [[ ! -d "${INFINITY_REPO_ROOT}" ]]; then
  echo "InfinityStar repo not found: ${INFINITY_REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  echo "CHECKPOINTS_DIR not found: ${CHECKPOINTS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${T5_PATH:-${CHECKPOINTS_DIR}/text_encoder/flan-t5-xl-official}" ]]; then
  echo "Missing T5 assets for local inference: ${T5_PATH:-${CHECKPOINTS_DIR}/text_encoder/flan-t5-xl-official}" >&2
  exit 1
fi

if [[ ! -f "${VAE_PATH:-${CHECKPOINTS_DIR}/infinitystar_videovae.pth}" ]]; then
  echo "Missing VAE checkpoint for local inference: ${VAE_PATH:-${CHECKPOINTS_DIR}/infinitystar_videovae.pth}" >&2
  exit 1
fi

if [[ ! -d "${ACTIONHEAD_REPO_ROOT}" ]]; then
  echo "TSformer repo not found: ${ACTIONHEAD_REPO_ROOT}" >&2
  exit 1
fi

if [[ -z "${INFINITY_CKPT:-}" ]]; then
  echo "INFINITY_CKPT is required. Export it to a local InfinityStar checkpoint." >&2
  exit 1
fi

case "${ACTION_HEAD_MODE}" in
  actionhead_ref_vit|actionhead_ref|actionhead_vit|ref_vit|actionhead)
    if [[ -z "${ACTIONHEAD_CKPT:-}" || -z "${ACTIONHEAD_RUN_CONFIG:-}" ]]; then
      echo "ACTION_HEAD_MODE=${ACTION_HEAD_MODE} requires ACTIONHEAD_CKPT and ACTIONHEAD_RUN_CONFIG." >&2
      exit 1
    fi
    ;;
esac

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m uvicorn infinity_tsformer_api_server:app --host "${HOST}" --port "${PORT}"
