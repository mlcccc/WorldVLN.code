#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8001}"
export ACTION_HEAD_MODE="${ACTION_HEAD_MODE:-tsformer_latent}"

export INFINITY_SERVER_CONFIG="${INFINITY_SERVER_CONFIG:-${SCRIPT_DIR}/config.json}"
export INFINITY_REPO_ROOT="${INFINITY_REPO_ROOT:-${REPO_ROOT}/Worldmodel/runtime}"
export INFINITY_RESET_SESSION_ON_ONE_FRAME="${INFINITY_RESET_SESSION_ON_ONE_FRAME:-1}"
export INFINITY_REQUIRE_TGT_HW="${INFINITY_REQUIRE_TGT_HW:-640,640}"
export INFINITY_LATENT_CACHE_ROOT="${INFINITY_LATENT_CACHE_ROOT:-${SCRIPT_DIR}/cache}"
export STAGE2_LATENT2ACTION_CKPT="${STAGE2_LATENT2ACTION_CKPT:-${SCRIPT_DIR}/checkpoints/stage2_latent2action_combined.pt}"

mkdir -p "${INFINITY_LATENT_CACHE_ROOT}"

exec "${PYTHON_BIN}" -m uvicorn \
  server:app \
  --host "${HOST}" \
  --port "${PORT}"

