#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec bash "${ROOT_DIR}/InfinityStar-main/scripts/offline_grpo_stageB_partialfreeze_train.sh" "$@"
