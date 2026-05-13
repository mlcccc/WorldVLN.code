#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
exec bash "${REPO_ROOT}/Worldmodel/runtime/scripts/GRPO_stageA_collect.sh" "$@"
