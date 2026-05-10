#!/usr/bin/env bash

offline_grpo_csv_count() {
  local csv="${1:-}"
  if [ -z "${csv}" ]; then
    printf '0\n'
    return 0
  fi
  local -a items=()
  IFS=',' read -r -a items <<< "${csv}"
  printf '%s\n' "${#items[@]}"
}

offline_grpo_make_gpu_id_csv() {
  local count="${1:-0}"
  local ids=""
  local idx
  for ((idx=0; idx<count; idx++)); do
    if [ -n "${ids}" ]; then
      ids+="," 
    fi
    ids+="${idx}"
  done
  printf '%s\n' "${ids}"
}

offline_grpo_has_glob_match() {
  local pattern="${1:-}"
  if [ -z "${pattern}" ]; then
    return 1
  fi
  compgen -G "${pattern}" > /dev/null 2>&1
}

offline_grpo_wait_for_path() {
  local target_path="${1:?missing target_path}"
  local timeout_sec="${2:-600}"
  local label="${3:-path}"
  local fail_glob="${4:-}"
  local waited=0
  while [ "${waited}" -lt "${timeout_sec}" ]; do
    if [ -e "${target_path}" ]; then
      return 0
    fi
    if offline_grpo_has_glob_match "${fail_glob}"; then
      echo "[offline_grpo][fatal] detected peer failure while waiting for ${label}: ${fail_glob}"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[offline_grpo][fatal] timed out after ${timeout_sec}s waiting for ${label}: ${target_path}"
  return 1
}

offline_grpo_wait_for_markers() {
  local root_dir="${1:?missing root_dir}"
  local prefix="${2:?missing prefix}"
  local expected_count="${3:?missing expected_count}"
  local timeout_sec="${4:-600}"
  local fail_glob="${5:-}"
  local waited=0
  local marker_count=0
  while [ "${waited}" -lt "${timeout_sec}" ]; do
    if offline_grpo_has_glob_match "${fail_glob}"; then
      echo "[offline_grpo][fatal] detected peer failure while waiting for markers: ${fail_glob}"
      return 1
    fi
    marker_count=$(find "${root_dir}" -maxdepth 1 -type f -name "${prefix}*" | wc -l)
    if [ "${marker_count}" -ge "${expected_count}" ]; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "[offline_grpo][fatal] timed out after ${timeout_sec}s waiting for ${expected_count} markers under ${root_dir}/${prefix}* (got=${marker_count})"
  return 1
}

offline_grpo_recommend_stageb_sp_size() {
  local total_gpus="${1:-${OFFLINE_GRPO_TOTAL_GPUS:-0}}"
  if [ -z "${total_gpus}" ] || [ "${total_gpus}" -le 0 ]; then
    total_gpus=1
  fi
  if [ "${total_gpus}" -ge 8 ]; then
    printf '8\n'
  else
    printf '%s\n' "${total_gpus}"
  fi
}

offline_grpo_recommend_stageb_ac() {
  local base_ac="${1:-4}"
  local total_gpus="${2:-${OFFLINE_GRPO_TOTAL_GPUS:-0}}"
  local sp_size="${3:-8}"
  local dp_groups
  local ac
  if [ -z "${sp_size}" ] || [ "${sp_size}" -le 0 ]; then
    printf '%s\n' "${base_ac}"
    return 0
  fi
  if [ -z "${total_gpus}" ] || [ "${total_gpus}" -le 0 ]; then
    total_gpus="${sp_size}"
  fi
  dp_groups=$((total_gpus / sp_size))
  if [ "${dp_groups}" -lt 1 ]; then
    dp_groups=1
  fi
  ac=$((base_ac / dp_groups))
  if [ "${ac}" -lt 1 ]; then
    ac=1
  fi
  printf '%s\n' "${ac}"
}

offline_grpo_resolve_cluster_env() {
  if [ "${OFFLINE_GRPO_CLUSTER_ENV_READY:-0}" = "1" ]; then
    return 0
  fi

  local default_nproc="8"
  local visible_count="0"
  if [ -n "${MLP_WORKER_GPU:-}" ]; then
    default_nproc="${MLP_WORKER_GPU}"
  elif [ -n "${NPROC_PER_NODE:-}" ]; then
    default_nproc="${NPROC_PER_NODE}"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    visible_count="$(offline_grpo_csv_count "${CUDA_VISIBLE_DEVICES}")"
    if [ "${visible_count}" -gt 0 ]; then
      default_nproc="${visible_count}"
    fi
  fi

  export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
  export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
  export NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
  export NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-${default_nproc}}"

  if [ -z "${OFFLINE_GRPO_LOCAL_GPU_IDS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      export OFFLINE_GRPO_LOCAL_GPU_IDS="${CUDA_VISIBLE_DEVICES}"
    else
      export OFFLINE_GRPO_LOCAL_GPU_IDS="$(offline_grpo_make_gpu_id_csv "${NPROC_PER_NODE}")"
    fi
  fi

  export OFFLINE_GRPO_LOCAL_GPU_COUNT="$(offline_grpo_csv_count "${OFFLINE_GRPO_LOCAL_GPU_IDS}")"
  if [ "${OFFLINE_GRPO_LOCAL_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]; then
    echo "[offline_grpo][fatal] local visible GPU count (${OFFLINE_GRPO_LOCAL_GPU_COUNT}) is smaller than NPROC_PER_NODE (${NPROC_PER_NODE})"
    return 1
  fi

  export OFFLINE_GRPO_TOTAL_GPUS="$((NNODES * NPROC_PER_NODE))"
  export OFFLINE_GRPO_NODE_TAG="node_${NODE_RANK}"
  export OFFLINE_GRPO_CLUSTER_ENV_READY=1

  echo "[offline_grpo] cluster_env master=${MASTER_ADDR}:${MASTER_PORT} nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE} total_gpus=${OFFLINE_GRPO_TOTAL_GPUS} local_gpu_ids=${OFFLINE_GRPO_LOCAL_GPU_IDS}"
  case "${OFFLINE_GRPO_TOTAL_GPUS}" in
    8|16|24)
      ;;
    *)
      echo "[offline_grpo][warn] auto topology defaults are tuned around total_gpus in {8,16,24}; current total_gpus=${OFFLINE_GRPO_TOTAL_GPUS}"
      ;;
  esac
}