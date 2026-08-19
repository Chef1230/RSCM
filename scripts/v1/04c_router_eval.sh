#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASK_MANIFEST_PATH="${BENCHMARK_TASK_MANIFEST:-${TASK_MANIFEST:-}}"
CHECKPOINT_PATH="${ROUTER_CHECKPOINT:-}"
EVAL_OUTPUT_PATH="${ROUTER_EVAL_OUTPUT:-${PROJECT_ROOT}/outputs/benchmark/router_eval}"

if [[ -z "${TASK_MANIFEST_PATH}" ]]; then
  echo "Set BENCHMARK_TASK_MANIFEST or TASK_MANIFEST to task/manifest.json." >&2
  exit 2
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set ROUTER_CHECKPOINT to router/checkpoints/best.pt or last.pt." >&2
  exit 2
fi
if [[ ! -f "${TASK_MANIFEST_PATH}" ]]; then
  echo "Benchmark task manifest does not exist: ${TASK_MANIFEST_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Router checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
ARGS=(
  router-eval
  --task-manifest "${TASK_MANIFEST_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --output-dir "${EVAL_OUTPUT_PATH}"
)
[[ -n "${NUM_TASKS:-}" ]] && ARGS+=(--count "${NUM_TASKS}")
[[ -n "${START_INDEX:-}" ]] && ARGS+=(--start-index "${START_INDEX}")
[[ -n "${DEVICE:-}" ]] && ARGS+=(--device "${DEVICE}")
[[ -n "${MIXED_PRECISION:-}" ]] && ARGS+=(--mixed-precision "${MIXED_PRECISION}")
[[ -n "${ARTIFACT_CACHE_SIZE:-}" ]] && ARGS+=(--artifact-cache-size "${ARTIFACT_CACHE_SIZE}")
[[ -n "${LOG_LEVEL:-}" ]] && ARGS+=(--log-level "${LOG_LEVEL}")
[[ -n "${LOG_FILE:-}" ]] && ARGS+=(--log-file "${LOG_FILE}")

case "${OVERWRITE:-}" in
  1|true|TRUE|yes|YES) ARGS+=(--overwrite) ;;
  0|false|FALSE|no|NO) ARGS+=(--no-overwrite) ;;
  "") ;;
  *) echo "OVERWRITE must be true/false" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m rdb_prior.cli "${ARGS[@]}" --progress "$@"
