#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v2.yaml}}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
ARGS=(router-train --config "${CONFIG_PATH}")

[[ -n "${TASK_MANIFEST:-}" ]] && ARGS+=(--task-manifest "${TASK_MANIFEST}")
[[ -n "${ROUTER_OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "${ROUTER_OUTPUT_DIR}")
[[ -n "${NUM_TASKS:-}" ]] && ARGS+=(--count "${NUM_TASKS}")
[[ -n "${START_INDEX:-}" ]] && ARGS+=(--start-index "${START_INDEX}")
[[ -n "${EPOCHS:-}" ]] && ARGS+=(--epochs "${EPOCHS}")
[[ -n "${DEVICE:-}" ]] && ARGS+=(--device "${DEVICE}")
ROUTER_BATCH_SIZE="${ROUTER_BATCH_SIZE:-${BATCH_SIZE:-}}"
[[ -n "${ROUTER_BATCH_SIZE}" ]] && ARGS+=(--batch-size "${ROUTER_BATCH_SIZE}")
[[ -n "${NUM_WORKERS:-}" ]] && ARGS+=(--num-workers "${NUM_WORKERS}")
[[ -n "${PREFETCH_FACTOR:-}" ]] && ARGS+=(--prefetch-factor "${PREFETCH_FACTOR}")
[[ -n "${MIXED_PRECISION:-}" ]] && ARGS+=(--mixed-precision "${MIXED_PRECISION}")
[[ -n "${LOG_LEVEL:-}" ]] && ARGS+=(--log-level "${LOG_LEVEL}")

case "${OVERWRITE:-}" in
  1|true|TRUE|yes|YES) ARGS+=(--overwrite) ;;
  0|false|FALSE|no|NO) ARGS+=(--no-overwrite) ;;
  "") ;;
  *) echo "OVERWRITE must be true/false" >&2; exit 2 ;;
esac

[[ "${VALIDATE_CONFIG_ONLY:-0}" == "1" ]] && ARGS+=(--validate-config-only)
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m rdb_prior.cli "${ARGS[@]}" "$@"
