#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v1.yaml}}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  task
  --config "${CONFIG_PATH}"
)

if [[ -n "${INSTANCE_MANIFEST:-}" ]]; then
  ARGS+=(--instance-manifest "${INSTANCE_MANIFEST}")
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--instance-manifest "${OUTPUT_DIR}/instance/manifest.json")
fi
if [[ -n "${TASK_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${TASK_OUTPUT_DIR}")
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}/task")
fi
if [[ -n "${NUM_DATABASES:-}" ]]; then
  ARGS+=(--database-count "${NUM_DATABASES}")
fi
if [[ -n "${TASKS_PER_DATABASE:-}" ]]; then
  ARGS+=(--tasks-per-database "${TASKS_PER_DATABASE}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${SHARD_ID:-}" ]]; then
  ARGS+=(--shard-id "${SHARD_ID}")
fi
if [[ -n "${NUM_SHARDS:-}" ]]; then
  ARGS+=(--num-shards "${NUM_SHARDS}")
fi
if [[ -n "${TASK_JOBS:-${JOBS:-}}" ]]; then
  ARGS+=(--jobs "${TASK_JOBS:-${JOBS}}")
fi
if [[ -n "${PROGRESS_EVERY:-}" ]]; then
  ARGS+=(--progress-every "${PROGRESS_EVERY}")
fi
if [[ -n "${LOG_LEVEL:-}" ]]; then
  ARGS+=(--log-level "${LOG_LEVEL}")
fi
if [[ -n "${LOG_FILE:-}" ]]; then
  ARGS+=(--log-file "${LOG_FILE}")
fi
if [[ -n "${PROGRESS_WIDTH:-}" ]]; then
  ARGS+=(--progress-width "${PROGRESS_WIDTH}")
fi

case "${PROGRESS_BAR:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--progress)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-progress)
    ;;
  "")
    ;;
  *)
    echo "PROGRESS_BAR must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

case "${OVERWRITE:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--overwrite)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-overwrite)
    ;;
  "")
    ;;
  *)
    echo "OVERWRITE must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

if [[ "${VALIDATE_CONFIG_ONLY:-0}" == "1" ]]; then
  ARGS+=(--validate-config-only)
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m rdb_prior.cli "${ARGS[@]}" "$@"
