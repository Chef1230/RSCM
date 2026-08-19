#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/local/local.yaml}}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  rdbpfn-export
  --config "${CONFIG_PATH}"
)

if [[ -n "${TASK_MANIFEST:-}" ]]; then
  ARGS+=(--task-manifest "${TASK_MANIFEST}")
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--task-manifest "${OUTPUT_DIR}/task/manifest.json")
fi
if [[ -n "${RDBPFN_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${RDBPFN_OUTPUT_DIR}")
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}/rdbpfn")
fi
if [[ -n "${NUM_EXPORTS:-}" ]]; then
  ARGS+=(--count "${NUM_EXPORTS}")
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
if [[ -n "${VALIDATION_FRACTION:-}" ]]; then
  ARGS+=(--validation-fraction "${VALIDATION_FRACTION}")
fi
if [[ -n "${MIN_VALIDATION_ROWS:-}" ]]; then
  ARGS+=(--min-validation-rows "${MIN_VALIDATION_ROWS}")
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
if [[ -n "${H5_OUTPUT:-}" ]]; then
  ARGS+=(--h5-output "${H5_OUTPUT}")
fi
if [[ -n "${RDBPFN_PREPROCESSING_DIR:-}" ]]; then
  ARGS+=(--rdbpfn-preprocessing-root "${RDBPFN_PREPROCESSING_DIR}")
fi
if [[ -n "${DFS_DEPTH:-}" ]]; then
  ARGS+=(--dfs-depth "${DFS_DEPTH}")
fi
if [[ -n "${DFS_JOBS:-}" ]]; then
  ARGS+=(--dfs-jobs "${DFS_JOBS}")
fi
if [[ -n "${H5_TOTAL_ROWS:-}" ]]; then
  ARGS+=(--h5-total-rows "${H5_TOTAL_ROWS}")
fi
if [[ -n "${H5_MAX_COLUMNS:-}" ]]; then
  ARGS+=(--h5-max-columns "${H5_MAX_COLUMNS}")
fi
if [[ -n "${H5_SEED:-}" ]]; then
  ARGS+=(--h5-seed "${H5_SEED}")
fi

case "${H5_EXPORT:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--h5)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-h5)
    ;;
  "")
    ;;
  *)
    echo "H5_EXPORT must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

case "${H5_RUN_DFS:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--h5-run-dfs)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-h5-run-dfs)
    ;;
  "")
    ;;
  *)
    echo "H5_RUN_DFS must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

case "${COMPRESS:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--compress)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-compress)
    ;;
  "")
    ;;
  *)
    echo "COMPRESS must be 1/0, true/false, or yes/no" >&2
    exit 2
    ;;
esac

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
