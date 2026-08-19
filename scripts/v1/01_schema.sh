#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v1.yaml}}"

# A non-option first argument is a convenient config-path override.
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  schema
  --config "${CONFIG_PATH}"
)

if [[ -n "${SCHEMA_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${SCHEMA_OUTPUT_DIR}")
elif [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}/schema")
fi
if [[ -n "${NUM_SCHEMAS:-}" ]]; then
  ARGS+=(--count "${NUM_SCHEMAS}")
fi
if [[ -n "${BASE_SEED:-}" ]]; then
  ARGS+=(--seed "${BASE_SEED}")
fi
if [[ -n "${START_INDEX:-}" ]]; then
  ARGS+=(--start-index "${START_INDEX}")
fi
if [[ -n "${SAMPLE_ID_PREFIX:-}" ]]; then
  ARGS+=(--sample-id-prefix "${SAMPLE_ID_PREFIX}")
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
if [[ -n "${SCHEMA_GRAPH_FORMAT:-}" ]]; then
  ARGS+=(--schema-graph-format "${SCHEMA_GRAPH_FORMAT}")
fi
if [[ -n "${GRAPHVIZ_COMMAND:-}" ]]; then
  ARGS+=(--graphviz-command "${GRAPHVIZ_COMMAND}")
fi

case "${SCHEMA_DOT:-}" in
  1|true|TRUE|yes|YES)
    ARGS+=(--schema-dot)
    ;;
  0|false|FALSE|no|NO)
    ARGS+=(--no-schema-dot)
    ;;
  "")
    ;;
  *)
    echo "SCHEMA_DOT must be 1/0, true/false, or yes/no" >&2
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
