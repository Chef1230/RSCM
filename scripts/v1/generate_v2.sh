#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/local/local.yaml}}"
BASH_BIN="${BASH_BIN:-bash}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "generate_v2.sh accepts only an optional config path" >&2
  exit 2
fi

if [[ -n "${SCHEMA_OUTPUT_DIR:-}" && -z "${SCHEMA_MANIFEST:-}" ]]; then
  export SCHEMA_MANIFEST="${SCHEMA_OUTPUT_DIR}/manifest.json"
fi
if [[ -n "${INSTANCE_OUTPUT_DIR:-}" && -z "${INSTANCE_MANIFEST:-}" ]]; then
  export INSTANCE_MANIFEST="${INSTANCE_OUTPUT_DIR}/manifest.json"
fi
if [[ -n "${TASK_OUTPUT_DIR:-}" && -z "${TASK_MANIFEST:-}" ]]; then
  export TASK_MANIFEST="${TASK_OUTPUT_DIR}/manifest.json"
fi

"${BASH_BIN}" "${SCRIPT_DIR}/01_schema.sh" "${CONFIG_PATH}"
"${BASH_BIN}" "${SCRIPT_DIR}/02_instance.sh" "${CONFIG_PATH}"
"${BASH_BIN}" "${SCRIPT_DIR}/03_task.sh" "${CONFIG_PATH}"
"${BASH_BIN}" "${SCRIPT_DIR}/04b_router_train.sh" "${CONFIG_PATH}"
"${BASH_BIN}" "${SCRIPT_DIR}/05_routed_h5.sh" "${CONFIG_PATH}"
