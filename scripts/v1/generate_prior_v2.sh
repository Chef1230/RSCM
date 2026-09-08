#!/usr/bin/env bash
set -euo pipefail

# Small, complete prior-v2 run: schema -> instance -> task -> RDBPFN export.
# Environment overrides such as OUTPUT_DIR, OVERWRITE, INSTANCE_JOBS and
# TASK_JOBS remain supported by the delegated stage scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/prior_v2_temporal_example.yaml}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "generate_prior_v2.sh accepts only an optional config path" >&2
  exit 2
fi

# The configured environment can still be overridden by the caller.
export PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/cf/envs/.rdb-pre/bin/python}"

exec bash "${SCRIPT_DIR}/generate_v1.sh" "${CONFIG_PATH}"
