#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v2.yaml}}"
BASH_BIN="${BASH_BIN:-bash}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "generate_v3.sh accepts only an optional config path" >&2
  exit 2
fi

# Stage manifest forwarding --------------------------------------------------
if [[ -n "${SCHEMA_OUTPUT_DIR:-}" && -z "${SCHEMA_MANIFEST:-}" ]]; then
  export SCHEMA_MANIFEST="${SCHEMA_OUTPUT_DIR}/manifest.json"
fi
if [[ -n "${INSTANCE_OUTPUT_DIR:-}" && -z "${INSTANCE_MANIFEST:-}" ]]; then
  export INSTANCE_MANIFEST="${INSTANCE_OUTPUT_DIR}/manifest.json"
fi
if [[ -n "${TASK_OUTPUT_DIR:-}" && -z "${TASK_MANIFEST:-}" ]]; then
  export TASK_MANIFEST="${TASK_OUTPUT_DIR}/manifest.json"
fi

# H5 output base directory ---------------------------------------------------
# Set H5_OUTPUT_BASE to override; otherwise it is derived from the config's
# output_root via a fast Python one-liner.
if [[ -z "${H5_OUTPUT_BASE:-}" ]]; then
  H5_OUTPUT_BASE="$(
    cd "${PROJECT_ROOT}"
    PYTHONPATH="${PROJECT_ROOT}/src" python -c "
from rdb_prior.config import load_rdbpfn_export_config
c = load_rdbpfn_export_config('${CONFIG_PATH}')
print(c.output_root.resolve())
" 2>/dev/null || echo ''
  )"
  if [[ -z "${H5_OUTPUT_BASE:-}" ]]; then
    echo "generate_v3.sh: could not derive rdbpfn output from ${CONFIG_PATH}" >&2
    echo "  set H5_OUTPUT_BASE explicitly, e.g." >&2
    echo "  H5_OUTPUT_BASE=outputs/my_run/rdbpfn bash scripts/v1/generate_v3.sh" >&2
    exit 2
  fi
fi
export H5_OUTPUT_BASE

# ---------------------------------------------------------------------------
# Stage 1–3: schema, instance, task
# ---------------------------------------------------------------------------
echo "=== Stage 01: schema ==="
"${BASH_BIN}" "${SCRIPT_DIR}/01_schema.sh" "${CONFIG_PATH}"

echo "=== Stage 02: instance ==="
"${BASH_BIN}" "${SCRIPT_DIR}/02_instance.sh" "${CONFIG_PATH}"

echo "=== Stage 03: task ==="
"${BASH_BIN}" "${SCRIPT_DIR}/03_task.sh" "${CONFIG_PATH}"

# ---------------------------------------------------------------------------
# Stage 04: DBB export + DFS depth 1 -> H5
# ---------------------------------------------------------------------------
echo "=== Stage 04: export + DFS depth 1 ==="
H5_RUN_DFS=true \
  H5_EXPORT=true \
  DFS_DEPTH=1 \
  LOG_LEVEL=WARNING \
  PROGRESS_BAR=1 \
  H5_OUTPUT="${H5_OUTPUT_BASE}/rdbpfn_tasks_dfs-1.h5" \
  "${BASH_BIN}" "${SCRIPT_DIR}/04_rdbpfn_export.sh" "${CONFIG_PATH}"

# ---------------------------------------------------------------------------
# Stage 04: DFS depth 2 -> H5 (reuses existing DBB datasets)
# ---------------------------------------------------------------------------
echo "=== Stage 04: export + DFS depth 2 ==="
H5_RUN_DFS=true \
  H5_EXPORT=true \
  DFS_DEPTH=2 \
  LOG_LEVEL=WARNING \
  PROGRESS_BAR=1 \
  H5_OUTPUT="${H5_OUTPUT_BASE}/rdbpfn_tasks_dfs-2.h5" \
  "${BASH_BIN}" "${SCRIPT_DIR}/04_rdbpfn_export.sh" "${CONFIG_PATH}"

echo "=== generate_v3 complete ==="
echo "  depth-1 H5: ${H5_OUTPUT_BASE}/rdbpfn_tasks_dfs-1.h5"
echo "  depth-2 H5: ${H5_OUTPUT_BASE}/rdbpfn_tasks_dfs-2.h5"
