#!/usr/bin/env bash
set -euo pipefail

# Small, complete prior-v2 run: schema -> instance -> task -> RDBPFN export.
# Environment overrides such as OUTPUT_DIR, OVERWRITE, INSTANCE_JOBS and
# TASK_JOBS remain supported by the delegated stage scripts.  Use --set to
# override any existing YAML mapping value for this invocation, for example:
#   bash scripts/v1/generate_prior_v2.sh configs/prior_v2_3k.yaml \
#     --set generation.num_schemas=10 --set prior.shared_state.dimension=6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/prior_v2_temporal_example.yaml}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_PATH="${1}"
  shift
fi

# The configured environment can still be overridden by the caller.
export PYTHON_BIN="${PYTHON_BIN:-/opt/data/private/cf/envs/.rdb-pre/bin/python}"

OVERRIDES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --set)
      if [[ $# -lt 2 ]]; then
        echo "--set requires PATH=YAML_VALUE" >&2
        exit 2
      fi
      OVERRIDES+=("$2")
      shift 2
      ;;
    --set=*)
      OVERRIDES+=("${1#--set=}")
      shift
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: generate_prior_v2.sh [CONFIG_PATH] [--set PATH=YAML_VALUE]...

Runs schema, instance, task, and RDBPFN export with one effective config.
Each --set overrides an existing YAML mapping key only for this invocation.
Values use YAML syntax, so strings should be quoted when necessary.

Examples:
  bash scripts/v1/generate_prior_v2.sh configs/prior_v2_3k.yaml \
    --set generation.num_schemas=10 \
    --set task_generation.tasks_per_database=1
  OUTPUT_DIR=outputs/debug OVERWRITE=1 \
    bash scripts/v1/generate_prior_v2.sh \
      --set prior.shared_state.dimension=6
USAGE
      exit 0
      ;;
    *)
      echo "unknown argument: $1 (expected --set PATH=YAML_VALUE)" >&2
      exit 2
      ;;
  esac
done

if [[ ${#OVERRIDES[@]} -eq 0 ]]; then
  exec bash "${SCRIPT_DIR}/generate_v1.sh" "${CONFIG_PATH}"
fi

EFFECTIVE_CONFIG="$(mktemp "${TMPDIR:-/tmp}/rdb-prior-v2-config.XXXXXX.yaml")"
cleanup() {
  rm -f "${EFFECTIVE_CONFIG}"
}
trap cleanup EXIT

"${PYTHON_BIN}" - "${CONFIG_PATH}" "${EFFECTIVE_CONFIG}" "${OVERRIDES[@]}" <<'PY'
from __future__ import annotations

import sys
from collections.abc import MutableMapping
from pathlib import Path

import yaml

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
overrides = sys.argv[3:]

with source_path.open("r", encoding="utf-8") as source_file:
    config = yaml.safe_load(source_file)
if not isinstance(config, MutableMapping):
    raise SystemExit(f"configuration root must be a mapping: {source_path}")

for override in overrides:
    path, separator, raw_value = override.partition("=")
    if not separator or not path:
        raise SystemExit(
            f"invalid --set value {override!r}; expected PATH=YAML_VALUE"
        )
    keys = path.split(".")
    if any(not key for key in keys):
        raise SystemExit(f"invalid override path: {path!r}")

    target = config
    for key in keys[:-1]:
        value = target.get(key)
        if not isinstance(value, MutableMapping):
            raise SystemExit(
                f"override path is not an existing mapping: {path!r}"
            )
        target = value
    leaf = keys[-1]
    if leaf not in target:
        raise SystemExit(f"override key does not exist: {path!r}")
    target[leaf] = yaml.safe_load(raw_value)

with output_path.open("w", encoding="utf-8") as output_file:
    yaml.safe_dump(config, output_file, allow_unicode=True, sort_keys=False)
PY

bash "${SCRIPT_DIR}/generate_v1.sh" "${EFFECTIVE_CONFIG}"
