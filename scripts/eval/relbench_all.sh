#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_CONFIG_PATH="${1:-${EVAL_CONFIG:-}}"

if [[ -n "${EVAL_CONFIG_PATH}" ]]; then
  # shellcheck source=scripts/eval/config.sh
  source "${SCRIPT_DIR}/config.sh"
  load_eval_config "${EVAL_CONFIG_PATH}"
fi

RELBENCH_DATASET="${RELBENCH_DATASET:-rel-amazon}"
RELBENCH_OUTPUT_ROOT="${RELBENCH_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/relbench/${RELBENCH_DATASET}}"

if [[ -n "${RELBENCH_TASKS:-}" ]]; then
  IFS=',' read -r -a tasks <<< "${RELBENCH_TASKS}"
else
  if [[ -z "${RELBENCH_CACHE_DIR:-}" ]]; then
    echo "Set RELBENCH_CACHE_DIR to the RelBench cache root." >&2
    exit 2
  fi
  task_root="${RELBENCH_CACHE_DIR}/${RELBENCH_DATASET}/tasks"
  if [[ ! -d "${task_root}" ]]; then
    echo "RelBench task directory does not exist: ${task_root}" >&2
    exit 2
  fi
  tasks=()
  while IFS= read -r task_path; do
    tasks+=("$(basename "${task_path}")")
  done < <(find "${task_root}" -mindepth 1 -maxdepth 1 -type d -print | sort)
fi

selected_tasks=()
for raw_task in "${tasks[@]}"; do
  task="${raw_task#${raw_task%%[![:space:]]*}}"
  task="${task%${task##*[![:space:]]}}"
  [[ -n "${task}" ]] && selected_tasks+=("${task}")
done
tasks=("${selected_tasks[@]}")

if [[ "${#tasks[@]}" == "0" ]]; then
  echo "No RelBench tasks were selected." >&2
  exit 2
fi

completed=0
skipped=0
failed=0
processed=0
total="${#tasks[@]}"
progress_line=""

clear_progress() {
  if [[ -n "${progress_line}" ]]; then
    printf '\r%*s\r' "${#progress_line}" ''
    progress_line=""
  fi
}

print_progress() {
  local task="$1"
  local outcome="$2"
  local width="${PROGRESS_WIDTH:-28}"
  local percent=$((processed * 100 / total))
  local filled=$((processed * width / total))
  local empty=$((width - filled))
  local filled_bar empty_bar
  printf -v filled_bar '%*s' "${filled}" ''
  printf -v empty_bar '%*s' "${empty}" ''
  filled_bar="${filled_bar// /#}"
  empty_bar="${empty_bar// /-}"
  printf -v progress_line '[relbench-all] [%s%s] %d/%d %3d%% | %s | %s' \
    "${filled_bar}" "${empty_bar}" "${processed}" "${total}" \
    "${percent}" "${task}" "${outcome}"
  printf '\r%s' "${progress_line}"
  if [[ "${processed}" == "${total}" ]]; then
    printf '\n'
    progress_line=""
  fi
}

for task in "${tasks[@]}"; do
  clear_progress
  output="${RELBENCH_OUTPUT_ROOT}/${task}"
  metadata="${output}/relbench_metadata.json"

  echo
  echo "[relbench-all] ${RELBENCH_DATASET}/${task}: inspect"
  if ! task_type="$(
      "${PYTHON_BIN}" -c \
        'import sys; from relbench.tasks import get_task; t=get_task(sys.argv[1], sys.argv[2], download=sys.argv[3].lower() in {"1", "true", "yes"}); print(getattr(t.task_type, "value", t.task_type))' \
        "${RELBENCH_DATASET}" "${task}" "${DOWNLOAD:-0}"
    )"; then
    echo "[relbench-all] ${RELBENCH_DATASET}/${task}: FAILED (task inspection)"
    failed=$((failed + 1))
    processed=$((processed + 1))
    print_progress "${task}" "failed:inspect"
    continue
  fi
  if [[ "${task_type}" != "binary_classification" ]]; then
    echo "[relbench-all] ${RELBENCH_DATASET}/${task}: SKIPPED (${task_type}; TFM is binary-only)"
    skipped=$((skipped + 1))
    processed=$((processed + 1))
    print_progress "${task}" "skipped:${task_type}"
    continue
  fi

  echo "[relbench-all] ${RELBENCH_DATASET}/${task}: convert"
  if ! RELBENCH_TASK="${task}" RELBENCH_OUTPUT="${output}" \
      bash "${SCRIPT_DIR}/relbench.sh" convert; then
    echo "[relbench-all] ${RELBENCH_DATASET}/${task}: FAILED (conversion)"
    failed=$((failed + 1))
    processed=$((processed + 1))
    print_progress "${task}" "failed:convert"
    continue
  fi

  converted_task_type="$(
    "${PYTHON_BIN}" -c \
      'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["relbench_task_type"])' \
      "${metadata}"
  )"
  if [[ "${converted_task_type}" != "${task_type}" ]]; then
    echo "[relbench-all] ${RELBENCH_DATASET}/${task}: FAILED (task type changed during conversion)"
    failed=$((failed + 1))
    processed=$((processed + 1))
    print_progress "${task}" "failed:type-mismatch"
    continue
  fi

  echo "[relbench-all] ${RELBENCH_DATASET}/${task}: routed H5"
  if ! RELBENCH_TASK="${task}" RELBENCH_OUTPUT="${output}" \
      TASK_MANIFEST="${output}/task/manifest.json" \
      ROUTED_H5_OUTPUT="${output}/routed.h5" \
      bash "${SCRIPT_DIR}/relbench.sh" h5; then
    echo "[relbench-all] ${RELBENCH_DATASET}/${task}: FAILED (H5 export)"
    failed=$((failed + 1))
    processed=$((processed + 1))
    print_progress "${task}" "failed:h5"
    continue
  fi

  echo "[relbench-all] ${RELBENCH_DATASET}/${task}: TFM inference + score"
  if ! RELBENCH_TASK="${task}" RELBENCH_OUTPUT="${output}" \
      RELBENCH_METADATA="${metadata}" \
      ROUTED_H5_OUTPUT="${output}/routed.h5" \
      TFM_PREDICTIONS_OUTPUT="${output}/tfm_eval/predictions.jsonl" \
      TFM_METRICS_OUTPUT="${output}/tfm_eval/relbench_metrics.json" \
      bash "${SCRIPT_DIR}/relbench.sh" tfm; then
    echo "[relbench-all] ${RELBENCH_DATASET}/${task}: FAILED (TFM eval)"
    failed=$((failed + 1))
    processed=$((processed + 1))
    print_progress "${task}" "failed:tfm"
    continue
  fi
  completed=$((completed + 1))
  processed=$((processed + 1))
  print_progress "${task}" "completed"
done

echo
echo "[relbench-all] completed=${completed} skipped=${skipped} failed=${failed} total=${#tasks[@]}"
if [[ "${failed}" != "0" ]]; then
  exit 1
fi
