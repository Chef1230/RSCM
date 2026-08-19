#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-convert}"
EVAL_CONFIG_PATH="${2:-${EVAL_CONFIG:-}}"

if [[ -n "${EVAL_CONFIG_PATH}" ]]; then
  # shellcheck source=scripts/eval/config.sh
  source "${SCRIPT_DIR}/config.sh"
  load_eval_config "${EVAL_CONFIG_PATH}"
fi

RELBENCH_DATASET="${RELBENCH_DATASET:-rel-amazon}"
RELBENCH_TASK="${RELBENCH_TASK:-user-churn}"
RELBENCH_OUTPUT="${RELBENCH_OUTPUT:-${PROJECT_ROOT}/outputs/relbench/${RELBENCH_DATASET}/${RELBENCH_TASK}}"
TASK_MANIFEST="${TASK_MANIFEST:-${RELBENCH_OUTPUT}/task/manifest.json}"
ROUTER_EVAL_OUTPUT="${ROUTER_EVAL_OUTPUT:-${RELBENCH_OUTPUT}/router_eval}"
RELBENCH_METADATA="${RELBENCH_METADATA:-${RELBENCH_OUTPUT}/relbench_metadata.json}"
RELBENCH_METRICS_OUTPUT="${RELBENCH_METRICS_OUTPUT:-${ROUTER_EVAL_OUTPUT}/relbench_metrics.json}"
ROUTED_H5_OUTPUT="${ROUTED_H5_OUTPUT:-${RELBENCH_OUTPUT}/routed.h5}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v2.yaml}}"
RDBPFN_ROOT="${RDBPFN_ROOT:-${PROJECT_ROOT}/../RDBPFN}"
MODEL_ROOT="${RDBPFN_ROOT}/model_pretrain"
TFM_MODEL_CONFIG="${TFM_MODEL_CONFIG:-${MODEL_ROOT}/conf_train/RDBPFN_routed.yaml}"
TFM_PREDICTIONS_OUTPUT="${TFM_PREDICTIONS_OUTPUT:-${RELBENCH_OUTPUT}/tfm_eval/predictions.jsonl}"
TFM_METRICS_OUTPUT="${TFM_METRICS_OUTPUT:-${RELBENCH_OUTPUT}/tfm_eval/relbench_metrics.json}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

bool_arg() {
  local value="$1"
  local positive="$2"
  local negative="$3"
  case "${value}" in
    1|true|TRUE|yes|YES) printf '%s\n' "${positive}" ;;
    0|false|FALSE|no|NO) printf '%s\n' "${negative}" ;;
    *) echo "Expected true/false, got: ${value}" >&2; exit 2 ;;
  esac
}

require_checkpoint() {
  if [[ -z "${ROUTER_CHECKPOINT:-}" ]]; then
    echo "Set ROUTER_CHECKPOINT to router/checkpoints/best.pt or last.pt." >&2
    exit 2
  fi
  if [[ ! -f "${ROUTER_CHECKPOINT}" ]]; then
    echo "Router checkpoint does not exist: ${ROUTER_CHECKPOINT}" >&2
    exit 2
  fi
}

run_convert() {
  case "${REUSE_CONVERTED:-0}" in
    1|true|TRUE|yes|YES)
      local required
      for required in \
          "${RELBENCH_METADATA}" \
          "${TASK_MANIFEST}" \
          "${RELBENCH_OUTPUT}/schema/manifest.json" \
          "${RELBENCH_OUTPUT}/instance/manifest.json"; do
        if [[ ! -f "${required}" ]]; then
          required=""
          break
        fi
      done
      if [[ -n "${required}" ]]; then
        echo "[relbench-import] reusing converted artifacts: ${RELBENCH_OUTPUT}"
        return 0
      fi
      ;;
    0|false|FALSE|no|NO) ;;
    *) echo "REUSE_CONVERTED must be true/false" >&2; exit 2 ;;
  esac
  local args=(
    relbench-import
    --dataset "${RELBENCH_DATASET}"
    --task "${RELBENCH_TASK}"
    --output-dir "${RELBENCH_OUTPUT}"
    --seed "${SEED:-0}"
    --max-rows-per-task "${MAX_ROWS_PER_TASK:-600}"
    --query-rows-per-task "${QUERY_ROWS_PER_TASK:-256}"
    --max-classes "${MAX_CLASSES:-16}"
    --max-text-length "${MAX_TEXT_LENGTH:-256}"
    "$(bool_arg "${DOWNLOAD:-1}" --download --no-download)"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
    --progress
  )
  [[ -n "${SUPPORT_ROWS:-}" ]] && args+=(--support-rows "${SUPPORT_ROWS}")
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
}

run_eval() {
  require_checkpoint
  if [[ ! -f "${TASK_MANIFEST}" ]]; then
    echo "RelBench task manifest does not exist: ${TASK_MANIFEST}" >&2
    exit 2
  fi
  local args=(
    router-eval
    --task-manifest "${TASK_MANIFEST}"
    --checkpoint "${ROUTER_CHECKPOINT}"
    --output-dir "${ROUTER_EVAL_OUTPUT}"
    --device "${DEVICE:-auto}"
    --mixed-precision "${MIXED_PRECISION:-none}"
    --artifact-cache-size "${ARTIFACT_CACHE_SIZE:-4}"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
    --progress
  )
  [[ -n "${NUM_TASKS:-}" ]] && args+=(--count "${NUM_TASKS}")
  [[ -n "${START_INDEX:-}" ]] && args+=(--start-index "${START_INDEX}")
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
  if [[ -n "${NUM_TASKS:-}" || ( -n "${START_INDEX:-}" && "${START_INDEX}" != "0" ) ]]; then
    echo "Skipping official RelBench metrics for a partial task selection."
  else
    run_score
  fi
}

run_score() {
  local predictions="${1:-${ROUTER_EVAL_OUTPUT}/predictions.jsonl}"
  local metrics_output="${2:-${RELBENCH_METRICS_OUTPUT}}"
  if [[ ! -f "${RELBENCH_METADATA}" ]]; then
    echo "RelBench metadata does not exist: ${RELBENCH_METADATA}" >&2
    exit 2
  fi
  if [[ ! -f "${predictions}" ]]; then
    echo "Predictions do not exist: ${predictions}" >&2
    exit 2
  fi
  local args=(
    relbench-score
    --metadata "${RELBENCH_METADATA}"
    --predictions "${predictions}"
    --output "${metrics_output}"
    "$(bool_arg "${SCORE_DOWNLOAD:-0}" --download --no-download)"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
  )
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
}

run_tfm() {
  if [[ -z "${TFM_CHECKPOINT:-}" ]]; then
    echo "Set TFM_CHECKPOINT to checkpoints/RDBPFN_routed/model.pt." >&2
    exit 2
  fi
  for path in "${ROUTED_H5_OUTPUT}" "${TFM_CHECKPOINT}" "${TFM_MODEL_CONFIG}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Required TFM inference file does not exist: ${path}" >&2
      exit 2
    fi
  done
  local args=(
    -m src.routed_eval
    --input "${ROUTED_H5_OUTPUT}"
    --checkpoint "${TFM_CHECKPOINT}"
    --model-config "${TFM_MODEL_CONFIG}"
    --output "${TFM_PREDICTIONS_OUTPUT}"
    --device "${DEVICE:-auto}"
    --mixed-precision "${MIXED_PRECISION:-none}"
    --progress-every "${PROGRESS_EVERY:-10}"
    --progress-width "${PROGRESS_WIDTH:-28}"
  )
  [[ -n "${NUM_TASKS:-}" ]] && args+=(--count "${NUM_TASKS}")
  [[ -n "${START_INDEX:-}" ]] && args+=(--start-index "${START_INDEX}")
  case "${OVERWRITE:-0}" in
    1|true|TRUE|yes|YES) args+=(--overwrite) ;;
    0|false|FALSE|no|NO) ;;
    *) echo "OVERWRITE must be true/false" >&2; exit 2 ;;
  esac
  local status=0
  if (
      cd "${MODEL_ROOT}"
      PYTHONPATH="${MODEL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" "${args[@]}"
  ); then
    status=0
  else
    status=$?
  fi
  if [[ "${status}" == "3" ]]; then
    echo "Skipping TFM scoring because routed H5 has no supported binary classification tasks."
    return 0
  fi
  if [[ "${status}" != "0" ]]; then
    return "${status}"
  fi
  if [[ -n "${NUM_TASKS:-}" || ( -n "${START_INDEX:-}" && "${START_INDEX}" != "0" ) ]]; then
    echo "Skipping official RelBench metrics for a partial TFM task selection."
  else
    run_score "${TFM_PREDICTIONS_OUTPUT}" "${TFM_METRICS_OUTPUT}"
  fi
}

run_h5() {
  require_checkpoint
  if [[ ! -f "${TASK_MANIFEST}" ]]; then
    echo "RelBench task manifest does not exist: ${TASK_MANIFEST}" >&2
    exit 2
  fi
  local args=(
    routed-h5
    --config "${CONFIG_PATH}"
    --task-manifest "${TASK_MANIFEST}"
    --checkpoint "${ROUTER_CHECKPOINT}"
    --output "${ROUTED_H5_OUTPUT}"
    --device "${DEVICE:-auto}"
    "$(bool_arg "${OVERWRITE:-0}" --overwrite --no-overwrite)"
    --progress
  )
  [[ -n "${NUM_TASKS:-}" ]] && args+=(--count "${NUM_TASKS}")
  [[ -n "${START_INDEX:-}" ]] && args+=(--start-index "${START_INDEX}")
  [[ -n "${LOG_LEVEL:-}" ]] && args+=(--log-level "${LOG_LEVEL}")
  [[ -n "${LOG_FILE:-}" ]] && args+=(--log-file "${LOG_FILE}")
  "${PYTHON_BIN}" -m rdb_prior.cli "${args[@]}"
}

cd "${PROJECT_ROOT}"
case "${ACTION}" in
  convert) run_convert ;;
  eval) run_eval ;;
  score) run_score ;;
  h5) run_h5 ;;
  tfm) run_tfm ;;
  pipeline)
    run_convert
    run_h5
    run_tfm
    ;;
  all)
    run_convert
    run_eval
    run_h5
    ;;
  *)
    echo "Usage: bash scripts/eval/relbench.sh [convert|eval|score|h5|tfm|pipeline|all] [config.yaml]" >&2
    exit 2
    ;;
esac
