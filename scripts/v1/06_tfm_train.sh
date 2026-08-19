#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-${RDB_PRIOR_CONFIG:-${PROJECT_ROOT}/configs/refactor_v2.yaml}}"

if [[ $# -gt 0 && ( "${1}" == *.yaml || "${1}" == *.yml ) ]]; then
  CONFIG_PATH="${1}"
  shift
fi

RDBPFN_ROOT="${RDBPFN_ROOT:-${PROJECT_ROOT}/../RDBPFN}"
if [[ ! -d "${RDBPFN_ROOT}/model_pretrain" && -d "${PROJECT_ROOT}/../RDB_PFN/model_pretrain" ]]; then
  RDBPFN_ROOT="${PROJECT_ROOT}/../RDB_PFN"
fi
MODEL_ROOT="${RDBPFN_ROOT}/model_pretrain"
TFM_CONFIG_NAME="${TFM_CONFIG_NAME:-RDBPFN_routed}"
ROUTED_H5_PATH="${ROUTED_H5_OUTPUT:-}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Synthetic-data config does not exist: ${CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "RDBPFN model_pretrain directory does not exist: ${MODEL_ROOT}" >&2
  echo "Set RDBPFN_ROOT=/absolute/path/to/RDBPFN if it is elsewhere." >&2
  exit 2
fi
if [[ ! -f "${MODEL_ROOT}/conf_train/${TFM_CONFIG_NAME}.yaml" ]]; then
  echo "TFM config does not exist: ${MODEL_ROOT}/conf_train/${TFM_CONFIG_NAME}.yaml" >&2
  exit 2
fi
if [[ ! -f "${MODEL_ROOT}/src/routed_dataloader.py" ]] || \
   ! grep -q "RoutedTokenDataset" "${MODEL_ROOT}/src/train.py"; then
  echo "RDBPFN does not contain routed-token training support: ${MODEL_ROOT}" >&2
  echo "Sync the routed-enabled model_pretrain sources before running stage 06." >&2
  exit 2
fi
if [[ -n "${ROUTED_H5_PATH}" && ! -f "${ROUTED_H5_PATH}" ]]; then
  echo "Routed H5 does not exist: ${ROUTED_H5_PATH}" >&2
  echo "Run scripts/v1/05_routed_h5.sh first or correct ROUTED_H5_OUTPUT." >&2
  exit 2
fi
if ! [[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_PROCESSES must be a positive integer" >&2
  exit 2
fi

LAUNCH_ARGS=(
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --dynamo_backend no
)
if (( NUM_PROCESSES > 1 )); then
  LAUNCH_ARGS+=(--multi_gpu)
fi
[[ -n "${MIXED_PRECISION:-}" ]] && LAUNCH_ARGS+=(--mixed_precision "${MIXED_PRECISION}")
[[ -n "${GPU_IDS:-}" ]] && LAUNCH_ARGS+=(--gpu_ids "${GPU_IDS}")

TRAIN_ARGS=(
  --config-name "${TFM_CONFIG_NAME}"
  "++train.datasets.0.format=routed_tokens"
  "train.num_gpus=${NUM_PROCESSES}"
  "train.batch_size=1"
  "++model.enable_routed_tokens=true"
)
[[ -n "${ROUTED_H5_PATH}" ]] && TRAIN_ARGS+=("train.datasets.0.path=${ROUTED_H5_PATH}")
[[ -n "${TFM_SAVE_EVERY_EVALS:-}" ]] && TRAIN_ARGS+=("train.save_every_evals=${TFM_SAVE_EVERY_EVALS}")
[[ -n "${TFM_FIND_UNUSED_PARAMETERS:-}" ]] && TRAIN_ARGS+=("train.find_unused_parameters=${TFM_FIND_UNUSED_PARAMETERS}")
[[ -n "${TFM_NUM_STEPS:-}" ]] && TRAIN_ARGS+=("train.num_steps=${TFM_NUM_STEPS}")
[[ -n "${TFM_NUM_EPOCHS:-}" ]] && TRAIN_ARGS+=("train.num_epochs=${TFM_NUM_EPOCHS}")
[[ -n "${TFM_LR:-}" ]] && TRAIN_ARGS+=("train.lr=${TFM_LR}")
[[ -n "${TFM_LOAD_CHECKPOINT:-}" ]] && TRAIN_ARGS+=("train.load_model_path=${TFM_LOAD_CHECKPOINT}")
[[ -n "${TFM_SAVE_CHECKPOINT:-}" ]] && TRAIN_ARGS+=("train.save_model_path=${TFM_SAVE_CHECKPOINT}")
[[ -n "${ROUTED_TOKEN_DIM:-}" ]] && TRAIN_ARGS+=("model.routed_token_dim=${ROUTED_TOKEN_DIM}")

echo "Synthetic config: ${CONFIG_PATH}"
if [[ -n "${ROUTED_H5_PATH}" ]]; then
  echo "Routed H5:       ${ROUTED_H5_PATH} (environment override)"
else
  echo "Routed H5:       from conf_train/${TFM_CONFIG_NAME}.yaml"
fi
echo "TFM root:        ${MODEL_ROOT}"
echo "Processes:       ${NUM_PROCESSES} (batch_size=1 per process)"

cd "${MODEL_ROOT}"
export PYTHONPATH="${MODEL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m accelerate.commands.launch \
  "${LAUNCH_ARGS[@]}" \
  -m src.train \
  "${TRAIN_ARGS[@]}" \
  "$@"
