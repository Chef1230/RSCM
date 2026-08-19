#!/usr/bin/env bash

load_eval_config() {
  local config_path="$1"
  if [[ "${config_path}" != /* && -f "${PROJECT_ROOT}/${config_path}" ]]; then
    config_path="${PROJECT_ROOT}/${config_path}"
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "Evaluation config does not exist: ${config_path}" >&2
    return 2
  fi
  local config_dump key value
  config_dump="$(mktemp)"
  if ! PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -m rdb_prior.eval_config \
      "${config_path}" "${PROJECT_ROOT}" > "${config_dump}"; then
    rm -f "${config_dump}"
    return 2
  fi
  while IFS= read -r -d '' key && IFS= read -r -d '' value; do
    if [[ ! -v "${key}" ]]; then
      printf -v "${key}" '%s' "${value}"
      export "${key}"
    fi
  done < "${config_dump}"
  rm -f "${config_dump}"
}
