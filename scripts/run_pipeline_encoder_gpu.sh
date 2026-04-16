#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${HOME}/.venvs/advnlp/bin/activate"

require_file() {
  local target="$1"
  if [[ ! -f "${target}" ]]; then
    echo "[ERROR] Missing required file: ${target}" >&2
    echo "[ERROR] Sync the repo again or update the workflow scripts if paths changed." >&2
    exit 1
  fi
}

if [[ ! -f "${VENV_PATH}" ]]; then
  echo "[ERROR] Expected virtual environment activation script not found: ${VENV_PATH}" >&2
  echo "[ERROR] Create the environment on the GPU server before running this pipeline." >&2
  exit 1
fi

require_file "${REPO_ROOT}/run_encoder.py"
require_file "${REPO_ROOT}/codes/06_results_analysis.py"
require_file "${REPO_ROOT}/configs/experiment.yaml"

source "${VENV_PATH}"
cd "${REPO_ROOT}"

echo "[INFO] Repo root: ${REPO_ROOT}"
VENV_PYTHON="${HOME}/.venvs/advnlp/bin/python"
echo "[INFO] Installing/verifying Python dependencies from requirements.txt ..."
"${VENV_PYTHON}" -m ensurepip --upgrade 2>/dev/null || true
"${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements.txt" --quiet
echo "[INFO] Dependencies OK."
echo "[INFO] GPU snapshot:"
nvidia-smi | head -n 15 || true

python - <<'PY'
import torch
print(f"[INFO] torch.__version__ = {torch.__version__}")
print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"[INFO] torch.cuda.device_count() = {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[INFO] current_device = {torch.cuda.current_device()}")
    print(f"[INFO] device_name = {torch.cuda.get_device_name(torch.cuda.current_device())}")
PY

echo "[INFO] Running encoder pipeline (train + attack)"
python run_encoder.py

echo "[INFO] Running results analysis"
python codes/06_results_analysis.py

echo "[DONE] Encoder pipeline completed successfully."
echo "[DONE] Outputs are written under: ${REPO_ROOT}/results/"
