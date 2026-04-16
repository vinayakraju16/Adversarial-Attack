#!/usr/bin/env bash
# =============================================================================
# run_all_attacks.sh
#
# Batch runner: iterates over all attacks defined in experiment.yaml, running
# run_encoder.py and run_decoder.py for each attack name in sequence.
# Edit configs/experiment.yaml to change the attack, then call this script to
# sweep all attacks in one go by patching the YAML between runs.
#
# Usage (on the GPU server, after the repo has been synced):
#   bash scripts/run_all_attacks.sh
#
# Outputs per run (inside results/):
#   results/{dataset}/{attack}/{model}/attacks.jsonl
#   results/{dataset}/{attack}/{model}/summary.csv
#   results/{dataset}/{attack}/{model}/stats.json
# Combined report (after all runs):
#   results/combined_summary.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/experiment.yaml"

# 15 supported attacks (must match ATTACK_REGISTRY in run_encoder.py / run_decoder.py)
ATTACKS=(
  "textfooler"
  "textbugger"
  "bertattack"
  "bae"
  "pwws"
  "a2t"
  "clare"
  "genetic"
  "faster_genetic"
  "pso"
  "hotflip"
  "deepwordbug"
  "pruthi"
  "morpheus"
  "input_reduction"
)

# ─── ACTIVATE VENV ───────────────────────────────────────────────────────────
VENV_PATH="${HOME}/.venvs/advnlp/bin/activate"
if [[ ! -f "${VENV_PATH}" ]]; then
  echo "[ERROR] Venv not found: ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}"
cd "${REPO_ROOT}"

# ─── BATCH LOOP ──────────────────────────────────────────────────────────────
TOTAL=${#ATTACKS[@]}
DONE=0
FAILED=()

for attack in "${ATTACKS[@]}"; do
  DONE=$(( DONE + 1 ))
  echo ""
  echo "========================================================="
  echo " [${DONE}/${TOTAL}]  attack=${attack}"
  echo "========================================================="

  # Patch the attack name in experiment.yaml in-place
  sed -i "s/^  name: .*/  name: ${attack}/" "${CONFIG}"

  python run_encoder.py --skip-training \
    && echo "[OK] encoder / ${attack}" \
    || { echo "[WARN] encoder / ${attack} failed — continuing."; FAILED+=("encoder/${attack}"); }

  python run_decoder.py --skip-training \
    && echo "[OK] decoder / ${attack}" \
    || { echo "[WARN] decoder / ${attack} failed — continuing."; FAILED+=("decoder/${attack}"); }

done

# ─── COMBINED REPORT ─────────────────────────────────────────────────────────
echo ""
echo "========================================================="
echo " Building combined_summary.csv"
echo "========================================================="
python codes/06_results_analysis.py

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
echo ""
echo "[DONE] Batch complete. ${#FAILED[@]} failure(s)."
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[FAILED RUNS]:"
  for entry in "${FAILED[@]}"; do
    echo "  - ${entry}"
  done
fi
echo "[DONE] Results under: ${REPO_ROOT}/results/"
