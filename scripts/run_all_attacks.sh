#!/usr/bin/env bash
# =============================================================================
# run_all_attacks.sh
#
# Batch runner: iterates over all 12 word-level attacks × 2 model checkpoints.
# Runs each combination as a separate run_attack.py call so every combination
# gets its own JSONL + CSV under results/.
#
# Usage (on the GPU server, after the repo has been synced):
#   bash scripts/run_all_attacks.sh
#   bash scripts/run_all_attacks.sh --num-examples 100   (smaller debug run)
#
# Outputs per run:
#   results/attacks/<run_id>.jsonl
#   results/summary/<run_id>_summary.csv
# Combined report (after all runs):
#   results/combined_summary.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── SETTINGS ────────────────────────────────────────────────────────────────
NUM_EXAMPLES="${1:-200}"   # override via first positional arg or env var
SEED=42
CONFIG="${REPO_ROOT}/configs/experiment.yaml"

# 12 core word-level attacks (all in TextAttack)
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
)

# Models: arch_key -> checkpoint_dir (relative to repo root)
declare -A CHECKPOINTS
CHECKPOINTS["bert"]="models/tlink_bert_final"
CHECKPOINTS["roberta"]="models/tlink_roberta_final"
# To add another model, add one line here and it will be included automatically:
# CHECKPOINTS["distilbert"]="models/tlink_distilbert_final"

# ─── ACTIVATE VENV ───────────────────────────────────────────────────────────
VENV_PATH="${HOME}/.venvs/advnlp/bin/activate"
if [[ ! -f "${VENV_PATH}" ]]; then
  echo "[ERROR] Venv not found: ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}"
cd "${REPO_ROOT}"

# ─── BATCH LOOP ──────────────────────────────────────────────────────────────
TOTAL=$(( ${#ATTACKS[@]} * ${#CHECKPOINTS[@]} ))
DONE=0
FAILED=()

for arch in "${!CHECKPOINTS[@]}"; do
  checkpoint="${CHECKPOINTS[$arch]}"

  if [[ ! -d "${REPO_ROOT}/${checkpoint}" ]]; then
    echo "[WARN] Checkpoint not found for ${arch}: ${checkpoint} — skipping model."
    continue
  fi

  for attack in "${ATTACKS[@]}"; do
    DONE=$(( DONE + 1 ))
    echo ""
    echo "========================================================="
    echo " [${DONE}/${TOTAL}]  arch=${arch}  attack=${attack}"
    echo "========================================================="

    python run.py \
      --config "${CONFIG}" \
      --arch "${arch}" \
      --attack "${attack}" \
      --checkpoint "${checkpoint}" \
      --num-examples "${NUM_EXAMPLES}" \
      --seed "${SEED}" \
      && {
        echo "[OK] ${arch} / ${attack}"
        python codes/06_results_analysis.py
      } \
      || {
        echo "[WARN] ${arch} / ${attack} failed — logged and continuing."
        FAILED+=("${arch}/${attack}")
      }
  done
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
