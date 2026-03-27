"""
06_results_analysis.py
Scans results/{dataset}/{attack}/{model}/summary.csv and prints a combined table.
Run after any attack:  python codes/06_results_analysis.py
"""

import csv
import json
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_all_summaries():
    rows = []
    for summary_path in sorted(RESULTS_DIR.glob("*/*/*/summary.csv")):
        # path: results/{dataset}/{attack}/{model}/summary.csv
        parts = summary_path.parts
        model   = parts[-2]
        attack  = parts[-3]
        dataset = parts[-4]
        try:
            with summary_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            print(f"[WARN] Could not read {summary_path}: {e}")
    return rows


def print_table(rows):
    if not rows:
        print("[INFO] No results found yet. Run python run.py first.")
        return

    cols = ["model", "attack", "dataset", "N",
            "original_accuracy", "accuracy_under_attack",
            "attack_success_rate", "avg_perturbed_word_pct", "avg_num_queries",
            "num_success", "num_fail", "num_skip"]

    # Build display rows
    display = []
    for r in rows:
        display.append({
            "model":                  r.get("model", ""),
            "attack":                 r.get("attack", ""),
            "dataset":                r.get("dataset", "").split("/")[-1],
            "N":                      r.get("N", ""),
            "original_accuracy":      f"{float(r.get('original_accuracy', 0))*100:.1f}%",
            "accuracy_under_attack":  f"{float(r.get('accuracy_under_attack', 0))*100:.1f}%",
            "attack_success_rate":    f"{float(r.get('attack_success_rate', 0))*100:.2f}%",
            "avg_perturbed_word_pct": f"{float(r.get('avg_perturbed_word_pct', 0)):.2f}%",
            "avg_num_queries":        f"{float(r.get('avg_num_queries', 0)):.1f}",
            "num_success":            r.get("num_success", ""),
            "num_fail":               r.get("num_fail", ""),
            "num_skip":               r.get("num_skip", ""),
        })

    # Column widths
    headers = {c: c for c in cols}
    widths  = {c: max(len(headers[c]), max(len(str(d[c])) for d in display)) for c in cols}

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    hdr = "| " + " | ".join(f"{c:<{widths[c]}}" for c in cols) + " |"

    print()
    print(sep)
    print(hdr)
    print(sep)
    for d in display:
        print("| " + " | ".join(f"{str(d[c]):<{widths[c]}}" for c in cols) + " |")
    print(sep)
    print(f"\n[INFO] Total runs: {len(display)}")
    print(f"[INFO] Results dir: {RESULTS_DIR}")


def save_combined_csv(rows):
    if not rows:
        return
    out = RESULTS_DIR / "combined_summary.csv"
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Combined summary saved to: {out}")


if __name__ == "__main__":
    rows = load_all_summaries()
    print_table(rows)
    save_combined_csv(rows)
