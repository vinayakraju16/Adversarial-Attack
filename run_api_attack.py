# =============================================================================
# run_api_attack.py — Adversarial Attack Pipeline for OpenRouter API Models
#
# Attacks a cloud LLM (GPT-4o-mini, Gemma, etc.) as a black-box text
# classifier using TextAttack. No fine-tuning — the model is queried
# zero-shot via OpenRouter.
#
# Usage:
#   python run_api_attack.py                      # uses configs/api_attack.yaml
#   python run_api_attack.py --config configs/api_attack.yaml
#   python run_api_attack.py --skip-prompt-test   # skip the sanity check API call
#
# Results saved to: results/{dataset}/{attack}/{model}/
#   attacks.jsonl  — per-example records
#   summary.csv    — attack metrics
#   stats.json     — full metrics incl. F1 / precision / recall (before & after)
#   run.log        — full console log
# =============================================================================

import argparse
import csv
import importlib
import json
import logging
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from codes.openrouter_wrapper import OpenRouterWrapper

# Black-box attacks only (no gradients required)
ATTACK_REGISTRY = {
    "textfooler":      ("textattack.attack_recipes", "TextFoolerJin2019"),
    "textbugger":      ("textattack.attack_recipes", "TextBuggerLi2018"),
    "bae":             ("textattack.attack_recipes", "BAEGarg2019"),
    "pwws":            ("textattack.attack_recipes", "PWWSRen2019"),
    "a2t":             ("textattack.attack_recipes", "A2TYoo2021"),
    "clare":           ("textattack.attack_recipes", "CLARELi2020"),
    "genetic":         ("textattack.attack_recipes", "GeneticAlgorithmAlzantot2018"),
    "faster_genetic":  ("textattack.attack_recipes", "FasterGeneticAlgorithmJia2019"),
    "pso":             ("textattack.attack_recipes", "PSOZang2020"),
    "deepwordbug":     ("textattack.attack_recipes", "DeepWordBugGao2018"),
    "pruthi":          ("textattack.attack_recipes", "Pruthi2019"),
    "morpheus":        ("textattack.attack_recipes", "MorpheusTan2020"),
    "input_reduction": ("textattack.attack_recipes", "InputReductionFeng2018"),
}


# =============================================================================
# Helpers
# =============================================================================

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def patch_textattack():
    import textattack
    _orig = textattack.search_methods.GreedyWordSwapWIR.__init__
    def _patched(self, *a, **kw):
        kw.pop("truncate_words_to", None)
        _orig(self, *a, **kw)
    textattack.search_methods.GreedyWordSwapWIR.__init__ = _patched


def get_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logger.propagate = False
    return logger


def build_label_mappings(dataset, label_field):
    labels = sorted(set(str(x) for x in dataset[label_field]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    return label2id, id2label


def pct_changed(orig, pert):
    if not pert:
        return None
    a, b = orig.split(), pert.split()
    if not a:
        return 0.0
    m = SequenceMatcher(a=a, b=b)
    changed = sum(max(i2 - i1, j2 - j1)
                  for tag, i1, i2, j1, j2 in m.get_opcodes() if tag != "equal")
    return 100.0 * changed / len(a)


# =============================================================================
# Phase — Attack
# =============================================================================

def attack_one(arch, wrapper, texts, labels, attack, attack_name,
               N, seed, dataset_name, log):

    out_dir = ROOT / "results" / dataset_name.split("/")[-1] / attack_name / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"  Model  : {arch}  |  Attack : {attack_name}  |  N : {N}")
    log.info(f"  Output : {out_dir}")
    log.info(f"{'='*60}")

    records = []

    for i, (text, label) in enumerate(zip(texts, labels)):
        label_id = wrapper.label2id.get(label, 0)
        result   = attack.attack(text, label_id)

        rtype = ("successful" if isinstance(result, SuccessfulAttackResult)
                 else "skipped" if isinstance(result, SkippedAttackResult)
                 else "failed")

        # Get original and perturbed predictions from the result object
        # (avoids extra API calls — TextAttack already queried the wrapper)
        orig_label_id = int(result.original_result.output)
        orig_pred     = wrapper.id2label.get(orig_label_id, str(orig_label_id))

        pert_text = pert_pred = None
        pa = getattr(result, "perturbed_result", None)
        if pa:
            pert_text      = str(pa.attacked_text.text)
            pert_label_id  = int(pa.output)
            pert_pred      = wrapper.id2label.get(pert_label_id, str(pert_label_id))

        nq = getattr(result, "num_queries", 0) or 0

        records.append({
            "original_text":       text,
            "perturbed_text":      pert_text,
            "original_output":     orig_pred,
            "perturbed_output":    pert_pred,
            "ground_truth_output": label,
            "num_queries":         int(nq),
            "result_type":         rtype,
            "pct_words_changed":   pct_changed(text, pert_text),
        })

        if (i + 1) % 5 == 0:
            correct = sum(1 for r in records if r["original_output"] == r["ground_truth_output"])
            log.info(f"  Progress: {i+1}/{len(texts)} | "
                     f"Acc so far: {correct/(i+1):.3f} | "
                     f"Total API calls: {wrapper.total_queries}")

    # ------------------------------------------------------------------
    # Save raw records
    # ------------------------------------------------------------------
    with (out_dir / "attacks.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Compute attack metrics
    # ------------------------------------------------------------------
    s    = sum(1 for r in records if r["result_type"] == "successful")
    fail = sum(1 for r in records if r["result_type"] == "failed")
    sk   = sum(1 for r in records if r["result_type"] == "skipped")
    tot  = len(records)

    gold      = [r["ground_truth_output"] for r in records]
    pred_orig = [r["original_output"] for r in records]
    pred_atk  = [r["perturbed_output"] or r["original_output"] for r in records]

    oa  = accuracy_score(gold, pred_orig)
    aa  = accuracy_score(gold, pred_atk)
    asr = s / (s + fail) if (s + fail) else 0.0

    pcts = [r["pct_words_changed"] for r in records if r["pct_words_changed"] is not None]
    ap   = float(np.mean(pcts)) if pcts else 0.0
    aw   = float(np.mean([len(r["original_text"].split()) for r in records]))
    aq   = float(np.mean([r["num_queries"] for r in records]))

    # ------------------------------------------------------------------
    # F1 / precision / recall — before attack and under attack
    # ------------------------------------------------------------------
    label_set = sorted(set(gold))

    def clf_metrics(y_true, y_pred):
        return {
            "accuracy":         round(accuracy_score(y_true, y_pred), 4),
            "f1_macro":         round(f1_score(y_true, y_pred, average="macro",
                                               labels=label_set, zero_division=0), 4),
            "f1_micro":         round(f1_score(y_true, y_pred, average="micro",
                                               labels=label_set, zero_division=0), 4),
            "f1_weighted":      round(f1_score(y_true, y_pred, average="weighted",
                                               labels=label_set, zero_division=0), 4),
            "precision_macro":  round(precision_score(y_true, y_pred, average="macro",
                                                       labels=label_set, zero_division=0), 4),
            "recall_macro":     round(recall_score(y_true, y_pred, average="macro",
                                                    labels=label_set, zero_division=0), 4),
            "per_class":        classification_report(y_true, y_pred,
                                                      labels=label_set,
                                                      zero_division=0,
                                                      output_dict=True),
        }

    metrics_before = clf_metrics(gold, pred_orig)
    metrics_after  = clf_metrics(gold, pred_atk)

    # ------------------------------------------------------------------
    # Save summary.csv  (compatible with 06_results_analysis.py)
    # ------------------------------------------------------------------
    summary = {
        "dataset":                dataset_name,
        "model":                  arch,
        "attack":                 attack_name,
        "N":                      tot,
        "seed":                   seed,
        "num_success":            s,
        "num_fail":               fail,
        "num_skip":               sk,
        "original_accuracy":      round(oa,  4),
        "accuracy_under_attack":  round(aa,  4),
        "attack_success_rate":    round(asr, 4),
        "avg_perturbed_word_pct": round(ap,  4),
        "avg_num_queries":        round(aq,  2),
    }
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # ------------------------------------------------------------------
    # Save stats.json  (full metrics including F1 / precision / recall)
    # ------------------------------------------------------------------
    stats = {
        **summary,
        "avg_words_per_input":   round(aw, 2),
        "total_api_calls":       wrapper.total_queries,
        "metrics_before_attack": metrics_before,
        "metrics_after_attack":  metrics_after,
        "formatted": {
            "original_accuracy":        f"{oa*100:.1f}%",
            "accuracy_under_attack":    f"{aa*100:.1f}%",
            "attack_success_rate":      f"{asr*100:.2f}%",
            "avg_perturbed_word_pct":   f"{ap:.2f}%",
            "avg_words_per_input":      f"{aw:.2f}",
            "avg_num_queries":          f"{aq:.2f}",
            "f1_macro_before":          f"{metrics_before['f1_macro']:.4f}",
            "f1_macro_after":           f"{metrics_after['f1_macro']:.4f}",
            "precision_macro_before":   f"{metrics_before['precision_macro']:.4f}",
            "precision_macro_after":    f"{metrics_after['precision_macro']:.4f}",
            "recall_macro_before":      f"{metrics_before['recall_macro']:.4f}",
            "recall_macro_after":       f"{metrics_after['recall_macro']:.4f}",
        },
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    rows = [
        ("Succeeded / Failed / Skipped:",  f"{s} / {fail} / {sk} / {tot}"),
        ("Original accuracy:",             f"{oa*100:.1f}%"),
        ("Accuracy under attack:",         f"{aa*100:.1f}%"),
        ("Attack success rate:",           f"{asr*100:.2f}%"),
        ("─── Before attack ───",          ""),
        ("  F1 macro:",                    f"{metrics_before['f1_macro']:.4f}"),
        ("  Precision macro:",             f"{metrics_before['precision_macro']:.4f}"),
        ("  Recall macro:",                f"{metrics_before['recall_macro']:.4f}"),
        ("─── Under attack ───",           ""),
        ("  F1 macro:",                    f"{metrics_after['f1_macro']:.4f}"),
        ("  Precision macro:",             f"{metrics_after['precision_macro']:.4f}"),
        ("  Recall macro:",                f"{metrics_after['recall_macro']:.4f}"),
        ("Avg words changed %:",           f"{ap:.2f}%"),
        ("Avg API queries / example:",     f"{aq:.2f}"),
        ("Total API calls:",               str(wrapper.total_queries)),
    ]
    w1  = max(len(r[0]) for r in rows)
    w2  = max(len(r[1]) for r in rows)
    sep = f"+{'-'*(w1+2)}+{'-'*(w2+2)}+"
    log.info(f"\n{sep}")
    log.info(f"| {'Metric':<{w1}} | {'Value':<{w2}} |")
    log.info(sep)
    for lbl, val in rows:
        log.info(f"| {lbl:<{w1}} | {val:<{w2}} |")
    log.info(sep)
    log.info(f"\n[attack] Results saved to: {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/api_attack.yaml")
    parser.add_argument("--skip-prompt-test", action="store_true",
                        help="Skip the one-example sanity check API call")
    args = parser.parse_args()

    api_cfg = load_config(args.config)
    exp_cfg = load_config(ROOT / "configs" / "experiment.yaml")

    dataset_name = exp_cfg["dataset"]["name"]
    text_field   = exp_cfg["dataset"]["text_field"]
    label_field  = exp_cfg["dataset"]["label_field"]

    attack_name  = api_cfg["attack"]["name"]
    N            = api_cfg["attack"]["num_examples"]
    seed         = api_cfg["attack"].get("seed", 42)
    models       = api_cfg.get("api_models") or []
    api_key      = api_cfg["openrouter"]["api_key"]
    base_url     = api_cfg["openrouter"]["base_url"]
    call_delay   = api_cfg.get("call_delay", 1.5)
    max_retries  = api_cfg.get("max_retries", 5)

    if not models:
        print("[run_api_attack] No api_models defined in api_attack.yaml — nothing to do.")
        return

    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack '{attack_name}'. "
            f"Supported: {sorted(ATTACK_REGISTRY)}"
        )

    set_seed(seed)
    patch_textattack()

    # Load dataset
    print(f"\n[run_api_attack] Loading dataset: {dataset_name}")
    ds_full          = load_dataset(dataset_name, split=exp_cfg["dataset"]["split"])
    label2id, id2label = build_label_mappings(ds_full, label_field)

    ds     = ds_full.select(range(min(N, len(ds_full))))
    texts  = [str(r[text_field])  for r in ds]
    labels = [str(r[label_field]) for r in ds]

    # Cost estimate
    avg_words      = sum(len(t.split()) for t in texts[:20]) / max(1, min(20, len(texts)))
    approx_queries = int(N * avg_words * 3)
    est_hours      = approx_queries * call_delay / 3600
    est_cost_usd   = approx_queries * 100 * 0.15 / 1_000_000   # rough at 100 tok/call

    print(f"\n[run_api_attack] dataset  = {dataset_name}")
    print(f"[run_api_attack] attack   = {attack_name}  |  N = {N}  |  seed = {seed}")
    print(f"[run_api_attack] labels   = {list(label2id.keys())}")
    print(f"[run_api_attack] models   = {[m['arch'] for m in models]}")
    print(f"\n[COST ESTIMATE]  ~{approx_queries:,} API calls")
    print(f"[COST ESTIMATE]  ~{est_hours:.1f} hours at {call_delay}s/call")
    print(f"[COST ESTIMATE]  ~${est_cost_usd:.2f} for GPT-4o-mini (rough)")

    mod, cls = ATTACK_REGISTRY[attack_name]

    for m_cfg in models:
        arch     = m_cfg["arch"]
        model_id = m_cfg["model_id"]

        # Build prompt — fill {labels} placeholder
        labels_str = ", ".join(label2id.keys())
        raw_tpl    = api_cfg.get("prompt_template", "").strip()
        if not raw_tpl:
            raw_tpl = (
                "You are a temporal relation classifier.\n"
                "Classify the temporal relation in the sentence.\n"
                f"Valid labels: {labels_str}\n"
                "Respond with ONLY one label. No explanation.\n\n"
                "Sentence: {text}\nLabel:"
            )
        prompt_template = raw_tpl.replace("{labels}", labels_str)

        print(f"\n{'='*60}")
        print(f"  Model : {arch}  ({model_id})")
        print(f"  Attack: {attack_name}")
        print(f"{'='*60}")

        wrapper = OpenRouterWrapper(
            model_id        = model_id,
            api_key         = api_key,
            base_url        = base_url,
            label2id        = label2id,
            id2label        = id2label,
            prompt_template = prompt_template,
            call_delay      = call_delay,
            max_retries     = max_retries,
        )

        # ------------------------------------------------------------------
        # Sanity check — one example before spending budget on full run
        # ------------------------------------------------------------------
        if not args.skip_prompt_test:
            print("\n[sanity check] Testing prompt on first example...")
            test_pred = wrapper.predict_label(texts[0])
            print(f"  Input : {texts[0][:100]}")
            print(f"  Gold  : {labels[0]}")
            print(f"  Pred  : {test_pred}")
            if test_pred not in label2id:
                print(f"\n  [WARN] Response '{test_pred}' is not in the label set.")
                print(f"  [WARN] Labels are: {list(label2id.keys())}")
                print(f"  [WARN] Edit prompt_template in api_attack.yaml to fix this.")
                print(f"  [WARN] Continuing anyway — results may be unreliable.\n")

        # ------------------------------------------------------------------
        # Build attack and run
        # ------------------------------------------------------------------
        atk = getattr(importlib.import_module(mod), cls).build(wrapper)

        out_dir = ROOT / "results" / dataset_name.split("/")[-1] / attack_name / arch
        log     = get_logger(out_dir / "run.log")

        attack_one(
            arch        = arch,
            wrapper     = wrapper,
            texts       = texts,
            labels      = labels,
            attack      = atk,
            attack_name = attack_name,
            N           = N,
            seed        = seed,
            dataset_name= dataset_name,
            log         = log,
        )

    print(f"\n[run_api_attack] All done.")
    print(f"[run_api_attack] Results in: results/{dataset_name.split('/')[-1]}/{attack_name}/")


if __name__ == "__main__":
    main()
