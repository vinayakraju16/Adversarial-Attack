# =============================================================================
# run.py — Edit configs/experiment.yaml, then:  python run.py
# =============================================================================

import csv
import importlib
import json
import logging
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT   = Path(__file__).parent
CONFIG = ROOT / "configs" / "experiment.yaml"

ATTACK_REGISTRY = {
    "textfooler":      ("textattack.attack_recipes", "TextFoolerJin2019"),
    "textbugger":      ("textattack.attack_recipes", "TextBuggerLi2018"),
    "bertattack":      ("textattack.attack_recipes", "BERTAttackLi2020"),
    "bae":             ("textattack.attack_recipes", "BAEGarg2019"),
    "pwws":            ("textattack.attack_recipes", "PWWSRen2019"),
    "a2t":             ("textattack.attack_recipes", "A2TYoo2021"),
    "clare":           ("textattack.attack_recipes", "CLARELi2020"),
    "genetic":         ("textattack.attack_recipes", "GeneticAlgorithmAlzantot2018"),
    "faster_genetic":  ("textattack.attack_recipes", "FasterGeneticAlgorithmJia2019"),
    "pso":             ("textattack.attack_recipes", "PSOZang2020"),
    "hotflip":         ("textattack.attack_recipes", "HotFlipEbrahimi2018"),
    "deepwordbug":     ("textattack.attack_recipes", "DeepWordBugGao2018"),
    "pruthi":          ("textattack.attack_recipes", "Pruthi2019"),
    "morpheus":        ("textattack.attack_recipes", "MorpheusTan2020"),
    "input_reduction": ("textattack.attack_recipes", "InputReductionFeng2018"),
}


def load_config():
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def predict(model, tokenizer, text, device, max_length):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    idx = int(probs.argmax())
    return model.config.id2label[idx], float(probs[idx])


def pct_changed(orig, pert):
    if not pert:
        return None
    a, b = orig.split(), pert.split()
    if not a:
        return 0.0
    m = SequenceMatcher(a=a, b=b)
    changed = sum(max(i2-i1, j2-j1)
                  for tag, i1, i2, j1, j2 in m.get_opcodes() if tag != "equal")
    return 100.0 * changed / len(a)


def print_summary_box(log, s, f, sk, total, oa, aa, asr, ap, aw, aq):
    rows = [
        ("Number of successful attacks:", str(s)),
        ("Number of failed attacks:",     str(f)),
        ("Number of skipped attacks:",    str(sk)),
        ("Original accuracy:",            f"{oa*100:.1f}%"),
        ("Accuracy under attack:",        f"{aa*100:.1f}%"),
        ("Attack success rate:",          f"{asr*100:.2f}%"),
        ("Average perturbed word %:",     f"{ap:.2f}%"),
        ("Average num. words per input:", f"{aw:.2f}"),
        ("Avg num queries:",              f"{aq:.2f}"),
    ]
    w1 = max(len(r[0]) for r in rows)
    w2 = max(len(r[1]) for r in rows)
    sep = f"+{'-'*(w1+2)}+{'-'*(w2+2)}+"
    log.info(f"\n[Succeeded / Failed / Skipped / Total] {s} / {f} / {sk} / {total}\n")
    log.info(sep)
    log.info(f"| {'Attack Results':<{w1}} | {'':<{w2}} |")
    log.info(sep)
    for label, val in rows:
        log.info(f"| {label:<{w1}} | {val:<{w2}} |")
    log.info(sep)


def run_single(arch, ckpt_path, dataset, texts, labels, attack, attack_name,
               N, seed, max_len, device, dataset_name):
    """Run the attack for one model and save all results."""

    out_dir = ROOT / "results" / dataset_name.split("/")[-1] / attack_name / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(out_dir / "run.log")

    log.info(f"\n{'='*60}")
    log.info(f"  Model : {arch}  |  Attack : {attack_name}  |  N : {N}")
    log.info(f"  Output: {out_dir}")
    log.info(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model     = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device).eval()
    label2id  = {str(k): int(v) for k, v in model.config.label2id.items()}

    records = []
    for text, label in zip(texts, labels):
        orig_out, orig_score = predict(model, tokenizer, text, device, max_len)
        result = attack.attack(text, label2id[label])
        rtype  = ("successful" if isinstance(result, SuccessfulAttackResult)
                  else "skipped" if isinstance(result, SkippedAttackResult)
                  else "failed")
        pert_text = pert_out = pert_score = None
        pa = getattr(result, "perturbed_result", None)
        if pa:
            pert_text = str(pa.attacked_text.text)
            pert_out, pert_score = predict(model, tokenizer, pert_text, device, max_len)
        nq = getattr(result, "num_queries", None) \
             or getattr(getattr(result, "original_result", None), "num_queries", 0) or 0
        records.append({
            "original_text":       text,
            "perturbed_text":      pert_text,
            "original_score":      orig_score,
            "perturbed_score":     pert_score,
            "original_output":     orig_out,
            "perturbed_output":    pert_out,
            "ground_truth_output": label,
            "num_queries":         int(nq),
            "result_type":         rtype,
            "pct_words_changed":   pct_changed(text, pert_text),
        })

    # Save per-example JSONL
    with (out_dir / "attacks.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute metrics
    s    = sum(1 for r in records if r["result_type"] == "successful")
    fail = sum(1 for r in records if r["result_type"] == "failed")
    sk   = sum(1 for r in records if r["result_type"] == "skipped")
    tot  = len(records)
    oa   = sum(1 for r in records if r["original_output"] == r["ground_truth_output"]) / tot
    aa   = sum(1 for r in records if (r["perturbed_output"] or r["original_output"]) == r["ground_truth_output"]) / tot
    asr  = s / (s + fail) if (s + fail) else 0.0
    pcts = [r["pct_words_changed"] for r in records if r["pct_words_changed"] is not None]
    ap   = float(np.mean(pcts)) if pcts else 0.0
    aw   = float(np.mean([len(t.split()) for t in texts]))
    aq   = float(np.mean([r["num_queries"] for r in records]))

    summary = {
        "dataset": dataset_name, "model": arch, "attack": attack_name,
        "N": tot, "seed": seed,
        "num_success": s, "num_fail": fail, "num_skip": sk,
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

    stats = {
        "dataset":                   dataset_name,
        "model":                     arch,
        "attack":                    attack_name,
        "N":                         tot,
        "seed":                      seed,
        "num_success":               s,
        "num_fail":                  fail,
        "num_skip":                  sk,
        "original_accuracy":         round(oa,  4),
        "accuracy_under_attack":     round(aa,  4),
        "attack_success_rate":       round(asr, 4),
        "avg_perturbed_word_pct":    round(ap,  4),
        "avg_words_per_input":       round(aw,  2),
        "avg_num_queries":           round(aq,  2),
        "formatted": {
            "original_accuracy":      f"{oa*100:.1f}%",
            "accuracy_under_attack":  f"{aa*100:.1f}%",
            "attack_success_rate":    f"{asr*100:.2f}%",
            "avg_perturbed_word_pct": f"{ap:.2f}%",
            "avg_words_per_input":    f"{aw:.2f}",
            "avg_num_queries":        f"{aq:.2f}",
        }
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print_summary_box(log, s, fail, sk, tot, oa, aa, asr, ap, aw, aq)
    log.info(f"\n[run] Results saved to: {out_dir}")


def main():
    cfg         = load_config()
    dataset_name = cfg["dataset"]["name"]
    attack_name  = cfg["attack"]["name"]
    N            = cfg["attack"]["num_examples"]
    seed         = cfg["attack"]["seed"]
    max_len      = cfg["model"]["max_length"]
    dev_cfg      = cfg["model"]["device"]
    models       = cfg["models"]  # list of {arch, checkpoint}

    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack '{attack_name}'. Options: {sorted(ATTACK_REGISTRY)}")

    set_seed(seed)
    patch_textattack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if dev_cfg == "auto" else torch.device(dev_cfg)

    # Load dataset once — shared across all models
    ds     = load_dataset(dataset_name, split=cfg["dataset"]["split"]).select(range(min(N, 999999)))
    texts  = [str(r[cfg["dataset"]["text_field"]])  for r in ds]
    labels = [str(r[cfg["dataset"]["label_field"]]) for r in ds]

    # Build attack once — shared across all models
    mod, cls = ATTACK_REGISTRY[attack_name]

    print(f"\n[run] dataset = {dataset_name}")
    print(f"[run] attack  = {attack_name}  |  N = {N}  |  seed = {seed}")
    print(f"[run] device  = {device}")
    print(f"[run] running on {len(models)} model(s): {[m['arch'] for m in models]}\n")

    for m in models:
        arch = m["arch"]
        ckpt = ROOT / m["checkpoint"]

        if not ckpt.exists():
            print(f"[WARN] Checkpoint not found for {arch}: {ckpt} — skipping.")
            continue

        # Rebuild attack wrapper per model (each model needs its own wrapper)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(ckpt)
        _model     = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
        atk = getattr(importlib.import_module(mod), cls).build(
                  HuggingFaceModelWrapper(_model, _tokenizer))

        run_single(
            arch=arch, ckpt_path=ckpt,
            dataset=ds, texts=texts, labels=labels,
            attack=atk, attack_name=attack_name,
            N=N, seed=seed, max_len=max_len, device=device,
            dataset_name=dataset_name,
        )

    print(f"\n[run] All done. Results in: results/{dataset_name.split('/')[-1]}/{attack_name}/")


if __name__ == "__main__":
    main()
