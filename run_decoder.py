# =============================================================================
# run_decoder.py — Full pipeline for decoder models (GPT-2, LLaMA, OPT, etc.)
# Usage:
#   python run_decoder.py                  # train + attack
#   python run_decoder.py --skip-training  # attack only (models already trained)
#
# Edit configs/experiment.yaml to change dataset, models, or attack.
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

import evaluate
import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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

DECODER_KEEP_COLS = {"input_ids", "attention_mask", "labels"}


# =============================================================================
# Shared helpers
# =============================================================================

def load_config():
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_label_mappings(dataset, label_field):
    labels = sorted(set(str(x) for x in dataset[label_field]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}
    return label2id, id2label


def get_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    logger.propagate = False
    return logger


def patch_textattack():
    import textattack
    _orig = textattack.search_methods.GreedyWordSwapWIR.__init__
    def _patched(self, *a, **kw):
        kw.pop("truncate_words_to", None)
        _orig(self, *a, **kw)
    textattack.search_methods.GreedyWordSwapWIR.__init__ = _patched


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


# =============================================================================
# PHASE 1 — Training
# =============================================================================

def train_one(arch, base_model, checkpoint, cfg):
    output_dir   = ROOT / checkpoint
    ckpt_dir     = ROOT / checkpoint.replace("_final", "_checkpoints")
    dataset_name = cfg["dataset"]["name"]
    text_field   = cfg["dataset"]["text_field"]
    label_field  = cfg["dataset"]["label_field"]
    epochs       = cfg["training"]["epochs"]
    batch_size   = cfg["training"]["batch_size"]
    max_length   = cfg["training"]["max_length"]
    seed         = cfg["training"]["seed"]
    dev_cfg      = cfg["training"]["device"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if dev_cfg == "auto" else torch.device(dev_cfg)

    print(f"\n[train] {arch} ({base_model})  →  {output_dir}  |  device: {device}")
    set_seed(seed)

    raw = load_dataset(dataset_name)

    # Decoder-specific: left-padding, pad token = eos token
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label2id, id2label = build_label_mappings(raw["train"], label_field)

    def tokenize(batch):
        return tokenizer(batch[text_field], padding="max_length",
                         truncation=True, max_length=max_length)

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.map(lambda ex: {"labels": label2id[str(ex[label_field])]})

    keep = [c for c in tokenized["train"].column_names if c in DECODER_KEEP_COLS]
    tokenized = DatasetDict({split: ds.select_columns(keep)
                             for split, ds in tokenized.items()})

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        load_in_4bit=True if "llama" in base_model.lower() else False,
        torch_dtype=torch.float16 if "llama" in base_model.lower() else torch.float32,
        device_map="auto" if "llama" in base_model.lower() else None
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return metric.compute(predictions=logits.argmax(axis=-1), references=labels)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    eval_results = trainer.evaluate()
    stats = {
        "model":                 arch,
        "base_model":            base_model,
        "dataset":               dataset_name,
        "seed":                  seed,
        "epochs":                epochs,
        "train_samples":         len(tokenized["train"]),
        "eval_samples":          len(tokenized["test"]),
        "final_train_loss":      round(train_result.training_loss, 4),
        "final_eval_accuracy":   round(eval_results.get("eval_accuracy", 0), 4),
        "final_eval_loss":       round(eval_results.get("eval_loss", 0), 4),
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 1),
    }
    with (output_dir / "training_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"[train] Saved model + stats to {output_dir}")
    print(f"[train] Final eval accuracy: {stats['final_eval_accuracy']*100:.2f}%")


# =============================================================================
# PHASE 2 — Attack
# =============================================================================

def attack_one(arch, ckpt_path, texts, labels, attack, attack_name,
               N, seed, max_len, device, dataset_name):

    out_dir = ROOT / "results" / dataset_name.split("/")[-1] / attack_name / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(out_dir / "run.log")

    log.info(f"\n{'='*60}")
    log.info(f"  Model : {arch}  |  Attack : {attack_name}  |  N : {N}")
    log.info(f"  Output: {out_dir}")
    log.info(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path,
        load_in_4bit=True if "llama" in arch.lower() else False,
        torch_dtype=torch.float16 if "llama" in arch.lower() else torch.float32,
        device_map="auto" if "llama" in arch.lower() else None
    ).eval()
    if not "llama" in arch.lower():
        model = model.to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    label2id = {str(k): int(v) for k, v in model.config.label2id.items()}

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

    with (out_dir / "attacks.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
        **summary,
        "avg_words_per_input": round(aw, 2),
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
    log.info(f"\n[attack] Results saved to: {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip fine-tuning and run attacks on existing checkpoints")
    args = parser.parse_args()

    cfg          = load_config()
    models       = cfg.get("decoder_models") or []
    dataset_name = cfg["dataset"]["name"]
    attack_name  = cfg["attack"]["name"]
    N            = cfg["attack"]["num_examples"]
    seed         = cfg["attack"]["seed"]
    max_len      = cfg["model"]["max_length"]
    dev_cfg      = cfg["model"]["device"]

    if not models:
        print("[run_decoder] No decoder_models defined (or all commented out) in experiment.yaml — nothing to do.")
        return

    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack '{attack_name}'. Options: {sorted(ATTACK_REGISTRY)}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if dev_cfg == "auto" else torch.device(dev_cfg)

    print(f"\n[run_decoder] dataset  = {dataset_name}")
    print(f"[run_decoder] attack   = {attack_name}  |  N = {N}  |  seed = {seed}")
    print(f"[run_decoder] device   = {device}")
    print(f"[run_decoder] models   = {[m['arch'] for m in models]}")
    print(f"[run_decoder] training = {'SKIPPED' if args.skip_training else 'enabled'}\n")

    # ------------------------------------------------------------------
    # PHASE 1 — Train
    # ------------------------------------------------------------------
    if not args.skip_training:
        print(f"\n{'='*60}\n  PHASE 1 — Training decoder models\n{'='*60}")
        for m in models:
            train_one(arch=m["arch"], base_model=m["base_model"],
                      checkpoint=m["checkpoint"], cfg=cfg)

    # ------------------------------------------------------------------
    # PHASE 2 — Attack
    # ------------------------------------------------------------------
    print(f"\n{'='*60}\n  PHASE 2 — Running {attack_name} attack\n{'='*60}")

    patch_textattack()
    ds     = load_dataset(dataset_name, split=cfg["dataset"]["split"]).select(range(min(N, 999999)))
    texts  = [str(r[cfg["dataset"]["text_field"]])  for r in ds]
    labels = [str(r[cfg["dataset"]["label_field"]]) for r in ds]
    mod, cls = ATTACK_REGISTRY[attack_name]

    for m in models:
        arch = m["arch"]
        ckpt = ROOT / m["checkpoint"]

        if not ckpt.exists():
            print(f"[WARN] Checkpoint not found for {arch}: {ckpt} — skipping.")
            continue

        _tokenizer = AutoTokenizer.from_pretrained(ckpt)
        _tokenizer.padding_side = "left"
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            load_in_4bit=True if "llama" in arch.lower() else False,
            torch_dtype=torch.float16 if "llama" in arch.lower() else torch.float32,
            device_map="auto" if "llama" in arch.lower() else None
        ).eval()
        if not "llama" in arch.lower():
            _model = _model.to(device)
        if _model.config.pad_token_id is None:
            _model.config.pad_token_id = _tokenizer.pad_token_id

        atk = getattr(importlib.import_module(mod), cls).build(
                  HuggingFaceModelWrapper(_model, _tokenizer))

        attack_one(arch=arch, ckpt_path=ckpt, texts=texts, labels=labels,
                   attack=atk, attack_name=attack_name,
                   N=N, seed=seed, max_len=max_len, device=device,
                   dataset_name=dataset_name)

    print(f"\n[run_decoder] All done. Results in: results/{dataset_name.split('/')[-1]}/{attack_name}/")


if __name__ == "__main__":
    main()
