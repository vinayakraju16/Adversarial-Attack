"""
run_causal_eval.py — Causal Reasoning Evaluation Pipeline

Evaluates LLMs on causal reasoning datasets via OpenRouter API.
Three prompting strategies:
  - inductive  : few-shot examples, model infers the pattern
  - deductive  : task description only, model applies rules
  - abductive  : chain-of-thought, model explains its reasoning

Usage:
    python run_causal_eval.py                          # all enabled datasets & models
    python run_causal_eval.py --datasets bigbench_cj   # single dataset
    python run_causal_eval.py --models gpt4o-mini      # single model
    python run_causal_eval.py --strategy inductive     # single strategy
    python run_causal_eval.py --dry-run                # print prompts, no API calls
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/causal_eval.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_hf_dataset(ds_cfg: dict, max_examples: int, seed: int):
    """Load a HuggingFace dataset and return a list of dicts."""
    kwargs = {}
    if ds_cfg.get("hf_subset"):
        kwargs["name"] = ds_cfg["hf_subset"]

    # Use explicit split if provided, otherwise try standard splits
    if ds_cfg.get("hf_split"):
        ds = load_dataset(ds_cfg["hf_path"], split=ds_cfg["hf_split"], **kwargs)
    else:
        for split in ["test", "validation", "train"]:
            try:
                ds = load_dataset(ds_cfg["hf_path"], split=split, **kwargs)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not load any split for {ds_cfg['hf_path']}")

    # Convert to list of dicts
    records = [dict(row) for row in ds]

    # Shuffle and limit
    random.seed(seed)
    random.shuffle(records)
    if max_examples and max_examples < len(records):
        records = records[:max_examples]

    return records


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_example(record: dict, ds_cfg: dict, include_answer: bool = True) -> str:
    """Format a single record into a prompt line."""
    text = record.get(ds_cfg["text_field"], "")
    if ds_cfg.get("hypothesis_field"):
        hypothesis = record.get(ds_cfg["hypothesis_field"], "")
        text = f"Premise: {text}\nHypothesis: {hypothesis}"
    label = record.get(ds_cfg["label_field"], "")
    if include_answer:
        return f"Input: {text}\nAnswer: {label}"
    else:
        return f"Input: {text}\nAnswer:"


def build_inductive_prompt(record: dict, few_shot_examples: list, ds_cfg: dict) -> str:
    """Few-shot prompt — show k examples, then ask the question."""
    lines = [
        f"Task: {ds_cfg['description']}",
        "",
        "Here are some examples:",
        "",
    ]
    for ex in few_shot_examples:
        lines.append(_format_example(ex, ds_cfg, include_answer=True))
        lines.append("")
    lines.append("Now answer the following:")
    lines.append("")
    lines.append(_format_example(record, ds_cfg, include_answer=False))
    return "\n".join(lines)


def build_deductive_prompt(record: dict, ds_cfg: dict) -> str:
    """Description-only prompt — no examples, just the task definition."""
    label_map = ds_cfg.get("label_map", {})
    valid_labels = list(label_map.keys()) if label_map else ["0", "1"]

    lines = [
        f"Task: {ds_cfg['description']}",
        f"Respond with ONLY one of these exact values: {valid_labels}",
        "Do not include any explanation.",
        "",
        _format_example(record, ds_cfg, include_answer=False),
    ]
    return "\n".join(lines)


def build_abductive_prompt(record: dict, ds_cfg: dict) -> str:
    """Chain-of-thought prompt — ask model to reason step by step."""
    label_map = ds_cfg.get("label_map", {})
    valid_labels = list(label_map.keys()) if label_map else ["0", "1"]

    lines = [
        f"Task: {ds_cfg['description']}",
        "",
        _format_example(record, ds_cfg, include_answer=False),
        "",
        "Think step by step, then give your final answer on the LAST line in this exact format:",
        f"Final Answer: <one of {valid_labels}>",
    ]
    return "\n".join(lines)


PROMPT_BUILDERS = {
    "inductive": build_inductive_prompt,
    "deductive": build_deductive_prompt,
    "abductive": build_abductive_prompt,
}


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

def call_openrouter(prompt: str, model_id: str, api_key: str, base_url: str,
                    max_retries: int = 3, timeout: int = 60) -> str:
    """Send a prompt to OpenRouter and return the response text."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/thesis-adversarial-nlp",
        "X-Title": "Causal Reasoning Eval",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Check for error in response body (some providers return 200 with error)
            if "error" in data:
                raise ValueError(f"API error: {data['error']}")
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if resp.status_code in (429, 503):
                wait = min(60, 5 * (2 ** attempt))
                print(f"  [RATE LIMIT {resp.status_code}] Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif resp.status_code == 402:
                raise RuntimeError(
                    "Insufficient OpenRouter credits. Add credits at https://openrouter.ai/settings/credits"
                )
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                raise
    return ""


# ---------------------------------------------------------------------------
# Answer parser
# ---------------------------------------------------------------------------

def parse_answer(response: str, ds_cfg: dict, strategy: str) -> str:
    """Extract the predicted label from the model response."""
    response = response.strip()

    if strategy == "abductive":
        # Look for "Final Answer: <value>" on the last non-empty line
        match = re.search(r"final\s+answer\s*:\s*(.+)", response, re.IGNORECASE)
        if match:
            response = match.group(1).strip()
        else:
            # Fall back to last line
            lines = [l.strip() for l in response.splitlines() if l.strip()]
            response = lines[-1] if lines else response

    label_map = ds_cfg.get("label_map", {})
    if not label_map:
        return response.strip().lower()

    # Exact match first (case-insensitive)
    for label_str in label_map:
        if str(label_str).lower() == response.lower():
            return str(label_map[label_str])

    # Partial match fallback
    for label_str in label_map:
        if str(label_str).lower() in response.lower():
            return str(label_map[label_str])

    # Unknown
    return response.strip()


def get_gold_label(record: dict, ds_cfg: dict) -> str:
    """Get the normalized gold label from a record."""
    raw = record.get(ds_cfg["label_field"], "")
    label_map = ds_cfg.get("label_map", {})
    if not label_map:
        return str(raw).strip().lower()
    for k, v in label_map.items():
        if str(k).lower() == str(raw).strip().lower() or str(k) == str(raw):
            return str(v)
    return str(raw)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gold: list, pred: list) -> dict:
    """Compute classification metrics."""
    # Align to common label set
    labels = sorted(set(gold + pred))
    try:
        metrics = {
            "accuracy": round(accuracy_score(gold, pred), 4),
            "f1_macro": round(f1_score(gold, pred, average="macro", labels=labels, zero_division=0), 4),
            "f1_micro": round(f1_score(gold, pred, average="micro", labels=labels, zero_division=0), 4),
            "precision_macro": round(precision_score(gold, pred, average="macro", labels=labels, zero_division=0), 4),
            "recall_macro": round(recall_score(gold, pred, average="macro", labels=labels, zero_division=0), 4),
        }
    except Exception as e:
        metrics = {"error": str(e)}
    return metrics


# ---------------------------------------------------------------------------
# Run one (dataset × model × strategy) combination
# ---------------------------------------------------------------------------

def run_eval(
    ds_cfg: dict,
    model_cfg: dict,
    strategy: str,
    records: list,
    few_shot_pool: list,
    cfg: dict,
    output_dir: Path,
    dry_run: bool = False,
):
    model_name = model_cfg["name"]
    ds_name = ds_cfg["name"]
    k = cfg.get("few_shot_k", 5)

    print(f"\n{'='*70}")
    print(f"  Dataset : {ds_name}  |  Model : {model_name}  |  Strategy : {strategy}")
    print(f"  Examples: {len(records)}  |  Few-shot k: {k if strategy == 'inductive' else 'N/A'}")
    print(f"{'='*70}")

    results = []
    gold_labels = []
    pred_labels = []
    skipped = 0

    for i, record in enumerate(records):
        # Build few-shot pool (exclude current example)
        if strategy == "inductive":
            pool = [r for r in few_shot_pool if r is not record]
            shots = random.sample(pool, min(k, len(pool)))
            prompt = build_inductive_prompt(record, shots, ds_cfg)
        elif strategy == "deductive":
            prompt = build_deductive_prompt(record, ds_cfg)
        else:  # abductive
            prompt = build_abductive_prompt(record, ds_cfg)

        gold = get_gold_label(record, ds_cfg)

        if dry_run:
            if i == 0:
                print(f"\n[DRY RUN] Sample prompt for record 0:\n{'-'*50}\n{prompt}\n{'-'*50}")
            pred = gold  # perfect score in dry run
            raw_response = "[dry-run]"
        else:
            try:
                raw_response = call_openrouter(
                    prompt,
                    model_cfg["id"],
                    cfg["openrouter"]["api_key"],
                    cfg["openrouter"]["base_url"],
                )
                pred = parse_answer(raw_response, ds_cfg, strategy)
            except Exception as e:
                print(f"  [ERROR] record {i}: {e}")
                raw_response = f"ERROR: {e}"
                pred = "error"
                skipped += 1

        gold_labels.append(gold)
        pred_labels.append(pred)

        results.append({
            "index": i,
            "gold": gold,
            "pred": pred,
            "correct": gold == pred,
            "raw_response": raw_response,
            "text": record.get(ds_cfg["text_field"], ""),
        })

        if (i + 1) % 10 == 0:
            correct_so_far = sum(1 for r in results if r["correct"])
            print(f"  Progress: {i+1}/{len(records)} | Acc so far: {correct_so_far/(i+1):.3f}")

        # Delay to respect rate limits (free models need more breathing room)
        if not dry_run:
            time.sleep(1.0)

    # Compute metrics
    valid_gold = [g for g, p in zip(gold_labels, pred_labels) if p != "error"]
    valid_pred = [p for g, p in zip(gold_labels, pred_labels) if p != "error"]
    metrics = compute_metrics(valid_gold, valid_pred) if valid_pred else {}

    summary = {
        "dataset": ds_name,
        "level": ds_cfg["level"],
        "model": model_name,
        "model_id": model_cfg["id"],
        "strategy": strategy,
        "n_total": len(records),
        "n_skipped": skipped,
        "n_evaluated": len(valid_pred),
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }

    # Print results table
    print(f"\n  Results for {ds_name} / {model_name} / {strategy}:")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*36}")
    for k_m, v in metrics.items():
        print(f"  {k_m:<25} {v:>10.4f}" if isinstance(v, float) else f"  {k_m:<25} {str(v):>10}")
    print(f"  {'n_evaluated':<25} {summary['n_evaluated']:>10}")
    print(f"  {'n_skipped':<25} {summary['n_skipped']:>10}")

    # Save outputs
    out_dir = output_dir / ds_name / model_name / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.jsonl", "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved to: {out_dir}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Causal Reasoning Evaluation Pipeline")
    parser.add_argument("--config", default="configs/causal_eval.yaml")
    parser.add_argument("--datasets", nargs="+", help="Dataset names to run (default: all enabled)")
    parser.add_argument("--models", nargs="+", help="Model names to run (default: all)")
    parser.add_argument("--strategy", choices=["inductive", "deductive", "abductive"],
                        help="Single strategy (default: all)")
    parser.add_argument("--max-examples", type=int, help="Override max_examples from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts and print sample — no API calls")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg.get("seed", 42))

    max_examples = args.max_examples or cfg.get("max_examples", 200)
    results_dir = Path(cfg.get("results_dir", "results/causal_eval"))
    strategies = [args.strategy] if args.strategy else cfg.get("strategies", ["inductive", "deductive", "abductive"])

    # Filter datasets
    all_datasets = [d for d in cfg["datasets"] if d.get("enabled", False)]
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]

    # Filter models
    all_models = cfg["openrouter"]["models"]
    if args.models:
        all_models = [m for m in all_models if m["name"] in args.models]

    if not all_datasets:
        print("[ERROR] No enabled datasets found. Check causal_eval.yaml — set enabled: true.")
        sys.exit(1)

    if not all_models:
        print("[ERROR] No models selected.")
        sys.exit(1)

    print(f"\nCausal Reasoning Evaluation Pipeline")
    print(f"  Datasets  : {[d['name'] for d in all_datasets]}")
    print(f"  Models    : {[m['name'] for m in all_models]}")
    print(f"  Strategies: {strategies}")
    print(f"  Max ex.   : {max_examples}")
    print(f"  Dry run   : {args.dry_run}")

    all_summaries = []

    for ds_cfg in all_datasets:
        print(f"\nLoading dataset: {ds_cfg['name']} ({ds_cfg['hf_path']})...")
        try:
            records = load_hf_dataset(ds_cfg, max_examples, cfg.get("seed", 42))
            print(f"  Loaded {len(records)} examples.")
        except Exception as e:
            print(f"  [ERROR] Could not load {ds_cfg['name']}: {e}")
            continue

        # Separate few-shot pool (use first 20% of full dataset for consistency)
        few_shot_pool = records[: max(20, len(records) // 5)]

        for model_cfg in all_models:
            for strategy in strategies:
                summary = run_eval(
                    ds_cfg=ds_cfg,
                    model_cfg=model_cfg,
                    strategy=strategy,
                    records=records,
                    few_shot_pool=few_shot_pool,
                    cfg=cfg,
                    output_dir=results_dir,
                    dry_run=args.dry_run,
                )
                all_summaries.append(summary)

    # Save combined summary
    if all_summaries:
        results_dir.mkdir(parents=True, exist_ok=True)
        combined_path = results_dir / "all_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\n\nAll results saved to: {combined_path}")

        # Print final summary table
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<18} {'Model':<14} {'Strategy':<12} {'Acc':>7} {'F1 Mac':>8} {'N':>5}")
        print(f"{'-'*80}")
        for s in all_summaries:
            m = s.get("metrics", {})
            acc = f"{m.get('accuracy', 0):.4f}" if "accuracy" in m else "  N/A "
            f1 = f"{m.get('f1_macro', 0):.4f}" if "f1_macro" in m else "  N/A "
            print(f"{s['dataset']:<18} {s['model']:<14} {s['strategy']:<12} {acc:>7} {f1:>8} {s['n_evaluated']:>5}")


if __name__ == "__main__":
    main()
