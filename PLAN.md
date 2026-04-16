# Thesis Work Plan — Adversarial Robustness & Causal Reasoning

**Last updated:** 2026-04-09

---

## Part A: Adversarial Attacks on NLP Models

### A1. Run Remaining 14 Attacks on Encoder Models (BERT + RoBERTa)
- **Status:** Pending (only PWWS done so far)
- **How:**
  1. In `configs/experiment.yaml`, change `attack.name` to next attack (e.g., `textfooler`)
  2. Run `scripts/run_all_encoder_remote.ps1` from local PowerShell
  3. Results auto-save to `results/{dataset}/{attack}/{model}/`
  4. Repeat for all 14 remaining attacks:
     `textfooler`, `textbugger`, `bertattack`, `bae`, `a2t`, `clare`, `genetic`, `faster_genetic`, `pso`, `hotflip`, `deepwordbug`, `pruthi`, `morpheus`, `input_reduction`
- **Dependencies:** GPU server access
- **Estimated runs:** 14 attacks x 2 models = 28 runs

### A2. Run All 15 Attacks on GPT-2
- **Status:** Pending (only PWWS done so far)
- **How:**
  1. Same as A1 but use `scripts/run_all_decoder_remote.ps1`
  2. GPT-2 is already fine-tuned (`models/tlink_gpt2_final`)
- **Dependencies:** GPU server access
- **Estimated runs:** 14 remaining attacks x 1 model = 14 runs

### A3. LLaMA Setup and Attack Runs
- **Status:** Blocked
- **What's needed:**
  1. Get fine-tuned LLaMA checkpoint from Sagni
  2. Add 4-bit quantization (bitsandbytes/QLoRA) to `run_decoder.py` for LLaMA support on 16GB GPU
  3. Test LLaMA inference on GPU server
  4. Run all 15 attacks on LLaMA
- **Dependencies:** Sagni's checkpoint, bitsandbytes working on GPU server
- **Note:** LLaMA-2-7B needs ~14GB at FP16, ~4-5GB at 4-bit quantized

### A4. Analysis and Comparison
- **Status:** Pending (waiting on attack results)
- **Deliverables:**
  1. Attack Success Rate (ASR) table across all 15 attacks x 4 models
  2. Per-attack vulnerability analysis (which attacks are strongest?)
  3. Encoder vs decoder robustness comparison
  4. Statistical significance tests (paired t-test using raw outputs from `attacks.jsonl`)

---

## Part B: Causal Reasoning Evaluation

### B1. Current Evaluation Run (In Progress)
- **Status:** Running in background
- **Dataset:** BigBench Causal Judgement (187 examples)
- **Models:** Gemma 3 12B (free), Gemma 3 27B (free)
- **Strategies:** Inductive, deductive, abductive
- **Partial result:** GPT-4o-mini inductive = 65% accuracy on 103/187 examples

### B2. CLadder Dataset Evaluation
- **Status:** Ready to run
- **How:**
  1. CLadder is already enabled in `configs/causal_eval.yaml`
  2. Set `max_examples: 200` first (full dataset is 10K)
  3. Run: `python run_causal_eval.py --datasets cladder`
- **Dependencies:** B1 completing first (avoid API rate limit stacking)

### B3. Download and Push Remaining 9 Datasets to HuggingFace
- **Status:** Pending
- **Datasets to push under `mdg-nlp` org:**

  | Dataset | Source | Size | Notes |
  |---------|--------|------|-------|
  | NatQuest | kaist-ai/NatQuest | 13.5K | Need to verify HF availability |
  | e-CARE | Causal-Reasoning/e-CARE | 20K | Need to verify HF availability |
  | FinCausal T1 | SemEval/FinCausal | unknown | Binary: has causal relation? |
  | FinCausal T2 | SemEval/FinCausal | unknown | Span extraction — different task type |
  | CounterBench | — | unknown | Need to find source |
  | BigBench CU | lukaemon/bbh (causal_understanding) | 187 | Already on HF, just verify subset name |
  | AC-Reason | — | 1K | Need to find source |
  | corr2cause | causalnlp/corr2cause | 208K | Already on HF, verify structure |
  | CausalBench | — | 60K | Need to find source |

- **Steps per dataset:**
  1. Find/download source data
  2. Standardize format (text, label columns)
  3. Create train/val/test splits
  4. Push to HuggingFace under `mdg-nlp` org
  5. Update `configs/causal_eval.yaml` with correct `hf_path` and field names
  6. Set `enabled: true` in config
- **Dependencies:** MDG-NLP HuggingFace org access

### B4. Get MDG-NLP Access
- **Status:** Blocked
- **Action:** Ask Sagni to add you to the MDG-NLP GitHub/HuggingFace org
- **Purpose:** Push datasets, study existing evaluation code in repo

### B5. Synthetic Data Generation (for tiny datasets)
- **Status:** Not started
- **Datasets needing augmentation:**
  - BigBench CJ: 187 examples
  - BigBench CU: 187 examples
  - AC-Reason: ~1K examples
- **Steps:**
  1. Design a prompt template for synthetic data generation
  2. Show the prompt to professor for approval BEFORE running
  3. Generate synthetic examples using OpenRouter API (GPT-4o or similar)
  4. Validate quality (manual spot-check + consistency metrics)
  5. Merge with original data, re-push to HuggingFace

### B6. Full Evaluation Matrix
- **Status:** Pending (depends on B2-B5)
- **Target evaluation matrix:**
  - 11 datasets x 3+ models x 3 strategies = 99+ evaluation runs
  - Per-run: `results.jsonl` (raw outputs) + `summary.json` (aggregate metrics)
- **Models to evaluate (adjust based on credit availability):**
  - Free: Gemma 3 12B, Gemma 3 27B, LLaMA 3.3 70B (when available)
  - Paid: GPT-4o-mini, GPT-4o (if credits allow)

### B7. Significance Testing and Analysis
- **Status:** Pending (depends on B6)
- **Deliverables:**
  1. Per-dataset performance comparison across models and strategies
  2. Per-level analysis (Level 1 vs 2 vs 3 — Pearl's ladder)
  3. Statistical significance tests from raw `results.jsonl` files
  4. Best strategy per dataset type analysis

---

## Part C: Writeup and Cleanup

### C1. Update Thesis Progress Report
- **File:** `Thesis_Progress_Report_Adversarial_Robustness.docx`
- **When:** After each major result batch (A4, B6)
- **Add:** Result tables, analysis, new pipeline descriptions

### C2. Code Cleanup
- **Delete old unused files:**
  - `codes/02_finetune_bert.py` (merged into `run_encoder.py`)
  - `codes/03_finetune_roberta.py` (merged into `run_encoder.py`)
  - `config.py` (replaced by `configs/experiment.yaml`)
  - `project_utils.py` (check if still referenced)

### C3. Standalone Evaluation Script
- **Purpose:** Quick inference on trained models without running full attack pipeline
- **Status:** Not started

---

## Priority Order

1. **A1 + A2** — Run remaining attacks on GPU (can batch overnight)
2. **B1** — Wait for current eval run to finish
3. **B2** — Run CLadder evaluation
4. **B3** — Download + push datasets (can do while attacks run on GPU)
5. **A3** — Get LLaMA checkpoint from Sagni + B4 (get MDG-NLP access)
6. **B5** — Synthetic data (needs professor approval)
7. **A4 + B6 + B7** — Analysis once data is collected
8. **C1-C3** — Cleanup and writeup
