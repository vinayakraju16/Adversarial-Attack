# Adversarial Robustness of Temporal Relation Classifiers

This project fine-tunes BERT and RoBERTa on a temporal relation dataset (TLINK) and tests how vulnerable they are to adversarial text attacks. The entire pipeline is controlled from a single config file.

---

## What This Project Does

1. **Train** — Fine-tune BERT or RoBERTa on the TLINK dataset
2. **Attack** — Run adversarial attacks to fool the trained model
3. **Analyse** — View results showing how often and how easily the model was fooled

---

## Project Structure

```
Thesis work/
├── run.py                        <- MAIN FILE: runs the attack
├── configs/
│   └── experiment.yaml           <- ALL CONFIG: change dataset, model, attack here
├── codes/
│   ├── 02_finetune_bert.py       <- trains BERT
│   ├── 03_finetune_roberta.py    <- trains RoBERTa
│   ├── 06_results_analysis.py    <- combines results across runs
│   └── predict.py                <- test the model locally
├── scripts/
│   ├── run_all_remote.ps1        <- Windows: sync + run on GPU server + pull results
│   ├── run_pipeline_gpu.sh       <- GPU server: runs the full pipeline
│   ├── run_all_attacks.sh        <- GPU server: runs all attacks on both models
│   └── pull_models.ps1           <- Windows: pull trained models from GPU server
├── models/                       <- saved model checkpoints (created after training)
├── results/                      <- attack results (created after running attacks)
├── requirements.txt
└── ATTACKS.md                    <- list of all 15 supported attack methods
```

---

## Step 0 — Prerequisites

- Python 3.10 or higher
- pip
- A GPU is recommended (attacks are slow on CPU)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1 — Configure the Experiment

Open `configs/experiment.yaml`. This is the **only file you need to edit**.

```yaml
dataset:
  name: mdg-nlp/tlink-extr-classification-sentence   # HuggingFace dataset
  text_field: text                                    # column with input text
  label_field: label                                  # column with the label
  split: test                                         # which split to attack

model:
  arch: bert                          # bert | roberta | distilbert | albert
  checkpoint_out_dir: models/tlink_bert_final  # where the trained model is saved
  max_length: 256
  device: auto                        # auto | cuda | cpu

attack:
  name: pwws                          # which attack to use (see ATTACKS.md)
  num_examples: 200                   # how many examples to attack

project:
  seed: 42
```

**To switch to RoBERTa**, change:
```yaml
model:
  arch: roberta
  checkpoint_out_dir: models/tlink_roberta_final
```

**To switch attack**, change:
```yaml
attack:
  name: bertattack    # or: bae, a2t, deepwordbug, textfooler, etc.
```

See `ATTACKS.md` for all 15 available attack names.

---

## Step 2 — Train the Model

> Skip this step if models are already trained and saved in `models/`.

**Train BERT:**
```bash
python codes/02_finetune_bert.py
```

**Train RoBERTa:**
```bash
python codes/03_finetune_roberta.py
```

Each script:
- Downloads the TLINK dataset from HuggingFace automatically
- Trains for 3 epochs
- Saves the model to `models/tlink_bert_final/` or `models/tlink_roberta_final/`
- Saves training statistics to `models/tlink_bert_final/training_stats.json`

Training takes around 90 minutes per model on a GPU.

---

## Step 3 — Run the Attack

Make sure `configs/experiment.yaml` has the right model and attack set, then:

```bash
python run.py
```

This will:
1. Load the trained model from the checkpoint folder
2. Load test examples from the dataset
3. Run the attack on each example
4. Print a summary box in the terminal
5. Save all results automatically

---

## Step 4 — View Results

Results are saved automatically in this structure:

```
results/
  tlink-extr-classification-sentence/    <- dataset name
    pwws/                                 <- attack name
      bert/                               <- model name
        attacks.jsonl    <- every example: original text, perturbed text, scores, labels
        summary.csv      <- overall metrics for this run
        stats.json       <- same metrics in JSON format for easy reuse
        run.log          <- full terminal output saved to file
      roberta/
        attacks.jsonl
        summary.csv
        stats.json
        run.log
```

### What the columns mean

| Column | Meaning |
|--------|---------|
| `original_text` | The original sentence before any attack |
| `perturbed_text` | The sentence after the attack changed some words |
| `original_score` | Model confidence on the original text (0 to 1) |
| `perturbed_score` | Model confidence after the attack (0 to 1) |
| `original_output` | What the model predicted originally (e.g. AFTER, BEFORE) |
| `perturbed_output` | What the model predicted after the attack |
| `ground_truth_output` | The correct label from the dataset |
| `num_queries` | How many times the attack called the model |
| `result_type` | `successful` = attack fooled the model, `failed` = model held correct, `skipped` = model was already wrong |
| `pct_words_changed` | Percentage of words that were changed |

### Terminal output after each run

```
[Succeeded / Failed / Skipped / Total] 148 / 24 / 28 / 200

+------------------------------+--------+
| Attack Results               |        |
+------------------------------+--------+
| Number of successful attacks:| 148    |
| Number of failed attacks:    | 24     |
| Number of skipped attacks:   | 28     |
| Original accuracy:           | 86.0%  |
| Accuracy under attack:       | 12.0%  |
| Attack success rate:         | 86.05% |
| Average perturbed word %:    | 9.96%  |
| Average num. words per input:| 35.24  |
| Avg num queries:             | 186.70 |
+------------------------------+--------+
```

---

## Running on the GPU Server (Recommended for Speed)

Training and attacks are slow on a local machine. Use the college GPU server instead.

### One-command workflow (recommended)

This syncs your code to the server, runs the pipeline, and pulls results back:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_remote.ps1
```

### Manual workflow

**Step 1** — SSH into the server:
```bash
ssh scr0179@129.120.60.102
```

**Step 2** — Run the attack on the server:
```bash
cd /home/UNT/scr0179/thesis_work/Adversarial-Attack
source ~/.venvs/advnlp/bin/activate
python run.py
```

**Step 3** — Pull results back to your local machine (run in a new local terminal):
```powershell
subst T: "D:\Thesis work"
scp -r scr0179@129.120.60.102:/home/UNT/scr0179/thesis_work/Adversarial-Attack/results/. T:\results_remote
```

### Pull trained models from GPU to local

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pull_models.ps1
```

---

## Running All Attacks at Once

To run all attacks on both BERT and RoBERTa in one go (on the GPU server):

```bash
bash scripts/run_all_attacks.sh
```

This loops through all 12 word-level attacks × 2 models and saves results for each combination separately.

---

## Test the Model Locally

To quickly check how the model performs on a few real examples:

```bash
python codes/predict.py
```

---

## Common Mistakes

| Problem | Fix |
|---------|-----|
| `No such file or directory: models/tlink_bert_final` | Run training first: `python codes/02_finetune_bert.py` |
| `Unknown attack 'xyz'` | Check `ATTACKS.md` for valid attack names |
| Attack is very slow | Run on the GPU server, not locally |
| `CUDA not available` | Set `device: cpu` in the YAML, or run on the GPU server |
| `scp` fails with spaces in path | Use `subst T: "D:\Thesis work"` first, then scp to `T:\...` |
