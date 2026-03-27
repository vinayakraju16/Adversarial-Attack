import sys
from pathlib import Path

import evaluate
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODELS_DIR
from project_utils import (
    DEFAULT_LABEL_FIELD,
    DEFAULT_TEXT_FIELD,
    TLINK_DATASET_NAME,
    build_label_mappings,
    encode_labels,
    ensure_project_dirs,
    load_tlink_dataset,
    resolve_device,
    set_seed,
)

MODEL_NAME = "bert-base-uncased"
MODEL_ARCH = "bert"
OUTPUT_DIR = MODELS_DIR / "tlink_bert_final"
CHECKPOINT_DIR = MODELS_DIR / "tlink_bert_checkpoints"
SEED = 42
MAX_LENGTH = 256


def main() -> None:
    ensure_project_dirs()
    set_seed(SEED)
    device = resolve_device()
    print(f"Training {MODEL_ARCH} on {device} with dataset {TLINK_DATASET_NAME}")

    dataset = load_tlink_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label2id, id2label = build_label_mappings(dataset["train"], DEFAULT_LABEL_FIELD)

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch[DEFAULT_TEXT_FIELD],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    tokenized = DatasetDict(
        {
            split: encode_labels(split_dataset, label2id, DEFAULT_LABEL_FIELD)
            for split, split_dataset in tokenized.items()
        }
    )
    tokenized = tokenized.remove_columns(
        [column for column in tokenized["train"].column_names if column not in {"input_ids", "attention_mask", "token_type_ids", "labels"}]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={idx: label for idx, label in id2label.items()},
    )
    model.config.label2id = label2id
    model.config.id2label = {idx: label for idx, label in id2label.items()}

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: tuple) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save training stats
    eval_results = trainer.evaluate()
    stats = {
        "model": MODEL_ARCH,
        "base_model": MODEL_NAME,
        "dataset": TLINK_DATASET_NAME,
        "seed": SEED,
        "epochs": training_args.num_train_epochs,
        "train_samples": len(tokenized["train"]),
        "eval_samples": len(tokenized["test"]),
        "final_train_loss": round(train_result.training_loss, 4),
        "final_eval_accuracy": round(eval_results.get("eval_accuracy", 0), 4),
        "final_eval_loss": round(eval_results.get("eval_loss", 0), 4),
        "train_runtime_seconds": round(train_result.metrics.get("train_runtime", 0), 1),
    }
    import json
    stats_path = OUTPUT_DIR / "training_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")
    print(f"Saved training stats to {stats_path}")
    print(f"Final eval accuracy: {stats['final_eval_accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()
