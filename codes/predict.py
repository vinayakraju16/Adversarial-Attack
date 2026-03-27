import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODELS_DIR

# ----------------------------------------------------------------
# Change MODEL_PATH to switch between bert and roberta
# ----------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "tlink_bert_final"
# MODEL_PATH = MODELS_DIR / "tlink_roberta_final"

NUM_EXAMPLES = 10   # how many real test examples to run


def predict(model, tokenizer, text, device):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    label = str(model.config.id2label[pred_id])
    confidence = float(probs[pred_id].item())
    return label, confidence


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from: {MODEL_PATH}\n")

    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()

    print(f"Label mapping: {model.config.id2label}\n")
    print("-" * 70)

    # Load real examples from the TLINK test split
    print(f"Loading {NUM_EXAMPLES} real examples from TLINK test split ...\n")
    dataset = load_dataset("mdg-nlp/tlink-extr-classification-sentence", split="test")

    correct = 0
    for i in range(NUM_EXAMPLES):
        row = dataset[i]
        text = row["text"]
        true_label = row["label"]

        pred_label, confidence = predict(model, tokenizer, text, device)
        is_correct = pred_label == true_label
        correct += int(is_correct)

        print(f"[{i+1}] Text       : {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"     True label : {true_label}")
        print(f"     Predicted  : {pred_label} ({confidence:.4f}) {'OK' if is_correct else 'WRONG'}")
        print()

    print("-" * 70)
    print(f"Accuracy on {NUM_EXAMPLES} examples: {correct}/{NUM_EXAMPLES} ({100*correct/NUM_EXAMPLES:.1f}%)")


if __name__ == "__main__":
    main()
