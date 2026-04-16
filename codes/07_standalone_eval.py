"""
07_standalone_eval.py — Standalone Inference Script

Quickly test a fine-tuned model checkpoint on a single input string.
Supports both Encoder (BERT/RoBERTa) and Decoder (GPT-2/LLaMA) architectures.

Usage:
    python codes/07_standalone_eval.py --model_path models/tlink_bert_final --text "Example sentence"
    python codes/07_standalone_eval.py --model_path models/tlink_gpt2_final --text "Example sentence" --is_decoder
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    parser = argparse.ArgumentParser(description="Standalone Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    parser.add_argument("--is_decoder", action="store_true", help="Flag if the model is a decoder (GPT-2, LLaMA)")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.is_decoder:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load model
    load_kwargs = {}
    if args.is_decoder and ("llama" in args.model_path.lower()):
        print("[INFO] LLaMA detected, using 4-bit quantization")
        load_kwargs = {
            "load_in_4bit": True,
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, **load_kwargs)
    
    if not load_kwargs.get("load_in_4bit"):
        model = model.to(device)
    
    model.eval()

    if args.is_decoder and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Inference
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, 
                       padding="max_length" if args.is_decoder else True, 
                       max_length=args.max_length)
    
    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    # Map back to label if available
    id2label = model.config.id2label
    label = id2label[pred_idx] if id2label else str(pred_idx)

    print(f"\n{'='*40}")
    print(f"Input text : {args.text}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.4f}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    main()
