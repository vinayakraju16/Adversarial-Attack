import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Path to your saved model
MODEL_PATH = "./models/bert_imdb_final"

# 1. Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_sentiment(text):
    # 2. Prepare the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
    
    label = "POSITIVE" if pred.item() == 1 else "NEGATIVE"
    print(f"\nReview: {text}")
    print(f"Result: {label} ({conf.item()*100:.2f}% confidence)")

# --- Change the text below to whatever you want to test! ---
test_sentences = [
    "I didn't expect to like this movie, and I wasn't wrong to be skeptical at first, but it turned out to be not that bad at all.",
    "The plot was very stagnant and the characters were fragile.",
    "This is the topmost cinema I have ever witnessed.",
    "Oh great, another two hours of my life I'll never get back. Truly a masterpiece of wasting time."
]
    
for text in test_sentences:
    get_sentiment(text)