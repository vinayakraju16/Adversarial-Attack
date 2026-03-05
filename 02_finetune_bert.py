import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Setup GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing BERT Training on: {device}")

# 2. Load Dataset and Tokenizer
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

print("Tokenizing data...")
tokenized_data = dataset.map(tokenize_data, batched=True)

# 3. Load Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./models/bert_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
)

# Subset data for faster testing (remove the .select() to train on full dataset)
train_subset = tokenized_data["train"].shuffle(seed=42).select(range(5000))
eval_subset = tokenized_data["test"].shuffle(seed=42).select(range(1000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset
)

# 5. Train and Save
print("Starting BERT Training...")
trainer.train()

trainer.save_model("./models/bert_imdb_final")
tokenizer.save_pretrained("./models/bert_imdb_final")
print("✅ BERT Model saved to ./models/bert_imdb_final")