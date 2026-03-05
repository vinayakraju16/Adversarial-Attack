import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing RoBERTa Training on: {device}")

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_data(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

print("Tokenizing data...")
tokenized_data = dataset.map(tokenize_data, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir="./models/roberta_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
)

train_subset = tokenized_data["train"].shuffle(seed=42).select(range(5000))
eval_subset = tokenized_data["test"].shuffle(seed=42).select(range(1000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=eval_subset
)

print("Starting RoBERTa Training...")
trainer.train()

trainer.save_model("./models/roberta_imdb_final")
tokenizer.save_pretrained("./models/roberta_imdb_final")
print("✅ RoBERTa Model saved to ./models/roberta_imdb_final")