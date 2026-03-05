from datasets import load_dataset
import pandas as pd

print("Downloading and loading the IMDB dataset...")
dataset = load_dataset("imdb")

# Convert to Pandas for easy exploration
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

print("\n--- Dataset Overview ---")
print(f"Training Samples: {len(df_train)}")
print(f"Testing Samples: {len(df_test)}")

print("\n--- Label Distribution (Train) ---")
# 0 = Negative, 1 = Positive
print(df_train['label'].value_counts())

print("\n--- Sample Review ---")
print(df_train['text'].iloc[0][:500] + "...")