import nltk
nltk.download('averaged_perceptron_tagger_eng')

import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- HOTFIX FOR TEXTATTACK BUG ---
original_init = textattack.search_methods.GreedyWordSwapWIR.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop("truncate_words_to", None)
    original_init(self, *args, **kwargs)
textattack.search_methods.GreedyWordSwapWIR.__init__ = patched_init
# ---------------------------------

print("Loading trained RoBERTa model...")
model_path = "./models/roberta_imdb_final"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

print("Building A2T Attack Recipe...")
attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper)

dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

attack_args = textattack.AttackArgs(
    num_examples=200, 
    log_to_csv="./results/roberta_a2t_results.csv",
    checkpoint_interval=50,
    disable_stdout=False
)

attacker = textattack.Attacker(attack, dataset, attack_args)

print("Starting A2T Attack on RoBERTa...")
attacker.attack_dataset()
print("✅ Attack finished. Results saved to ./results/roberta_a2t_results.csv")