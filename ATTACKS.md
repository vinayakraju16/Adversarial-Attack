# Adversarial Attack Methods

Complete reference for all attacks supported by this project.
Change the `attack.name` field in `configs/experiment.yaml` — or pass `--attack <key>` on the command line — to switch methods without touching any other file.

---

## Recommended 12 (TextAttack, word-level focus)

| # | YAML key | Method | Level | Year | Paper |
|---|----------|--------|-------|------|-------|
| 1 | `textfooler` | TextFooler | Word sub | 2019 | [Jin et al., AAAI 2020](https://arxiv.org/abs/1907.11932) |
| 2 | `textbugger` | TextBugger | Word+Char | 2019 | [Li et al., NDSS 2019](https://arxiv.org/abs/1812.05271) |
| 3 | `bertattack` | BERT-Attack | Word sub (MLM) | 2020 | [Li et al., EMNLP 2020](https://arxiv.org/abs/2004.09984) |
| 4 | `bae` | BAE | Word sub (MLM) | 2020 | [Garg & Ramakrishna, EMNLP 2020](https://arxiv.org/abs/2004.01970) |
| 5 | `pwws` | PWWS | Word sub (WordNet) | 2019 | [Ren et al., ACL 2019](https://arxiv.org/abs/1907.11932) |
| 6 | `a2t` | A2T | Word sub (constrained) | 2021 | [Yoo & Qi, EMNLP 2021](https://arxiv.org/abs/2109.00544) |
| 7 | `clare` | CLARE | Word sub/ins/del | 2021 | [Li et al., EMNLP 2021](https://arxiv.org/abs/2009.07502) |
| 8 | `genetic` | Genetic Algorithm | Word sub | 2018 | [Alzantot et al., EMNLP 2018](https://arxiv.org/abs/1804.07998) |
| 9 | `faster_genetic` | Faster Genetic | Word sub | 2019 | [Jia et al., ACL 2019](https://arxiv.org/abs/1905.02175) |
| 10 | `pso` | PSO | Word sub (particle swarm) | 2020 | [Zang et al., ACL 2020](https://arxiv.org/abs/2004.01830) |
| 11 | `hotflip` | HotFlip | Char/Word (gradient) | 2018 | [Ebrahimi et al., ACL 2018](https://arxiv.org/abs/1712.06751) |
| 12 | `deepwordbug` | DeepWordBug | Char edit | 2018 | [Gao et al., S&P 2018](https://arxiv.org/abs/1801.04354) |

---

## Additional TextAttack Recipes

| YAML key | Method | Level | Year | Paper |
|----------|--------|-------|------|-------|
| `pruthi` | Pruthi typos | Char | 2019 | [Pruthi et al., ACL 2019](https://arxiv.org/abs/1903.12136) |
| `morpheus` | Morpheus | Morphological | 2020 | [Tan & Motani, ACL 2020](https://arxiv.org/abs/2007.02814) |
| `input_reduction` | Input Reduction | Deletion | 2018 | [Feng et al., EMNLP 2018](https://arxiv.org/abs/1804.07781) |

---

## Post-2022 Attacks (Outside TextAttack)

These are word-level / semantic attacks published after 2022 with public implementations.
They are **not** prompt-based.

| Method | Level | Year | Paper | Implementation |
|--------|-------|------|-------|----------------|
| **SemAttack** | Semantic space | 2022 | [Wang et al., NAACL 2022](https://arxiv.org/abs/2205.01060) | [JHL-HUST/SemAttack](https://github.com/JHL-HUST/SemAttack) |
| **SAFER** | Word sub (certified) | 2021 | [Ye et al., ACL 2020](https://arxiv.org/abs/2005.14424) | [lushleaf/SAFER](https://github.com/lushleaf/SAFER) |
| **OpenAttack** library | Multiple | 2021+ | [Zeng et al., ACL 2021](https://arxiv.org/abs/2009.09191) | [thunlp/OpenAttack](https://github.com/thunlp/OpenAttack) |
| **AdvFooler** | Word sub | 2023 | [Yang et al., 2023](https://arxiv.org/abs/2310.11709) | See paper |
| **VIPER** | Visual char | 2022 | [Eger et al., 2022](https://arxiv.org/abs/2204.02664) | [VISualPerturb](https://github.com/s-helm/TextFooler) |

> **To integrate SemAttack or OpenAttack attacks:** add an entry to `_ATTACK_REGISTRY` in `run_attack.py` pointing to the recipe class, then wrap the model with `HuggingFaceModelWrapper` exactly as the existing recipes do.

---

## IRT (Item Response Theory) for Attack Evaluation

IRT treats each test example as an "item" and each attack as an "respondent." The difficulty parameter of an item measures how hard it is for all attacks to fool the model on that example. This can be used to:

1. Rank examples by robustness difficulty.
2. Compare attacks on a common scale beyond raw ASR.
3. Identify systematically hard/easy label classes (useful for TLINK, which has asymmetric class difficulty).

Recommended library: [`py-irt`](https://github.com/JohnLangford/py-irt) or `girth`.

---

## Model Architecture Notes

| Type | Examples | Notes |
|------|----------|-------|
| Encoder-only | BERT, RoBERTa, DistilBERT, ALBERT, ELECTRA | Default. Bidirectional attention. Use `padding_side='right'`. |
| Decoder-only | GPT-2, OPT, LLaMA | Works with `AutoModelForSequenceClassification` but requires `tokenizer.padding_side = 'left'` and `tokenizer.pad_token = tokenizer.eos_token`. |

All TextAttack attack recipes wrap the model through `HuggingFaceModelWrapper` — they are **architecture-agnostic**. Adding a decoder model only requires changing `arch` and `checkpoint_out_dir` in the YAML.

---

## Naming Convention

Every run produces files named with the pattern:

```
{dataset}_{model}_{attack}_n{N}_seed{seed}
```

Example: `mdg-nlp_tlink-extr-classification-sentence_bert_textfooler_n200_seed42`

| Token | Meaning |
|-------|---------|
| `dataset` | HuggingFace dataset path (slashes → underscores) |
| `model` | `arch` value from YAML |
| `attack` | `name` value from YAML |
| `N` | Number of examples attacked |
| `seed` | Random seed |
