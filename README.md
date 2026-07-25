# 🛡️ Toxic Comment Classification — Hybrid Attention–Convolution Memory-Augmented Neural Network

> A multi-label deep learning system that flags toxic, obscene, threatening, and hateful comments with **99.66% accuracy** — outperforming BiLSTM+CNN, BERT, and standard embedding baselines.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![NLTK](https://img.shields.io/badge/NLP-NLTK-green)
![Status](https://img.shields.io/badge/Paper-Under%20Review-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Online platforms generate huge volumes of user comments every day, and a meaningful fraction of them are toxic, obscene, threatening, or hateful. Manually moderating this content doesn't scale — automated, multi-label classification is the only realistic path to safer online spaces.

This repository contains the implementation of a **Hybrid Attention–Convolution Memory-Augmented Neural Network** that classifies each comment across **six overlapping toxicity categories simultaneously**:

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

The model combines four complementary ideas into a single pipeline — sequence memory, attention, local feature extraction, and dense classification — instead of relying on any one architecture alone.

This work is currently **under academic review** (SASTRA Deemed University).

---

## 🏆 Key Results

| Metric | Baseline (before tuning) | **Proposed Model** |
|---|---|---|
| Overall Accuracy | 82.37% | **99.66%** |
| Precision | — | **99.33%** |
| Recall | — | **99.83%** |
| F1-Score | — | **99.83%** |

**Per-label accuracy (all labels ≥ 0.99):**

| toxic | severe_toxic | obscene | threat | insult | identity_hate |
|---|---|---|---|---|---|
| 0.9931 | 0.9958 | 0.9969 | 0.9993 | 0.9957 | 0.9988 |

**ROC-AUC** is near-perfect across every label, with `threat` and `identity_hate` reaching a perfect **1.0000**.

Benchmarked against BiLSTM+CNN, bare BERT, and GloVe/word2vec/fastText/Intel 300-dim embedding baselines, this architecture consistently comes out ahead — while still being lightweight enough to train on a single GPU.

---

## 🧠 How It Works

```
Raw Comment
    │
    ▼
┌─────────────────────┐
│ 1.Text Preprocessing│   lowercase → strip URLs/mentions/hashtags → remove punctuation
│                     │   → expand contractions → stopword removal (negations kept)
│                     │   → WordNet lemmatization
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 2. GloVe Embedding  │   100-dim pretrained vectors, fine-tuned during training
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 3.Memory-Augmented  │   Sequential gated memory cell (forget / input / output gates)
│   Layer (LSTM-based)│   captures short- and long-range dependencies across the comment
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 4.Attention Layer   │   Learns per-timestep importance weights and produces a single
│                     │   context vector focused on the most decisive words
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 5. Conv1D Layer     │   Extracts local n-gram patterns from the attended representation
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 6. Dense + Sigmoid  │   Independent probability per label → multi-label classification
└─────────────────────┘
```

**Why this combination?** Memory cells alone treat every word equally; attention alone can miss local phrasing (e.g. "not funny" vs "funny"); convolution alone has no long-range context. Stacking them lets the model capture global dependencies, weight the important words, *and* pick up short, punchy insult/obscenity patterns — which is exactly the mix that toxic comments contain.

---

## 📊 Dataset

- **Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle / Wikipedia talk-page edits)
- **Size:** 159,571 labeled comments
- **Labels:** 6 non-mutually-exclusive toxicity categories (multi-label)
- **Challenge handled:** Severe class imbalance — solved with a combination of majority-class under-sampling and minority-class over-sampling before training

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| NLP Preprocessing | NLTK, `contractions` |
| Embeddings | GloVe (100d) |
| Data Handling | Pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Dharsh62/toxic-comment-classification.git
cd toxic-comment-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place train.csv in a /data folder,
#    and GloVe 100d vectors (glove.6B.100d.txt) alongside it

# 4. Open the notebook
jupyter notebook notebooks/Final_Code.ipynb
```

**`requirements.txt`**
```
tensorflow>=2.12
pandas
numpy
scikit-learn
nltk
contractions
matplotlib
seaborn
```

---

## 📈 Evaluation

The model is evaluated with:
- **Confusion matrices** for each of the 6 labels
- **Overall confusion matrix** across all labels
- **ROC curves + AUC** per label
- **Accuracy/loss curves** across training epochs (with early stopping + learning-rate reduction on plateau)

All plots are generated directly inside the notebook and saved to `/images` for quick reference without needing to re-run training.

---

## 🔮 Future Work

- Explore SMOTE-based oversampling as an alternative to random resampling
- Cross-domain evaluation on non-Wikipedia toxic comment datasets
- Explainability layer (e.g. attention-weight visualization per prediction) for content-moderation transparency

---

## 👥 Authors

**Dharshanaa N** · Pooja A · Shobana Vaishnavi B
Guided by **Prof. Pradeepa Sampath**, SASTRA Deemed University

📧 dharshanaa62nr@gmail.com &nbsp;|&nbsp; [LinkedIn](https://www.linkedin.com/in/dharshanaa-n/) &nbsp;|&nbsp; [GitHub](https://github.com/Dharsh62/)

---

*If you use this work, a citation link to the paper will be added here once publication is finalized.*
