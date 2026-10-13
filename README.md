# Question-Answer Text Classification: TF-IDF vs Skip-gram Neural Models

Group project for **CSE440 (Natural Language Processing)**, BRAC University.
Authors: Tasneem Tuhfa, Maimuna Anjum, MD. Samsul Arafin Anan.

This project classifies question-answer text into categories using two feature-extraction approaches (TF-IDF and Skip-gram Word2Vec embeddings) and compares 9 models in total, from classical machine learning to recurrent neural networks.

## Overview

| Track | Features | Models |
|-------|----------|--------|
| TF-IDF (2 experiments) | TF-IDF, 5000 features, unigrams | Logistic Regression, DNN |
| Skip-gram (7 experiments) | Word2Vec Skip-gram (100-dim), frozen embeddings | Pooled DNN, SimpleRNN, GRU, LSTM, BiSimpleRNN, BiGRU, BiLSTM |

Hyperparameters for every model are tuned manually (grid search) using **validation macro-F1**. The provided test set is used only once per model for final evaluation.

## Dataset

- **Train:** 93,333 rows. **Test:** 59,999 rows.
- Columns: `QA Text` (the text) and `Class` (the label, 10 classes).
- Both files are included in the `data/` folder:
  - `data/Question Answer Classification Dataset 6(1).csv` (train)
  - `data/[Updated] Question Answer Classification Dataset[Test].csv` (test)

## Pipeline

1. **EDA:** missing values, class distribution, text length distribution, most frequent words (raw and cleaned).
2. **Preprocessing:** removes `Question Title:`, `Question Body:` and `Answer:` markers, HTML tags and URLs, and special characters. Then lowercases, removes stopwords, and lemmatizes (verb, then noun). Digits are kept.
3. **Split:** 90% train / 10% validation (stratified, seed 42). The provided test CSV is used for final evaluation.
4. **Modeling:** TF-IDF models and Skip-gram models with grid search over hyperparameters (sequence length, units, dropout, learning rate, regularization `C`).
5. **Evaluation:** accuracy, macro-F1, confusion matrices, classification reports, and a best-ML vs best-NN comparison.

## Results (test set)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| SkipGram + BiLSTM | 0.7140 | 0.7098 |
| SkipGram + GRU | 0.7126 | 0.7076 |
| SkipGram + BiGRU | 0.7117 | 0.7070 |
| SkipGram + LSTM | 0.7074 | 0.7021 |
| TF-IDF + DNN | 0.6843 | 0.6795 |
| SkipGram + DNN (Pooled) | 0.6832 | 0.6777 |
| TF-IDF + Logistic Regression | 0.6756 | 0.6734 |
| SkipGram + BiSimpleRNN | 0.6749 | 0.6694 |
| SkipGram + SimpleRNN | 0.3492 | 0.2780 |

- **Best model:** SkipGram + BiLSTM (macro-F1 0.7098).
- **Best classical/TF-IDF model:** TF-IDF + DNN (macro-F1 0.6795).
- **Worst model:** SkipGram + SimpleRNN. Likely causes are difficulty with long sequences and the post-padding used without masking; this was not investigated further.

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

NLTK data (downloaded automatically by the notebook):

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## How to Run

### Option A: Google Colab (as originally written)
1. Upload the notebook to Colab.
2. Put both CSV files in a Drive folder named `CSE440_Project`.
3. Run all cells. The notebook mounts Google Drive and reads the files from `/content/drive/MyDrive/CSE440_Project/`.

### Option B: Locally
1. Install the requirements (see above).
2. In the notebook, remove the Google Drive cell (`from google.colab import drive ...`).
3. Change `BASE_PATH` to `"./data/"` (the dataset is already in this repo).
4. Launch Jupyter and run all cells:
   ```bash
   jupyter notebook qa_text_classification.ipynb
   ```

> **Note:** the full run takes a long time (the Skip-gram grid search took roughly 15 to 40 minutes per model on Colab). Set `FAST_MODE = True` in the notebook to shrink the search and run faster.

## Repository Structure

```
.
├── qa_text_classification.ipynb   # Full code: EDA, preprocessing, models, evaluation
├── report.pdf                                # Written project report
├── data/                                     # Train and test CSV files
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

Random seeds are fixed (`SEED = 42`) for Python, NumPy and TensorFlow. Results can still vary slightly across hardware and library versions.

## Authors

- Tasneem Tuhfa
- Maimuna Anjum
- MD. Samsul Arafin Anan
