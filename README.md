# Sentiment Analysis – Spotify App Reviews

A machine learning project that classifies Spotify app reviews into three sentiment categories — **positive**, **neutral**, and **negative** — based solely on the text content of the review.

The project covers a full NLP pipeline: from raw text preprocessing through classical ML modelling to iterative hyperparameter tuning, with a focus on handling class imbalance and improving classification of the minority neutral class.

---

## Goal

Build a model that, based on the content of reviews, can correctly classify them into one of three sentiment classes: positive, neutral, or negative.

Sentiment labels were derived from star ratings:

| Rating | Sentiment |
|--------|-----------|
| ⭐⭐⭐⭐⭐ / ⭐⭐⭐⭐ | Positive |
| ⭐⭐⭐ | Neutral |
| ⭐⭐ / ⭐ | Negative |

---

## Dataset

**Source:** [Spotify App Reviews 2022 – Kaggle](https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022)

- 61,594 reviews with no empty entries
- Average review length: ~31 words; three-quarters of reviews contain fewer than 40 words
- Class distribution is highly imbalanced, with neutral reviews representing the smallest class
  
---

## Pipeline

```
Raw review text
  → Cleaning         (HTML removal, lowercasing, emoticon handling, punctuation removal)
  → Tokenisation     (custom regex tokenizer — contractions preserved for negation context)
  → Lemmatisation    (WordNetLemmatizer)
  → Stopword removal (NLTK English stopwords + custom domain-specific terms)
  → Representation   (Bag of Words / N-gram / TF-IDF)
  → Classification   (TextBlob / Naive Bayes / Logistic Regression)
```

**Key preprocessing decision:** Domain-specific words appearing frequently across *all* sentiment classes (e.g. *app*, *music*, *song*, *playlist*, *premium*) were identified through corpus analysis and added to the stopword list. This reduced noise and improved the model's ability to focus on sentiment-bearing terms.

**Custom tokenizer:** The default NLTK tokenizer splits contractions such as *didn't* into *did* and *n't*. A custom regex-based tokenizer was implemented to preserve contractions, retaining negation information that is important for classifying negative sentiment.

---

## Models & Results

### Naive Bayes

| Text Representation | Accuracy | F1 Positive | F1 Negative | F1 Neutral |
|---|---|---|---|---|
| Bag of Words | 0.77 | 0.85 | 0.79 | 0.08 |
| N-gram | 0.78 | 0.86 | 0.79 | 0.01 |
| TF-IDF | 0.77 | 0.85 | 0.78 | 0.00 |
| **BoW + tuned α = 0.1** | **0.76** | **0.84** | **0.77** | **0.22** |

### Logistic Regression

| Text Representation | Accuracy | F1 Positive | F1 Negative | F1 Neutral |
|---|---|---|---|---|
| Bag of Words | 0.77 | 0.85 | 0.78 | 0.17 |
| N-gram | 0.78 | 0.86 | 0.79 | 0.18 |
| TF-IDF | 0.78 | 0.86 | 0.80 | 0.09 |
| **N-gram + balanced weights + C = 0.1** | **0.76** | **0.86** | **0.78** | **0.28** |

---

## Key Findings

- **TextBlob** (lexicon-based baseline) produced similar results regardless of preprocessing, confirming that its polarity scoring is largely preprocessing-agnostic. It classified positive reviews reasonably well but performed poorly on negative and especially neutral reviews.
- **Machine learning models substantially outperformed TextBlob**, demonstrating the advantage of training classifiers on domain-specific data.
- **TF-IDF generally produced the weakest neutral-class F1-scores** among the evaluated text representations.
- **Hyperparameter tuning improved neutral-class performance** in both models:
  - Naive Bayes: reducing α from 1.0 to 0.1 increased neutral F1-score from **0.08 → 0.22**
  - Logistic Regression: class weight balancing + C = 0.1 increased neutral F1-score from **0.18 → 0.28**
- The **best overall model** is the tuned Logistic Regression with n-gram representation, class weight balancing, and C = 0.1. However, neutral reviews remained the most challenging category throughout — the model may be more suitable for applications focused on distinguishing positive from negative feedback than for precise three-class sentiment classification.

---

## Tech Stack

- **Python**
- `nltk` — tokenisation, lemmatisation, stopwords
- `scikit-learn` — Naive Bayes, Logistic Regression, TF-IDF, GridSearchCV
- `textblob` — lexicon-based baseline
- `matplotlib`, `seaborn` — visualisation
- `pandas`, `tqdm`

---

## Project Structure

```
spotify-sentiment-analysis/
├── README.md
├── plot_confusion_matrix.py         # confusion matrix helper function
├── requirements.txt
├── reviews.csv                      # Spotify reviews dataset
└── spotify_sentiment_analysis.ipynb # main notebook (with outputs)
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Dwis97/spotify-sentiment-analysis.git
cd spotify-sentiment-analysis

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook spotify_sentiment_analysis.ipynb
# or open in VS Code and select the venv kernel
```

---

## Suggested Further Improvements

- Analyse neutral reviews in more detail to better understand how they differ from positive and negative reviews
- Test other methods for handling class imbalance, especially for Naive Bayes where automatic class weighting is unavailable
- Review the sentiment labelling strategy — adjusting the rating-to-sentiment mapping to increase the number of neutral reviews may improve classification
- Test additional classifiers such as Support Vector Machines (SVM) or transformer-based models (e.g. BERT) for better contextual understanding
