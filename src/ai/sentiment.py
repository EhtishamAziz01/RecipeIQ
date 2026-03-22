"""
Sentiment analysis for recipe reviews — TF-IDF + Logistic Regression.

Provides:
    - Text preprocessing pipeline (HTML removal, stopwords, normalization)
    - Sentiment labelling from ratings (positive / neutral / negative)
    - Model training with class-weight balancing
    - Batch prediction for scoring large review sets

Usage:
    from src.ai.sentiment import (
        load_reviews, preprocess_text, rating_to_sentiment,
        build_sentiment_model, predict_sentiment_batch,
    )
"""

import logging
import re
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ── Database path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"

# ── Stopwords (cached) ──
STOP_WORDS = set(stopwords.words("english"))


# ────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────

def load_reviews(min_length: int = 10) -> pd.DataFrame:
    """
    Load reviews from DuckDB for sentiment analysis.
    Filters to reviews with text longer than min_length characters.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT
            ReviewId, RecipeId, AuthorId,
            Rating, Review, review_length
        FROM reviews
        WHERE Review IS NOT NULL
          AND LENGTH(Review) > {min_length}
          AND Rating IS NOT NULL
    """).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df):,} reviews for sentiment analysis")
    return df


def rating_to_sentiment(rating: int) -> str:
    """Map a numeric rating to a sentiment label."""
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"


# ────────────────────────────────────────────────────
# Text preprocessing
# ────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalize review text for ML.

    Steps:
        1. Lowercase
        2. Remove HTML tags
        3. Remove non-alphabetic characters
        4. Remove stopwords
        5. Drop short words (≤ 2 chars)
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)       # non-letters
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)


# ────────────────────────────────────────────────────
# Model training
# ────────────────────────────────────────────────────

def build_sentiment_model(
    X_train: pd.Series,
    y_train: pd.Series,
    max_features: int = 10_000,
) -> tuple[TfidfVectorizer, LogisticRegression, np.ndarray]:
    """
    Build a TF-IDF + Logistic Regression sentiment classifier.

    Args:
        X_train: Series of preprocessed review texts.
        y_train: Series of sentiment labels.
        max_features: Number of TF-IDF features.

    Returns:
        (fitted TfidfVectorizer, fitted LogisticRegression, X_train_tfidf)
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)

    logger.info(f"TF-IDF matrix: {X_train_tfidf.shape}")

    classes = np.array(["negative", "neutral", "positive"])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weights}")

    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        C=1.0,
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    return tfidf, model, X_train_tfidf


# ────────────────────────────────────────────────────
# Batch prediction
# ────────────────────────────────────────────────────

def predict_sentiment_batch(
    texts: pd.Series,
    tfidf: TfidfVectorizer,
    model: LogisticRegression,
    batch_size: int = 10_000,
) -> list[str]:
    """
    Score reviews in batches to avoid memory issues.

    Args:
        texts: Series of raw review texts (not preprocessed).
        tfidf: Fitted TfidfVectorizer.
        model: Fitted LogisticRegression.
        batch_size: Number of reviews per batch.

    Returns:
        List of predicted sentiment labels.
    """
    sentiments = []
    for i in range(0, len(texts), batch_size):
        batch = texts.iloc[i : i + batch_size]
        batch_clean = batch.apply(preprocess_text)
        batch_tfidf = tfidf.transform(batch_clean)
        preds = model.predict(batch_tfidf)
        sentiments.extend(preds)
        if (i // batch_size) % 10 == 0:
            logger.info(f"Scored {i + len(batch):,} / {len(texts):,}")

    return sentiments
