"""
Feature engineering pipeline for RecipeIQ.
Transforms raw recipe data into ML-ready feature matrices.
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Database path ──
DB_PATH = Path("data/processed/recipeiq.duckdb")

# ── Feature definitions ──
NUMERIC_FEATURES = [
    "ingredient_count",
    "instruction_steps",
    "calories_per_serving",
    "protein_per_serving",
    "fat_per_serving",
    "sugar_per_serving",
    "fiber_per_serving",
    "sodium_per_serving",
    "CookTime_minutes",
    "PrepTime_minutes",
    "TotalTime_minutes",
    "ReviewCount",
]

CATEGORICAL_FEATURES = [
    "RecipeCategory",
]

TARGET = "AggregatedRating"


def load_ml_data() -> pd.DataFrame:
    """
    Load recipe data from DuckDB for ML.
    Filters out recipes with no ratings.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT
            {', '.join(NUMERIC_FEATURES)},
            {', '.join(CATEGORICAL_FEATURES)},
            {TARGET}
        FROM recipes
        WHERE {TARGET} IS NOT NULL
          AND ingredient_count IS NOT NULL
          AND calories_per_serving IS NOT NULL
          AND CookTime_minutes IS NOT NULL
          AND RecipeCategory IS NOT NULL
    """).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df):,} recipes for ML")
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
    - Scales numeric features to zero mean, unit variance
    - One-hot encodes categorical features

    This ensures all features are on the same scale and in numeric form.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drop any columns not listed above
    )
    return preprocessor


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load data, split into train/test, and return everything needed for training.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_ml_data()

    # Separate features (X) from target (y)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor
