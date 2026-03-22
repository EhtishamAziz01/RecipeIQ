"""
Recipe clustering by nutrition profile — K-Means on per-serving nutrition.
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ── Database path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"

# ── Feature definitions ──
NUTRITION_FEATURES = [
    "calories_per_serving",
    "protein_per_serving",
    "fat_per_serving",
    "sugar_per_serving",
    "fiber_per_serving",
    "sodium_per_serving",
]


def load_clustering_data() -> pd.DataFrame:
    """
    Load recipes from DuckDB for nutrition clustering.
    Filters to recipes with valid nutrition data and reasonable calorie range.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT
            RecipeId, Name, RecipeCategory,
            calories_per_serving, protein_per_serving, fat_per_serving,
            sugar_per_serving, fiber_per_serving, sodium_per_serving,
            ingredient_count, complexity_score
        FROM recipes
        WHERE calories_per_serving IS NOT NULL
          AND calories_per_serving BETWEEN 10 AND 2000
          AND protein_per_serving IS NOT NULL
    """).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df):,} recipes for clustering")
    return df


def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Scale feature columns to zero-mean, unit-variance.
    Critical for K-Means because it uses Euclidean distances.
    """
    if feature_cols is None:
        feature_cols = NUTRITION_FEATURES

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    logger.info(f"Scaled feature matrix: {X_scaled.shape}")
    return X_scaled, scaler


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 11),
    sample_size: int = 10_000,
) -> tuple[list[float], list[float]]:
    """
    Compute inertia and silhouette score for each k in k_range.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels, sample_size=sample_size)
        silhouettes.append(sil)
        logger.info(
            f"k={k:2d} | Inertia: {kmeans.inertia_:>12,.0f} | "
            f"Silhouette: {sil:.4f}"
        )

    return inertias, silhouettes


def fit_clusters(
    X_scaled: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, KMeans]:
    """
    Fit K-Means with the chosen k.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    logger.info(f"Fitted K-Means with k={k}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        logger.info(f"  Cluster {u}: {c:,} recipes")

    return labels, kmeans


def get_cluster_profiles(
    recipes: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean nutrition values per cluster.
    Requires 'cluster' column in recipes DataFrame.
    """
    if feature_cols is None:
        feature_cols = NUTRITION_FEATURES

    return recipes.groupby("cluster")[feature_cols].mean()


def reduce_pca(
    X_scaled: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, PCA]:
    """
    Reduce features to n_components dimensions using PCA.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    logger.info(
        f"PCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.1%}"
    )
    return X_pca, pca
