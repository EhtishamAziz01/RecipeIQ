"""
Recipe recommendation engine — content-based and collaborative filtering.
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# ── Database path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"

# ── Columns returned by load_recipes ──
RECIPE_COLUMNS = [
    "RecipeId", "Name", "RecipeCategory",
    "AggregatedRating", "ReviewCount",
    "Description", "ingredient_count",
    "complexity_score", "calories_per_serving",
]


# ────────────────────────────────────────────────────
# Content-based filtering
# ────────────────────────────────────────────────────

def load_recipes(limit: int = 50000) -> pd.DataFrame:
    """
    Load recipes from DuckDB for recommendation.
    Filters to recipes with descriptions, ratings, and 3+ reviews.
    Orders by ReviewCount DESC so we keep the most-reviewed recipes.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"""
        SELECT {', '.join(RECIPE_COLUMNS)}
        FROM recipes
        WHERE Description IS NOT NULL
          AND AggregatedRating IS NOT NULL
          AND ReviewCount >= 3
        ORDER BY ReviewCount DESC
        LIMIT {limit}
    """).fetchdf()
    con.close()

    logger.info(f"Loaded {len(df):,} recipes for recommendation")
    return df


def build_tfidf(
    recipes: pd.DataFrame,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> tuple[TfidfVectorizer, "sparse matrix"]:
    """
    Build a TF-IDF matrix from recipe descriptions.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range,
    )
    tfidf_matrix = tfidf.fit_transform(recipes["Description"].fillna(""))

    logger.info(f"TF-IDF matrix: {tfidf_matrix.shape}")
    return tfidf, tfidf_matrix


def recommend_similar(
    recipe_name: str,
    recipes: pd.DataFrame,
    tfidf_matrix,
    n: int = 10,
) -> pd.DataFrame:
    """
    Find n recipes most similar to the given recipe (content-based).
    Uses cosine similarity on TF-IDF vectors.
    """
    matches = recipes[
        recipes["Name"].str.contains(recipe_name, case=False, na=False)
    ]
    if matches.empty:
        logger.warning(f"No recipe found matching '{recipe_name}'")
        return pd.DataFrame()

    idx = matches.index[0]
    recipe = recipes.loc[idx]
    logger.info(
        f"Finding recipes similar to: {recipe['Name']} "
        f"({recipe['RecipeCategory']})"
    )

    similarities = cosine_similarity(
        tfidf_matrix[idx : idx + 1], tfidf_matrix
    ).flatten()

    similar_indices = similarities.argsort()[::-1][1 : n + 1]

    results = recipes.iloc[similar_indices][
        ["Name", "RecipeCategory", "AggregatedRating", "ReviewCount"]
    ].copy()
    results["Similarity"] = similarities[similar_indices]

    return results


# ────────────────────────────────────────────────────
# Collaborative filtering
# ────────────────────────────────────────────────────

def load_reviews(
    min_ratings: int = 5,
    recipe_ids: list | None = None,
) -> pd.DataFrame:
    """
    Load user-recipe ratings and filter to active users.

    Args:
        min_ratings: Minimum number of ratings a user must have
                     to be included (default 5).
        recipe_ids: Optional list of RecipeIds to restrict to.
                    Use this to limit the matrix size for collaborative
                    filtering — 5K recipes is manageable (~200MB),
                    112K recipes creates a ~100GB matrix and crashes.

    Returns:
        Filtered DataFrame with columns: AuthorId, RecipeId, Rating.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)

    if recipe_ids is not None:
        # Filter to only specified recipes
        ids_str = ", ".join(str(int(r)) for r in recipe_ids)
        query = f"""
            SELECT AuthorId, RecipeId, Rating
            FROM reviews
            WHERE Rating IS NOT NULL
              AND RecipeId IN ({ids_str})
        """
    else:
        query = """
            SELECT AuthorId, RecipeId, Rating
            FROM reviews
            WHERE Rating IS NOT NULL
              AND RecipeId IN (
                  SELECT RecipeId FROM recipes WHERE ReviewCount >= 3
              )
        """

    reviews = con.execute(query).fetchdf()
    con.close()

    logger.info(
        f"Loaded {len(reviews):,} ratings from "
        f"{reviews['AuthorId'].nunique():,} users "
        f"({reviews['RecipeId'].nunique():,} recipes)"
    )

    # Filter to active users
    counts = reviews["AuthorId"].value_counts()
    active = counts[counts >= min_ratings].index
    filtered = reviews[reviews["AuthorId"].isin(active)]

    logger.info(
        f"Active users ({min_ratings}+ ratings): {len(active):,} — "
        f"{len(filtered):,} ratings"
    )
    return filtered


def build_user_item_matrix(
    reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a user-item rating matrix and compute item-item similarity.
    """
    user_item = reviews.pivot_table(
        index="AuthorId",
        columns="RecipeId",
        values="Rating",
        aggfunc="mean",
    ).fillna(0)

    logger.info(f"User-Item matrix: {user_item.shape}")
    sparsity = (user_item == 0).sum().sum() / user_item.size
    logger.info(f"Sparsity: {sparsity:.2%}")

    user_item_sparse = csr_matrix(user_item.values)
    item_similarity = cosine_similarity(user_item_sparse.T)

    logger.info(f"Item similarity matrix: {item_similarity.shape}")
    return user_item, item_similarity


def recommend_collaborative(
    recipe_id: int,
    user_item: pd.DataFrame,
    item_similarity: np.ndarray,
    recipes: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Recommend recipes based on user rating patterns.
    "Users who rated this recipe also rated these recipes highly."
    """
    if recipe_id not in user_item.columns:
        logger.warning(f"Recipe {recipe_id} not in user-item matrix")
        return pd.DataFrame()

    col_idx = user_item.columns.get_loc(recipe_id)
    sim_scores = item_similarity[col_idx]

    similar_indices = sim_scores.argsort()[::-1][1 : n + 1]
    similar_recipe_ids = user_item.columns[similar_indices]
    similar_scores = sim_scores[similar_indices]

    results = recipes[recipes["RecipeId"].isin(similar_recipe_ids)][
        ["RecipeId", "Name", "RecipeCategory", "AggregatedRating", "ReviewCount"]
    ].copy()

    score_map = dict(zip(similar_recipe_ids, similar_scores))
    results["Similarity"] = results["RecipeId"].map(score_map)
    results = results.sort_values("Similarity", ascending=False)

    return results
