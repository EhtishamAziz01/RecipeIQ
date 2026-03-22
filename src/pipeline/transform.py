"""
Transformation module — creates derived features from cleaned data.

Transformation responsibilities:
  1. Create count-based features (ingredient_count, instruction_steps)
  2. Calculate normalized nutrition metrics (per-serving)
  3. Build a composite complexity score
  4. Add time-based features (recipe age)
  5. Drop raw columns that are no longer needed in the warehouse
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def transform_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """Transform cleaned recipes into analytics-ready format."""

    logger.info("Transformation Recipe...")

    df = df.copy()
    rows_before = len(df)
 # ──- Count based features ──-
    if "RecipeIngredientParts" in df.columns:
        df["ingredient_count"] = df["RecipeIngredientParts"].apply(len)

        logger.info(
            f"  ingredient_count — "
            f"min: {df['ingredient_count'].min()}, "
            f"max: {df['ingredient_count'].max()}, "
            f"mean: {df['ingredient_count'].mean():.1f}"
        )

    # How many steps in the instructions?
    if "RecipeInstructions" in df.columns:
        df["instruction_steps"] = df["RecipeInstructions"].apply(len)

        logger.info(
            f"--- instruction_steps ---"
            f"min: {df['instruction_steps'].min()}, "
            f"max: {df['instruction_steps'].max()}, "
            f"mean: {df['instruction_steps'].mean():.1f}"
        )

    # Does the recipe have at least one image?
    if "Images" in df.columns:
        df["has_image"] = df["Images"].apply(lambda x: len(x) > 0)

        logger.info(
            f"  has_image — "
            f"{df['has_image'].sum():,} recipes with images "
            f"({df['has_image'].mean() * 100:.1f}%)"
        )

    # How many Keywords/tags?
    if "Keywords" in df.columns:
        df["keyword_count"] = df["Keywords"].apply(len)

    # ──- Nutrition per serving ───
    nutrition_cols = [
        "Calories",
        "FatContent",
        "SaturatedFatContent",
        "CholesterolContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "ProteinContent",
    ]

    if "RecipeServings" in df.columns:
        safe_servings = df["RecipeServings"].replace(0, np.nan)

        for col in nutrition_cols:
            if col in df.columns:
                new_col = f"{col.replace('Content', '').lower()}_per_serving"
                df[new_col] = (df[col] / safe_servings)

        logger.info(f"  Created {len(nutrition_cols)} per-serving nutrition columns")

    # ──- Complexity Score ───
    if all(col in df.columns for col in ["ingredient_count", "instruction_steps"]):
        conditions = [
            (df["ingredient_count"] <= 5) & (df["instruction_steps"] <= 4),
            (df["ingredient_count"] > 12) | (df["instruction_steps"] > 10),
        ]
        choices = [1, 3] # 1 = simple, 3 = complex

        df["complexity_score"] = np.select(conditions, choices, default=2)

        logger.info(
            f"--- complexity_score distribution ---"
            f"simple: {(df['complexity_score'] == 1).sum():,}, "
            f"medium: {(df['complexity_score'] == 2).sum():,}, "
            f"complex: {(df['complexity_score'] == 3).sum():,}"
        )

    # ── 4. Time-based features ───

    if "DatePublished" in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        df["recipe_age_days"] = (now - df["DatePublished"]).dt.days

        logger.info(
            f"  recipe_age_days — "
            f"oldest: {df['recipe_age_days'].max():,} days, "
            f"newest: {df['recipe_age_days'].min():,} days"
        )

    # ── 5. Drop raw list/text columns ───

    columns_to_drop = [
        "RecipeIngredientParts",
        "RecipeIngredientQuantities",
        "RecipeInstructions",
        "Images",
        "Keywords",
        "CookTime",
        "PrepTime",
        "TotalTime",
        "RecipeYield",
    ]

    existing_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_to_drop)

    logger.info(f" Dropped {len(existing_to_drop)} raw columns")

    # --- LOG SUMMARY ---
    logger.info(
        f"Transformation complete. "
        f"Rows: {rows_before:,} → {len(df):,}. "
        f"Columns: {len(df.columns)} "
        f"(was 31 after cleaning)"
    )

    return df


def transform_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Transform reviews into analytics-ready format."""

    logger.info("Transforming reviews...")

    df = df.copy()


    # Review text length - useful for filtering quality reviews

    if "Review" in df.columns:
        df["review_length"] = df["Review"].str.len()
        logger.info(
            f"  review_length — "
            f"min: {df['review_length'].min()}, "
            f"max: {df['review_length'].max():,}, "
            f"mean: {df['review_length'].mean():.0f} chars"
        )

    logger.info(
        f"Reviews transformation complete. "
        f"Columns: {len(df.columns)}"
    )

    return df

# ════════════════════════════════════════════════════════════════
# MAIN — Run transformation standalone for testing
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from ingest import load_recipes, load_reviews
    from clean import clean_recipes, clean_reviews

    # Load -> Clean -> transform_reviews

    recipes = load_recipes()
    recipes = clean_recipes(recipes)
    recipes = transform_recipes(recipes)

    reviews = load_reviews()
    reviews = clean_reviews(reviews)
    reviews = transform_reviews(reviews)

    # Show Final Schema
    print("\n" + "=" * 60)
    print(" FINAL RECIPES SCHEMA")
    print("=" * 60)
    print(f"Shape: {recipes.shape[0]:,} rows × {recipes.shape[1]} columns\n")
    for col in recipes.columns:
        null_count = recipes[col].isnull().sum()
        null_info = f"  ({null_count:,} nulls)" if null_count > 0 else ""
        print(f"  {col:.<40} {str(recipes[col].dtype):<20}{null_info}")

    print("\n" + "=" * 60)
    print(" FINAL REVIEWS SCHEMA")
    print("=" * 60)
    print(f"Shape: {reviews.shape[0]:,} rows × {reviews.shape[1]} columns\n")
    for col in reviews.columns:
        null_count = reviews[col].isnull().sum()
        null_info = f"  ({null_count:,} nulls)" if null_count > 0 else ""
        print(f"  {col:.<40} {str(reviews[col].dtype):<20}{null_info}")

    # Show sample of derived features
    print("\n── Sample Derived Features ──")
    sample_cols = [
        "Name", "ingredient_count", "instruction_steps",
        "complexity_score", "calories_per_serving", "protein_per_serving"
    ]
    existing = [c for c in sample_cols if c in recipes.columns]
    print(recipes[existing].head(10).to_string())
