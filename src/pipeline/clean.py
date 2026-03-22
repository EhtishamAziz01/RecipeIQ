import html
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# PARSING FUNCTIONS
# ════════════════════════════════════════════════════════════════
def parse_r_list(value: Optional[str]) -> list[str]:
    """ Parse R-style c("item1", "item2") strings into Python lists."""

    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":

        return []
    items = re.findall(r'"([^"]*)"', value)
    return items


def parse_duration(value: Optional[str]) -> Optional[float]:
    """ Converts ISO 8601 duration string to total minutes."""

    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return None

    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", value)

    if not match:
        logger.warning(f"Could not parse duration: {value}")
        return None

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0

    total_minutes = hours * 60 + minutes
    if total_minutes == 0 and value != "PT0M":
        return None

    return float(total_minutes)


# ════════════════════════════════════════════════════════════════
# MAIN CLEANING FUNCTIONS
# ════════════════════════════════════════════════════════════════

def clean_recipes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw recipes DataFrame.

    Cleaning steps (in order):
        1. Parse R-style list columns into Python lists
        2. Convert ISO 8601 durations to numeric minutes
        3. Parse date strings into datetime objects
        4. Log a summary of what was cleaned
    """

    logger.info(f"Cleaning recipes...")

    df = df.copy()

    rows_before = len(df)

    # Decode HTML entities in text columns (&amp; → &, &quot; → ", etc.)
    # Loop until stable to handle multi-level encoding
    def _unescape(val):
        if not isinstance(val, str):
            return val
        prev = None
        while prev != val:
            prev = val
            val = html.unescape(val)
        return val

    for col in ["Name", "Description"]:
        if col in df.columns:
            df[col] = df[col].apply(_unescape)
    logger.info("Decoded HTML entities in Name, Description")

    # Parse R-style list columns

    list_columns = [
        "RecipeIngredientParts", "RecipeIngredientQuantities",
        "Keywords",
        "RecipeInstructions",
        "Images",
    ]

    for col in list_columns:
        if col in df.columns:
            logger.info(f"Parsing R-style list column: {col}")
            df[col] = df[col].apply(parse_r_list)

    # Convert duration columns to minutes

    duration_columns = [
        "CookTime",
        "PrepTime",
        "TotalTime"
    ]

    for col in duration_columns:
        if col in df.columns:
            logger.info(f"Parsing duration column: {col}")

            new_col = f"{col}_minutes"
            df[new_col] = df[col].apply(parse_duration)

    # Parse date columns

    if "DatePublished" in df.columns:
        logger.info("Parsing DatePublished to datetime...")
        df["DatePublished"] = pd.to_datetime(df["DatePublished"], errors="coerce")

    # Log clean summary
    rows_after = len(df)
    logger.info(
        f"Recipe cleaning complete. "
        f"Rows: {rows_before:,} -> {rows_after:,} "
        f"Columns: {len(df.columns)}"
    )

    return df

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean the raw reviews DataFrame.

    Cleaning steps (in order):
        1. Parse date strings into datetime objects
        2. Drop reviews with null review text
        3. Log a summary of what was cleaned
    """

    logger.info(f"Cleaning reviews...")

    df = df.copy()
    rows_before = len(df)

    # Parse date columns
    for col in ["DateSubmitted", "DateModified"]:
        if col in df.columns:
            logger.info(f"Parsing {col} to datetime...")
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop reviews with no text
    if "Review" in df.columns:
        null_reviews = df["Review"].isnull().sum()
        if null_reviews > 0:
            logger.info(f"Dropping {null_reviews:,} reviews with null text...")
            df = df.dropna(subset=["Review"])

    # Log cleaning summary
    rows_after = len(df)
    logger.info(
        f"Reviews cleaning complete. "
        f"Rows: {rows_before:,} -> {rows_after:,} "
        f"Columns: {len(df.columns)}"
    )

    return df


# ════════════════════════════════════════════════════════════════
# MAIN — Run cleaning standalone for testing
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from ingest import load_recipes, load_reviews, preview_data

    # Load raw data
    recipes_raw = load_recipes()
    reviews_raw = load_reviews()

    # Clean Both
    recipes_clean = clean_recipes(recipes_raw)
    reviews_clean = clean_reviews(reviews_raw)

    # Preview the cleaned data
    preview_data(recipes_clean, "cleaned recipes")
    preview_data(reviews_clean, "cleaned reviews")

    # Show examples of parsed fields
    print("\n--- Parsed Ingredients (first 3)---")
    for i, ingredients in enumerate(recipes_clean["RecipeIngredientParts"].head(3)):
        print(f" Recipe {i}: {ingredients[:5]}...")

    print("\n--- Duration Examples ---")
    sample = recipes_clean[["CookTime", "CookTime_minutes"]].head(5)
    print(sample.to_string())
