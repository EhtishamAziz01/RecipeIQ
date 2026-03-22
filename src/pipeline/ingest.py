# Imports
import logging
from pathlib import Path
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Constants
RAW_DATA_DIR = Path("data/raw")
RECIPES_FILE = RAW_DATA_DIR / "recipes.csv"
REVIEWS_FILE = RAW_DATA_DIR / "reviews.csv"

def load_recipes() -> pd.DataFrame:
    """
    Load the raw recipes dataset from CSV.

    Returns:
        pd.DataFrame with all recipe columns as-is from the raw file.

    Raises:
        FileNotFoundError: If the recipes file doesn't exist.
    """

    if not RECIPES_FILE.exists():
        raise FileNotFoundError(
            f"Recipe file not found at {RECIPES_FILE}."
            f"Make sure you've placed the raw data in {RAW_DATA_DIR}/"
        )

    logger.info(f"Loading recipes from {RECIPES_FILE}...")

    df = pd.read_csv(RECIPES_FILE)

    logger.info(
        f"Loaded {df.shape[0]:,} recipes with {df.shape[1]} columns. "
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB"
    )

    return df

def load_reviews() -> pd.DataFrame:
    """
    Load the raw reviews dataset from CSV.

    Returns:
        pd.DataFrame with all review columns as-is from the raw file.

    Raises:
        FileNotFoundError: If the reviews file doesn't exist.
    """
    if not REVIEWS_FILE.exists():
        raise FileNotFoundError(
            f"Reviews file not found at {REVIEWS_FILE}."
            f"Make sure you've placed the raw data in {RAW_DATA_DIR}/"
        )

    logger.info(f"Loading reviews from {REVIEWS_FILE}...")

    df = pd.read_csv(REVIEWS_FILE)

    logger.info(
        f"Loaded {df.shape[0]:,} reviews with {df.shape[1]} columns. "
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB"
    )

    return df

def preview_data(df: pd.DataFrame, name: str, nrows: int = 5) -> None:
    """
    Print a summary of a DataFrame for inspection.

    Args:
        df: The DataFrame to preview.
        name: A label for the output (e.g., "recipes").
        n_rows: Number of sample rows to show.
    """
    print(f"\n{'='*60}")
    print(f"{name.upper()} PREVIEW")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum()/ 1e6:.1f} MB")

    print(f"\n── Columns and Types ──")
    for col in df.columns:
        # Count nulls for this column
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        null_info = f" ({null_count:,} nulls, {null_pct:.1f}%)" if null_count > 0 else ""
        print(f"  {col:.<40} {str(df[col].dtype):<15}{null_info}")

    print(f"\n── Sample Rows ──")
    print(df.head(nrows))


if __name__ == "__main__":
    # Setup up basic logging so we see INFO messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    recipes = load_recipes()
    preview_data(recipes, "recipes")

    reviews = load_reviews()
    preview_data(reviews, "reviews")
