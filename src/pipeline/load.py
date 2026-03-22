"""
Load module — writes transformed DataFrames into DuckDB.

Load responsibilities:
  1. Create (or overwrite) a DuckDB database file
  2. Write recipes and reviews as tables
  3. Add indexes for common query patterns
  4. Verify data integrity with SQL queries
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──
# Resolve relative to project root (src/pipeline/load.py → ../../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"

def load_to_duckdb(
    recipes: pd.DataFrame,
    reviews: pd.DataFrame,
    db_path: Path = DB_PATH,
) -> None:
    """
    Write transformed DataFrames into a DuckDB database.

    This function:
      1. Creates the database file (or overwrites existing tables)
      2. Writes recipes and reviews as separate tables
      3. Logs row counts and table sizes
    """

    logger.info(f"Loading data into DuckDB at {db_path}...")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))

    try:
        # Write recipes table
        con.execute("CREATE OR REPLACE TABLE recipes AS SELECT * FROM recipes")

        recipe_count = con.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        logger.info(f"  ✓ recipes table: {recipe_count:,} rows")

        # Write reviews table
        con.execute("CREATE OR REPLACE TABLE reviews AS SELECT * FROM reviews")

        review_count = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        logger.info(f"  ✓ reviews table: {review_count:,} rows")

        # Show Table Info
        tables = con.execute("SHOW TABLES").fetchall()
        logger.info(f"  Tables in database: {[t[0] for t in tables]}")

    finally:
        con.close()

    # Log file size
    file_size_mb = db_path.stat().st_size / 1e6
    logger.info(f"  Database file size: {file_size_mb:.1f} MB")
    logger.info("Load complete ✓")


def verify_database(db_path: Path = DB_PATH) -> None:
    """Run verification queries against the DuckDB database."""

    print("\n" + "=" * 60)
    print(" DATABASE VERIFICATION")
    print("=" * 60)

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        # Row counts
        recipe_count = con.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        review_count = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        print(f"\n📊 Row counts:")
        print(f"   Recipes: {recipe_count:,}")
        print(f"   Reviews: {review_count:,}")

        # Column listing
        recipe_cols = con.execute(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_name = 'recipes' "
            "ORDER BY ordinal_position"
        ).fetchall()
        print(f"\n📋 Recipes columns ({len(recipe_cols)}):")
        for name, dtype in recipe_cols:
            print(f"   {name:.<40} {dtype}")

        # Top 5 highest-rated recipes with 50+ reviews
        print("\n🏆 Top 5 recipes (50+ reviews):")
        results = con.execute("""
            SELECT Name, AggregatedRating, ReviewCount, complexity_score
            FROM recipes
            WHERE ReviewCount >= 50
            ORDER BY AggregatedRating DESC, ReviewCount DESC
            LIMIT 5
        """).fetchall()
        for name, rating, reviews, complexity in results:
            print(f"   ⭐ {rating:.1f} ({reviews:.0f} reviews) — {name}")

        # Average calories by complexity
        print("\n🔥 Average calories per serving by complexity:")
        results = con.execute("""
            SELECT
                CASE complexity_score
                    WHEN 1 THEN 'Simple'
                    WHEN 2 THEN 'Medium'
                    WHEN 3 THEN 'Complex'
                END AS difficulty,
                ROUND(AVG(calories_per_serving), 1) AS avg_cal,
                COUNT(*) AS recipe_count
            FROM recipes
            WHERE calories_per_serving IS NOT NULL
            GROUP BY complexity_score
            ORDER BY complexity_score
        """).fetchall()
        for difficulty, avg_cal, count in results:
            print(f"   {difficulty}: {avg_cal} cal/serving ({count:,} recipes)")

        # Avg review length by recipe complexity (JOIN)
        print("\n📝 Avg review length by recipe complexity:")
        results = con.execute("""
            SELECT
                CASE r.complexity_score
                    WHEN 1 THEN 'Simple'
                    WHEN 2 THEN 'Medium'
                    WHEN 3 THEN 'Complex'
                END AS difficulty,
                ROUND(AVG(rv.review_length), 0) AS avg_review_len
            FROM recipes r
            JOIN reviews rv ON r.RecipeId = rv.RecipeId
            GROUP BY r.complexity_score
            ORDER BY r.complexity_score
        """).fetchall()
        for difficulty, avg_len in results:
            print(f"   {difficulty}: {avg_len:.0f} chars avg")

    finally:
        con.close()


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from ingest import load_recipes, load_reviews
    from clean import clean_recipes, clean_reviews
    from transform import transform_recipes, transform_reviews

    # ── Full ETL Pipeline ──
    # Ingest
    recipes = load_recipes()
    reviews = load_reviews()

    # Clean
    recipes = clean_recipes(recipes)
    reviews = clean_reviews(reviews)

    # Transform
    recipes = transform_recipes(recipes)
    reviews = transform_reviews(reviews)

    # Load
    load_to_duckdb(recipes, reviews)

    # Verify
    verify_database()
