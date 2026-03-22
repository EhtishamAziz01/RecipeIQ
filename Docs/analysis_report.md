# RecipeIQ: Data Analysis Report

> **Author:** [EhtishamAziz01]
> **Date:** [20.03.2025]
> **Dataset:** Food.com Recipes & Reviews (Kaggle)
> **Tools:** DuckDB, Pandas, Matplotlib, Seaborn, Plotly, Streamlit

---

## 1. Executive Summary

This report summarizes the analysis of **522,517 recipes** and **1.4 million reviews**
from Food.com, spanning 1999–2023. The analysis reveals significant patterns in recipe
complexity, rating behavior, nutrition profiles, and user engagement that will inform
the next phase: building a machine learning recommendation engine.

**Key takeaway:** Recipe ratings are heavily biased toward 5 stars,
making raw ratings unreliable for recommendations. Alternative signals —
review count, review length, and ingredient-based features — will be more
valuable for the ML model.

---

## 2. Dataset Overview

| Metric | Value |
|--------|-------|
| Total recipes | 522,517 |
| Total reviews | 1,401,982 |
| Unique categories | 312 |
| Date range | 1999–2023 |
| Avg rating (all recipes) | 4.63 / 5.0 |
| Median rating | 5.0 |
| Recipes with images | ~XX% |
| Avg ingredients per recipe | 7.9 |

**Data quality notes:**
- All recipes have parsed ingredient counts, instruction steps, and nutrition data
- Derived features include `complexity_score`, `calories_per_serving`, and `review_length`
- HTML entities (e.g., `&amp;`) were decoded during the ETL pipeline

---

## 3. Key Findings

### Finding 1: Dessert Dominates the Platform
Dessert recipes account for **62,072** entries (11.9% of all recipes), nearly double
the second-place category (Lunch/Snacks at 32,586). This suggests a strong
self-selection bias — home bakers are the platform's core users.

**Implication:** Any recommendation model must handle category imbalance.
A model trained naively would over-recommend desserts.

### Finding 2: Ratings Are Nearly Useless
The median rating is **5.0** and 77%+ of recipes are rated 4–5 stars. The rating
distribution is extremely left-skewed. The difference between "good" and "great"
recipes is invisible in the ratings.

**Implication:** Star ratings alone cannot power a recommendation engine. We need
alternative quality signals — review count, review sentiment, and engagement metrics.

### Finding 3: Complexity Barely Affects Ratings
Average ratings across complexity levels (Simple: 4.66, Medium: 4.62, Complex: 4.64)
are nearly identical. However, **complex recipes get 35% more reviews** (5.8 vs 4.3
avg reviews per recipe) — suggesting that complexity drives engagement, not quality
perception.

**Implication:** Recipe complexity is a valuable feature for recommendations, but as
an engagement predictor rather than a quality signal.

### Finding 4: The Golden Era Was 2006–2008
Recipe publishing peaked at ~70,000/year during 2006–2008, then sharply declined.
Post-2015, fewer than 5,000 recipes were published annually. Average ratings
increased post-2018 to ~4.9 — likely survivorship bias (only dedicated users remain).

**Implication:** The model should weight recency carefully. Older recipes have more
reviews but may be outdated.

### Finding 5: Happy Reviewers Write More
Contrary to the common assumption that angry customers write longer reviews, our data
shows the opposite. Median review length increases with rating: 0-star reviews average
193 characters while 4-star reviews average 248 characters. Satisfied cooks share
modifications, tips, and stories.

**Implication:** Review length is a positive engagement signal, not a complaint
indicator. Longer reviews correlate with recipe success.

### Finding 6: Nutrition Profiles Cluster Predictably
The heatmap reveals expected patterns — seafood categories (Lobster, Mussels) are
highest in protein and sodium, while Shakes/Smoothies lead in sugar. However,
**Japanese and Thai categories show unusually high sodium** (483 and 314 per serving),
which could be a data quality issue (soy sauce measurements).

**Implication:** Nutrition-based filtering and clustering are viable features for
the recommendation engine (e.g., "healthy alternatives to your favorites").

### Finding 7: The Ingredient Sweet Spot Is Minimal
Recipes with 1–3 ingredients have the highest average ratings (4.67–4.69), with a
slight decline through 5–10 ingredients before recovering at 20+. The trend is subtle
but consistent — simplicity correlates with satisfaction.

**Implication:** Ingredient count is a lightweight feature that can approximate
recipe accessibility for cold-start recommendations.

---

## 4. Visual Evidence

The following charts were generated in `notebooks/02_visualizations.ipynb` and saved
to `data/processed/figures/`.

| Figure | File | Key Insight |
|--------|------|-------------|
| Top 15 Categories | `01_top_categories.png` | Dessert = 2x the next category |
| Rating Distribution | `02_rating_distribution.png` | Extreme left skew, median = 5.0 |
| Complexity Comparison | `03_complexity_comparison.png` | Ratings flat, reviews increase |
| Nutrition Heatmap | `04_nutrition_heatmap.png` | Category nutrition clusters |
| Publishing Trends | `05_publishing_trends.png` | Peak 2006–2008, sharp decline |
| Review Length vs Rating | `06_review_length_vs_rating.png` | Happy reviewers write more |
| Ingredient Sweet Spot | `07_ingredient_sweet_spot.png` | Fewer ingredients → higher rating |
| Calorie Violins | `08_calorie_violins.png` | Similar distributions across complexity |

An interactive version of these charts is available in the Streamlit dashboard:
```bash
uv run streamlit run src/dashboard/app.py
```

---

## 5. Limitations & Caveats

1. **Self-selection bias** — Users who rate recipes are not representative of all cooks.
   People who had a bad experience may simply not rate (or not even try the recipe).

2. **Rating inflation** — The 5-star ceiling and social pressure inflate ratings.
   This is common across recipe platforms (Allrecipes shows the same pattern).

3. **Nutrition data accuracy** — Nutrition values are user-submitted and not verified.
   Outliers exist (e.g., recipes with 0 calories or 50,000mg sodium).

4. **Temporal bias** — The 2006–2008 peak means most recipes are 15+ years old.
   Food trends, demographics, and dietary preferences have shifted significantly.

5. **Missing data** — Not all recipes have images, complete nutrition data, or reviews.
   The ML model will need to handle missing features gracefully.

---

## 6. Recommendations for Machine Learning

Based on the analysis, the following features and approaches are recommended for the
recommendation engine:

### Feature Engineering Priorities

| Feature | Source | Rationale |
|---------|--------|-----------|
| Review count (log) | `ReviewCount` | Better quality signal than rating |
| Review length (avg) | `reviews.review_length` | Engagement indicator |
| Ingredient count | `ingredient_count` | Accessibility proxy |
| Complexity score | `complexity_score` | Engagement predictor |
| Calories per serving | `calories_per_serving` | Health-conscious filtering |
| Category embedding | `RecipeCategory` | Handle 312 categories via embeddings |
| Recipe age | `DatePublished` | Recency weighting |

### Model Considerations

1. **Don't use raw ratings as the target** — use a composite engagement score
   (weighted combination of review count, review length, and rating)
2. **Handle category imbalance** — stratified sampling or category weighting
3. **Cold-start strategy** — for recipes with no reviews, fall back to
   content-based features (ingredients, complexity, nutrition)
4. **Evaluation metric** — NDCG (Normalized Discounted Cumulative Gain) rather than
   accuracy, since this is a ranking problem

---

## 7. Appendix

### A. SQL Queries Used
Full SQL queries are available in `notebooks/01_sql_analysis.ipynb`.

### B. Visualization Code
Chart generation code is in `notebooks/02_visualizations.ipynb`.

### C. Interactive Dashboard
The Streamlit dashboard is in `src/dashboard/app.py`.
Run with: `uv run streamlit run src/dashboard/app.py`

### D. ETL Pipeline
Data was processed through the pipeline in `src/pipeline/`:
- `ingest.py` — CSV loading
- `clean.py` — Parsing, HTML decoding, type conversion
- `transform.py` — Derived features (complexity, per-serving nutrition)
- `load.py` — DuckDB loading

---
