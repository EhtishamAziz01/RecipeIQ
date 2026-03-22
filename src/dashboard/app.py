"""
RecipeIQ Dashboard — Interactive Recipe Intelligence
Run with: uv run streamlit run src/dashboard/app.py
"""

# IMPORTS
import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="RecipeIQ Dashboard",
    page_icon="🍳",
    layout="wide",            # Use full browser width
    initial_sidebar_state="expanded",
)


# DATA LOADING WTIH CACHING

# Resolve relative to project root (src/dashboard/app.py → ../../)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"

@st.cache_data
def load_recipes():
    """Load recipes from DuckDB. Cached so it only runs once."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT
            Name, RecipeCategory, AggregatedRating, ReviewCount,
            ingredient_count, instruction_steps, complexity_score,
            calories_per_serving, protein_per_serving, fat_per_serving,
            sugar_per_serving, fiber_per_serving, sodium_per_serving,
            CookTime_minutes, PrepTime_minutes, TotalTime_minutes,
            DatePublished, has_image
        FROM recipes
        WHERE AggregatedRating IS NOT NULL
    """).fetchdf()
    con.close()
    return df

@st.cache_data
def load_review_stats():
    """Load aggregated review stats. Cached."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""
        SELECT
            Rating,
            COUNT(*) AS review_count,
            ROUND(AVG(review_length), 0) AS avg_length
        FROM reviews
        GROUP BY Rating
        ORDER BY Rating
    """).fetchdf()
    con.close()
    return df

recipes = load_recipes()
review_stats = load_review_stats()

# ════════════════════════════════════════════════════════════
# SIDEBAR — Filters
# ════════════════════════════════════════════════════════════

st.sidebar.title("🔍 Filters")
st.sidebar.markdown("Adjust filters to explore the data")

# ── Category filter ──
all_categories = sorted(recipes["RecipeCategory"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "Recipe Categories",
    options=all_categories,
    default=[],                     # Empty = show all
    placeholder="All categories",
)

# ── Complexity filter ──
complexity_map = {1: "🟢 Simple", 2: "🟡 Medium", 3: "🔴 Complex"}
selected_complexity = st.sidebar.multiselect(
    "Complexity",
    options=[1, 2, 3],
    format_func=lambda x: complexity_map[x],
    default=[1, 2, 3],
)

# ── Calorie range slider ──
cal_min, cal_max = st.sidebar.slider(
    "Calories per Serving",
    min_value=0,
    max_value=2000,
    value=(0, 1000),                # Default range
    step=50,
)

# ── Rating filter ──
min_rating = st.sidebar.slider(
    "Minimum Rating",
    min_value=1.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
)

# ── Apply filters ──
filtered = recipes.copy()

if selected_categories:
    filtered = filtered[filtered["RecipeCategory"].isin(selected_categories)]

filtered = filtered[filtered["complexity_score"].isin(selected_complexity)]

filtered = filtered[
    (filtered["calories_per_serving"] >= cal_min)
    & (filtered["calories_per_serving"] <= cal_max)
]

filtered = filtered[filtered["AggregatedRating"] >= min_rating]

# ════════════════════════════════════════════════════════════
# MAIN AREA — Header and KPIs
# ════════════════════════════════════════════════════════════

st.title("🍳 RecipeIQ Dashboard")
st.markdown("Explore **522K+ recipes** — filter, compare, and discover patterns")

# ── KPI Cards ──
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    label="📊 Recipes",
    value=f"{len(filtered):,}",
    delta=f"{len(filtered) - len(recipes):,} from filters" if len(filtered) != len(recipes) else None,
)

kpi2.metric(
    label="⭐ Avg Rating",
    value=f"{filtered['AggregatedRating'].mean():.2f}",
)

kpi3.metric(
    label="🍽️ Categories",
    value=f"{filtered['RecipeCategory'].nunique():,}",
)

kpi4.metric(
    label="🧂 Avg Ingredients",
    value=f"{filtered['ingredient_count'].mean():.1f}",
)

st.divider()

# ════════════════════════════════════════════════════════════
# CHARTS — Row 1
# ════════════════════════════════════════════════════════════

chart1, chart2 = st.columns(2)

# ── Chart 1: Top Categories ──
with chart1:
    st.subheader("Top Categories")

    cat_counts = (
        filtered["RecipeCategory"]
        .value_counts()
        .head(15)
        .reset_index()
    )
    cat_counts.columns = ["Category", "Count"]

    fig = px.bar(
        cat_counts,
        x="Count",
        y="Category",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        coloraxis_showscale=False,
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

# ── Chart 2: Rating Distribution ──
with chart2:
    st.subheader("Rating Distribution")

    fig = px.histogram(
        filtered,
        x="AggregatedRating",
        nbins=20,
        color_discrete_sequence=["#4C72B0"],
    )
    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Count",
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

# ════════════════════════════════════════════════════════════
# CHARTS — Row 2
# ════════════════════════════════════════════════════════════

chart3, chart4 = st.columns(2)

# ── Chart 3: Complexity Breakdown ──
with chart3:
    st.subheader("Complexity Breakdown")

    complexity_data = (
        filtered.groupby("complexity_score")
        .agg(
            count=("Name", "size"),
            avg_rating=("AggregatedRating", "mean"),
            avg_reviews=("ReviewCount", "mean"),
        )
        .reset_index()
    )
    complexity_data["label"] = complexity_data["complexity_score"].map(
        {1: "Simple", 2: "Medium", 3: "Complex"}
    )

    fig = px.bar(
        complexity_data,
        x="label",
        y="count",
        color="label",
        color_discrete_map={
            "Simple": "#55A868",
            "Medium": "#F0C75E",
            "Complex": "#C44E52",
        },
        text="count",
        hover_data={"avg_rating": ":.2f", "avg_reviews": ":.1f"},
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Recipe Count",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    st.plotly_chart(fig, width="stretch")

# ── Chart 4: Nutrition Comparison ──
with chart4:
    st.subheader("Nutrition by Complexity")

    nutrition_by_complexity = (
        filtered.groupby("complexity_score")
        .agg(
            calories=("calories_per_serving", "mean"),
            protein=("protein_per_serving", "mean"),
            fat=("fat_per_serving", "mean"),
            sugar=("sugar_per_serving", "mean"),
        )
        .reset_index()
    )
    nutrition_by_complexity["label"] = nutrition_by_complexity[
        "complexity_score"
    ].map({1: "Simple", 2: "Medium", 3: "Complex"})

    # Melt to long format for grouped bar
    nutrition_long = nutrition_by_complexity.melt(
        id_vars=["label"],
        value_vars=["calories", "protein", "fat", "sugar"],
        var_name="Nutrient",
        value_name="Per Serving",
    )

    fig = px.bar(
        nutrition_long,
        x="Nutrient",
        y="Per Serving",
        color="label",
        barmode="group",
        color_discrete_map={
            "Simple": "#55A868",
            "Medium": "#F0C75E",
            "Complex": "#C44E52",
        },
    )
    fig.update_layout(
        legend_title="Complexity",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

# ════════════════════════════════════════════════════════════
# CHARTS — Row 3
# ════════════════════════════════════════════════════════════

chart5, chart6 = st.columns(2)

# ── Chart 5: Publishing Timeline ──
with chart5:
    st.subheader("Publishing Timeline")

    timeline = filtered.copy()
    timeline["year"] = pd.to_datetime(timeline["DatePublished"]).dt.year
    timeline = timeline[timeline["year"].between(1999, 2023)]
    yearly = (
        timeline.groupby("year")
        .agg(count=("Name", "size"), avg_rating=("AggregatedRating", "mean"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly["year"],
            y=yearly["count"],
            name="Recipes",
            marker_color="#4C72B0",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["year"],
            y=yearly["avg_rating"],
            name="Avg Rating",
            yaxis="y2",
            line=dict(color="#C44E52", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        yaxis=dict(title="Recipes Published"),
        yaxis2=dict(title="Avg Rating", overlaying="y", side="right", range=[4, 5]),
        legend=dict(x=0.01, y=0.99),
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")

# ── Chart 6: Ingredient Sweet Spot ──
with chart6:
    st.subheader("Ingredient Sweet Spot")

    ingredient_data = (
        filtered[filtered["ingredient_count"].between(1, 30)]
        .groupby("ingredient_count")
        .agg(
            count=("Name", "size"),
            avg_rating=("AggregatedRating", "mean"),
            avg_cal=("calories_per_serving", "mean"),
        )
        .reset_index()
    )
    ingredient_data = ingredient_data[ingredient_data["count"] >= 50]

    fig = px.scatter(
        ingredient_data,
        x="ingredient_count",
        y="avg_rating",
        size="count",
        color="avg_cal",
        color_continuous_scale="YlOrRd",
        labels={
            "ingredient_count": "Number of Ingredients",
            "avg_rating": "Average Rating",
            "count": "Recipe Count",
            "avg_cal": "Avg Calories",
        },
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════
# DATA EXPLORER
# ════════════════════════════════════════════════════════════

st.divider()
st.subheader("🔎 Data Explorer")
st.markdown(f"Showing **{len(filtered):,}** recipes matching your filters")

# ── Search box ──
search = st.text_input("Search recipes by name", placeholder="e.g. banana bread")

display_df = filtered.copy()
if search:
    display_df = display_df[
        display_df["Name"].str.contains(search, case=False, na=False)
    ]

# ── Display columns (clean names for the table) ──
display_cols = {
    "Name": "Recipe",
    "RecipeCategory": "Category",
    "AggregatedRating": "Rating",
    "ReviewCount": "Reviews",
    "ingredient_count": "Ingredients",
    "calories_per_serving": "Cal/Serving",
    "complexity_score": "Complexity",
}

st.dataframe(
    display_df[list(display_cols.keys())]
    .rename(columns=display_cols)
    .sort_values("Reviews", ascending=False)
    .head(100)
    .style.format({"Rating": "{:.1f}", "Cal/Serving": "{:.0f}", "Reviews": "{:.0f}"}),
    width="stretch",
    height=400,
)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "RecipeIQ Dashboard • Built with Streamlit & Plotly • "
    f"Data: {len(recipes):,} recipes from Food.com"
)
