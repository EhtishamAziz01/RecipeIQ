# 🧠 RecipeIQ

An end-to-end food intelligence platform built on 960K+ recipes and 1.6M+ reviews.

## Project Overview

| Phase | Role | What's Built |
|-------|------|-------------|
| 1 | Data Engineer | ETL pipeline → DuckDB warehouse |
| 2 | Data Analyst | EDA + Streamlit dashboard |
| 3 | Data Scientist | Rating predictor + recipe recommender |
| 4 | AI Engineer | Sentiment analysis + Gemini-powered recipe chatbot |

## Tech Stack
- **Python 3.12** (uv)
- **Pandas / NumPy** — data processing
- **DuckDB** — analytical database
- **Prefect** — orchestration
- **Streamlit** — dashboards
- **Scikit-learn / MLflow** — ML models
- **Gemini + ChromaDB** — AI / RAG
- **FastAPI** — API serving

## Setup
```bash
git clone <repo-url>
cd RecipeIQ
uv sync
# Place raw data files in data/raw/

## Status
- [ ] Phase 1: Data Engineering
- [ ] Phase 2: Data Analysis
- [ ] Phase 3: Data Science
- [ ] Phase 4: AI Engineering
