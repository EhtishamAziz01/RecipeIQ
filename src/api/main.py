"""
RecipeIQ FastAPI application.

Run locally:
    uv run uvicorn src.api.main:app --reload --port 8000

API docs available at:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import duckdb
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"
MODELS_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="RecipeIQ API",
    description="Recipe intelligence platform — search, recommend, and chat about recipes.",
    version="1.0.0",
)

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──

class RecipeResponse(BaseModel):
    recipe_id: int
    name: str
    category: str | None
    rating: float | None
    review_count: int | None
    calories_per_serving: float | None
    protein_per_serving: float | None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    limit: int = Field(5, ge=1, le=20)

class SearchResult(BaseModel):
    name: str
    category: str
    rating: float
    reviews: int
    distance: float

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field("default", description="Chat session identifier")

class ChatResponse(BaseModel):
    reply: str
    sources: list[dict]


# ── State ──
# Store chatbot sessions in memory (for simplicity)
_chat_sessions: dict = {}


# ── Endpoints ──

@app.get("/health")
def health_check():
    """Service health check."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/recipe/{recipe_id}", response_model=RecipeResponse)
def get_recipe(recipe_id: int):
    """Get details for a specific recipe."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    row = con.execute("""
        SELECT RecipeId, Name, RecipeCategory, AggregatedRating,
               ReviewCount, calories_per_serving, protein_per_serving
        FROM recipes
        WHERE RecipeId = ?
    """, [recipe_id]).fetchone()
    con.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")

    return RecipeResponse(
        recipe_id=row[0], name=row[1], category=row[2],
        rating=row[3], review_count=int(row[4]) if row[4] else None,
        calories_per_serving=row[5], protein_per_serving=row[6],
    )


@app.get("/categories")
def list_categories():
    """List all recipe categories with counts."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("""
        SELECT RecipeCategory, COUNT(*) as count,
               ROUND(AVG(AggregatedRating), 2) as avg_rating
        FROM recipes
        WHERE RecipeCategory IS NOT NULL
        GROUP BY RecipeCategory
        ORDER BY count DESC
    """).fetchall()
    con.close()

    return [
        {"category": r[0], "count": r[1], "avg_rating": r[2]}
        for r in rows
    ]


@app.post("/search", response_model=list[SearchResult])
def semantic_search(request: SearchRequest):
    """
    Semantic recipe search using Gemini embeddings.
    Finds recipes by meaning, not just keywords.
    """
    try:
        from src.ai.embeddings import search_recipes
        results = search_recipes(request.query, n=request.limit)
        return [
            SearchResult(
                name=r["name"], category=r["category"],
                rating=r["rating"], reviews=r["reviews"],
                distance=r["distance"],
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Conversational recipe assistant.
    Maintains separate chat history per session_id.
    """
    try:
        from src.ai.chatbot import RecipeChatbot

        if request.session_id not in _chat_sessions:
            _chat_sessions[request.session_id] = RecipeChatbot()

        bot = _chat_sessions[request.session_id]
        answer = bot.chat(request.message)

        # Get last mentioned sources
        recent_sources = bot.mentioned_recipes[-5:] if bot.mentioned_recipes else []

        return ChatResponse(
            reply=answer,
            sources=[
                {"name": s["name"], "category": s["category"], "rating": s["rating"]}
                for s in recent_sources
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.get("/search/keyword")
def keyword_search(q: str, limit: int = 10):
    """
    Simple keyword search for recipes by name.
    Fallback when vector search is unavailable.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("""
        SELECT RecipeId, Name, RecipeCategory, AggregatedRating, ReviewCount
        FROM recipes
        WHERE Name ILIKE ?
        ORDER BY ReviewCount DESC
        LIMIT ?
    """, [f"%{q}%", limit]).fetchall()
    con.close()

    return [
        {
            "recipe_id": r[0], "name": r[1], "category": r[2],
            "rating": r[3], "review_count": int(r[4]) if r[4] else 0,
        }
        for r in rows
    ]
