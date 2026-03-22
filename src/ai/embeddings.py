"""
Recipe embedding pipeline using Gemini Embedding API + ChromaDB.
"""

import duckdb
import chromadb
from google import genai
from pathlib import Path
import os
import time
from dotenv import load_dotenv

# ── Config ──
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "recipeiq.duckdb"
VECTOR_DIR = PROJECT_ROOT / "data" / "vectors" / "chroma_db"

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBEDDING_MODEL = "gemini-embedding-001"


def get_recipes(limit: int = 50000) -> list[dict]:
    """
    Load recipes from DuckDB and format for embedding.
    We create a rich text representation combining name, category,
    description, and nutrition highlights.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(f"""
        SELECT
            RecipeId, Name, RecipeCategory, Description,
            AggregatedRating, ReviewCount,
            calories_per_serving, protein_per_serving,
            fat_per_serving, ingredient_count
        FROM recipes
        WHERE Description IS NOT NULL
          AND AggregatedRating IS NOT NULL
          AND ReviewCount >= 3
        ORDER BY ReviewCount DESC
        LIMIT {limit}
    """).fetchdf()
    con.close()

    recipes = []
    for _, row in rows.iterrows():
        # Create embedding text: rich representation for the model
        text = (
            f"{row['Name']}. "
            f"Category: {row['RecipeCategory']}. "
            f"{row['Description']} "
            f"Nutrition per serving: {row['calories_per_serving']:.0f} cal, "
            f"{row['protein_per_serving']:.0f}g protein, "
            f"{row['fat_per_serving']:.0f}g fat. "
            f"{row['ingredient_count']} ingredients."
        )
        recipes.append({
            "id": str(row["RecipeId"]),
            "text": text[:2000],  # Gemini has input limits
            "metadata": {
                "name": row["Name"],
                "category": row["RecipeCategory"] or "Unknown",
                "rating": float(row["AggregatedRating"]),
                "reviews": int(row["ReviewCount"]),
                "calories": float(row["calories_per_serving"]),
                "protein": float(row["protein_per_serving"]),
            },
        })
    return recipes


def embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embed a batch of texts using Gemini Embedding API.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config={
            "task_type": task_type,
        },
    )
    return [e.values for e in result.embeddings]


def build_vector_store(batch_size: int = 100):
    """
    Build ChromaDB collection from all recipes.
    Embeds recipes in batches and stores them.
    """
    recipes = get_recipes()
    print(f"Embedding {len(recipes):,} recipes...")

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))

    # Delete existing collection if rebuilding
    try:
        chroma_client.delete_collection("recipes")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="recipes",
        metadata={"description": "Recipe embeddings for semantic search"},
    )

    # Process in batches (Gemini API: 3000 req/min limit)
    for i in range(0, len(recipes), batch_size):
        batch = recipes[i:i + batch_size]
        texts = [r["text"] for r in batch]
        ids = [r["id"] for r in batch]
        metadatas = [r["metadata"] for r in batch]

        for attempt in range(3):
            try:
                embeddings = embed_batch(texts)

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

                if (i // batch_size) % 10 == 0:
                    print(f"  Embedded {i + len(batch):,} / {len(recipes):,}")

                break  # Success — next batch

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 60
                    print(f"  Rate limited at batch {i}, waiting {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                else:
                    print(f"  Error at batch {i}: {e}")
                    break  # Non-rate-limit error, skip batch

    print(f"✅ Built vector store with {collection.count()} recipes")
    return collection


def search_recipes(query: str, n: int = 5) -> list[dict]:
    """
    Semantic search: find recipes most similar to the query.
    """
    # Embed the query (different task_type for queries)
    query_embedding = embed_batch([query], task_type="RETRIEVAL_QUERY")[0]

    # Search ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))
    collection = chroma_client.get_collection("recipes")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "name": results["metadatas"][0][i]["name"],
            "category": results["metadatas"][0][i]["category"],
            "rating": results["metadatas"][0][i]["rating"],
            "reviews": results["metadatas"][0][i]["reviews"],
            "distance": results["distances"][0][i],
            "snippet": results["documents"][0][i][:200],
        })

    return formatted


# ── CLI ──
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.ai.embeddings build")
        print('  python -m src.ai.embeddings search "quick healthy chicken"')
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        build_vector_store()

    elif command == "search":
        query = " ".join(sys.argv[2:])
        print(f"Searching for: '{query}'\n")
        results = search_recipes(query, n=5)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['name']} ({r['category']})")
            print(f"   ★ {r['rating']:.1f} ({r['reviews']} reviews) | Distance: {r['distance']:.4f}")
            print(f"   {r['snippet']}...")
            print()
