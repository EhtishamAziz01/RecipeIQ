"""
RAG pipeline for recipe Q&A.

Retrieves relevant recipes from ChromaDB, augments with context,
then generates a response using Gemini.
"""

from google import genai
from pathlib import Path
import os
from dotenv import load_dotenv
from src.ai.embeddings import search_recipes

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are RecipeIQ, a knowledgeable recipe assistant powered by a database of real recipes.

RULES:
1. Only recommend recipes that appear in the PROVIDED CONTEXT below.
2. Never invent recipe names, nutrition facts, or ingredients — use only what's given.
3. Include the recipe name, category, rating, and key details in your answer.
4. If the context doesn't contain relevant recipes, say so honestly.
5. Be conversational, helpful, and enthusiastic about food.
6. Keep responses concise — 2-4 recipes max unless asked for more.
7. When mentioning nutrition, cite the exact numbers from the context.
"""


def build_context(recipes: list[dict]) -> str:
    """
    Format retrieved recipes into a context string for the LLM.
    """
    context_parts = []
    for i, r in enumerate(recipes, 1):
        context_parts.append(
            f"Recipe {i}: {r['name']}\n"
            f"  Category: {r['category']}\n"
            f"  Rating: {r['rating']:.1f}★ ({r['reviews']} reviews)\n"
            f"  Details: {r['snippet']}"
        )
    return "\n\n".join(context_parts)


def ask(question: str, n_results: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.
    """
    # 1. RETRIEVE — semantic search
    sources = search_recipes(question, n=n_results)

    # 2. AUGMENT — build prompt with context
    context = build_context(sources)

    prompt = f"""CONTEXT (recipes from the database):
{context}

USER QUESTION: {question}

Answer the question using ONLY the recipes from the context above. Cite recipe names and details."""

    # 3. GENERATE — call Gemini
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.7,
            "max_output_tokens": 1024,
        },
    )

    return {
        "question": question,
        "answer": response.text,
        "sources": [{"name": s["name"], "category": s["category"], "rating": s["rating"]} for s in sources],
    }


# ── CLI ──
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print('Usage: python -m src.ai.rag "What\'s a good chicken recipe?"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask(question)

    print(f"\n{'='*60}")
    print(f"Q: {result['question']}")
    print(f"{'='*60}")
    print(f"\n{result['answer']}")
    print(f"\n{'─'*60}")
    print(f"Sources:")
    for s in result["sources"]:
        print(f"  • {s['name']} ({s['category']}) — {s['rating']:.1f}★")
