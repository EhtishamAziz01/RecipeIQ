"""
Conversational recipe chatbot with RAG + memory.
"""

from google import genai
from pathlib import Path
import os
from dotenv import load_dotenv
from src.ai.embeddings import search_recipes

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GENERATION_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are RecipeIQ Chef, a friendly and knowledgeable AI recipe assistant.

You have access to a database of 50,000+ real recipes. When answering:
1. Only reference recipes that appear in your CONTEXT
2. Be warm, enthusiastic, and encouraging about cooking
3. If asked follow-up questions, refer to previously mentioned recipes
4. Suggest recipe modifications when appropriate
5. Include ratings and key nutrition facts when relevant
6. If you don't have relevant recipes, admit it and suggest alternatives

Keep responses conversational but concise (2-3 paragraphs max)."""


class RecipeChatbot:
    """
    Stateful chatbot that maintains conversation history
    and retrieves recipes relevant to each turn.
    """

    def __init__(self):
        self.history: list[dict] = []
        self.mentioned_recipes: list[dict] = []

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        Retrieves fresh recipes for each turn to stay relevant.
        """
        # Retrieve relevant recipes
        sources = search_recipes(user_message, n=5)
        self.mentioned_recipes.extend(sources)

        # Build context from retrieved + previously mentioned recipes
        context_parts = []
        seen_ids = set()
        for r in sources + self.mentioned_recipes[-15:]:  # Keep last 15
            if r["id"] not in seen_ids:
                context_parts.append(
                    f"- {r['name']} ({r['category']}) "
                    f"★{r['rating']:.1f} ({r['reviews']} reviews). "
                    f"{r['snippet'][:150]}"
                )
                seen_ids.add(r["id"])

        context = "\n".join(context_parts[:10])  # Cap at 10 recipes

        # Build conversation messages
        messages = []

        # Add conversation history (last 6 turns)
        for turn in self.history[-6:]:
            messages.append(turn)

        # Add current turn with context
        augmented_message = (
            f"AVAILABLE RECIPES:\n{context}\n\n"
            f"USER: {user_message}"
        )
        messages.append({"role": "user", "parts": [{"text": augmented_message}]})

        # Generate response
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=messages,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.7,
                "max_output_tokens": 1024,
            },
        )

        answer = response.text

        # Store in history (without the context, to save tokens)
        self.history.append({"role": "user", "parts": [{"text": user_message}]})
        self.history.append({"role": "model", "parts": [{"text": answer}]})

        return answer

    def reset(self):
        """Clear conversation history."""
        self.history.clear()
        self.mentioned_recipes.clear()


# ── Interactive CLI ──
if __name__ == "__main__":
    bot = RecipeChatbot()
    print("🍳 RecipeIQ Chef — Ask me anything about recipes!")
    print("   Type 'quit' to exit, 'reset' to start over.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye! Happy cooking! 👋")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("🔄 Conversation reset.\n")
            continue

        response = bot.chat(user_input)
        print(f"\n🤖 Chef: {response}\n")
