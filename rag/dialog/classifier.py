"""Intent classifier for incoming user queries.

Decides whether a query is concrete enough to search the RAG index directly
or whether the bot should first ask 1–3 clarifying questions. This file is
a stub for the dialog-pipeline scaffolding (Prompt 0) — it always returns
the "fast path" verdict. Real LLM-based classification lands in Prompt 1.
"""

from rag.generator import _make_llm_client  # noqa: F401 — used by later stages


async def classify_intent(question: str) -> dict:
    return {"needs_clarification": False, "reason": "stub"}
