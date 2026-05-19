"""Query enrichment for the dialog pipeline.

Given the original user query and the clarifying Q&A pairs collected during
the dialog, this module asks the LLM to produce a single dense search-query
string suitable for the vector retriever. User-profile extraction (topics,
user_type, document_hints, temporal_context) is deferred to Stage 4.
"""

import logging

from config import settings
from rag.dialog.prompts import ENRICH_SYSTEM_PROMPT
from rag.generator import _make_llm_client

logger = logging.getLogger(__name__)


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'", "«"):
        return text[1:-1].strip()
    return text


async def enrich_query(original: str, answers: list[dict]) -> str:
    if not answers:
        return original

    filtered = []
    for a in answers:
        ans = a.get("answer")
        if ans is None:
            continue
        if isinstance(ans, str) and not ans.strip():
            continue
        filtered.append(a)
    if not filtered:
        return original

    try:
        lines = [f"original_query: {original}", "answers:"]
        for a in filtered:
            q = (a.get("question") or "").strip()
            ans_text = str(a.get("answer")).strip()
            lines.append(f"- {q} → {ans_text}")
        user_content = "\n".join(lines)

        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": ENRICH_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        content = (response.choices[0].message.content or "").strip()
        content = _strip_wrapping_quotes(content).strip()
        return content or original
    except Exception:
        logger.warning(
            "enrich_query failed for original=%.80r; falling back to original",
            original,
            exc_info=True,
        )
        return original
