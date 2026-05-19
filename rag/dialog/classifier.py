"""Intent classifier for incoming user queries.

Decides whether a query is concrete enough to search the RAG index directly
or whether the bot should first ask 1–3 clarifying questions. At this stage
the classifier runs in read-only mode — the result is logged but does not
yet change handler behavior (see bot/handlers/user.py).
"""

import json
import logging
from typing import TypedDict

from config import settings
from rag.dialog.prompts import CLASSIFY_SYSTEM_PROMPT
from rag.generator import _make_llm_client

logger = logging.getLogger(__name__)


class ClassificationResult(TypedDict):
    needs_clarification: bool
    reason: str


_VALID_REASONS = {"specific", "vague_topic", "ambiguous"}


async def classify_intent(question: str) -> ClassificationResult:
    try:
        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=50,
        )
        content = response.choices[0].message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in classifier response: {content!r}")

        data = json.loads(content[start:end])
        needs = bool(data.get("needs_clarification"))
        reason = str(data.get("reason", "")).strip()
        if reason not in _VALID_REASONS:
            reason = "vague_topic" if needs else "specific"
        return ClassificationResult(needs_clarification=needs, reason=reason)
    except Exception:
        logger.warning(
            "classify_intent failed for q=%.80r; falling back to needs_clarification=False",
            question,
            exc_info=True,
        )
        return ClassificationResult(needs_clarification=False, reason="error")
