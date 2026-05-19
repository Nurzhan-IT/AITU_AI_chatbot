"""LLM-driven generation of clarifying questions.

Given the original user query, the indexed-corpus snapshot (doc_title /
section_title) and the dialog history, this module asks the LLM to produce
the next single clarifying question with 2–4 Telegram-friendly options.
"""

import json
import logging
import time
from typing import TypedDict

from config import settings
from rag.dialog.prompts import QUESTION_GEN_SYSTEM_PROMPT
from rag.generator import _make_llm_client

logger = logging.getLogger(__name__)

_OPTION_LABEL_LIMIT = 50
_MAX_OPTIONS = 4
_MAX_DOCS_IN_PAYLOAD = 200

_DOCS_TTL = 600  # seconds (10 minutes)
_docs_cache: dict = {"items": None, "expires_at": 0.0}


class ClarificationQuestion(TypedDict):
    question: str
    options: list[str]
    stop: bool


async def _get_cached_docs(retriever) -> list[dict]:
    """Return cached unique (doc_title, section_title) pairs, or refresh the cache.

    Performs a SINGLE Qdrant scroll across the collection (no second call to
    Retriever.get_all_documents()). Falls back to an empty list on any error
    so the question generator can still run.
    """
    now = time.time()
    cached = _docs_cache.get("items")
    if cached is not None and _docs_cache.get("expires_at", 0.0) > now:
        return cached

    items: list[dict] = []
    try:
        seen: set[tuple[str, str]] = set()
        offset = None
        while True:
            records, next_offset = await retriever._client.scroll(
                collection_name=settings.qdrant_collection,
                limit=200,
                offset=offset,
                with_payload=["doc_title", "section_title"],
                with_vectors=False,
            )
            for record in records:
                p = record.payload or {}
                doc_title = (p.get("doc_title") or "").strip()
                section_title = (p.get("section_title") or "").strip()
                if not doc_title:
                    continue
                key = (doc_title, section_title)
                if key in seen:
                    continue
                seen.add(key)
                items.append({"doc_title": doc_title, "section_title": section_title})
            if next_offset is None:
                break
            offset = next_offset
        logger.info("dialog docs cache refreshed: %d unique entries", len(items))
    except Exception:
        logger.warning("Failed to refresh dialog docs cache; using empty list", exc_info=True)
        items = []

    _docs_cache["items"] = items
    _docs_cache["expires_at"] = now + _DOCS_TTL
    return items


def _normalize_options(raw_opts) -> list[str]:
    options: list[str] = []
    if not isinstance(raw_opts, list):
        return options
    for raw in raw_opts[:_MAX_OPTIONS]:
        label = str(raw).strip()
        if not label:
            continue
        if len(label) > _OPTION_LABEL_LIMIT:
            label = label[: _OPTION_LABEL_LIMIT - 1] + "…"
        options.append(label)
    return options


async def next_clarification(
    state_data: dict, available_docs: list[dict]
) -> ClarificationQuestion:
    try:
        payload = {
            "original_query": state_data.get("original_query", ""),
            "rounds_done": state_data.get("rounds_done", 0),
            "answers": state_data.get("answers", []),
            "available_docs": list(available_docs)[:_MAX_DOCS_IN_PAYLOAD],
        }
        user_content = json.dumps(payload, ensure_ascii=False)

        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": QUESTION_GEN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        content = response.choices[0].message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in next_clarification response: {content!r}")

        data = json.loads(content[start:end])
        question = str(data.get("question", "")).strip()
        options = _normalize_options(data.get("options", []))
        stop = bool(data.get("stop", False))
        return ClarificationQuestion(question=question, options=options, stop=stop)
    except Exception:
        logger.warning(
            "next_clarification failed for original_query=%.80r; returning stop=True",
            state_data.get("original_query", ""),
            exc_info=True,
        )
        return ClarificationQuestion(question="", options=[], stop=True)
