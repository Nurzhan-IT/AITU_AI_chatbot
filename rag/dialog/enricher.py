"""Query enrichment and user-profile extraction for the dialog pipeline.

Given the original user query and the clarifying Q&A pairs, this module:
- ``enrich_query`` — produces a dense search-query string for the retriever.
- ``extract_profile`` — produces a structured ``UserProfile`` (topics,
  user_type, document_hints, temporal_context) for multi-factor scoring.
"""

import json
import logging
from typing import Optional, TypedDict

from config import settings
from rag.dialog.prompts import (
    ENRICH_AND_PROFILE_SYSTEM_PROMPT,
    ENRICH_SYSTEM_PROMPT,
    EXTRACT_PROFILE_SYSTEM_PROMPT,
)
from rag.generator import _make_llm_client

logger = logging.getLogger(__name__)

_ALLOWED_USER_TYPES = {"бакалавр", "магистрант", "сотрудник"}
_MAX_LIST_ITEMS = 10


class UserProfile(TypedDict):
    topics: list[str]
    user_type: Optional[str]
    document_hints: list[str]
    temporal_context: Optional[str]


def _empty_profile() -> UserProfile:
    return UserProfile(
        topics=[],
        user_type=None,
        document_hints=[],
        temporal_context=None,
    )


def _coerce_str_list(raw, limit: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            cleaned.append(s)
        if len(cleaned) >= limit:
            break
    return cleaned


def _coerce_optional_str(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


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


async def enrich_and_profile(
    original: str, answers: list[dict]
) -> tuple[str, UserProfile]:
    """Single LLM call returning (enriched_query, profile).

    Replaces the parallel enrich_query + extract_profile calls.
    Falls back to (original, empty_profile) on any failure.
    """
    filtered = [
        a for a in answers
        if a.get("answer") is not None and str(a.get("answer", "")).strip()
    ]
    if not filtered:
        return original, _empty_profile()

    try:
        lines = [f"original: {original}", "answers:"]
        for a in filtered:
            q = (a.get("question") or "").strip()
            ans_text = str(a.get("answer")).strip()
            lines.append(f'  {{"question": {json.dumps(q, ensure_ascii=False)}, "answer": {json.dumps(ans_text, ensure_ascii=False)}}}')
        user_content = "\n".join(lines)

        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": ENRICH_AND_PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in enrich_and_profile response: {content!r}")

        data = json.loads(content[start:end])

        enriched = _strip_wrapping_quotes(str(data.get("enriched") or "").strip()) or original

        raw_profile = data.get("profile") or {}
        topics = _coerce_str_list(raw_profile.get("topics"), _MAX_LIST_ITEMS)
        document_hints = _coerce_str_list(raw_profile.get("document_hints"), _MAX_LIST_ITEMS)
        temporal_context = _coerce_optional_str(raw_profile.get("temporal_context"))
        user_type: Optional[str] = None
        ut = _coerce_optional_str(raw_profile.get("user_type"))
        if ut and ut.lower() in _ALLOWED_USER_TYPES:
            user_type = ut.lower()

        profile = UserProfile(
            topics=topics,
            user_type=user_type,
            document_hints=document_hints,
            temporal_context=temporal_context,
        )
        return enriched, profile
    except Exception:
        logger.warning(
            "enrich_and_profile failed for original=%.80r; using fallback",
            original,
            exc_info=True,
        )
        return original, _empty_profile()


async def extract_profile(original: str, answers: list[dict]) -> UserProfile:
    try:
        lines = [f"original_query: {original}", "answers:"]
        for a in answers:
            ans = a.get("answer")
            if ans is None:
                continue
            ans_text = str(ans).strip()
            if not ans_text:
                continue
            q = (a.get("question") or "").strip()
            lines.append(f"- {q} → {ans_text}")
        user_content = "\n".join(lines)

        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": EXTRACT_PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        content = response.choices[0].message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in extract_profile response: {content!r}")

        data = json.loads(content[start:end])

        topics = _coerce_str_list(data.get("topics"), _MAX_LIST_ITEMS)
        document_hints = _coerce_str_list(data.get("document_hints"), _MAX_LIST_ITEMS)
        temporal_context = _coerce_optional_str(data.get("temporal_context"))

        user_type_raw = data.get("user_type")
        user_type: Optional[str] = None
        if user_type_raw is not None:
            ut = str(user_type_raw).strip().lower()
            if ut in _ALLOWED_USER_TYPES:
                user_type = ut

        return UserProfile(
            topics=topics,
            user_type=user_type,
            document_hints=document_hints,
            temporal_context=temporal_context,
        )
    except Exception:
        logger.warning(
            "extract_profile failed for original=%.80r; returning empty profile",
            original,
            exc_info=True,
        )
        return _empty_profile()
