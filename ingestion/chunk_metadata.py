"""LLM-based per-chunk metadata extraction for AITU regulation documents.

One LLM call per chunk (or per batch) classifies the audience (`applies_to`),
academic calendar (`admission_type`), and topic keywords (`topic_tags`).
Result is stored in the Qdrant payload so retrieval can apply hard filters
via ``Filter(must=[FieldCondition(...)])`` instead of substring matching.

See section 1 of ``chunk_selection_prompts.md`` for the source prompt and
section 2.1 of the parallel execution plan for the ingestion wiring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from config import settings

logger = logging.getLogger("ingestion.chunk_metadata")


CHUNK_METADATA_SYSTEM_PROMPT = """\
You are a metadata extractor for AITU regulation documents. You will receive ONE
chunk of text from a university policy/regulation PDF. Your job: classify which
audience and which academic calendar the chunk applies to, based ONLY on what
the text itself says.

Return EXACTLY this JSON object (one line, no markdown, no commentary):
{"applies_to": [...], "admission_type": [...], "topic_tags": [...]}

Field rules:

- "applies_to": subset of ["бакалавр", "магистрант", "докторант", "сотрудник",
  "all"]. Use "all" ONLY when the chunk explicitly states it covers every level
  (e.g. "для всех обучающихся"). When the text mentions specific levels by
  name (or by unambiguous synonym: "bachelor"/"бакалавриат" → "бакалавр",
  "master"/"магистратура"/"PhD"/"докторантура" → corresponding value),
  list those. NEVER infer a level from indirect signals like the word
  "триместр" alone — bachelor, master AND doctoral programs all have
  trimesters at AITU. If the text gives no signal, return [].

- "admission_type": subset of ["обычный", "зимний", "all"]. Use "all" only when
  the text explicitly covers both. Use "зимний" only when the text says
  "зимний приём" / "winter admission" / equivalent. Use "обычный" only when
  the text says "обычный приём" / "regular admission" / equivalent OR when
  the chunk concerns бакалавриат exclusively (бакалавриат has no winter
  admission at AITU — see BACKGROUND). If the text gives no signal, return [].

- "topic_tags": 1–5 short Russian keywords describing the subject of THIS
  chunk (e.g. "рубежный контроль", "оплата обучения", "академический
  отпуск"). Use noun phrases, lowercase. Do NOT echo the document title.

Hard rules:
- Empty array [] is the correct answer when the chunk gives no signal. Do NOT
  guess. Do NOT default to "бакалавр" or "обычный".
- Never invent a category that is not in the allowed lists above.
- Judge the text only — ignore the document title unless the chunk text
  itself repeats it.
"""

CHUNK_METADATA_BATCH_SYSTEM_PROMPT = (
    CHUNK_METADATA_SYSTEM_PROMPT.rstrip()
    + "\n\nYou will receive multiple chunks indexed [1], [2], ..., respond with "
      "a JSON array of the same length, one metadata object per chunk in the same order. "
      "Output the JSON array on a single line, no markdown, no commentary."
)


_EMPTY_METADATA: dict[str, list[str]] = {
    "applies_to": [],
    "admission_type": [],
    "topic_tags": [],
}

_ALLOWED_APPLIES_TO = {"бакалавр", "магистрант", "докторант", "сотрудник", "all"}
_ALLOWED_ADMISSION_TYPE = {"обычный", "зимний", "all"}

# Match a single LLM call cost-bound; chunks are ~400 tokens so 4000 chars is
# a safe upper bound that survives outliers without exploding cost.
_MAX_CHUNK_CHARS = 4000

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _empty_metadata() -> dict[str, list[str]]:
    return {k: list(v) for k, v in _EMPTY_METADATA.items()}


def _strip_code_fence(content: str) -> str:
    match = _FENCE_RE.search(content)
    if match:
        return match.group(1).strip()
    return content.strip()


def _sanitize_list(value: Any, allowed: set[str] | None) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        if allowed is not None and s not in allowed:
            continue
        if s not in out:
            out.append(s)
    return out


def _sanitize_topic_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip().lower()
        if not s:
            continue
        if s not in out:
            out.append(s)
        if len(out) >= 5:
            break
    return out


def _coerce_metadata(obj: Any) -> dict[str, list[str]]:
    if not isinstance(obj, dict):
        return _empty_metadata()
    return {
        "applies_to":     _sanitize_list(obj.get("applies_to"), _ALLOWED_APPLIES_TO),
        "admission_type": _sanitize_list(obj.get("admission_type"), _ALLOWED_ADMISSION_TYPE),
        "topic_tags":     _sanitize_topic_tags(obj.get("topic_tags")),
    }


def _parse_object(content: str) -> dict[str, list[str]]:
    body = _strip_code_fence(content)
    start = body.find("{")
    end = body.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in response: {content!r}")
    return _coerce_metadata(json.loads(body[start:end]))


def _parse_array(content: str, expected_len: int) -> list[dict[str, list[str]]]:
    body = _strip_code_fence(content)
    start = body.find("[")
    end = body.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in response: {content!r}")
    arr = json.loads(body[start:end])
    if not isinstance(arr, list):
        raise ValueError(f"Expected JSON array, got {type(arr).__name__}")
    if len(arr) != expected_len:
        raise ValueError(
            f"Expected JSON array of length {expected_len}, got {len(arr)}"
        )
    return [_coerce_metadata(x) for x in arr]


def _make_llm_client() -> Any:
    """Mirror rag.generator._make_llm_client so this module owns no provider state."""
    if settings.llm_provider == "groq":
        from groq import AsyncGroq
        return AsyncGroq(api_key=settings.groq_api_key)
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


async def extract_chunk_metadata(chunk_text: str) -> dict[str, list[str]]:
    """Classify a single chunk. Fail-open: returns empty arrays on any error."""
    text = (chunk_text or "").strip()
    if not text:
        return _empty_metadata()
    excerpt = text[:_MAX_CHUNK_CHARS]

    try:
        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": CHUNK_METADATA_SYSTEM_PROMPT},
                {"role": "user", "content": f"CHUNK:\n{excerpt}"},
            ],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or ""
        return _parse_object(content)
    except Exception:
        logger.warning(
            "extract_chunk_metadata failed for chunk (len=%d); returning empty metadata",
            len(text),
            exc_info=True,
        )
        return _empty_metadata()


async def extract_chunk_metadata_batch(
    chunks: list[str],
    batch_size: int = 10,
) -> list[dict[str, list[str]]]:
    """Classify many chunks in batches of ``batch_size`` to cut LLM round-trips.

    Each batched LLM call returns a JSON array of the same length as its input.
    On any error within a batch, that batch falls back to per-chunk extraction
    so a single bad batch never produces empty metadata for an entire ingest.
    """
    if not chunks:
        return []
    if batch_size < 1:
        batch_size = 1

    results: list[dict[str, list[str]]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        try:
            results.extend(await _extract_batch(batch))
        except Exception:
            logger.warning(
                "extract_chunk_metadata_batch: batch %d-%d failed, falling back to per-chunk",
                start, start + len(batch) - 1,
                exc_info=True,
            )
            for ch in batch:
                results.append(await extract_chunk_metadata(ch))
    return results


async def _extract_batch(batch: list[str]) -> list[dict[str, list[str]]]:
    parts = []
    for i, ch in enumerate(batch, 1):
        excerpt = (ch or "").strip()[:_MAX_CHUNK_CHARS]
        parts.append(f"[{i}]\n{excerpt}")
    user_content = "\n\n".join(parts)

    client = _make_llm_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CHUNK_METADATA_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=200 * len(batch),
    )
    content = response.choices[0].message.content or ""
    return _parse_array(content, expected_len=len(batch))


__all__ = [
    "CHUNK_METADATA_SYSTEM_PROMPT",
    "CHUNK_METADATA_BATCH_SYSTEM_PROMPT",
    "extract_chunk_metadata",
    "extract_chunk_metadata_batch",
]
