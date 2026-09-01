"""Intent classifier for incoming user queries.

Decides whether a query is concrete enough to search the RAG index directly
or whether the bot should first ask 1–3 clarifying questions. At this stage
the classifier runs in read-only mode — the result is logged but does not
yet change handler behavior (see bot/handlers/user.py).
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Optional, TypedDict

from config import settings
from rag.dialog.prompts import CLASSIFY_SYSTEM_PROMPT
from rag.generator import _make_llm_client, supports_json_object, supports_json_schema

logger = logging.getLogger(__name__)


class ClassificationResult(TypedDict):
    needs_clarification: bool
    reason: str
    required_slots: list[str]
    confidence: float


_VALID_REASONS = {"specific", "vague_topic", "ambiguous"}
_VALID_SLOTS = ("level", "admission_type", "year", "topic", "document")
_VALID_SLOTS_SET = set(_VALID_SLOTS)

# ---------------------------------------------------------------------------
# In-memory classification cache
# ---------------------------------------------------------------------------
_CACHE_TTL   = 86_400.0   # 24 hours
_CACHE_MAX   = 1_000      # evict oldest when exceeded

_cache: OrderedDict[str, tuple["ClassificationResult", float]] = OrderedDict()


def _cache_key(question: str) -> str:
    normalized = " ".join(question.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _cache_get(key: str) -> Optional["ClassificationResult"]:
    entry = _cache.get(key)
    if entry is None:
        return None
    result, expires_at = entry
    if time.time() > expires_at:
        del _cache[key]
        return None
    _cache.move_to_end(key)
    return result


def _cache_put(key: str, result: "ClassificationResult") -> None:
    if key in _cache:
        _cache.move_to_end(key)
    _cache[key] = (result, time.time() + _CACHE_TTL)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)

# Strict JSON Schema for providers with structured-output support. It eliminates
# the "model wrapped JSON in markdown" failure class at the source. Providers
# without support fall back to brace-slicing the raw text response below.
_CLASSIFY_JSON_SCHEMA = {
    "name": "intent_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "needs_clarification": {"type": "boolean"},
            "reason": {"type": "string", "enum": sorted(_VALID_REASONS)},
            "required_slots": {
                "type": "array",
                "items": {"type": "string", "enum": list(_VALID_SLOTS)},
                "uniqueItems": True,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["needs_clarification", "reason", "required_slots", "confidence"],
        "additionalProperties": False,
    },
}


def _coerce_required_slots(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s in _VALID_SLOTS_SET and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _describe_choice(choice) -> str:
    """Diagnostic summary of an LLM completion choice.

    Used when ``message.content`` is empty so we can tell *why* — was the
    response truncated (``finish_reason='length'``), returned via tool calls,
    refused, or did the model put the answer in a non-content field like
    ``reasoning`` (OpenRouter reasoning models)?
    """
    parts: list[str] = []
    finish = getattr(choice, "finish_reason", None)
    if finish is not None:
        parts.append(f"finish_reason={finish!r}")

    msg = getattr(choice, "message", None)
    if msg is None:
        return " | ".join(parts) if parts else "no message in choice"

    dump = None
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(msg, attr, None)
        if callable(fn):
            try:
                dump = fn()
                break
            except Exception:
                continue

    if isinstance(dump, dict):
        # Drop content (already in the surrounding error) and empty fields.
        filtered = {k: v for k, v in dump.items() if k != "content" and v}
        if filtered:
            s = repr(filtered)
            if len(s) > 500:
                s = s[:500] + "...<truncated>"
            parts.append(f"message={s}")

    return " | ".join(parts) if parts else "no diagnostic fields"


async def classify_intent(question: str) -> ClassificationResult:
    key = _cache_key(question)
    cached = _cache_get(key)
    if cached is not None:
        logger.debug("classify_intent cache hit q=%.80r", question)
        return cached

    try:
        client = _make_llm_client()
        request_kwargs = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": 0,
            "max_tokens": 300,
        }
        if supports_json_schema():
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": _CLASSIFY_JSON_SCHEMA,
            }
        elif supports_json_object():
            request_kwargs["response_format"] = {"type": "json_object"}
        response = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=6.0,
        )
        choice = response.choices[0]
        content = choice.message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(
                f"No JSON object in classifier response: {content!r} "
                f"[{_describe_choice(choice)}]"
            )

        data = json.loads(content[start:end])
        if "needs_clarification" not in data:
            raise ValueError(f"Classifier response missing 'needs_clarification': {data!r}")
        needs = bool(data["needs_clarification"])
        reason = str(data.get("reason", "")).strip()
        if reason not in _VALID_REASONS:
            reason = "vague_topic" if needs else "specific"
        required_slots = _coerce_required_slots(data.get("required_slots"))
        if not needs:
            required_slots = []
        try:
            confidence = float(data["confidence"])
            if not (0.0 <= confidence <= 1.0):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            confidence = 0.5

        if confidence >= settings.classify_conf_high:
            conf_label = "dialog" if needs else "direct-search"
        elif confidence >= settings.classify_conf_low:
            conf_label = "borderline"
        else:
            conf_label = "direct-search"
        logger.debug(
            "classify_intent q=%.80r → needs_clarification=%s reason=%s required_slots=%s confidence=%.2f (%s)",
            question, needs, reason, required_slots, confidence, conf_label,
        )
        result = ClassificationResult(
            needs_clarification=needs,
            reason=reason,
            required_slots=required_slots,
            confidence=confidence,
        )
        _cache_put(key, result)
        return result
    except Exception:
        logger.warning(
            "classify_intent failed for q=%.80r; falling back to triage Stage 1 heuristics",
            question,
            exc_info=True,
        )
        try:
            from rag.dialog.triage import _STOPWORDS, _tokenize
            tokens = _tokenize(question)
            if len(tokens) <= 1 or all(t in _STOPWORDS for t in tokens):
                return ClassificationResult(
                    needs_clarification=True,
                    reason="ambiguous",
                    required_slots=[],
                    confidence=0.5,
                )
        except Exception:
            logger.debug("classify_intent Stage 1 fallback also failed", exc_info=True)
        return ClassificationResult(
            needs_clarification=False,
            reason="specific",
            required_slots=[],
            confidence=0.5,
        )
