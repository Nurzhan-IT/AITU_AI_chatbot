"""Coverage pre-check gate (§3 of answer_quality_prompts.md).

One cheap LLM call placed BEFORE the main generator: given the user's question,
the clarified profile slots, and the retrieved fragments, decide whether the
fragments are enough to answer for *this* user's cohort. Also flags fragments
that belong to a different cohort (e.g. magistracy fragment for a bachelor
question) so the generator can drop them before composing the answer.

Fail-open: any error in the LLM call or JSON parsing returns a "sufficient"
verdict with empty lists, so an outage of this gate never blocks the pipeline.
"""

import asyncio
import json
import logging
from typing import Any

from config import settings

logger = logging.getLogger("rag.coverage_gate")


COVERAGE_GATE_PROMPT = """You are a retrieval coverage checker for a university Q&A assistant.

You will receive:
- QUESTION: the user's original question.
- PROFILE: user-clarified slots (level, admission_type, topics, ...). Unknown
  slots are marked UNKNOWN — treat them as a missing constraint, not a default.
- FRAGMENTS: the chunks the retriever returned (numbered, with doc_title and
  section_title).

Decide whether the fragments contain enough information to answer the QUESTION
under the constraints of PROFILE.

A profile constraint is SATISFIED when at least one fragment is explicitly
about that cohort (e.g. PROFILE.level=бакалавр and a fragment is a chapter from
the bachelor academic policy). A fragment that does not name the cohort but
plausibly belongs to it (judged from doc_title) counts as PARTIAL.

Output STRICT JSON, no markdown:
{
  "verdict": "<sufficient | partial | insufficient>",
  "missing": ["<one short phrase per missing slot or topic>"],
  "wrong_cohort_indices": [<int>, ...]
}

Verdict rules:
- "sufficient"   — at least one fragment fully addresses the question for the
                   user's exact (level, admission_type) cohort.
- "partial"      — fragments cover the topic but only for a different cohort,
                   or cover the cohort but not the specific question.
- "insufficient" — fragments are off-topic or contradict the profile entirely.

"wrong_cohort_indices" lists fragment numbers that belong to a DIFFERENT
cohort than PROFILE specifies — the generator should drop these before
composing the answer.

Respond ONLY with the JSON object.
"""


_FAIL_OPEN_RESULT: dict = {
    "verdict": "sufficient",
    "missing": [],
    "wrong_cohort_indices": [],
}

_FRAGMENT_EXCERPT_LEN = 500
_VALID_VERDICTS = frozenset({"sufficient", "partial", "insufficient"})


def _make_llm_client() -> Any:
    """Same provider switch as rag/generator.py — kept local to avoid import cycles."""
    if settings.llm_provider == "groq":
        from groq import AsyncGroq
        return AsyncGroq(api_key=settings.groq_api_key)
    else:
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )


def _format_profile_slot(value: Any) -> str:
    """Render a profile slot value for the prompt. Empty / None → UNKNOWN."""
    if value is None:
        return "UNKNOWN"
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ", ".join(items) if items else "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _build_user_message(
    question: str, profile: dict, fragments: list[dict]
) -> str:
    profile_block = (
        f"- user_type: {_format_profile_slot(profile.get('user_type'))}\n"
        f"- admission_type: {_format_profile_slot(profile.get('admission_type'))}\n"
        f"- topics: {_format_profile_slot(profile.get('topics'))}\n"
        f"- document_hints: {_format_profile_slot(profile.get('document_hints'))}\n"
        f"- temporal_context: {_format_profile_slot(profile.get('temporal_context'))}"
    )

    fragment_lines: list[str] = []
    for i, frag in enumerate(fragments, 1):
        doc_title = (frag.get("doc_title") or "").strip()
        section_title = (frag.get("section_title") or "").strip()
        text = (frag.get("text") or "").strip()
        excerpt = text[:_FRAGMENT_EXCERPT_LEN]
        fragment_lines.append(
            f'[{i}] doc_title="{doc_title}", section_title="{section_title}"\n'
            f"    {excerpt}"
        )
    fragments_block = "\n".join(fragment_lines) if fragment_lines else "(none)"

    return (
        f"QUESTION:\n{question}\n\n"
        f"PROFILE:\n{profile_block}\n\n"
        f"FRAGMENTS:\n{fragments_block}"
    )


def _extract_json_object(content: str) -> dict:
    """Parse a JSON object from a model response, tolerating ```json fences."""
    text = content.strip()
    if text.startswith("```"):
        # Strip leading fence (``` or ```json) and the trailing fence.
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in coverage gate response: {content!r}")
    return json.loads(text[start:end])


def _normalize_result(data: dict) -> dict:
    verdict = data.get("verdict")
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"Invalid verdict from coverage gate: {verdict!r}")

    missing_raw = data.get("missing", []) or []
    if not isinstance(missing_raw, list):
        raise ValueError(f"'missing' is not a list: {missing_raw!r}")
    missing = [str(m).strip() for m in missing_raw if str(m).strip()]

    wrong_raw = data.get("wrong_cohort_indices", []) or []
    if not isinstance(wrong_raw, list):
        raise ValueError(f"'wrong_cohort_indices' is not a list: {wrong_raw!r}")
    wrong_cohort_indices: list[int] = []
    for v in wrong_raw:
        try:
            wrong_cohort_indices.append(int(v))
        except (TypeError, ValueError):
            continue

    return {
        "verdict": verdict,
        "missing": missing,
        "wrong_cohort_indices": wrong_cohort_indices,
    }


async def check_coverage(
    question: str, profile: dict, fragments: list[dict]
) -> dict:
    """Ask the LLM whether the retrieved fragments cover the question for this profile.

    Args:
        question: the user's original query.
        profile: clarified slots — at least ``user_type``, ``admission_type``,
            ``topics``, ``document_hints``, ``temporal_context``. Missing or
            empty values are rendered as ``UNKNOWN`` in the prompt.
        fragments: retriever output. Each item must have at least ``text``;
            ``doc_title`` and ``section_title`` are used when present.

    Returns:
        ``{"verdict": "sufficient"|"partial"|"insufficient",
            "missing": [str, ...], "wrong_cohort_indices": [int, ...]}``.

        On any LLM or parsing failure: ``{"verdict": "sufficient", "missing": [],
        "wrong_cohort_indices": []}`` (fail-open — the gate never blocks the
        pipeline on its own outage).
    """
    user_message = _build_user_message(question, profile or {}, fragments or [])

    try:
        client = _make_llm_client()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": COVERAGE_GATE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                max_tokens=400,
            ),
            timeout=10.0,
        )
        content = response.choices[0].message.content or ""
        data = _extract_json_object(content)
        result = _normalize_result(data)
        logger.debug(
            "check_coverage: q=%.60r verdict=%s missing=%d wrong=%d",
            question,
            result["verdict"],
            len(result["missing"]),
            len(result["wrong_cohort_indices"]),
        )
        return result
    except Exception:
        logger.warning(
            "check_coverage failed for q=%.80r; fail-open with verdict=sufficient",
            question,
            exc_info=True,
        )
        return dict(_FAIL_OPEN_RESULT)
