"""Post-generation citation auditor (§4 of answer_quality_prompts.md, section 2).

One LLM call after answer generation: given the answer text with inline
[1], [2], ... citations and the list of source fragments, classify each
citation as supported or invalid and return an overall verdict.

Fail-open: on any error (parse failure, empty response, network) the function
returns a clean verdict so a verifier outage never blocks an answer.
"""

import json
import logging
import re

from config import settings
from rag.generator import _make_llm_client, supports_json_object, supports_json_schema

logger = logging.getLogger("rag.citation_verifier")


CITATION_AUDITOR_PROMPT = """You are a citation auditor for a university Q&A assistant.

You will receive:
- ANSWER: a generated response that contains inline citations like [1], [2], [3].
- FRAGMENTS: a numbered list of source chunks; index matches the citation number.

For EACH citation [N] in the ANSWER, decide whether the claim it supports is
actually present in FRAGMENT[N].

A citation is VALID when fragment N literally contains the fact being cited
(date, number, rule, procedure, definition). Paraphrase is fine; invention is not.

A citation is INVALID when:
- Fragment N talks about a different subject, cohort (level/admission), or year
  than what the answer claims.
- The cited number/date/threshold is NOT present in fragment N (even if a similar
  one is — different document, different program counts as INVALID).
- The fragment only mentions the topic in passing without stating the claim.

Output STRICT JSON, no markdown:
{
  "invalid_citations": [
    {"index": <N>, "claim": "<short excerpt of the answer text that cites [N]>",
     "reason": "<one-line reason>"}
  ],
  "verdict": "<clean | minor | unsupported>"
}

Verdict rules:
- "clean"       — no invalid citations.
- "minor"       — at most one invalid citation AND the answer's main facts are
                  supported by other valid citations.
- "unsupported" — two or more invalid citations, OR the central claim of the
                  answer rests on an invalid one.

Respond ONLY with the JSON object.
"""


_CITATION_AUDITOR_JSON_SCHEMA = {
    "name": "citation_audit",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "invalid_citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "claim": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["index", "claim", "reason"],
                    "additionalProperties": False,
                },
            },
            "verdict": {
                "type": "string",
                "enum": ["clean", "minor", "unsupported"],
            },
        },
        "required": ["invalid_citations", "verdict"],
        "additionalProperties": False,
    },
}


_VALID_VERDICTS = frozenset({"clean", "minor", "unsupported"})

# Cap fragment length so a verifier call stays bounded even when a chunk is
# unusually long. The audit only needs enough context to verify a quoted fact.
_FRAGMENT_MAX_CHARS = 2000


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` (or plain ```) wrappers around a JSON blob."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?[ \t]*\r?\n?", "", s)
        s = re.sub(r"\r?\n?```[ \t]*$", "", s)
    return s.strip()


def _format_fragments(fragments: list[str]) -> str:
    lines = []
    for i, frag in enumerate(fragments or [], 1):
        text = (frag or "").strip()
        if len(text) > _FRAGMENT_MAX_CHARS:
            text = text[:_FRAGMENT_MAX_CHARS] + " …"
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


async def verify_citations(answer: str, fragments: list[str]) -> dict:
    """Audit inline [N] citations in ``answer`` against ``fragments``.

    Args:
        answer: the generated response text containing [1], [2], ... markers.
        fragments: source chunk texts; ``fragments[N-1]`` backs citation [N].

    Returns:
        ``{"invalid_citations": [{"index": int, "claim": str, "reason": str}, ...],
           "verdict": "clean" | "minor" | "unsupported"}``

    Fail-open contract: on empty answer, empty LLM output, JSON parse failure,
    or any exception, returns ``{"invalid_citations": [], "verdict": "clean"}``
    so a verifier outage never blocks the user's answer.
    """
    fail_open: dict = {"invalid_citations": [], "verdict": "clean"}

    if not (answer or "").strip():
        return fail_open

    user_content = (
        f"ANSWER:\n{answer}\n\n"
        f"FRAGMENTS:\n{_format_fragments(fragments)}"
    )

    try:
        client = _make_llm_client()
        request_kwargs: dict = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": CITATION_AUDITOR_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        }
        if supports_json_schema():
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": _CITATION_AUDITOR_JSON_SCHEMA,
            }
        elif supports_json_object():
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**request_kwargs)
        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.warning("verify_citations: empty LLM response, returning clean")
            return fail_open

        cleaned = _strip_json_fences(content)
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning(
                "verify_citations: no JSON object in response: %r", content[:200]
            )
            return fail_open

        data = json.loads(cleaned[start:end])
    except Exception:
        logger.warning(
            "verify_citations: LLM call or JSON parse failed, returning clean",
            exc_info=True,
        )
        return fail_open

    verdict = data.get("verdict")
    if verdict not in _VALID_VERDICTS:
        logger.warning(
            "verify_citations: invalid verdict %r, defaulting to clean", verdict
        )
        verdict = "clean"

    invalid: list[dict] = []
    for item in data.get("invalid_citations") or []:
        try:
            invalid.append({
                "index": int(item["index"]),
                "claim": str(item.get("claim", "")),
                "reason": str(item.get("reason", "")),
            })
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "verify_citations: skipping malformed invalid_citation %r", item
            )

    return {"invalid_citations": invalid, "verdict": verdict}
