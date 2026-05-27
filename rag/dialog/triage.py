"""Retrieval-grounded triage cascade — phase C, §3.2 Stages 1–4.

Classifies an incoming query *before* the LLM intent classifier runs, with a
four-stage cascade:

    Stage 1  cheap heuristics      — no LLM, no network
    Stage 2  probe retrieval       — one embed + one Qdrant top-K search
    Stage 3  score-distribution    — no LLM, rules A/B/C/D
    Stage 4  LLM verdict           — borderline queries (rule D)

A rule-B "specific" verdict from Stage 3 is not trusted blindly: one cheap LLM
call verifies that the top1 chunk actually answers the query (§3.5 R1). If it
does not, the verdict is demoted to rule D and handed to Stage 4 — a high
cosine score alone never yields a direct answer.

Rollout (see dialog_classification_improvements.md §7): this module is wired
into bot/handlers/user.py in SHADOW mode only. The cascade runs off the hot
path and its verdict is logged next to the legacy ``classify_intent`` result
for offline comparison; it never changes the answer the user receives. The
``settings.triage_enabled`` flag (default False) is the master switch reserved
for a later phase that lets the verdict drive handler behavior.

Stage 3 thresholds are read from config.py and are never hardcoded here. They
describe the *shape* of the probe score distribution (gap-to-std ratio,
normalized entropy, document spread) rather than absolute cosine values, so
they survive an embedder or corpus change (§3.3). Until they are calibrated
from the phase-B score distribution (scripts/score_distribution.py, task C2),
``settings.triage_calibrated`` is False and Stage 3 unconditionally returns
rule D — an uncalibrated deploy therefore behaves exactly like the legacy
LLM-only path.

The Stage 2 probe vector and hits are carried out in the result so a later
phase can reuse them for the final retrieval without a second embed call
(§3.5 R3).
"""

import asyncio
import json
import logging
import math
import re
from typing import TypedDict

from config import settings
from rag.dialog.prompts import CLASSIFY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# --- Stage 1: stop-word vocabulary ----------------------------------------
# A query is "too short" when it is a single word OR consists entirely of these
# low-signal filler words (pronouns, prepositions, question words, generic
# verbs). Two-word noun phrases like "академический отпуск" must NOT be caught
# here — they are the most search-friendly queries (§3.2 note on Stage 1).
_STOPWORDS: frozenset[str] = frozenset({
    # Russian
    "что", "чо", "чё", "как", "какой", "какая", "какие", "какое", "каков",
    "где", "когда", "кто", "почему", "зачем", "чем", "это", "этот", "эта",
    "эти", "мне", "меня", "мной", "я", "ты", "вы", "он", "она", "оно", "они",
    "мы", "про", "о", "об", "обо", "на", "в", "во", "по", "с", "со", "к", "у",
    "и", "а", "но", "или", "же", "ли", "бы", "не", "ну", "да", "нет", "вот",
    "расскажи", "скажи", "подскажи", "помоги", "хочу", "нужно", "надо", "дай",
    "есть", "быть", "был", "пожалуйста", "плиз",
    # English
    "what", "how", "who", "where", "when", "why", "which", "whose", "is",
    "are", "am", "be", "the", "a", "an", "of", "to", "in", "on", "for", "at",
    "and", "or", "but", "me", "i", "you", "we", "they", "he", "she", "it",
    "tell", "give", "help", "want", "need", "do", "does", "did", "can",
    "could", "please", "about", "this", "that", "these", "those",
    # Kazakh
    "қалай", "кім", "қайда", "қашан", "неге", "бұл", "осы", "маған", "мен",
    "сен", "сіз", "ол", "біз", "туралы", "және", "бар", "айт", "айтшы",
    "көмектес",
})

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Appended to CLASSIFY_SYSTEM_PROMPT for Stage 4 so the model knows what the
# CORPUS MATCHES block in the user message is. Keeping it here leaves the
# shared prompt in rag/dialog/prompts.py untouched.
_STAGE4_SYSTEM_SUFFIX = """

The user message may be followed by a "CORPUS MATCHES" block listing the \
documents and sections a vector search retrieved for this question. It is NOT \
part of the user's question — use it only as evidence about the knowledge \
base: matches concentrated in one document suggest the question is specific, \
while matches scattered across several unrelated documents suggest it is \
ambiguous. Decide primarily from the question itself and use the matches to \
break ties."""


class TriageResult(TypedDict):
    """Verdict of the triage cascade for one query.

    ``needs_clarification`` is always a concrete bool — Stage 4 resolves the
    borderline (rule D) case. ``query_vector`` and ``probe_hits`` carry the
    Stage 2 probe outputs outward so a later phase can reuse them for the final
    retrieval without a second embed call (§3.5 R3).
    """
    needs_clarification: bool
    reason: str            # too_short | out_of_scope | specific | ambiguous |
                           # vague_topic | error
    confidence: float
    decided_by: str        # stage1 | stage2 | stage3 | stage4
    rule: str              # "" | A | B | C | D
    features: dict[str, float]
    probe_hits: list[dict]
    query_vector: list[float] | None


# --- Stage 1 helpers ------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# --- Stage 3 helpers ------------------------------------------------------

def _normalized_entropy(scores: list[float]) -> float:
    """Shannon entropy of the score vector, normalized to [0, 1].

    1.0 = perfectly flat (the retriever could not discriminate between hits);
    lower = a few chunks dominate. Mirrors scripts/score_distribution.py so the
    triage feature matches the one measured in phase B. NaN when fewer than two
    positive scores (entropy is undefined / uninformative).
    """
    pos = [s for s in scores if s > 0.0]
    n = len(pos)
    if n < 2:
        return float("nan")
    total = sum(pos)
    if total <= 0.0:
        return float("nan")
    h = -sum((s / total) * math.log(s / total) for s in pos)
    return h / math.log(n)


def _score_features(hits: list[dict]) -> dict[str, float]:
    """Derive the scale-independent Stage-3 feature vector from probe hits.

    ``gap_ratio`` (= (top1 - top2) / std) and ``entropy`` describe the *shape*
    of the distribution, not absolute cosines — see §3.3. NaN marks a feature
    that cannot be computed from the available hits.
    """
    scores = sorted((float(h.get("score", 0.0) or 0.0) for h in hits), reverse=True)
    n = len(scores)

    feats: dict[str, float] = {
        "n": float(n),
        "top1": float("nan"),
        "top2": float("nan"),
        "gap": float("nan"),
        "std": float("nan"),
        "gap_ratio": float("nan"),
        "entropy": float("nan"),
        "doc_spread": 0.0,
    }
    # doc_spread: distinct documents among the top-5 hits (Qdrant score order).
    top5_docs = {
        (h.get("doc_title") or "").strip()
        for h in hits[:5]
        if (h.get("doc_title") or "").strip()
    }
    feats["doc_spread"] = float(len(top5_docs))
    if n == 0:
        return feats

    feats["top1"] = scores[0]
    feats["entropy"] = _normalized_entropy(scores)
    if n >= 2:
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
        feats["top2"] = scores[1]
        feats["gap"] = scores[0] - scores[1]
        feats["std"] = std
        # No separation when the scores are effectively identical → gap_ratio 0
        # (cannot be "specific"), never an inf that would falsely fire rule B.
        feats["gap_ratio"] = (scores[0] - scores[1]) / std if std > 1e-9 else 0.0
    return feats


def _stage3(features: dict[str, float]) -> tuple[str, bool | None, str]:
    """Triage by probe score distribution. Returns ``(rule, needs, reason)``.

    ``needs`` is None for rule D — the borderline case handed to Stage 4.
    """
    # Uncalibrated thresholds → never decide here; defer everything to the LLM
    # so the bot keeps its legacy behavior until task C2 lands real numbers.
    if not settings.triage_calibrated:
        return ("D", None, "borderline")

    top1 = features.get("top1", float("nan"))
    if top1 != top1:  # NaN → empty probe result
        return ("D", None, "borderline")

    # Rule A — out of corpus scope: even the best match is below the shared
    # noise floor (min_chunk_score, §3.3 point 3), so clarification cannot help.
    if top1 < settings.min_chunk_score:
        return ("A", False, "out_of_scope")

    gap_ratio = features.get("gap_ratio", float("nan"))
    entropy = features.get("entropy", float("nan"))
    doc_spread = features.get("doc_spread", 0.0)

    # Rule B — SPECIFIC: the top hit stands clearly apart from the rest and the
    # distribution is concentrated → safe to search directly.
    if (gap_ratio == gap_ratio and entropy == entropy
            and gap_ratio >= settings.triage_specific_gap_ratio
            and entropy <= settings.triage_specific_max_entropy):
        return ("B", False, "specific")

    # Rule C — AMBIGUOUS: a flat distribution spread across several documents
    # → a clarifying question narrows the search.
    if (entropy == entropy
            and entropy >= settings.triage_ambiguous_min_entropy
            and doc_spread >= settings.triage_ambiguous_doc_spread):
        return ("C", True, "ambiguous")

    # Rule D — everything else is genuinely borderline.
    return ("D", None, "borderline")


# --- Stage 3 -> rule-B verification (§3.5 R1) -----------------------------

async def _verify_rule_b(question: str, probe_hits: list[dict]) -> bool:
    """Confirm a rule-B "specific" verdict via the single top1 chunk.

    A high top1 cosine score does not prove the chunk answers the query, so one
    cheap LLM call checks it (see rag.generator.verify_chunk_answers_query).
    Returns True only on an explicit positive; a negative — or any failure, or
    no probe hits — returns False so the caller demotes the verdict to rule D.
    """
    if not probe_hits:
        return False
    try:
        # Lazy import: keep triage.py importable without the LLM client stack.
        from rag.generator import verify_chunk_answers_query
        return await verify_chunk_answers_query(question, probe_hits[0])
    except Exception:
        logger.warning(
            "triage rule-B verification crashed for q=%.80r; treating as unconfirmed",
            question, exc_info=True,
        )
        return False


# --- Stage 4 helpers ------------------------------------------------------

def _format_probe_context(hits: list[dict]) -> str:
    """Build the CORPUS MATCHES block for the Stage-4 prompt from top-5 hits.

    Distinct doc/section pairs only — the top-5 chunks often repeat a document.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for h in hits[:5]:
        doc = (h.get("doc_title") or "").strip()
        section = (h.get("section_title") or "").strip()
        label = f"{doc} — {section}" if doc and section else (doc or section)
        if not label or label in seen:
            continue
        seen.add(label)
        lines.append(f"{len(lines) + 1}. {label}")
    if not lines:
        return ""
    return (
        "CORPUS MATCHES (top vector-search results for this question):\n"
        + "\n".join(lines)
    )


async def _llm_verdict(question: str, probe_hits: list[dict]) -> tuple[bool, str, float]:
    """Stage 4 — LLM intent verdict, grounded in the top-5 probe results.

    Reuses the legacy classifier's system prompt and JSON schema so the verdict
    stays comparable, but augments the prompt with a CORPUS MATCHES block built
    from the probe (doc + section titles).
    """
    # Lazy imports: keep triage.py importable in offline/script contexts and
    # avoid import-order coupling with the LLM client.
    from rag.dialog.classifier import _CLASSIFY_JSON_SCHEMA, _VALID_REASONS, _describe_choice
    from rag.generator import _make_llm_client, supports_json_object, supports_json_schema

    context = _format_probe_context(probe_hits)
    user_content = f"{question}\n\n{context}" if context else question

    client = _make_llm_client()
    request_kwargs: dict = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT + _STAGE4_SYSTEM_SUFFIX},
            {"role": "user", "content": user_content},
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
            f"No JSON object in Stage-4 response: {content!r} "
            f"[{_describe_choice(choice)}]"
        )
    data = json.loads(content[start:end])

    # Explicit failure on a missing key — never let an absent verdict decay
    # silently into "no clarification needed" (§4.8).
    if "needs_clarification" not in data:
        raise ValueError(f"Stage-4 response missing needs_clarification: {data!r}")
    needs = bool(data["needs_clarification"])

    reason = str(data.get("reason", "")).strip()
    if reason not in _VALID_REASONS:
        reason = "vague_topic" if needs else "specific"

    try:
        confidence = float(data["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        confidence = 0.5

    return needs, reason, confidence


# --- Cascade entry point --------------------------------------------------

async def triage(question: str, retriever) -> TriageResult:
    """Run the Stage 1–4 triage cascade for one query.

    ``retriever`` must expose ``probe_search`` (see rag/retriever.py). The
    returned :class:`TriageResult` always carries a concrete verdict; any
    failure degrades to ``needs_clarification=False`` so a caller in shadow
    mode can log it without risk.
    """
    # --- Stage 1: cheap heuristics ----------------------------------------
    tokens = _tokenize(question)
    if len(tokens) <= 1 or all(t in _STOPWORDS for t in tokens):
        return TriageResult(
            needs_clarification=True, reason="too_short", confidence=1.0,
            decided_by="stage1", rule="", features={},
            probe_hits=[], query_vector=None,
        )

    # --- Stage 2: probe retrieval -----------------------------------------
    try:
        probe_hits, query_vector = await retriever.probe_search(question)
    except Exception:
        logger.warning("triage Stage 2 probe failed for q=%.80r", question, exc_info=True)
        return TriageResult(
            needs_clarification=False, reason="error", confidence=0.5,
            decided_by="stage2", rule="", features={},
            probe_hits=[], query_vector=None,
        )
    features = _score_features(probe_hits)

    # --- Stage 3: score-distribution triage -------------------------------
    rule, needs, reason = _stage3(features)

    # §3.5 R1 — a rule-B "specific" verdict means "answer directly", but a high
    # top1 cosine is not proof the chunk is on-point. Verify the top1 chunk
    # with one cheap LLM call; on a negative or failed check, demote to rule D
    # so the query still gets the thorough Stage-4 verdict.
    if rule == "B" and not await _verify_rule_b(question, probe_hits):
        logger.info(
            "triage: rule B demoted to D — top1 chunk did not verify for q=%.80r",
            question,
        )
        rule, needs, reason = "D", None, "borderline"

    if needs is not None:
        return TriageResult(
            needs_clarification=needs, reason=reason, confidence=1.0,
            decided_by="stage3", rule=rule, features=features,
            probe_hits=probe_hits, query_vector=query_vector,
        )

    # --- Stage 4: LLM verdict (borderline / rule D) -----------------------
    try:
        needs, reason, confidence = await _llm_verdict(question, probe_hits)
    except Exception:
        logger.warning("triage Stage 4 verdict failed for q=%.80r", question, exc_info=True)
        return TriageResult(
            needs_clarification=False, reason="error", confidence=0.5,
            decided_by="stage4", rule="D", features=features,
            probe_hits=probe_hits, query_vector=query_vector,
        )
    return TriageResult(
        needs_clarification=needs, reason=reason, confidence=confidence,
        decided_by="stage4", rule="D", features=features,
        probe_hits=probe_hits, query_vector=query_vector,
    )
