"""LLM reranker for the clarification-dialog retrieval path.

Takes the top-N candidates produced by ``Retriever.search_with_profile`` and
asks the LLM to re-order them from most to least relevant to the *original*
user question. The reasoning string for each top chunk is attached to the
chunk as ``rerank_reason`` for DEBUG logging only — it is not surfaced to
the user and not persisted.

Designed to fail safe: any timeout, network error, or malformed JSON results
in the original top-k order being returned unchanged.
"""

import json
import logging

from config import settings
from rag.dialog.prompts import RERANK_SYSTEM_PROMPT
from rag.generator import _make_llm_client

logger = logging.getLogger(__name__)

_TEXT_EXCERPT_LEN = 300


def _build_user_content(query: str, chunks: list[dict]) -> str:
    lines: list[str] = [f"User question: {query}", "", "Candidates:"]
    for i, c in enumerate(chunks):
        title = (c.get("doc_title") or "(untitled)").strip()
        section = (c.get("section_title") or "").strip()
        text = (c.get("text") or "").replace("\n", " ").strip()
        excerpt = text[:_TEXT_EXCERPT_LEN]
        header = f"[{i}] {title}"
        if section:
            header += f" — {section}"
        lines.append(header)
        lines.append(excerpt)
        lines.append("")
    return "\n".join(lines)


def _parse_order(raw, n: int) -> list[int]:
    if not isinstance(raw, list):
        raise ValueError(f"'order' is not a list: {raw!r}")
    if len(raw) != n:
        raise ValueError(f"'order' length {len(raw)} != chunks {n}")
    seen: set[int] = set()
    cleaned: list[int] = []
    for x in raw:
        idx = int(x)  # may raise — caught by caller
        if idx < 0 or idx >= n:
            raise ValueError(f"'order' index out of range: {idx}")
        if idx in seen:
            raise ValueError(f"'order' has duplicate index: {idx}")
        seen.add(idx)
        cleaned.append(idx)
    return cleaned


async def rerank_chunks(query: str, chunks: list[dict], k: int = 5) -> list[dict]:
    if len(chunks) <= k:
        return chunks

    try:
        client = _make_llm_client()
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_content(query, chunks)},
            ],
            temperature=0,
            max_tokens=500,
        )
        content = response.choices[0].message.content or ""

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON object in reranker response: {content!r}")
        data = json.loads(content[start:end])

        order = _parse_order(data.get("order"), len(chunks))
        reasons_raw = data.get("reasons", [])
        reasons: list[str] = (
            [str(r) for r in reasons_raw] if isinstance(reasons_raw, list) else []
        )

        reranked = [chunks[i] for i in order[:k]]
        for rank, c in enumerate(reranked):
            if rank < len(reasons):
                c["rerank_reason"] = reasons[rank]

        logger.debug(
            "rerank: query=%.60r order_full=%s top%d_reasons=%s",
            query, order, k, reasons[:k],
        )
        return reranked
    except Exception:
        logger.warning(
            "rerank_chunks failed for query=%.80r; returning original top-%d",
            query, k, exc_info=True,
        )
        return chunks[:k]
