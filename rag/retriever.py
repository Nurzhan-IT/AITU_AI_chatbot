import logging
import math
import re
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from config import settings, TZ_UTC5
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 1024

# --- multi-factor scoring (Stage 4 Part B) --------------------------------

_FACTOR_WEIGHTS: dict[str, float] = {
    "semantic_sim":    0.35,
    "user_type_match": 0.20,
    "doc_match":       0.15,
    "position_bonus":  0.10,
    "recency":         0.10,
    "section_sim":     0.10,
}

# Stems that match document text for the user_type_match factor.
# Profile values are the lowercase Russian canonical forms from extract_profile.
_USER_TYPE_KEYWORDS: dict[str, str] = {
    "бакалавр":   "бакалавр",
    "магистрант": "магистр",
    "сотрудник":  "сотрудник",
}

_POSITION_TEXT_PREFIX = 1500
_POSITION_MIN_WORD = 3
_RECENCY_HALF_DECAY_DAYS = 365.0


def _clamp01(x: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    return max(0.0, min(1.0, x))


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _compute_factors(
    chunk: dict,
    user_keyword: str | None,
    lowered_hints: list[str],
    query_vec: list[float],
    section_cache: dict[str, list[float]],
    now: datetime,
) -> dict[str, float]:
    text = chunk.get("text") or ""
    section_title = (chunk.get("section_title") or "").strip()
    doc_title = (chunk.get("doc_title") or "").strip()

    # semantic_sim
    semantic_sim = _clamp01(float(chunk.get("score", 0.0) or 0.0))

    # user_type_match
    if user_keyword is None:
        user_type_match = 0.5
    else:
        kw = user_keyword.lower()
        haystack = (text.lower(), section_title.lower())
        user_type_match = 1.0 if any(kw in h for h in haystack) else 0.0

    # doc_match
    if not lowered_hints:
        doc_match = 0.5
    else:
        dt_lower = doc_title.lower()
        doc_match = _clamp01(
            max(SequenceMatcher(None, h, dt_lower).ratio() for h in lowered_hints)
        )

    # position_bonus
    if section_title:
        prefix = text[:_POSITION_TEXT_PREFIX].lower()
        section_words = [
            w for w in re.split(r"\W+", section_title.lower())
            if len(w) >= _POSITION_MIN_WORD
        ]
        position_bonus = 1.0 if section_words and any(w in prefix for w in section_words) else 0.5
    else:
        position_bonus = 0.5

    # recency
    uploaded_at = chunk.get("uploaded_at") or ""
    if uploaded_at:
        try:
            dt = datetime.fromisoformat(uploaded_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=now.tzinfo)
            age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
            recency = _clamp01(math.exp(-age_days / _RECENCY_HALF_DECAY_DAYS))
        except (ValueError, TypeError):
            recency = 0.5
    else:
        recency = 0.5

    # section_sim
    if section_title:
        section_vec = section_cache.get(section_title)
        section_sim = _clamp01(_cosine(section_vec, query_vec)) if section_vec is not None else 0.5
    else:
        section_sim = 0.5

    return {
        "semantic_sim":    semantic_sim,
        "user_type_match": user_type_match,
        "doc_match":       doc_match,
        "position_bonus":  position_bonus,
        "recency":         recency,
        "section_sim":     section_sim,
    }


def mmr(
    query_vec: list[float],
    candidates: list[dict],
    k: int = 5,
    lambda_: float = 0.5,
) -> list[dict]:
    """Maximal Marginal Relevance re-ranking.
    candidates must each have a '_vector' key with the chunk embedding.
    """
    if not candidates:
        return []
    vecs = np.array([c["_vector"] for c in candidates], dtype=float)
    q = np.array(query_vec, dtype=float)
    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))
    for _ in range(min(k, len(remaining))):
        if not selected_indices:
            scores = vecs[remaining] @ q
            best = remaining[int(np.argmax(scores))]
        else:
            best_score = -np.inf
            best = remaining[0]
            sel_vecs = vecs[selected_indices]
            for i in remaining:
                rel = float(vecs[i] @ q)
                red = float(np.max(sel_vecs @ vecs[i]))
                score = lambda_ * rel - (1 - lambda_) * red
                if score > best_score:
                    best_score = score
                    best = i
        selected_indices.append(best)
        remaining.remove(best)
    result = [candidates[i] for i in selected_indices]
    for c in result:
        c.pop("_vector", None)   # don't leak raw vectors downstream
    return result


class Retriever:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=60,
        )
        self._embedder = Embedder()
        # Persistent cache of section_title → embedding for the section_sim factor
        # in search_with_profile. Keyed by raw section_title string.
        self._section_emb_cache: dict[str, list[float]] = {}

    async def _ensure_collection(self) -> None:
        exists = await self._client.collection_exists(settings.qdrant_collection)
        if not exists:
            await self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s'", settings.qdrant_collection)

    async def ping(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def search(self, query: str) -> list[dict]:
        await self._ensure_collection()

        vector = await self._embedder.embed_query(query)
        results = await self._client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=settings.top_k * 3,
            with_payload=True,
            with_vectors=True,
        )

        hits = []
        for point in results:
            p = point.payload or {}
            filename = p.get("filename", "")
            url = f"{settings.pdf_base_url.rstrip('/')}/{filename}" if filename else ""
            page = p.get("page", 0)
            hits.append(
                {
                    "text": p.get("text", ""),
                    "doc_title": p.get("doc_title", ""),
                    "filename": filename,
                    "url": url,
                    "uploaded_at": p.get("uploaded_at", ""),
                    "page": page,
                    "page_end": p.get("page_end", page),
                    "section_title": p.get("section_title", ""),
                    "paragraph_range": p.get("paragraph_range", ""),
                    "score": point.score,
                    "_vector": point.vector,
                }
            )

        hits = mmr(vector, hits, k=settings.top_k)
        logger.debug("search('%s'): %d hits", query[:60], len(hits))
        return hits

    async def search_multilingual(
        self,
        question: str,
        detected_lang: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """Search with the original query plus a translation, merge results by score."""
        from rag.generator import translate_query

        await self._ensure_collection()
        limit = top_k if top_k is not None else settings.top_k

        other_lang = "English" if detected_lang == "Russian" else "Russian"
        translation = await translate_query(question, other_lang)

        queries = [question]
        if translation:
            queries.append(translation)

        seen_ids: set = set()
        all_results: list[dict] = []
        original_query_vec: list[float] | None = None
        for q in queries:
            vector = await self._embedder.embed_query(q)
            if original_query_vec is None:
                original_query_vec = vector
            results = await self._client.search(
                collection_name=settings.qdrant_collection,
                query_vector=vector,
                limit=limit * 3,
                with_payload=True,
                with_vectors=True,
            )
            for point in results:
                if point.id in seen_ids:
                    continue
                seen_ids.add(point.id)
                p = point.payload or {}
                filename = p.get("filename", "")
                url = f"{settings.pdf_base_url.rstrip('/')}/{filename}" if filename else ""
                page = p.get("page", 0)
                all_results.append({
                    "text": p.get("text", ""),
                    "doc_title": p.get("doc_title", ""),
                    "filename": filename,
                    "url": url,
                    "uploaded_at": p.get("uploaded_at", ""),
                    "page": page,
                    "page_end": p.get("page_end", page),
                    "section_title": p.get("section_title", ""),
                    "paragraph_range": p.get("paragraph_range", ""),
                    "score": point.score,
                    "_vector": point.vector,
                })

        merged = mmr(original_query_vec, all_results, k=limit)
        logger.debug(
            "search_multilingual('%s', lang=%s): %d queries → %d merged hits",
            question[:60], detected_lang, len(queries), len(merged),
        )
        return merged

    async def search_with_profile(
        self,
        enriched_query: str,
        profile: dict,
        k: int | None = None,
    ) -> list[dict]:
        """Profile-aware multi-factor retrieval.

        Pulls ``settings.top_k * 5`` candidates by cosine, then re-ranks them
        with a weighted blend of six [0,1]-scaled factors (see _FACTOR_WEIGHTS).
        Returns the top ``k`` chunks, each annotated with a ``factor_scores``
        dict for debug logging. Falls back to neutral 0.5 on any missing
        profile slot, so this method is safe to call even when ``profile`` is
        sparse.
        """
        await self._ensure_collection()

        query_vec = await self._embedder.embed_query(enriched_query)
        candidate_limit = settings.top_k * 5
        final_k = k if k is not None else settings.top_k

        results = await self._client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vec,
            limit=candidate_limit,
            with_payload=True,
            with_vectors=True,
        )

        candidates: list[dict] = []
        for point in results:
            p = point.payload or {}
            filename = p.get("filename", "")
            url = f"{settings.pdf_base_url.rstrip('/')}/{filename}" if filename else ""
            page = p.get("page", 0)
            candidates.append({
                "text":            p.get("text", ""),
                "doc_title":       p.get("doc_title", ""),
                "filename":        filename,
                "url":             url,
                "uploaded_at":     p.get("uploaded_at", ""),
                "page":            page,
                "page_end":        p.get("page_end", page),
                "section_title":   p.get("section_title", ""),
                "paragraph_range": p.get("paragraph_range", ""),
                "score":           point.score,
                "_vector":         point.vector,
            })

        # Resolve profile-derived inputs once per call
        user_type = profile.get("user_type") if isinstance(profile, dict) else None
        user_keyword = _USER_TYPE_KEYWORDS.get(user_type) if isinstance(user_type, str) else None
        hints_raw = profile.get("document_hints") if isinstance(profile, dict) else None
        lowered_hints = [
            h.strip().lower() for h in (hints_raw or [])
            if isinstance(h, str) and h.strip()
        ]

        # Embed unique section titles (cached) — single batch of misses per call
        unique_sections = {
            (c.get("section_title") or "").strip()
            for c in candidates
            if (c.get("section_title") or "").strip()
        }
        for st in unique_sections:
            if st in self._section_emb_cache:
                continue
            try:
                self._section_emb_cache[st] = await self._embedder.embed_query(st)
            except Exception:
                logger.warning(
                    "search_with_profile: failed to embed section_title=%r", st,
                    exc_info=True,
                )
                # Don't cache failures — let it retry on a future request.

        now = datetime.now(TZ_UTC5)

        for c in candidates:
            factors = _compute_factors(
                chunk=c,
                user_keyword=user_keyword,
                lowered_hints=lowered_hints,
                query_vec=query_vec,
                section_cache=self._section_emb_cache,
                now=now,
            )
            c["factor_scores"] = factors
            c["final_score"] = sum(_FACTOR_WEIGHTS[k] * v for k, v in factors.items())

        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        top = candidates[:final_k]
        for c in top:
            c.pop("_vector", None)

        logger.debug(
            "search_with_profile('%s'): %d candidates → top-%d (profile=%s)",
            enriched_query[:60], len(candidates), len(top), profile,
        )
        return top

    async def get_all_documents(self) -> list[dict]:
        """Return unique documents (one entry per filename) stored in the collection."""
        await self._ensure_collection()

        seen: set[str] = set()
        docs: list[dict] = []
        offset = None

        while True:
            records, next_offset = await self._client.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=["doc_title", "filename", "url"],
                with_vectors=False,
            )

            for record in records:
                p = record.payload or {}
                filename = p.get("filename", "")
                if filename and filename not in seen:
                    seen.add(filename)
                    docs.append(
                        {
                            "doc_title": p.get("doc_title", ""),
                            "filename": filename,
                            "url": p.get("url", ""),
                        }
                    )

            if next_offset is None:
                break
            offset = next_offset

        logger.info("get_all_documents: %d unique documents", len(docs))
        return docs

    async def get_document_chunks(self, filename: str, limit: int = 60) -> list[dict]:
        """Return up to `limit` chunks for the given filename, used for FAQ generation."""
        await self._ensure_collection()

        file_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        )

        chunks: list[dict] = []
        offset = None

        while len(chunks) < limit:
            records, next_offset = await self._client.scroll(
                collection_name=settings.qdrant_collection,
                scroll_filter=file_filter,
                limit=min(100, limit - len(chunks)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                p = record.payload or {}
                page = p.get("page", 0)
                chunks.append({
                    "text": p.get("text", ""),
                    "doc_title": p.get("doc_title", ""),
                    "filename": p.get("filename", filename),
                    "page": page,
                    "page_end": p.get("page_end", page),
                    "section_title": p.get("section_title", ""),
                    "paragraph_range": p.get("paragraph_range", ""),
                    "score": 1.0,
                })
            if next_offset is None:
                break
            offset = next_offset

        logger.info("get_document_chunks('%s'): %d chunks", filename, len(chunks))
        return chunks

    async def delete_document(self, filename: str) -> int:
        """Delete all chunks for the given filename. Returns number of deleted points."""
        await self._ensure_collection()

        file_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        )

        count_result = await self._client.count(
            collection_name=settings.qdrant_collection,
            count_filter=file_filter,
            exact=True,
        )
        n_to_delete = count_result.count

        if n_to_delete == 0:
            logger.info("delete_document('%s'): not found", filename)
            return 0

        await self._client.delete(
            collection_name=settings.qdrant_collection,
            points_selector=file_filter,
        )

        logger.info("delete_document('%s'): removed %d chunks", filename, n_to_delete)
        return n_to_delete
