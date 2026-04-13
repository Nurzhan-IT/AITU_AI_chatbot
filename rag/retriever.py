import logging

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from config import settings
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 1024


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

    async def _ensure_collection(self) -> None:
        exists = await self._client.collection_exists(settings.qdrant_collection)
        if not exists:
            await self._client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s'", settings.qdrant_collection)

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
