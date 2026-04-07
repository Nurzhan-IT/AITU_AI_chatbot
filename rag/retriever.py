import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from config import settings
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 1024


class Retriever:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
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
            limit=settings.top_k,
            with_payload=True,
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
                    "page": page,
                    "page_end": p.get("page_end", page),
                    "section_title": p.get("section_title", ""),
                    "paragraph_range": p.get("paragraph_range", ""),
                    "score": point.score,
                }
            )

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
        for q in queries:
            vector = await self._embedder.embed_query(q)
            results = await self._client.search(
                collection_name=settings.qdrant_collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
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
                    "page": page,
                    "page_end": p.get("page_end", page),
                    "section_title": p.get("section_title", ""),
                    "paragraph_range": p.get("paragraph_range", ""),
                    "score": point.score,
                })

        all_results.sort(key=lambda h: h["score"], reverse=True)
        merged = all_results[:limit]
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
