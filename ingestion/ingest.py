import argparse
import asyncio
import bisect
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import quote

import fitz  # PyMuPDF raw text extraction for accurate page offsets
import pymupdf4llm
import tiktoken
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import TZ_UTC5, settings
from ingestion.chunk_metadata import extract_chunk_metadata_batch
from ingestion.page_classifier import PageType, classify_document
from ingestion.page_processor import process_page
from ingestion.post_processor import postprocess
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 1024
_ENCODING = "cl100k_base"

# Regex для section-aware chunking.
# Поддерживает как plain ("2. Магистерская…"), так и markdown ("## 2. Магистерская…") заголовки.
# Группа 1 захватывает только текст заголовка без символов '#'.
_SECTION_RE = re.compile(
    r'^(?:#{1,6}\s*)?(\d+\.\s+[А-ЯA-Z].+|Термины и сокращения|Общие положения)',
    re.MULTILINE,
)
_PARAGRAPH_RE = re.compile(r'(?=^\s{0,4}\d{1,3}[\.\)]\s)', re.MULTILINE)


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def build_fitz_page_map(filepath: str | Path) -> dict[int, int]:
    """Build char-offset → page-number map using fitz raw text.

    Returns a dict where each key is the cumulative character offset at the
    START of a page, and the value is the 1-based page number.
    This mirrors the format expected by _page_for_offset().
    """
    doc = fitz.open(str(filepath))
    page_map: dict[int, int] = {}
    cumulative = 0
    for page in doc:
        page_num = page.number + 1          # fitz is 0-indexed
        page_map[cumulative] = page_num
        text = page.get_text("text")
        cumulative += len(text)
        page_map[cumulative] = page_num     # end-of-page offset also maps to this page
    doc.close()
    return page_map


def parse_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    classifications = classify_document(pdf_path)

    # Single pymupdf4llm call for all non-SCAN pages
    non_scan = [pn for pn, pt in classifications if pt != PageType.SCAN]
    if non_scan:
        full_md = pymupdf4llm.to_markdown(pdf_path, pages=non_scan)
        sections = re.split(r'\n-----\n', full_md)
        while len(sections) < len(non_scan):
            sections.append("")
        pre_md_map: dict[int, str] = dict(zip(non_scan, sections))
    else:
        pre_md_map = {}

    pages: list[tuple[int, str]] = []
    for page_num, page_type in classifications:
        page = doc[page_num]
        markdown = process_page(page, page_type, pre_md_map.get(page_num, ""), doc)
        pages.append((page_num + 1, markdown))
    doc.close()
    return postprocess(pages)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    enc = tiktoken.get_encoding(_ENCODING)
    tokens = enc.encode(text)
    chunks: list[str] = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def _extract_para_num(text: str) -> str | None:
    m = re.match(r'^\s*(\d+)[\.\)]', text)
    return m.group(1) if m else None


def _add_chunk(
    chunks: list[dict],
    text: str,
    section_title: str,
    para_range: str | None,
    char_offset: int,
    page_offsets: dict[int, int],
) -> None:
    raw_len = len(text)
    text = text.strip()
    if not text:
        return
    page = _page_for_offset(char_offset, page_offsets)
    page_end = _page_for_offset(char_offset + raw_len, page_offsets)
    full_text = f"[{section_title}]"
    if para_range:
        full_text += f" [{para_range}]"
    full_text += f"\n{text}"
    chunks.append({
        "text": full_text,
        "section_title": section_title,
        "paragraph_range": para_range or "",
        "page": page,
        "page_end": page_end,
        "char_offset": char_offset,
    })


def _token_chunks_fallback(
    text: str,
    page_offsets: dict[int, int],
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict]:
    """Fallback: обычный токенный чанкинг, оборачивает результат в dict-формат."""
    raw_chunks = chunk_text(text, max_tokens, overlap_tokens)
    result = []
    char_offset = 0
    for raw in raw_chunks:
        page = _page_for_offset(char_offset, page_offsets)
        page_end = _page_for_offset(char_offset + len(raw), page_offsets)
        result.append({
            "text": raw,
            "section_title": "",
            "paragraph_range": "",
            "page": page,
            "page_end": page_end,
            "char_offset": char_offset,
        })
        char_offset += len(raw)
    return result


def chunk_by_sections(
    text: str,
    page_offsets: dict[int, int],
    max_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[dict]:
    """Иерархическое разбиение по разделам и пронумерованным пунктам.

    Возвращает список dict: {text, section_title, paragraph_range, page}.
    Если структура документа не распознана — fallback на токенный чанкинг.
    """
    enc = tiktoken.get_encoding(_ENCODING)
    chunks: list[dict] = []

    section_splits = list(_SECTION_RE.finditer(text))
    if not section_splits:
        return _token_chunks_fallback(text, page_offsets, max_tokens, overlap_tokens)

    sections = []
    for i, match in enumerate(section_splits):
        start = match.start()
        end = section_splits[i + 1].start() if i + 1 < len(section_splits) else len(text)
        sections.append({
            "title": match.group(1).strip(),
            "text": text[start:end],
            "char_offset": start,
        })

    for section in sections:
        sec_text = section["text"]
        sec_title = section["title"]
        sec_char_offset = section["char_offset"]

        para_splits = list(_PARAGRAPH_RE.finditer(sec_text))

        if not para_splits:
            _add_chunk(chunks, sec_text, sec_title, None, sec_char_offset, page_offsets)
            continue

        # Собираем позиции всех пунктов
        para_positions = []
        for i, match in enumerate(para_splits):
            start = match.start()
            end = para_splits[i + 1].start() if i + 1 < len(para_splits) else len(sec_text)
            para_text = sec_text[start:end]
            para_num = _extract_para_num(para_text)
            para_positions.append((start, para_text, para_num))

        # Жадно группируем пункты в чанки
        # Храним (sec_relative_offset, text) чтобы знать позицию первого пункта
        current_paras: list[tuple[int, str]] = []
        current_tokens = 0
        first_para_num: str | None = None
        last_para_num: str | None = None

        for (start, para_text, para_num) in para_positions:
            tokens = len(enc.encode(para_text))

            if current_tokens + tokens > max_tokens and current_paras:
                para_range = (
                    f"п.{first_para_num}–{last_para_num}" if first_para_num else None
                )
                _add_chunk(
                    chunks,
                    "\n".join(t for _, t in current_paras),
                    sec_title,
                    para_range,
                    sec_char_offset + current_paras[0][0],  # позиция первого пункта чанка
                    page_offsets,
                )
                # Overlap: оставляем последний пункт
                overlap = current_paras[-1]
                current_paras = [overlap, (start, para_text)]
                current_tokens = sum(len(enc.encode(t)) for _, t in current_paras)
                first_para_num = _extract_para_num(overlap[1])
                last_para_num = para_num
            else:
                current_paras.append((start, para_text))
                current_tokens += tokens
                if first_para_num is None:
                    first_para_num = para_num
                last_para_num = para_num

        if current_paras:
            para_range = (
                f"п.{first_para_num}–{last_para_num}" if first_para_num else None
            )
            _add_chunk(
                chunks,
                "\n".join(t for _, t in current_paras),
                sec_title,
                para_range,
                sec_char_offset + current_paras[0][0],  # позиция первого пункта чанка
                page_offsets,
            )

    return chunks


def _page_for_offset(char_offset: int, page_offsets: dict[int, int]) -> int:
    """Return the page number that contains char_offset."""
    sorted_offsets = sorted(page_offsets.items())
    page = sorted_offsets[0][1] if sorted_offsets else 1
    for offset, page_num in sorted_offsets:
        if char_offset >= offset:
            page = page_num
        else:
            break
    return page


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

async def _ensure_collection(client: AsyncQdrantClient) -> None:
    exists = await client.collection_exists(settings.qdrant_collection)
    if exists:
        info = await client.get_collection(settings.qdrant_collection)
        need_recreate = False

        if settings.hybrid_search_enabled:
            # Hybrid schema: named dense vector + sparse vector.
            named = isinstance(info.config.params.vectors, dict)
            has_dense = named and "dense" in info.config.params.vectors
            dense_ok = (
                has_dense and info.config.params.vectors["dense"].size == _VECTOR_SIZE
            )
            has_sparse = bool(
                info.config.params.sparse_vectors
                and "sparse" in info.config.params.sparse_vectors
            )
            if not (dense_ok and has_sparse):
                logger.warning(
                    "Collection '%s' schema is incompatible with hybrid search "
                    "(dense_ok=%s, has_sparse=%s) — recreating. "
                    "Re-index all documents after this run.",
                    settings.qdrant_collection, dense_ok, has_sparse,
                )
                need_recreate = True
        else:
            # Dense-only schema: single unnamed VectorParams.
            if isinstance(info.config.params.vectors, dict):
                # Collection has named vectors but hybrid is disabled — mismatched state.
                logger.warning(
                    "Collection '%s' uses named vectors but HYBRID_SEARCH_ENABLED=false "
                    "— recreating for clean state. Re-index all documents.",
                    settings.qdrant_collection,
                )
                need_recreate = True
            else:
                existing_size = info.config.params.vectors.size
                if existing_size != _VECTOR_SIZE:
                    logger.warning(
                        "Collection '%s' has vector size %d, expected %d — recreating.",
                        settings.qdrant_collection, existing_size, _VECTOR_SIZE,
                    )
                    need_recreate = True

        if need_recreate:
            await client.delete_collection(settings.qdrant_collection)
            exists = False

    if not exists:
        if settings.hybrid_search_enabled:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={"dense": VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )},
            )
        else:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
            )
        logger.info(
            "Created collection '%s' (hybrid=%s)",
            settings.qdrant_collection, settings.hybrid_search_enabled,
        )

    # Payload indices for profile-aware filtering (§2.1). Both fields are
    # populated by ingestion.chunk_metadata, and the retriever uses them with
    # Filter(must=[FieldCondition(...)]) to hard-filter by user profile.
    # Calls are idempotent in practice but wrapped defensively so an existing
    # index never aborts ingestion.
    for field_name in ("applies_to", "admission_type"):
        try:
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.debug(
                "create_payload_index('%s') skipped (likely already exists)",
                field_name, exc_info=True,
            )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def _notify(cb: Callable[[str], Awaitable[None]] | None, msg: str) -> None:
    if cb is not None:
        try:
            await cb(msg)
        except Exception:
            pass


async def ingest_pdf(
    filepath: str | Path,
    title: str,
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
) -> int:
    """Ingest a PDF into Qdrant. Returns the number of chunks upserted."""
    path = Path(filepath)
    filename = path.name
    url = f"{settings.pdf_base_url.rstrip('/')}/{quote(filename)}"

    logger.info("Parsing '%s'...", filename)
    text = parse_pdf(str(path))
    page_offsets = build_fitz_page_map(path)

    _marker_re = re.compile(r'<!-- page: (\d+) -->')
    _marker_pairs = [(m.start(), int(m.group(1))) for m in _marker_re.finditer(text)]
    _marker_offsets = [pos for pos, _ in _marker_pairs]
    _marker_pages = [pnum for _, pnum in _marker_pairs]

    await _notify(progress_cb, "⚙️ Разбиваю на чанки...")

    chunks = chunk_by_sections(text, page_offsets, settings.chunk_size, settings.chunk_overlap)
    logger.info("Split into %d chunks", len(chunks))

    clean_texts = [re.sub(r'<!--.*?-->', '', c["text"]).strip() for c in chunks]

    await _notify(progress_cb, f"🔢 Создаю эмбеддинги для {len(chunks)} фрагментов...")

    embedder = Embedder()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)

    await _ensure_collection(client)

    # BM25 sparse vectors (hybrid mode only)
    bm25_token_lists: list[list[str]] = []
    if settings.hybrid_search_enabled:
        from pathlib import Path as _Path

        from rag.bm25 import BM25Stats
        from rag.bm25 import tokenize as bm25_tokenize
        bm25_stats = BM25Stats.load(_Path(settings.bm25_stats_path))
        bm25_token_lists = [bm25_tokenize(t) for t in clean_texts]
        # Register new chunks into stats so avg_len reflects this batch.
        # avg_len is used below to compute the TF component, so populate stats
        # before building doc vectors.
        for tokens in bm25_token_lists:
            bm25_stats.add_chunk(tokens)

    points: list[PointStruct] = []

    uploaded_at = datetime.now(TZ_UTC5).isoformat()

    for chunk_index, chunk in enumerate(chunks):
        idx = bisect.bisect_right(_marker_offsets, chunk["char_offset"]) - 1
        page_number = _marker_pages[idx] if idx >= 0 and _marker_pages else 1
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[],  # filled below after batched embed
                payload={
                    "doc_title": title,
                    "filename": filename,
                    "url": url,
                    "uploaded_at": uploaded_at,
                    "page": chunk["page"],
                    "page_end": chunk["page_end"],
                    "chunk_index": chunk_index,
                    "section_title": chunk["section_title"],
                    "paragraph_range": chunk["paragraph_range"],
                    "text": clean_texts[chunk_index],
                    "page_number": page_number,
                },
            )
        )

    logger.info("Embedding %d chunks...", len(chunks))
    vectors = await embedder.embed_passages(clean_texts)

    # Per-chunk LLM metadata (§2.1). Gated by settings.enable_chunk_metadata so
    # debug runs can skip the LLM cost. extract_chunk_metadata_batch is
    # fail-open: on any error it returns {"applies_to": [], "admission_type": [],
    # "topic_tags": []} for the affected chunk and logs a warning.
    if settings.enable_chunk_metadata:
        await _notify(progress_cb, "🏷️ Классифицирую чанки (метаданные)...")
        metadata_list = await extract_chunk_metadata_batch(
            clean_texts, batch_size=settings.chunk_metadata_batch_size,
        )
        if len(metadata_list) != len(points):
            logger.warning(
                "chunk_metadata length mismatch: got %d, expected %d — padding with empty",
                len(metadata_list), len(points),
            )
            empty = {"applies_to": [], "admission_type": [], "topic_tags": []}
            metadata_list = (metadata_list + [empty] * len(points))[:len(points)]
        for point, meta in zip(points, metadata_list):
            point.payload["applies_to"] = meta.get("applies_to", [])
            point.payload["admission_type"] = meta.get("admission_type", [])
            point.payload["topic_tags"] = meta.get("topic_tags", [])
    else:
        logger.info("enable_chunk_metadata=False — skipping LLM metadata extraction")

    await _notify(progress_cb, "💾 Сохраняю в базу...")

    if settings.hybrid_search_enabled:
        from rag.bm25 import bm25_doc_vector
        avg_len = bm25_stats.avg_len
        for point, dense_vec, tokens in zip(points, vectors, bm25_token_lists):
            sp_indices, sp_values = bm25_doc_vector(
                tokens, avg_len, settings.bm25_k1, settings.bm25_b
            )
            point.vector = {
                "dense": dense_vec,
                "sparse": SparseVector(indices=sp_indices, values=sp_values),
            }
    else:
        for point, vector in zip(points, vectors):
            point.vector = vector

    await client.upsert(collection_name=settings.qdrant_collection, points=points)

    if settings.hybrid_search_enabled:
        bm25_stats.save(_Path(settings.bm25_stats_path))
        logger.info("BM25 stats updated: %d total chunks", bm25_stats.total_chunks)
    logger.info(
        "Upserted %d chunks for '%s' into collection '%s'",
        len(points),
        filename,
        settings.qdrant_collection,
    )

    from rag.dialog.question_gen import invalidate_docs_cache
    invalidate_docs_cache()

    # Generate and persist a 1–2 sentence document summary (§4.7 / D3).
    # Failures are logged and silently swallowed — indexing must not abort.
    try:
        from rag.dialog.summary_store import upsert_doc_summary
        from rag.generator import generate_doc_summary
        excerpt = "\n\n".join(clean_texts[:5])[:3000]
        summary = await generate_doc_summary(title, excerpt)
        if summary:
            await upsert_doc_summary(filename, title, summary)
            logger.info("Saved doc summary for '%s': %.100s", filename, summary)
    except Exception:
        logger.warning("Doc summary generation failed for '%s'; skipping", filename, exc_info=True)

    await embedder.aclose()
    await client.close()
    return len(points)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDF documents into Qdrant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", metavar="PATH", help="Single PDF file to ingest")
    group.add_argument("--dir", metavar="DIR", help="Directory of PDF files to ingest")
    parser.add_argument("--title", metavar="TITLE", help="Document title (only with --file)")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_file():
            parser.error(f"File not found: {args.file}")
        title = args.title or path.stem
        await ingest_pdf(path, title)

    elif args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            parser.error(f"Directory not found: {args.dir}")
        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in '%s'", args.dir)
            return
        for pdf_path in pdf_files:
            await ingest_pdf(pdf_path, title=pdf_path.stem)


if __name__ == "__main__":
    logging.Formatter.converter = lambda *args: datetime.now(TZ_UTC5).timetuple()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())
