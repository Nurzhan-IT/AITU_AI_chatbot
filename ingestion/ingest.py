import argparse
import asyncio
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable
from urllib.parse import quote

import pymupdf4llm
import fitz                  # PyMuPDF raw text extraction for accurate page offsets
import tiktoken
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings, TZ_UTC5
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

def _build_fitz_page_map(filepath: str | Path) -> dict[int, int]:
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


def parse_pdf(filepath: str | Path) -> tuple[str, dict[int, int]]:
    """Return (markdown_text, page_char_offsets).

    page_char_offsets maps page_number → char offset where that page starts,
    so we can approximate which page a chunk belongs to.
    """
    path = Path(filepath)
    pages: list[dict] = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    full_text = "".join(page.get("text", "") for page in pages)
    # NOTE: page_offsets is built from fitz raw text (plain characters per page),
    # while full_text is pymupdf4llm Markdown. Markdown adds heading markers,
    # table syntax, etc., so char counts differ slightly. Page assignment for
    # chunks spanning page boundaries remains approximate, but is more accurate
    # than the previous pymupdf4llm-derived offsets.
    page_offsets = _build_fitz_page_map(path)
    return full_text, page_offsets


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
        existing_size = info.config.params.vectors.size
        if existing_size != _VECTOR_SIZE:
            logger.warning(
                "Collection '%s' has vector size %d, expected %d — recreating.",
                settings.qdrant_collection, existing_size, _VECTOR_SIZE,
            )
            await client.delete_collection(settings.qdrant_collection)
            exists = False
    if not exists:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'", settings.qdrant_collection)


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
    text, page_offsets = parse_pdf(path)
    await _notify(progress_cb, "⚙️ Разбиваю на чанки...")

    chunks = chunk_by_sections(text, page_offsets, settings.chunk_size, settings.chunk_overlap)
    logger.info("Split into %d chunks", len(chunks))
    await _notify(progress_cb, f"🔢 Создаю эмбеддинги для {len(chunks)} фрагментов...")

    embedder = Embedder()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)

    await _ensure_collection(client)

    points: list[PointStruct] = []

    uploaded_at = datetime.now(TZ_UTC5).isoformat()

    for chunk_index, chunk in enumerate(chunks):
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
                    "text": chunk["text"],
                },
            )
        )

    logger.info("Embedding %d chunks...", len(chunks))
    chunk_texts = [c["text"] for c in chunks]
    vectors = await embedder.embed_passages(chunk_texts)
    await _notify(progress_cb, "💾 Сохраняю в базу...")

    for point, vector in zip(points, vectors):
        point.vector = vector

    await client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(
        "Upserted %d chunks for '%s' into collection '%s'",
        len(points),
        filename,
        settings.qdrant_collection,
    )

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
