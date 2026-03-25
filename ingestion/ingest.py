import argparse
import asyncio
import logging
import uuid
from pathlib import Path
from urllib.parse import quote

import pymupdf4llm
import tiktoken
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

_VECTOR_SIZE = 1536
_ENCODING = "cl100k_base"


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_pdf(filepath: str | Path) -> tuple[str, dict[int, int]]:
    """Return (markdown_text, page_char_offsets).

    page_char_offsets maps page_number → char offset where that page starts,
    so we can approximate which page a chunk belongs to.
    """
    path = Path(filepath)
    pages: list[dict] = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    full_text = ""
    page_offsets: dict[int, int] = {}
    for page in pages:
        page_num: int = page.get("metadata", {}).get("page", 0) + 1  # 1-based
        page_offsets[len(full_text)] = page_num
        full_text += page.get("text", "")

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


def _page_for_offset(char_offset: int, page_offsets: dict[int, int]) -> int:
    """Return the page number that contains char_offset."""
    page = 1
    for offset, page_num in sorted(page_offsets.items()):
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
    if not exists:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'", settings.qdrant_collection)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def ingest_pdf(filepath: str | Path, title: str) -> int:
    """Ingest a PDF into Qdrant. Returns the number of chunks upserted."""
    path = Path(filepath)
    filename = path.name
    url = f"{settings.pdf_base_url.rstrip('/')}/{quote(filename)}"

    logger.info("Parsing '%s'...", filename)
    text, page_offsets = parse_pdf(path)

    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    logger.info("Split into %d chunks", len(chunks))

    embedder = Embedder()
    client = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    await _ensure_collection(client)

    # Build char offsets per chunk to estimate page numbers
    char_offset = 0
    points: list[PointStruct] = []

    for chunk_index, chunk in enumerate(chunks):
        page = _page_for_offset(char_offset, page_offsets)
        char_offset += len(chunk)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[],  # filled below after batched embed
                payload={
                    "doc_title": title,
                    "filename": filename,
                    "url": url,
                    "page": page,
                    "chunk_index": chunk_index,
                    "text": chunk,
                },
            )
        )

    logger.info("Embedding %d chunks...", len(chunks))
    vectors = await embedder.embed(chunks)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_main())
