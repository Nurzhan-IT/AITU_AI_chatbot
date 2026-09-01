"""Core duplicate/stale detection algorithm."""
import asyncio
import logging
from pathlib import Path

from aiogram import Bot
from groq import AsyncGroq
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import settings
from duplicate_detection import notifier, repository
from ingestion.ingest import build_fitz_page_map, chunk_by_sections, parse_pdf
from rag.embedder import Embedder

logger = logging.getLogger(__name__)

_SNIPPET_LEN = 120       # chars shown in log messages for chunk previews
_MIN_ANALYSIS_CHARS = 150  # header-only chunks are ~40-80 chars; skip them


def _snip(text: str, n: int = _SNIPPET_LEN) -> str:
    """Return a single-line snippet of text for log readability."""
    flat = " ".join(text.split())
    return flat[:n] + ("…" if len(flat) > n else "")


# ---------------------------------------------------------------------------
# LLM classification for the ambiguous similarity zone
# ---------------------------------------------------------------------------

async def _classify_with_llm(
    new_chunk: str,
    existing_chunk: str,
    new_doc_title: str,
    existing_doc_title: str,
    similarity: float,
    chunk_index: int,
) -> tuple[str, str]:
    """Return ("STALE" | "SIMILAR_ONLY", reason_string)."""
    logger.debug(
        "[chunk #%d] Sending to LLM for classification "
        "(similarity=%.4f, in ambiguous zone %.2f–%.2f)\n"
        "  NEW  (%s): %s\n"
        "  EXISTING (%s): %s",
        chunk_index, similarity,
        settings.stale_threshold_low, settings.duplicate_threshold,
        new_doc_title, _snip(new_chunk),
        existing_doc_title, _snip(existing_chunk),
    )

    client = AsyncGroq(api_key=settings.groq_api_key)
    system = (
        "You are a document analysis assistant for a university knowledge base. "
        "Determine whether two text fragments indicate that one document supersedes "
        "or updates another (STALE), or whether they are merely similar in topic "
        "without a version/update relationship (SIMILAR_ONLY).\n\n"
        "Reply with EXACTLY one of:\n"
        "STALE: <one sentence explaining why the new text likely supersedes the old>\n"
        "SIMILAR_ONLY: <one sentence explaining why they are independent similar topics>\n\n"
        "Do not add any other text. Do not use markdown."
    )
    user = (
        f"Cosine similarity between these fragments: {similarity:.2f}\n\n"
        f'EXISTING document ("{existing_doc_title}"):\n"""\n{existing_chunk[:600]}\n"""\n\n'
        f'NEW document ("{new_doc_title}"):\n"""\n{new_chunk[:600]}\n"""\n\n'
        "Does the NEW fragment supersede or replace information in the EXISTING fragment?"
    )
    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=100,
        )
        raw_text = (response.choices[0].message.content or "").strip()
        logger.debug("[chunk #%d] LLM raw response: %r", chunk_index, raw_text)

        if raw_text.startswith("STALE:"):
            reason = raw_text[len("STALE:"):].strip()
            logger.debug(
                "[chunk #%d] LLM verdict: STALE — %s", chunk_index, reason
            )
            return "STALE", reason

        logger.debug(
            "[chunk #%d] LLM verdict: SIMILAR_ONLY — not a version relationship, discarding.",
            chunk_index,
        )
        return "SIMILAR_ONLY", ""

    except Exception as exc:
        logger.warning(
            "[chunk #%d] LLM call failed (%s). Defaulting to SIMILAR_ONLY to avoid false positives.",
            chunk_index, exc,
        )
        return "SIMILAR_ONLY", ""


# ---------------------------------------------------------------------------
# Warning deduplication — keep best match per (type, new_file, existing_file)
# ---------------------------------------------------------------------------

def _deduplicate(raw: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for w in raw:
        key = (w["warning_type"], w["new_filename"], w["existing_filename"])
        if key not in best:
            best[key] = w
        elif w["similarity"] > best[key]["similarity"]:
            logger.debug(
                "Dedup [%s | %s → %s]: replacing similarity %.4f with higher %.4f",
                w["warning_type"], w["new_filename"], w["existing_filename"],
                best[key]["similarity"], w["similarity"],
            )
            best[key] = w
        else:
            logger.debug(
                "Dedup [%s | %s → %s]: keeping existing similarity %.4f, discarding %.4f",
                w["warning_type"], w["new_filename"], w["existing_filename"],
                best[key]["similarity"], w["similarity"],
            )
    return list(best.values())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def analyze_new_document(
    filepath: Path,
    filename: str,
    doc_title: str,
    bot: Bot,
    admin_ids: list[int],
) -> list[dict]:
    """Analyse a just-uploaded document for duplicates/stale content.

    Returns a list of warning dicts that were saved to SQLite.
    Never raises — all errors are caught and logged.
    """
    try:
        return await _run_analysis(filepath, filename, doc_title, bot, admin_ids)
    except Exception as exc:
        logger.error(
            "Duplicate detection pipeline crashed for '%s': %s",
            filename, exc, exc_info=True,
        )
        return []


async def _run_analysis(
    filepath: Path,
    filename: str,
    doc_title: str,
    bot: Bot,
    admin_ids: list[int],
) -> list[dict]:
    logger.info("=" * 60)
    logger.info("START analysis: '%s' (title='%s', path=%s)", filename, doc_title, filepath)
    logger.info(
        "Thresholds — DUPLICATE: ≥%.2f | STALE zone: %.2f–%.2f | below %.2f: ignored",
        settings.duplicate_threshold,
        settings.stale_threshold_low, settings.duplicate_threshold,
        settings.stale_threshold_low,
    )

    # ------------------------------------------------------------------
    # 1. Re-parse + chunk
    # ------------------------------------------------------------------
    logger.debug("Step 1: Parsing PDF '%s'…", filepath)
    text = parse_pdf(filepath)
    page_offsets = build_fitz_page_map(filepath)
    logger.debug(
        "Parsed: %d chars of markdown, %d page offset entries.",
        len(text), len(page_offsets),
    )

    chunks = chunk_by_sections(
        text, page_offsets,
        max_tokens=settings.chunk_size,
        overlap_tokens=settings.chunk_overlap,
    )

    if not chunks:
        logger.warning(
            "No chunks extracted from '%s' — PDF may be empty or unreadable. Aborting detection.",
            filename,
        )
        return []

    logger.info("Step 1 done: extracted %d chunk(s) from '%s'.", len(chunks), filename)
    for i, c in enumerate(chunks):
        logger.debug(
            "  chunk #%d | section='%s' | para='%s' | pages %s–%s | %s",
            i,
            c.get("section_title", "—"),
            c.get("paragraph_range", "—"),
            c.get("page", "?"), c.get("page_end", "?"),
            _snip(c["text"]),
        )

    # Filter out header-only chunks (section title with no body text).
    # These are ~40–80 chars and produce inflated cosine scores against
    # structurally similar documents, causing false DUPLICATE flags.
    filterable = [(i, c) for i, c in enumerate(chunks) if len(c["text"]) >= _MIN_ANALYSIS_CHARS]
    skipped_header = len(chunks) - len(filterable)
    if skipped_header:
        logger.debug(
            "Skipping %d header-only/short chunk(s) (<%d chars) from duplicate detection.",
            skipped_header, _MIN_ANALYSIS_CHARS,
        )
    if not filterable:
        logger.info(
            "RESULT for '%s': no analysable chunks (all %d chunk(s) are header-only). "
            "Skipping duplicate detection.",
            filename, len(chunks),
        )
        logger.info("=" * 60)
        return []

    # ------------------------------------------------------------------
    # 2. Batch embed
    # ------------------------------------------------------------------
    logger.debug("Step 2: Embedding %d chunk(s) via OpenRouter…", len(filterable))
    embedder = Embedder()
    try:
        chunk_texts = [c["text"] for _, c in filterable]
        vectors = await embedder.embed_passages(chunk_texts)
    finally:
        await embedder.aclose()
    logger.info(
        "Step 2 done: obtained %d embedding vector(s) (%d header-only chunk(s) skipped).",
        len(vectors), skipped_header,
    )

    # ------------------------------------------------------------------
    # 3. Per-chunk Qdrant search
    # ------------------------------------------------------------------
    logger.info(
        "Step 3: Searching Qdrant for nearest neighbours (score_threshold=%.2f, excluding '%s')…",
        settings.stale_threshold_low, filename,
    )

    qdrant = AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    exclude_self = Filter(
        must_not=[FieldCondition(key="filename", match=MatchValue(value=filename))]
    )

    # Counters for the summary log
    n_below_threshold = 0
    n_duplicate = 0
    n_stale = 0
    n_similar_only = 0

    # limit=3: ищем топ-3 совпадения на каждый чанк, чтобы найти ВСЕ
    # релевантные файлы (не только ближайший), иначе при наличии нескольких
    # похожих документов предупреждение может указать на неверный файл.
    async def _search_one(chunk: dict, vector: list[float], idx: int) -> list[dict]:
        nonlocal n_below_threshold, n_duplicate, n_stale, n_similar_only

        results = await qdrant.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            query_filter=exclude_self,
            limit=3,
            with_payload=True,
            score_threshold=settings.stale_threshold_low,
        )

        if not results:
            n_below_threshold += 1
            logger.debug(
                "[chunk #%d] No match above threshold %.2f — IGNORED. "
                "Chunk: %s",
                idx, settings.stale_threshold_low, _snip(chunk["text"]),
            )
            return []

        chunk_warnings: list[dict] = []

        for hit_rank, hit in enumerate(results):
            score: float = hit.score
            payload: dict = hit.payload or {}
            existing_filename: str = payload.get("filename", "unknown")
            existing_text: str = payload.get("text", "")
            existing_title: str = payload.get("doc_title", existing_filename)
            existing_section: str = payload.get("section_title", "—")
            existing_para: str = payload.get("paragraph_range", "—")
            existing_page: int = payload.get("page", 0)

            logger.debug(
                "[chunk #%d] Hit #%d: score=%.4f | file='%s' | section='%s' | para='%s' | page=%d\n"
                "  NEW chunk      : %s\n"
                "  EXISTING chunk : %s",
                idx, hit_rank + 1, score,
                existing_filename, existing_section, existing_para, existing_page,
                _snip(chunk["text"]),
                _snip(existing_text),
            )

            # --- DUPLICATE ---
            if score >= settings.duplicate_threshold:
                n_duplicate += 1
                logger.info(
                    "[chunk #%d] Hit #%d → DUPLICATE (score=%.4f ≥ threshold=%.2f)\n"
                    "  New file     : '%s'\n"
                    "  Existing file: '%s' | section='%s' | para='%s' | page=%d\n"
                    "  New text     : %s\n"
                    "  Existing text: %s",
                    idx, hit_rank + 1, score, settings.duplicate_threshold,
                    filename,
                    existing_filename, existing_section, existing_para, existing_page,
                    _snip(chunk["text"]),
                    _snip(existing_text),
                )
                chunk_warnings.append({
                    "warning_type": "DUPLICATE",
                    "new_filename": filename,
                    "existing_filename": existing_filename,
                    "new_chunk_text": chunk["text"],
                    "existing_chunk_text": existing_text,
                    "similarity": score,
                    "llm_reason": None,
                })
                continue

            # --- AMBIGUOUS ZONE → LLM ---
            logger.debug(
                "[chunk #%d] Hit #%d score %.4f is in ambiguous zone (%.2f–%.2f) → sending to LLM.",
                idx, hit_rank + 1, score,
                settings.stale_threshold_low, settings.duplicate_threshold,
            )
            verdict, reason = await _classify_with_llm(
                new_chunk=chunk["text"],
                existing_chunk=existing_text,
                new_doc_title=doc_title,
                existing_doc_title=existing_title,
                similarity=score,
                chunk_index=idx,
            )

            if verdict == "STALE":
                n_stale += 1
                logger.info(
                    "[chunk #%d] Hit #%d → STALE (score=%.4f, LLM confirmed)\n"
                    "  New file     : '%s'\n"
                    "  Existing file: '%s' | section='%s' | para='%s' | page=%d\n"
                    "  LLM reason   : %s\n"
                    "  New text     : %s\n"
                    "  Existing text: %s",
                    idx, hit_rank + 1, score,
                    filename,
                    existing_filename, existing_section, existing_para, existing_page,
                    reason,
                    _snip(chunk["text"]),
                    _snip(existing_text),
                )
                chunk_warnings.append({
                    "warning_type": "STALE",
                    "new_filename": filename,
                    "existing_filename": existing_filename,
                    "new_chunk_text": chunk["text"],
                    "existing_chunk_text": existing_text,
                    "similarity": score,
                    "llm_reason": reason,
                })
            else:
                n_similar_only += 1
                logger.debug(
                    "[chunk #%d] Hit #%d → SIMILAR_ONLY (score=%.4f, LLM: not a version relationship) — IGNORED.\n"
                    "  New text     : %s\n"
                    "  Existing text: %s",
                    idx, hit_rank + 1, score,
                    _snip(chunk["text"]),
                    _snip(existing_text),
                )

        return chunk_warnings

    search_tasks = [_search_one(c, v, i) for (i, c), v in zip(filterable, vectors)]
    results_list = await asyncio.gather(*search_tasks)

    raw_warnings = [w for chunk_warnings in results_list for w in chunk_warnings]

    await qdrant.close()

    # Per-chunk summary
    logger.info(
        "Step 3 done: %d chunk(s) processed.\n"
        "  Below threshold (%.2f)  : %d chunk(s) — no match, ignored\n"
        "  DUPLICATE (≥%.2f)       : %d chunk(s)\n"
        "  STALE (LLM confirmed)    : %d chunk(s)\n"
        "  SIMILAR_ONLY (discarded) : %d chunk(s)",
        len(chunks),
        settings.stale_threshold_low, n_below_threshold,
        settings.duplicate_threshold, n_duplicate,
        n_stale,
        n_similar_only,
    )

    # ------------------------------------------------------------------
    # 4. Deduplicate
    # ------------------------------------------------------------------
    logger.debug(
        "Step 4: Deduplicating %d raw warning(s) → keep 1 per (type, new_file, existing_file)…",
        len(raw_warnings),
    )
    deduped = _deduplicate(raw_warnings)
    logger.info(
        "Step 4 done: %d raw warning(s) → %d after deduplication.",
        len(raw_warnings), len(deduped),
    )

    if not deduped:
        logger.info(
            "RESULT for '%s': no warnings generated. "
            "(%d chunk(s) below threshold, %d SIMILAR_ONLY by LLM)",
            filename, n_below_threshold, n_similar_only,
        )
        logger.info("=" * 60)
        return []

    # ------------------------------------------------------------------
    # 5. Persist to SQLite
    # ------------------------------------------------------------------
    logger.debug("Step 5: Persisting %d warning(s) to SQLite…", len(deduped))
    saved: list[dict] = []
    for w in deduped:
        w_id = await repository.create_warning(
            warning_type=w["warning_type"],
            new_filename=w["new_filename"],
            existing_filename=w["existing_filename"],
            new_chunk_text=w["new_chunk_text"],
            existing_chunk_text=w["existing_chunk_text"],
            similarity=w["similarity"],
            llm_reason=w.get("llm_reason"),
        )
        saved.append({**w, "id": w_id})
        logger.debug(
            "  Saved warning #%d: %s | '%s' → '%s' | similarity=%.4f",
            w_id, w["warning_type"], w["new_filename"], w["existing_filename"], w["similarity"],
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    dup_list = [w for w in saved if w["warning_type"] == "DUPLICATE"]
    stale_list = [w for w in saved if w["warning_type"] == "STALE"]

    logger.info(
        "RESULT for '%s': %d warning(s) saved.\n"
        "  DUPLICATE (%d): %s\n"
        "  STALE     (%d): %s",
        filename, len(saved),
        len(dup_list),
        ", ".join(f"#{w['id']} vs '{w['existing_filename']}' ({w['similarity']:.0%})" for w in dup_list) or "—",
        len(stale_list),
        ", ".join(f"#{w['id']} vs '{w['existing_filename']}' ({w['similarity']:.0%})" for w in stale_list) or "—",
    )
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 6. Notify admin
    # ------------------------------------------------------------------
    await notifier.send_upload_warnings(bot, admin_ids, filename, saved)

    return saved
