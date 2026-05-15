# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AITU Chatbot — a Telegram bot for AITU university students. It answers questions in Russian, English, and Kazakh by searching indexed university documents via RAG (Retrieval-Augmented Generation). Only verified AITU students (email `@astanait.edu.kz`) can use it.

## Development Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # Then fill in all API keys

# Start dependencies (Qdrant + Nginx for PDF serving)
docker compose up -d

# Index documents
python -m ingestion.ingest --dir pdfs/

# Run bot (development)
python -m bot.main

# Production (all services including bot in Docker)
docker compose -f docker-compose.prod.yml up -d
```

There are no tests or linter configs in this project.

## Architecture

### Request Flow

```
User → Telegram → aiogram → Auth check (SQLite) → Handler
                                                      ↓
                                              rag/retriever.py
                                           (embed query → Qdrant MMR search)
                                                      ↓
                                              rag/generator.py
                                           (Groq/OpenRouter LLM → answer with citations)
                                                      ↓
                                         feedback.py logs query + awaits thumbs up/down
```

### Key Modules

**`bot/`** — All Telegram logic (aiogram 3.13)
- `bot/main.py` — Entry point: initializes Qdrant collection, SQLite DB, registers all routers, starts polling
- `bot/auth/` — Email verification flow using Brevo API; users must verify `@astanait.edu.kz` email before accessing the bot
- `bot/faq_repository.py` — Shared FAQ DB access used across FAQ handlers
- `bot/keyboards/` — ReplyKeyboardMarkup definitions: `user.py` (FAQ button), `admin.py` (full admin menu + FAQ/AI FAQ buttons)
- `bot/handlers/user.py` — Handles text queries and uploaded `.docx`/`.doc` files; rate limiting (10 req/min, in-memory), language detection, calls RAG pipeline
- `bot/handlers/admin.py` — Document management: `/upload`, `/list`, `/delete`, `/stats`, `/warnings`, `/resolve`, `/history`, `/health`, `/report`; triggers duplicate detection on upload
- `bot/handlers/admin_faq.py` — Manual FAQ CRUD via Telegram UI
- `bot/handlers/admin_ai_faq.py` — AI-generated FAQ per document (summarize chunks → generate Q&A pairs via LLM)
- `bot/handlers/feedback.py` — Logs every query to `query_logs` table; stores thumbs up/down feedback

**`rag/`** — Retrieval-Augmented Generation core
- `rag/embedder.py` — Embeds text via OpenRouter (default: `multilingual-e5-large`, 1024 dims)
- `rag/retriever.py` — Qdrant cosine similarity search with MMR reranking for diverse top-K results
- `rag/generator.py` — Sends retrieved chunks + query to LLM (Groq or OpenRouter); returns answer with source citations

**`ingestion/`** — Document ingestion pipeline
- `ingestion/ingest.py` — Orchestrates full pipeline: classify → extract per-page → postprocess → chunk → embed → upsert into Qdrant. Stores metadata in `file_history` table and triggers duplicate/stale detection.
- `ingestion/page_classifier.py` — Classifies each PDF page as `DIGITAL`, `SCAN`, or `MIXED` based on character density and image-to-page-area ratio
- `ingestion/page_processor.py` — Routes each page to the appropriate extractor: PyMuPDF4LLM for digital, Vision API for scans, hybrid for mixed
- `ingestion/post_processor.py` — Cleans per-page text output: merges multi-page tables, deduplicates repeated table headers, normalizes look-alike Latin→Cyrillic chars, inserts `<!-- page: N -->` markers
- `ingestion/vision_processor.py` — OCRs scan/mixed pages by rendering to PNG (300 DPI) and calling Gemini Vision via OpenRouter; falls back to `vision_model_fallback` on bad output

**`duplicate_detection/`** — Quality control for uploaded documents
- `db.py` — DB init and rotating file logger setup (called once on startup)
- `detector.py` — Compares new document chunks against existing via cosine similarity; LLM adjudicates ambiguous cases (0.75–0.90 similarity range); emits DUPLICATE or STALE warnings
- `repository.py` — CRUD for `warnings` table (SQLite)
- `notifier.py` — Sends Telegram alerts to admin when warnings are found
- `report_generator.py` — Generates PDF report of all unresolved warnings (fpdf2)

**`config.py`** — Single `Settings` class (pydantic-settings) loaded from `.env`. All modules import `settings` from here.

### Data Storage

| Store | Purpose |
|-------|---------|
| Qdrant (`qdrant_data/`) | Vector index of document chunks (cosine, 1024-dim) |
| SQLite (`data/bot.db`) | Users, query logs, FAQs, document history, duplicate warnings |
| `pdfs/` | Raw uploaded files, served by Nginx on port 8080 for citation links |

### LLM / Embedding Providers

Controlled by `LLM_PROVIDER` env var:
- `groq` → `llama-3.3-70b-versatile` (default)
- `openrouter` → configurable model (e.g., `gpt-oss-120b`, `google/gemini-2.0-flash`)

Embeddings always go through OpenRouter (`EMBEDDING_MODEL` env var).

Vision OCR (for scanned PDF pages) uses OpenRouter with:
- `VISION_MODEL_PRIMARY` (default: `google/gemini-2.0-flash-001`)
- `VISION_MODEL_FALLBACK` (default: `google/gemini-2.5-flash-preview`)

### Admin vs User Access

Admin is identified by `ADMIN_TELEGRAM_ID` in `.env`. Admin-only handlers check `message.from_user.id == settings.admin_telegram_id`. Regular users must complete email verification before any interaction.

Admin keyboard menu buttons map to commands: `📋 Документы` → `/list`, `⚠️ Предупреждения` → `/warnings`, `🩺 Состояние` → `/health`, `📊 Статистика` → `/stats`, `📄 Отчёт` → `/report`, `📤 Загрузить` → `/upload`.

### Adding a New Handler

1. Create/edit a file in `bot/handlers/`
2. Define an `aiogram.Router`
3. Register it in `bot/main.py` with `dp.include_router(...)`
