import logging
from collections import defaultdict
from typing import Any

import tiktoken
from langdetect import detect

from config import settings

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return {"ru": "Russian", "en": "English", "kk": "Kazakh"}.get(lang, "Russian")
    except Exception:
        return "Russian"


async def translate_query(question: str, target_lang: str) -> str | None:
    """Translate question to target_lang using the configured LLM. Returns None on failure."""
    try:
        client = _make_llm_client()
        prompt = (
            f"Translate the following question to {target_lang}. "
            f"Return ONLY the translated question, no explanation.\n\n"
            f"Question: {question}"
        )
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


_SYSTEM_PROMPT = """You are a university consultation assistant for AITU (Astana IT University).

Rules:
1. Answer ONLY based on the provided document fragments. Do not use outside knowledge.
2. ALWAYS cite the source using the document title, section, and page number from the fragment
   headers. For English documents use "Section X, p.Y"; for Russian documents use "п.X, стр.Y".
3. Quote exact figures: dates, deadlines, counts, thresholds — do not paraphrase numbers.
4. Use official terms as they appear in the documents.
5. If the question cannot be answered from the provided fragments, write exactly [NO_ANSWER] as
   the very first token of your response, then explain why in the same language as the question.
6. LANGUAGE RULE: The user message will state the detected query language. You MUST respond
   in that exact language (Russian, English, or Kazakh), regardless of the document language.
"""

_NO_CONTEXT_PROMPT = """В базе документов не найдено релевантной информации по данному вопросу.
Пожалуйста, переформулируйте запрос или уточните, к какому документу относится вопрос.
Отвечай на том же языке, что и вопрос."""


MAX_CONTEXT_TOKENS = 6000


def _build_context(chunks: list[dict], max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    parts = []
    used = 0
    for i, chunk in enumerate(chunks, 1):
        doc = chunk.get("doc_title", "Документ")
        page = chunk.get("page", 0)
        page_end = chunk.get("page_end", page)
        section = chunk.get("section_title", "")
        para = chunk.get("paragraph_range", "")
        text = chunk.get("text", "").strip()

        header_parts = [f"[{i}] {doc}"]
        if section:
            header_parts.append(f"Раздел: «{section}»")
        if para:
            header_parts.append(para)
        if page and page_end and page != page_end:
            header_parts.append(f"стр. {page}–{page_end}")
        elif page:
            header_parts.append(f"стр. {page}")

        block = f"{', '.join(header_parts)}:\n{text}\n\n---\n\n"
        block_tokens = len(enc.encode(block))
        if used + block_tokens > max_tokens:
            break
        parts.append(block)
        used += block_tokens
    return "".join(parts)


def _deduplicate_sources(chunks: list[dict]) -> list[dict]:
    """Group chunks by filename → {doc_title, url, uploaded_at, pages: sorted list}."""
    grouped: dict[str, dict] = {}
    pages: dict[str, set[int]] = defaultdict(set)

    for chunk in chunks:
        filename = chunk.get("filename", "")
        if not filename:
            continue
        if filename not in grouped:
            grouped[filename] = {
                "doc_title": chunk.get("doc_title", ""),
                "filename": filename,
                "url": chunk.get("url", ""),
                "uploaded_at": chunk.get("uploaded_at", ""),
            }
        page = chunk.get("page", 0)
        page_end = chunk.get("page_end", page)
        if page:
            pages[filename].add(page)
        if page_end and page_end != page:
            pages[filename].add(page_end)

    sources = []
    for filename, doc in grouped.items():
        sources.append({**doc, "pages": sorted(pages[filename])})

    return sources


def _make_llm_client() -> Any:
    if settings.llm_provider == "groq":
        from groq import AsyncGroq
        return AsyncGroq(api_key=settings.groq_api_key)
    else:
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )


MIN_CHUNK_SCORE = 0.55

_FAQ_SYSTEM_PROMPT = """You are an assistant that creates FAQ entries for a university document.

Given the document content, generate exactly 10 frequently asked questions with clear, detailed answers.
Cover the most important topics a student would ask about.

Respond ONLY with a valid JSON array of exactly 10 objects — no markdown, no explanation:
[
  {"question": "...", "answer": "..."},
  ...
]

Rules:
- Write questions and answers in the same language as the document.
- Base answers strictly on the provided document content.
- Make questions practical and relevant for students.
"""


async def generate_document_faq(chunks: list[dict]) -> list[dict]:
    """Generate 10 FAQ entries from document chunks. Returns list of {question, answer}."""
    import json

    if not chunks:
        return []

    client = _make_llm_client()
    context = _build_context(chunks, max_tokens=8000)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _FAQ_SYSTEM_PROMPT},
            {"role": "user", "content": f"Document content:\n{context}\n\nGenerate 10 FAQ entries."},
        ],
        temperature=0.3,
        max_tokens=3000,
    )
    content = response.choices[0].message.content or ""

    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        logger.error("generate_document_faq: no JSON array in LLM response")
        raise ValueError("LLM did not return valid JSON FAQ array")

    faqs = json.loads(content[start:end])
    return [{"question": str(f["question"]), "answer": str(f["answer"])} for f in faqs[:10]]


class Generator:
    def __init__(self) -> None:
        self._client = _make_llm_client()
        self._provider = settings.llm_provider
        self._model = settings.llm_model

    async def ping(self) -> bool:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            return bool(response.choices)
        except Exception:
            return False

    async def generate(self, question: str, chunks: list[dict]) -> dict:
        """
        Returns:
            {
                "answer": str,
                "detected_lang": str,
                "sources": [{"doc_title": str, "filename": str, "url": str, "pages": [int, ...]}, ...]
            }
        """
        detected_lang = detect_language(question)

        chunks = [c for c in chunks if c.get("score", 0) >= MIN_CHUNK_SCORE]
        if not chunks:
            logger.info(
                "generate: all chunks below score threshold (%.2f), returning no-context answer",
                MIN_CHUNK_SCORE,
            )
            answer = await self._call_llm(question, context=None, detected_lang=detected_lang)
            return {"answer": answer, "detected_lang": detected_lang, "sources": []}

        if not chunks:
            logger.info("generate: no chunks provided, returning no-context answer")
            answer = await self._call_llm(question, context=None, detected_lang=detected_lang)
            return {"answer": answer, "detected_lang": detected_lang, "sources": []}

        _NO_ANSWER_MARKER = "[NO_ANSWER]"

        context = _build_context(chunks)
        answer = await self._call_llm(question, context=context, detected_lang=detected_lang)

        if answer.lstrip().startswith(_NO_ANSWER_MARKER):
            answer = answer.lstrip()[len(_NO_ANSWER_MARKER):].strip()
            sources = []
        else:
            sources = _deduplicate_sources(chunks)

        logger.info(
            "generate: question='%.60s' lang=%s → %d chunks, %d sources",
            question, detected_lang, len(chunks), len(sources),
        )
        return {"answer": answer, "detected_lang": detected_lang, "sources": sources}

    async def _call_llm(self, question: str, context: str | None, detected_lang: str = "Russian") -> str:
        if context:
            user_content = (
                f"Контекст:\n{context}\n\n"
                f"Язык вопроса пользователя: {detected_lang}\n"
                f"Вопрос: {question}"
            )
            system = _SYSTEM_PROMPT
        else:
            user_content = f"Вопрос: {question}"
            system = _NO_CONTEXT_PROMPT

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM API call failed (provider=%s, model=%s): %s", self._provider, self._model, e)
            raise
