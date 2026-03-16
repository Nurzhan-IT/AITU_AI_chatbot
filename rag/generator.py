import logging
from collections import defaultdict

from groq import AsyncGroq

from config import settings

logger = logging.getLogger(__name__)

_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = (
    "Ты — университетский консультант-ассистент. Отвечай ТОЛЬКО на основе "
    "предоставленного контекста из официальных документов университета.\n"
    "Если ответа нет в контексте — честно скажи об этом.\n"
    "Отвечай на том языке, на котором задан вопрос (RU/EN/KZ).\n"
    "Будь точным, лаконичным и вежливым."
)

_NO_CONTEXT_PROMPT = (
    "Контекст не предоставлен. Сообщи пользователю, что в базе документов "
    "не найдено релевантной информации по его вопросу, и предложи переформулировать."
)


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        doc = chunk.get("doc_title", "Документ")
        page = chunk.get("page", 0)
        text = chunk.get("text", "").strip()
        header = f"[{i}] {doc}, стр. {page}" if page else f"[{i}] {doc}"
        lines.append(f"{header}:\n{text}")
    return "\n\n".join(lines)


def _deduplicate_sources(chunks: list[dict]) -> list[dict]:
    """Group chunks by filename → {doc_title, url, pages: sorted list}."""
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
            }
        page = chunk.get("page", 0)
        if page:
            pages[filename].add(page)

    sources = []
    for filename, doc in grouped.items():
        sources.append({**doc, "pages": sorted(pages[filename])})

    return sources


class Generator:
    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def generate(self, question: str, chunks: list[dict]) -> dict:
        """
        Returns:
            {
                "answer": str,
                "sources": [{"doc_title": str, "filename": str, "url": str, "pages": [int, ...]}, ...]
            }
        """
        if not chunks:
            logger.info("generate: no chunks provided, returning no-context answer")
            answer = await self._call_llm(question, context=None)
            return {"answer": answer, "sources": []}

        context = _build_context(chunks)
        answer = await self._call_llm(question, context=context)
        sources = _deduplicate_sources(chunks)

        logger.info(
            "generate: question='%.60s' → %d chunks, %d sources",
            question, len(chunks), len(sources),
        )
        return {"answer": answer, "sources": sources}

    async def _call_llm(self, question: str, context: str | None) -> str:
        if context:
            user_content = f"Контекст:\n{context}\n\nВопрос: {question}"
            system = _SYSTEM_PROMPT
        else:
            user_content = f"Вопрос: {question}"
            system = _NO_CONTEXT_PROMPT

        try:
            response = await self._client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            raise
