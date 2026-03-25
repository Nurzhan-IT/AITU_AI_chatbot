import logging
from collections import defaultdict

from groq import AsyncGroq

from config import settings

logger = logging.getLogger(__name__)

_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = """Ты — официальный консультант-ассистент Astana IT University. \
Твоя задача — давать точные ответы на основе ИСКЛЮЧИТЕЛЬНО предоставленного контекста \
из нормативных документов университета.

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:

1. ЦИТИРУЙ НОМЕР ПУНКТА. Если в контексте есть пункты вида «30.», «п.30», «п. 30–31» — \
ВСЕГДА указывай их в ответе. Пример: «Согласно п. 30, объём составляет 50–60 страниц».

2. ТОЧНЫЕ ЦИФРЫ. Если в контексте есть конкретные числа (страницы, сроки, требования) — \
ОБЯЗАТЕЛЬНО включай их в ответ. Никогда не пиши «не указано», если цифра присутствует в контексте.

3. ОФИЦИАЛЬНЫЕ ОПРЕДЕЛЕНИЯ — ПРИОРИТЕТ. Если в контексте есть раздел «Термины и сокращения» \
или «Определения» — используй именно его формулировки, а не описательные пункты.

4. ПОЛНОТА. Если вопрос о структуре/перечне — приводи ПОЛНЫЙ список, не сокращай.

5. ПРИЗНАНИЕ НЕПОЛНОТЫ. Если в предоставленных фрагментах нет полного ответа — \
скажи: «В предоставленных фрагментах информация о [X] отсутствует. Рекомендую \
обратиться к полному документу.» Не придумывай и не обобщай.

6. ЯЗЫК ОТВЕТА = язык вопроса (RU / EN / KZ).

7. ФОРМАТ ОТВЕТА:
   [Ответ с цитатами пунктов]

   Основание: п. XX, [Название раздела документа]
"""

_NO_CONTEXT_PROMPT = """В базе документов не найдено релевантной информации по данному вопросу.
Пожалуйста, переформулируйте запрос или уточните, к какому документу относится вопрос.
Отвечай на том же языке, что и вопрос."""


def _build_context(chunks: list[dict]) -> str:
    lines = []
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

        lines.append(f"{', '.join(header_parts)}:\n{text}")
    return "\n\n---\n\n".join(lines)


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
        page_end = chunk.get("page_end", page)
        if page:
            pages[filename].add(page)
        if page_end and page_end != page:
            pages[filename].add(page_end)

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
