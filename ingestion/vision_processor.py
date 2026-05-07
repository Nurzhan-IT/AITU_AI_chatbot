import base64
import re

import fitz
import openai

from config import settings

VISION_PROMPT = """Документ на русском или английском языке.
Извлеки весь текст точно как написано.
Таблицы конвертируй в markdown (| col1 | col2 |).
Сохрани структуру и порядок элементов.
Не добавляй ничего от себя. Только содержимое документа."""


def _render_page_to_base64(page: fitz.Page, dpi: int = settings.page_render_dpi) -> str:
    pix = page.get_pixmap(dpi=dpi)
    return base64.b64encode(pix.tobytes("png")).decode()


def _call_gemini(base64_image: str, model: str) -> str:
    client = openai.OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def _is_bad_result(text: str) -> bool:
    if not text or len(text) < 20:
        return True
    lines = text.splitlines()
    has_pipe = any("|" in line for line in lines)
    has_separator = any(re.search(r"\|[\s:]*-+[\s:|-]*\|", line) for line in lines)
    return has_pipe and not has_separator


def extract_text_via_vision(page: fitz.Page) -> str:
    base64_image = _render_page_to_base64(page)
    text = _call_gemini(base64_image, settings.vision_model_primary)
    if _is_bad_result(text):
        text = _call_gemini(base64_image, settings.vision_model_fallback)
    return text if not _is_bad_result(text) else ""


def extract_images_from_page(page: fitz.Page, doc: fitz.Document) -> list[str]:
    results = []
    for img in page.get_images(full=True):
        xref = img[0]
        image_bytes = doc.extract_image(xref)["image"]
        if len(image_bytes) < settings.vision_min_image_bytes:
            continue
        base64_image = base64.b64encode(image_bytes).decode()
        description = _call_gemini(base64_image, settings.vision_model_primary)
        results.append(description)
    return results
