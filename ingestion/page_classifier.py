from enum import Enum

import fitz

from config import settings


class PageType(str, Enum):
    DIGITAL = "DIGITAL"
    SCAN = "SCAN"
    MIXED = "MIXED"


def classify_page(page: fitz.Page) -> PageType:
    text = page.get_text("text")
    char_count = len(text.strip())

    images = page.get_images(full=True)
    page_area = page.rect.get_area()

    image_area = 0.0
    for img in images:
        xref = img[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            image_area += rect.get_area()

    image_ratio = image_area / page_area if page_area > 0 else 0.0

    if char_count < settings.scan_text_threshold and image_ratio > settings.scan_image_ratio:
        return PageType.SCAN

    if char_count >= settings.scan_text_threshold and images:
        for img in images:
            xref = img[0]
            try:
                pix = fitz.Pixmap(page.parent, xref)
                size = len(pix.tobytes())
                pix = None
                if size >= settings.vision_min_image_bytes:
                    return PageType.MIXED
            except Exception:
                continue

    return PageType.DIGITAL


def classify_document(pdf_path: str) -> list[tuple[int, PageType]]:
    doc = fitz.open(pdf_path)
    result = []
    for i, page in enumerate(doc):
        result.append((i, classify_page(page)))
    doc.close()
    return result
