import fitz
import pymupdf4llm

from config import settings
from ingestion.page_classifier import PageType
from ingestion.vision_processor import extract_images_from_page, extract_text_via_vision


def _has_table_artifacts(markdown: str) -> bool:
    table_lines = [line for line in markdown.splitlines() if "|" in line]
    if not table_lines:
        return False

    total_cells = 0
    empty_cells = 0
    col_counts: list[int] = []

    for line in table_lines:
        if all(c in "|-: " for c in line):  # separator row
            continue
        cells = line.strip().strip("|").split("|")
        col_counts.append(len(cells))
        total_cells += len(cells)
        empty_cells += sum(1 for c in cells if c.strip() == "")

    if not col_counts or total_cells == 0:
        return False

    if empty_cells / total_cells > 0.30:
        return True
    if max(col_counts) - min(col_counts) > 1:
        return True

    return False


def process_digital_page(page: fitz.Page, pre_markdown: str) -> str:
    if _has_table_artifacts(pre_markdown):
        return extract_text_via_vision(page)
    return pre_markdown


def process_scan_page(page: fitz.Page) -> str:
    return extract_text_via_vision(page)


def process_mixed_page(page: fitz.Page, pre_markdown: str, doc: fitz.Document) -> str:
    text_md = pre_markdown
    image_markdowns = extract_images_from_page(page, doc)

    raw_blocks = page.get_text("blocks")
    images = page.get_images(full=True)

    # (y0, content)
    blocks: list[tuple[float, str]] = []

    for block in raw_blocks:
        if block[6] == 0:  # text block
            blocks.append((block[1], block[4]))

    img_md_index = 0
    for img in images:
        xref = img[0]
        image_bytes = doc.extract_image(xref)["image"]
        if len(image_bytes) < settings.vision_min_image_bytes:
            continue
        if img_md_index >= len(image_markdowns):
            break
        md = image_markdowns[img_md_index]
        img_md_index += 1
        if md:
            bbox = page.get_image_bbox(img)
            blocks.append((bbox.y0, md))

    if not blocks:
        return text_md

    blocks.sort(key=lambda b: b[0])
    return "\n\n".join(content for _, content in blocks if content.strip())


def process_page(page: fitz.Page, page_type: PageType, pre_markdown: str, doc: fitz.Document) -> str:
    if page_type == PageType.SCAN:
        return process_scan_page(page)
    if page_type == PageType.MIXED:
        return process_mixed_page(page, pre_markdown, doc)
    return process_digital_page(page, pre_markdown)
