import re
import sys

import fitz
import pymupdf4llm

from ingestion.ingest import parse_pdf
from ingestion.page_classifier import PageType, classify_document
from ingestion.page_processor import process_page


def _image_ratio(page: fitz.Page) -> float:
    page_area = page.rect.get_area()
    if page_area == 0:
        return 0.0
    image_area = 0.0
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            image_area += rect.get_area()
    return image_area / page_area


def main(pdf_path: str) -> None:
    # --- Classification table ---
    print(f"\n=== Page Classification: {pdf_path} ===\n")
    classifications = classify_document(pdf_path)
    doc = fitz.open(pdf_path)

    print(f"{'Страница':>9} | {'Тип':>7} | {'Символов':>9} | {'Image ratio':>11}")
    print("-" * 47)
    for page_num, page_type in classifications:
        page = doc[page_num]
        chars = len(page.get_text("text").strip())
        ratio = _image_ratio(page)
        print(f"{page_num + 1:>9} | {page_type.value:>7} | {chars:>9} | {ratio:>11.3f}")

    # --- First 3 pages of each type ---
    print("\n=== Page Processing Samples ===")
    seen: dict[PageType, int] = {PageType.DIGITAL: 0, PageType.SCAN: 0, PageType.MIXED: 0}
    sample_pages = []
    for page_num, page_type in classifications:
        if seen[page_type] >= 3:
            continue
        seen[page_type] += 1
        sample_pages.append((page_num, page_type))

    non_scan_sample = [pn for pn, pt in sample_pages if pt != PageType.SCAN]
    if non_scan_sample:
        sample_md = pymupdf4llm.to_markdown(pdf_path, pages=non_scan_sample)
        sample_sections = re.split(r'\n-----\n', sample_md)
        while len(sample_sections) < len(non_scan_sample):
            sample_sections.append("")
        sample_pre_md: dict[int, str] = dict(zip(non_scan_sample, sample_sections))
    else:
        sample_pre_md = {}

    for page_num, page_type in sample_pages:
        page = doc[page_num]
        print(f"\n--- Page {page_num + 1} ({page_type.value}) ---")
        result = process_page(page, page_type, sample_pre_md.get(page_num, ""), doc)
        print(result[:500])

    doc.close()

    # --- Full pipeline ---
    print("\n=== Full Pipeline Output ===\n")
    full_markdown = parse_pdf(pdf_path)
    marker_count = len(re.findall(r'<!-- page: \d+ -->', full_markdown))
    print(f"Общее количество символов : {len(full_markdown)}")
    print(f"Количество page-маркеров  : {marker_count}")
    print(f"\nПервые 1000 символов:\n{full_markdown[:1000]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.test_vision_pipeline path/to/test.pdf")
        sys.exit(1)
    main(sys.argv[1])
