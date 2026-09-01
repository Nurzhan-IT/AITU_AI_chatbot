import re

CYRILLIC_MAP: dict[str, str] = {
    "c": "с", "e": "е", "o": "о", "p": "р", "a": "а", "x": "х",
    "C": "С", "E": "Е", "O": "О", "P": "Р", "A": "А", "X": "Х",
}

_LATIN_SET = set(CYRILLIC_MAP.keys())
_CYRILLIC_CHAR_RE = re.compile(r"[а-яёА-ЯЁ]")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _is_table_row(line: str) -> bool:
    if "|" not in line:
        return False
    return not all(c in "|-: " for c in line)


def _is_separator(line: str) -> bool:
    return "|" in line and all(c in "|-: " for c in line)


def _merge_multipage_tables(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    result = list(pages)

    for i in range(len(result) - 1):
        _, text_n = result[i]
        page_n1_num, text_n1 = result[i + 1]

        lines_n = text_n.splitlines()
        lines_n1 = text_n1.splitlines()

        if not lines_n or not lines_n1:
            continue
        if not _is_table_row(lines_n[-1]):
            continue
        if not _is_table_row(lines_n1[0]):
            continue

        # N+1 should not already have a separator near the start
        if any(_is_separator(ln) for ln in lines_n1[:3]):
            continue

        # Find the last separator in page N and the header row above it
        sep_idx = next(
            (j for j in range(len(lines_n) - 1, -1, -1) if _is_separator(lines_n[j])),
            None,
        )
        if sep_idx is None or sep_idx == 0:
            continue

        header_row = lines_n[sep_idx - 1]
        separator_row = lines_n[sep_idx]
        new_text_n1 = header_row + "\n" + separator_row + "\n" + text_n1
        result[i + 1] = (page_n1_num, new_text_n1)

    return result


def _deduplicate_headers(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    def _first_table_row(text: str) -> str | None:
        for line in text.splitlines():
            if _is_table_row(line):
                return line
        return None

    result = list(pages)
    n = len(result)
    i = 0

    while i < n:
        header = _first_table_row(result[i][1])
        if header is None:
            i += 1
            continue

        j = i + 1
        while j < n and _first_table_row(result[j][1]) == header:
            j += 1

        if j - i >= 3:
            for k in range(i + 1, j):
                page_num, text = result[k]
                lines = text.splitlines()
                new_lines: list[str] = []
                removed = False
                skip_sep = False

                for line in lines:
                    if skip_sep:
                        skip_sep = False
                        if _is_separator(line):
                            continue
                    if not removed and line == header:
                        removed = True
                        skip_sep = True
                        continue
                    new_lines.append(line)

                result[k] = (page_num, "\n".join(new_lines))

        i = j if j - i >= 3 else i + 1

    return result


def _normalize_cyrillic(text: str) -> str:
    def _fix_word(match: re.Match) -> str:
        word = match.group()
        latin_lookalikes = [c for c in word if c in _LATIN_SET]
        if not latin_lookalikes:
            return word
        cyrillic_chars = [c for c in word if _CYRILLIC_CHAR_RE.match(c)]
        if not cyrillic_chars or len(cyrillic_chars) <= len(latin_lookalikes):
            return word
        return "".join(CYRILLIC_MAP.get(c, c) for c in word)

    return _WORD_RE.sub(_fix_word, text)


def _add_page_markers(pages: list[tuple[int, str]]) -> str:
    parts = [f"<!-- page: {page_num} -->\n{text}" for page_num, text in pages]
    return "\n\n".join(parts)


def postprocess(pages: list[tuple[int, str]]) -> str:
    pages = _merge_multipage_tables(pages)
    pages = _deduplicate_headers(pages)
    pages = [(n, _normalize_cyrillic(text)) for n, text in pages]
    return _add_page_markers(pages)
