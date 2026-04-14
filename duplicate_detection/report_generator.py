"""Generate a PDF report of duplicate/stale detection warnings (in Russian)."""
import difflib
import logging
import os
import platform
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from config import TZ_UTC5
from duplicate_detection import repository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FONT = "Report"
_MARGIN = 15
_A4_W = 210
_CONTENT_W = _A4_W - 2 * _MARGIN

# Colors (R, G, B)
_C_DARK = (33, 37, 41)
_C_RED = (220, 53, 69)
_C_ORANGE = (253, 126, 20)
_C_GREY_LIGHT = (248, 249, 250)
_C_GREY_MED = (236, 240, 241)
_C_YELLOW_LIGHT = (255, 243, 205)
_C_MUTED = (108, 117, 125)
_C_WHITE = (255, 255, 255)


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def _find_fonts() -> tuple[str, str]:
    """Return (regular_ttf, bold_ttf) paths for a Cyrillic-capable font.

    Docker/Ubuntu note: run `apt-get install -y fonts-dejavu-core` to ensure
    DejaVu fonts are available.
    """
    system = platform.system()
    if system == "Windows":
        candidates = [
            (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
            (r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf"),
        ]
    else:
        candidates = [
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ),
            (
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            ),
            (
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ),
        ]

    for reg, bold in candidates:
        if os.path.exists(reg):
            return reg, bold if os.path.exists(bold) else reg

    raise FileNotFoundError(
        "Не найден шрифт с поддержкой кириллицы. "
        "На Ubuntu/Docker выполните: apt-get install -y fonts-dejavu-core"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_date(iso: str) -> str:
    """Convert ISO-8601 UTC string to human-readable Russian format."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%d.%m.%Y, %H:%M UTC")
    except Exception:
        return iso


def _truncate(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    return (text[:max_chars] + "…") if len(text) > max_chars else text


def _similarity_bar(similarity: float, width: int = 16) -> str:
    """Return a Unicode block bar like '████████████░░░░  94%'."""
    filled = round(similarity * width)
    return "█" * filled + "░" * (width - filled) + f"  {similarity:.0%}"


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    def __init__(self, font_reg: str, font_bold: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.add_font(_FONT, "", font_reg)
        self.add_font(_FONT, "B", font_bold)
        self.set_margins(_MARGIN, _MARGIN, _MARGIN)
        self.set_auto_page_break(auto=True, margin=20)
        self._is_cover = True  # suppresses header/footer on cover page

    # ------------------------------------------------------------------
    # Header / Footer (called automatically by FPDF on every page)
    # ------------------------------------------------------------------

    def header(self) -> None:
        if self._is_cover:
            return
        self.set_font(_FONT, size=8)
        self.set_text_color(*_C_MUTED)
        self.cell(
            0, 5, "AITU Chatbot — Мониторинг документов",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_draw_color(*_C_GREY_MED)
        self.line(_MARGIN, self.get_y(), _A4_W - _MARGIN, self.get_y())
        self.ln(2)
        self.set_text_color(*_C_DARK)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font(_FONT, size=8)
        self.set_text_color(*_C_MUTED)
        self.cell(0, 6, f"Страница {self.page_no()}", align="C")

    # ------------------------------------------------------------------
    # Cover / summary page
    # ------------------------------------------------------------------

    def cover_page(self, warnings: list[dict], generated_at: str) -> None:
        self.add_page()
        self._is_cover = True  # header() skips on this page

        # Dark header band
        self.set_fill_color(*_C_DARK)
        self.rect(0, 0, _A4_W, 58, "F")

        # Main title
        self.set_xy(0, 14)
        self.set_font(_FONT, "B", 19)
        self.set_text_color(*_C_WHITE)
        self.cell(
            _A4_W, 11,
            "ОТЧЁТ О ДУБЛИРОВАНИИ И УСТАРЕВШИХ ДАННЫХ",
            align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        # Subtitle
        self.set_x(0)
        self.set_font(_FONT, size=10)
        self.set_text_color(180, 190, 200)
        self.cell(
            _A4_W, 8,
            "AITU Chatbot — Система мониторинга знаний",
            align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        # Date line (below band)
        self.set_xy(_MARGIN, 68)
        self.set_font(_FONT, size=10)
        self.set_text_color(*_C_MUTED)
        self.cell(
            _CONTENT_W, 7,
            f"Дата формирования: {generated_at}",
            align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(6)

        # Summary stats table
        n_dup = sum(1 for w in warnings if w["warning_type"] == "DUPLICATE")
        n_stale = sum(1 for w in warnings if w["warning_type"] == "STALE")
        self._draw_summary(len(warnings), n_dup, n_stale)

        # Allow header/footer on all subsequent pages
        self._is_cover = False

    def _draw_summary(self, total: int, n_dup: int, n_stale: int) -> None:
        col_l = 110
        col_r = _CONTENT_W - col_l

        # Table header row
        self.set_fill_color(*_C_DARK)
        self.set_text_color(*_C_WHITE)
        self.set_font(_FONT, "B", 11)
        self.cell(
            col_l, 9, "  Сводка предупреждений",
            fill=True, border=0,
            new_x=XPos.RIGHT, new_y=YPos.TOP,
        )
        self.cell(col_r, 9, "", fill=True, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        data = [
            ("Нерешённых предупреждений (в отчёте)", str(total)),
            ("  в т.ч. Дубликаты (DUPLICATE)", str(n_dup)),
            ("  в т.ч. Устаревшие данные (STALE)", str(n_stale)),
        ]
        for i, (label, val) in enumerate(data):
            fill = i % 2 == 1
            self.set_fill_color(*_C_GREY_MED)
            self.set_text_color(*_C_DARK)
            self.set_font(_FONT, size=10)
            self.cell(
                col_l, 8, f"  {label}",
                border="B", fill=fill, align="L",
                new_x=XPos.RIGHT, new_y=YPos.TOP,
            )
            self.set_font(_FONT, "B", 10)
            self.cell(
                col_r, 8, f"  {val}",
                border="B", fill=fill, align="L",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

        if total == 0:
            self.ln(6)
            self.set_font(_FONT, size=10)
            self.set_text_color(*_C_MUTED)
            self.cell(
                _CONTENT_W, 8,
                "Нерешённых предупреждений не найдено.",
                align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

    # ------------------------------------------------------------------
    # Per-warning section
    # ------------------------------------------------------------------

    def warning_section(self, w: dict, idx: int) -> None:
        """Render one warning: colored header, meta table, reason box, chunk excerpts."""
        if idx > 0:
            # Horizontal separator between warnings
            self.set_draw_color(*_C_GREY_MED)
            self.line(_MARGIN, self.get_y() + 4, _A4_W - _MARGIN, self.get_y() + 4)
            self.ln(10)

        wtype = w["warning_type"]
        color = _C_RED if wtype == "DUPLICATE" else _C_ORANGE
        label = "ДУБЛИКАТ" if wtype == "DUPLICATE" else "УСТАРЕВШИЕ ДАННЫЕ"

        # Colored header bar
        self.set_fill_color(*color)
        self.set_text_color(*_C_WHITE)
        self.set_font(_FONT, "B", 12)
        self.cell(
            _CONTENT_W, 10,
            f"  \u26a0 ПРЕДУПРЕЖДЕНИЕ #{w['id']} — {label}",
            fill=True, align="L",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(3)

        # Meta info table
        type_desc = (
            "ДУБЛИКАТ — идентичное содержимое"
            if wtype == "DUPLICATE"
            else "УСТАРЕВШИЕ ДАННЫЕ — заменяет существующий документ"
        )
        status = "Решено ✓" if w.get("resolved") else "Нерешённое"
        meta_rows = [
            ("Тип", type_desc),
            ("Новый документ", w["new_filename"]),
            ("Совпадает с", w["existing_filename"]),
            ("Схожесть", _similarity_bar(w["similarity"])),
            ("Дата обнаружения", _fmt_date(w["created_at"])),
            ("Статус", status),
        ]
        self._meta_table(meta_rows, sim_color=color)

        # LLM reason box (STALE only)
        if wtype == "STALE" and w.get("llm_reason"):
            self.ln(3)
            self._reason_box(w["llm_reason"])

        # Chunk excerpts side-by-side (vertically stacked)
        self.ln(4)
        self._chunk_box(
            f"Фрагмент нового документа  [{w['new_filename']}]",
            w.get("new_chunk_text", ""),
        )
        self.ln(3)
        self._chunk_box(
            f"Совпадающий фрагмент  [{w['existing_filename']}]",
            w.get("existing_chunk_text", ""),
        )
        self.ln(2)

    def _meta_table(self, rows: list[tuple[str, str]], sim_color: tuple) -> None:
        col_l = 60
        col_r = _CONTENT_W - col_l
        for i, (label, val) in enumerate(rows):
            fill = i % 2 == 1
            self.set_fill_color(*_C_GREY_MED)
            self.set_font(_FONT, size=9)
            self.set_text_color(*_C_MUTED)
            self.cell(
                col_l, 7, f"  {label}",
                border="B", fill=fill,
                new_x=XPos.RIGHT, new_y=YPos.TOP,
            )
            # Similarity row gets colored + bold text
            if label == "Схожесть":
                self.set_text_color(*sim_color)
                self.set_font(_FONT, "B", 9)
            else:
                self.set_text_color(*_C_DARK)
                self.set_font(_FONT, size=9)
            self.cell(
                col_r, 7, f"  {val}",
                border="B", fill=fill,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

    def _reason_box(self, reason: str) -> None:
        self.set_font(_FONT, "B", 9)
        self.set_text_color(*_C_DARK)
        self.cell(
            _CONTENT_W, 7, "  Причина от ИИ:",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_font(_FONT, size=9)
        self.set_fill_color(*_C_YELLOW_LIGHT)
        self.set_text_color(*_C_DARK)
        self.multi_cell(
            _CONTENT_W, 6,
            f"  {reason.strip()}",
            border=1, fill=True, align="L",
        )

    def _chunk_box(self, label: str, text: str) -> None:
        self.set_font(_FONT, "B", 9)
        self.set_text_color(*_C_DARK)
        self.cell(
            _CONTENT_W, 7, label,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_font(_FONT, size=8.5)
        self.set_fill_color(*_C_GREY_LIGHT)
        self.set_text_color(*_C_DARK)
        self.multi_cell(
            _CONTENT_W, 5.5,
            f"  {_truncate(text, 500)}",
            border=1, fill=True, align="L",
        )

    def _chunk_box_full(self, label: str, text: str) -> None:
        self.set_font(_FONT, "B", 9)
        self.set_text_color(*_C_DARK)
        self.cell(
            _CONTENT_W, 7, label,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_font(_FONT, size=8.5)
        self.set_fill_color(*_C_GREY_LIGHT)
        self.set_text_color(*_C_DARK)
        self.multi_cell(
            _CONTENT_W, 5.5,
            f"  {(text or '').strip()}",
            border=1, fill=True, align="L",
        )

    def _diff_section(self, new_text: str, existing_text: str) -> None:
        self.set_font(_FONT, "B", 10)
        self.set_text_color(*_C_DARK)
        self.cell(
            _CONTENT_W, 7, "Сравнение текстов (различия выделены)",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(2)
        new_lines = (new_text or "").splitlines()
        existing_lines = (existing_text or "").splitlines()
        diff_lines = list(difflib.ndiff(existing_lines, new_lines))
        self.set_font(_FONT, size=7.5)
        for line in diff_lines:
            if line.startswith("? "):
                continue
            if line.startswith("+ "):
                self.set_text_color(*_C_RED)
            elif line.startswith("- "):
                self.set_text_color(52, 101, 164)   # blue
            else:
                self.set_text_color(*_C_MUTED)
            self.multi_cell(_CONTENT_W, 5, f"  {line}", border=0, align="L")
        self.set_text_color(*_C_DARK)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_report() -> bytes:
    """Fetch all unresolved warnings from SQLite and build a PDF report.

    Returns raw PDF bytes. Raises FileNotFoundError if no Cyrillic font is
    found on the system (install fonts-dejavu-core on Ubuntu/Docker).
    """
    warnings, _ = await repository.get_warnings_for_report()
    font_reg, font_bold = _find_fonts()
    generated_at = datetime.now(TZ_UTC5).strftime("%d.%m.%Y, %H:%M UTC+5")

    pdf = _ReportPDF(font_reg, font_bold)
    pdf.cover_page(warnings, generated_at)

    if warnings:
        pdf.add_page()  # _is_cover=False at this point → header/footer active
        for idx, w in enumerate(warnings):
            pdf.warning_section(w, idx)

    return bytes(pdf.output())


async def generate_warning_report(warning_id: int) -> bytes:
    """Generate a detailed single-warning PDF report.

    Returns raw PDF bytes.
    Raises ValueError if the warning is not found.
    """
    w = await repository.get_warning_by_id(warning_id)
    if w is None:
        raise ValueError(f"Warning #{warning_id} not found")

    new_history = await repository.get_file_history(w["new_filename"], limit=5)
    existing_history = await repository.get_file_history(w["existing_filename"], limit=5)

    font_reg, font_bold = _find_fonts()
    pdf = _ReportPDF(font_reg, font_bold)
    pdf._is_cover = False
    pdf.add_page()

    wtype = w["warning_type"]
    color = _C_RED if wtype == "DUPLICATE" else _C_ORANGE
    label = "ДУБЛИКАТ" if wtype == "DUPLICATE" else "УСТАРЕВШИЕ ДАННЫЕ"

    # ------------------------------------------------------------------
    # Section 1: Header banner
    # ------------------------------------------------------------------
    resolved = w.get("resolved")
    resolved_at = w.get("resolved_at")
    status_str = (
        (f"Решено ({_fmt_date(resolved_at)})" if resolved_at else "Решено")
        if resolved else "Нерешённое"
    )
    pdf.set_fill_color(*color)
    pdf.set_text_color(*_C_WHITE)
    pdf.set_font(_FONT, "B", 14)
    pdf.cell(
        _CONTENT_W, 12,
        f"  \u26a0 {label}  —  Предупреждение #{warning_id}",
        fill=True, align="L",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_font(_FONT, size=10)
    pdf.cell(
        _CONTENT_W, 8,
        f"  Статус: {status_str}",
        fill=True, align="L",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.ln(5)

    # ------------------------------------------------------------------
    # Section 2: Metadata table
    # ------------------------------------------------------------------
    type_desc = (
        "ДУБЛИКАТ — идентичное содержимое"
        if wtype == "DUPLICATE"
        else "УСТАРЕВШИЕ ДАННЫЕ — заменяет существующий документ"
    )
    pdf._meta_table(
        [
            ("Тип предупреждения", type_desc),
            ("Новый документ", w["new_filename"]),
            ("Существующий документ", w["existing_filename"]),
            ("Степень схожести", _similarity_bar(w["similarity"])),
            ("Дата обнаружения", _fmt_date(w["created_at"])),
        ],
        sim_color=color,
    )
    pdf.ln(5)

    # ------------------------------------------------------------------
    # Section 3: Timeline
    # ------------------------------------------------------------------
    def _find_upload(history: list[dict]) -> str | None:
        for ev in reversed(history):
            if ev.get("event") == "uploaded":
                return ev.get("timestamp")
        return None

    existing_upload = _find_upload(existing_history)
    new_upload = _find_upload(new_history)

    if existing_upload and new_upload:
        try:
            d1 = datetime.fromisoformat(existing_upload.replace("Z", "+00:00"))
            d2 = datetime.fromisoformat(new_upload.replace("Z", "+00:00"))
            delta_str = f"{abs((d2 - d1).days)} дн."
        except Exception:
            delta_str = "—"
    else:
        delta_str = "—"

    pdf.set_font(_FONT, "B", 10)
    pdf.set_text_color(*_C_DARK)
    pdf.cell(_CONTENT_W, 7, "Временная шкала", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf._meta_table(
        [
            ("Загрузка существующего документа", _fmt_date(existing_upload) if existing_upload else "неизвестно"),
            ("Загрузка нового документа", _fmt_date(new_upload) if new_upload else "неизвестно"),
            ("Разница во времени", delta_str),
        ],
        sim_color=color,
    )
    pdf.ln(5)

    # ------------------------------------------------------------------
    # Section 4: AI reason (STALE only)
    # ------------------------------------------------------------------
    if wtype == "STALE" and w.get("llm_reason"):
        pdf._reason_box(w["llm_reason"])
        pdf.ln(5)

    # ------------------------------------------------------------------
    # Section 5: Visual diff
    # ------------------------------------------------------------------
    pdf._diff_section(w.get("new_chunk_text", ""), w.get("existing_chunk_text", ""))
    pdf.ln(3)

    # ------------------------------------------------------------------
    # Section 6: Full chunk texts
    # ------------------------------------------------------------------
    pdf._chunk_box_full(
        f"Полный текст нового фрагмента  [{w['new_filename']}]",
        w.get("new_chunk_text", ""),
    )
    pdf.ln(3)
    pdf._chunk_box_full(
        f"Полный текст существующего фрагмента  [{w['existing_filename']}]",
        w.get("existing_chunk_text", ""),
    )
    pdf.ln(5)

    # ------------------------------------------------------------------
    # Section 7: Document history
    # ------------------------------------------------------------------
    pdf.set_font(_FONT, "B", 10)
    pdf.set_text_color(*_C_DARK)
    pdf.cell(_CONTENT_W, 7, "История документов", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    _EV_ICON = {"uploaded": "\u2b06", "deleted": "\u2715"}
    for hist_label, history in [
        (f"История: {w['new_filename']}", new_history),
        (f"История: {w['existing_filename']}", existing_history),
    ]:
        pdf.set_font(_FONT, "B", 9)
        pdf.set_text_color(*_C_DARK)
        pdf.cell(_CONTENT_W, 6, hist_label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if history:
            for ev in history:
                icon = _EV_ICON.get(ev.get("event", ""), "\u2022")
                ev_date = _fmt_date(ev.get("timestamp", ""))
                chunks = ev.get("chunk_count", 0)
                ev_type = ev.get("event", "")
                pdf.set_font(_FONT, size=8.5)
                pdf.set_fill_color(*_C_GREY_LIGHT)
                pdf.set_text_color(*_C_DARK)
                pdf.cell(
                    _CONTENT_W, 6,
                    f"  {icon}  {ev_date}  |  {ev_type}  |  чанков: {chunks}",
                    border="B", fill=True,
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                )
        else:
            pdf.set_font(_FONT, size=8.5)
            pdf.set_text_color(*_C_MUTED)
            pdf.cell(
                _CONTENT_W, 6, "  История не найдена",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        pdf.ln(3)

    pdf.ln(2)

    # ------------------------------------------------------------------
    # Section 8: Recommended action
    # ------------------------------------------------------------------
    rec_color = _C_RED if wtype == "DUPLICATE" else _C_ORANGE
    rec_text = (
        "Рекомендуется удалить новый документ как дубликат"
        if wtype == "DUPLICATE"
        else "Рекомендуется заменить существующий документ новым"
    )
    pdf.set_fill_color(*rec_color)
    pdf.set_text_color(*_C_WHITE)
    pdf.set_font(_FONT, "B", 11)
    pdf.cell(
        _CONTENT_W, 10, f"  {rec_text}",
        fill=True, align="L",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_fill_color(*_C_GREY_LIGHT)
    pdf.set_text_color(*_C_DARK)
    pdf.set_font(_FONT, size=9)
    pdf.multi_cell(
        _CONTENT_W, 6,
        f"  /resolve {warning_id} — закрыть предупреждение\n"
        f"  /delete {w['new_filename']} — удалить документ",
        border=1, fill=True, align="L",
    )

    return bytes(pdf.output())
