import pytest

from duplicate_detection import report_generator as rg
from duplicate_detection import repository as dd_repo

try:
    rg._find_fonts()
    _FONTS_AVAILABLE = True
except FileNotFoundError:
    _FONTS_AVAILABLE = False

# generate_report()/generate_warning_report() render real PDFs via FPDF with a
# Cyrillic-capable TTF font. On Windows this is arial.ttf; on Linux CI it
# requires `apt-get install -y fonts-dejavu-core`. Skip the smoke tests where
# no such font is present instead of failing the whole suite.
requires_fonts = pytest.mark.skipif(
    not _FONTS_AVAILABLE,
    reason="No Cyrillic-capable font found (needs Windows or fonts-dejavu-core on Linux CI)",
)


# ---------------------------------------------------------------------------
# _fmt_date
# ---------------------------------------------------------------------------

class TestFmtDate:
    def test_valid_iso_with_z_suffix(self):
        assert rg._fmt_date("2024-01-15T10:30:00Z") == "15.01.2024, 10:30 UTC"

    def test_valid_iso_with_offset(self):
        assert rg._fmt_date("2024-01-15T10:30:00+05:00") == "15.01.2024, 10:30 UTC"

    def test_invalid_string_returned_as_is(self):
        assert rg._fmt_date("not-a-date") == "not-a-date"

    def test_empty_string_returned_as_is(self):
        assert rg._fmt_date("") == ""


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_shorter_than_max_unchanged(self):
        assert rg._truncate("short text", max_chars=500) == "short text"

    def test_longer_than_max_gets_ellipsis(self):
        text = "a" * 600
        result = rg._truncate(text, max_chars=500)
        assert result == "a" * 500 + "…"

    def test_exact_length_no_ellipsis(self):
        text = "a" * 500
        assert rg._truncate(text, max_chars=500) == text

    def test_strips_surrounding_whitespace(self):
        assert rg._truncate("   padded text   ", max_chars=500) == "padded text"


# ---------------------------------------------------------------------------
# _similarity_bar
# ---------------------------------------------------------------------------

class TestSimilarityBar:
    def test_zero_similarity_all_empty_blocks(self):
        bar = rg._similarity_bar(0.0, width=16)
        assert bar == "░" * 16 + "  0%"

    def test_full_similarity_all_filled_blocks(self):
        bar = rg._similarity_bar(1.0, width=16)
        assert bar == "█" * 16 + "  100%"

    def test_partial_similarity_mixes_blocks(self):
        bar = rg._similarity_bar(0.5, width=16)
        assert bar.count("█") == 8
        assert bar.count("░") == 8
        assert "50%" in bar

    def test_default_width_is_16(self):
        bar = rg._similarity_bar(1.0)
        assert bar.count("█") == 16


# ---------------------------------------------------------------------------
# generate_report() / generate_warning_report() — smoke tests with real FPDF
# ---------------------------------------------------------------------------

@requires_fonts
class TestGenerateReport:
    async def test_generate_report_with_warnings_returns_pdf_bytes(self, tmp_sqlite_db):
        await dd_repo.create_warning(
            "DUPLICATE", "new.pdf", "existing.pdf", "new chunk text", "existing chunk text", 0.95
        )
        await dd_repo.create_warning(
            "STALE", "new2.pdf", "existing2.pdf", "new chunk 2", "existing chunk 2", 0.8, "reason text"
        )

        pdf_bytes = await rg.generate_report()

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

    async def test_generate_report_with_no_warnings_returns_pdf_bytes(self, tmp_sqlite_db):
        pdf_bytes = await rg.generate_report()

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

    async def test_generate_warning_report_returns_pdf_bytes(self, tmp_sqlite_db):
        wid = await dd_repo.create_warning(
            "STALE", "new.pdf", "existing.pdf", "new chunk text", "existing chunk text", 0.8, "reason text"
        )

        pdf_bytes = await rg.generate_warning_report(wid)

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

    async def test_generate_warning_report_missing_id_raises_value_error(self, tmp_sqlite_db):
        with pytest.raises(ValueError):
            await rg.generate_warning_report(9999)
