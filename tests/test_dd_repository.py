from duplicate_detection import repository as dd_repo


def _patch_now(monkeypatch, count):
    """Give record_file_event/create_warning strictly increasing timestamps.

    _now_utc() has one-second resolution, so back-to-back calls within a test
    can tie -- DESC ordering across ties is not guaranteed by SQLite. Patching
    it keeps the ordering assertions deterministic.
    """
    timestamps = iter(f"2024-01-01T00:00:{i:02d}+05:00" for i in range(count))
    monkeypatch.setattr(dd_repo, "_now_utc", lambda: next(timestamps))


# ---------------------------------------------------------------------------
# record_file_event / get_file_history
# ---------------------------------------------------------------------------

class TestFileHistory:
    async def test_get_file_history_orders_desc_by_timestamp(self, tmp_sqlite_db, monkeypatch):
        _patch_now(monkeypatch, 2)
        await dd_repo.record_file_event("a.pdf", "Doc A", "uploaded")
        await dd_repo.record_file_event("b.pdf", "Doc B", "uploaded")

        history = await dd_repo.get_file_history()

        assert [h["filename"] for h in history] == ["b.pdf", "a.pdf"]

    async def test_filter_by_filename(self, tmp_sqlite_db, monkeypatch):
        _patch_now(monkeypatch, 3)
        await dd_repo.record_file_event("a.pdf", "Doc A", "uploaded")
        await dd_repo.record_file_event("b.pdf", "Doc B", "uploaded")
        await dd_repo.record_file_event("a.pdf", "Doc A", "deleted")

        history = await dd_repo.get_file_history(filename="a.pdf")

        assert len(history) == 2
        assert all(h["filename"] == "a.pdf" for h in history)

    async def test_limit_caps_result_count(self, tmp_sqlite_db, monkeypatch):
        _patch_now(monkeypatch, 5)
        for i in range(5):
            await dd_repo.record_file_event(f"f{i}.pdf", "Doc", "uploaded")

        history = await dd_repo.get_file_history(limit=2)

        assert len(history) == 2


# ---------------------------------------------------------------------------
# create_warning / get_warning_by_id / resolve_warning
# ---------------------------------------------------------------------------

class TestWarningCrud:
    async def test_create_then_get_by_id(self, tmp_sqlite_db):
        wid = await dd_repo.create_warning(
            "DUPLICATE", "new.pdf", "existing.pdf", "new text", "existing text", 0.95, "reason"
        )

        warning = await dd_repo.get_warning_by_id(wid)

        assert warning is not None
        assert warning["warning_type"] == "DUPLICATE"
        assert warning["new_filename"] == "new.pdf"
        assert warning["resolved"] == 0

    async def test_get_warning_by_id_missing_returns_none(self, tmp_sqlite_db):
        assert await dd_repo.get_warning_by_id(9999) is None

    async def test_resolve_warning_succeeds_once(self, tmp_sqlite_db):
        wid = await dd_repo.create_warning("DUPLICATE", "n.pdf", "e.pdf", "a", "b", 0.9)

        assert await dd_repo.resolve_warning(wid) is True
        warning = await dd_repo.get_warning_by_id(wid)
        assert warning["resolved"] == 1
        assert warning["resolved_at"] is not None

    async def test_resolve_warning_twice_returns_false_second_time(self, tmp_sqlite_db):
        wid = await dd_repo.create_warning("DUPLICATE", "n.pdf", "e.pdf", "a", "b", 0.9)

        assert await dd_repo.resolve_warning(wid) is True
        assert await dd_repo.resolve_warning(wid) is False

    async def test_resolve_nonexistent_warning_returns_false(self, tmp_sqlite_db):
        assert await dd_repo.resolve_warning(9999) is False


# ---------------------------------------------------------------------------
# list_warnings
# ---------------------------------------------------------------------------

class TestListWarnings:
    async def test_pagination_and_total_count(self, tmp_sqlite_db, monkeypatch):
        _patch_now(monkeypatch, 5)
        for i in range(5):
            await dd_repo.create_warning("DUPLICATE", f"n{i}.pdf", f"e{i}.pdf", "a", "b", 0.9)

        page1, total1 = await dd_repo.list_warnings(resolved=False, limit=2, offset=0)
        page2, total2 = await dd_repo.list_warnings(resolved=False, limit=2, offset=2)

        assert total1 == 5
        assert total2 == 5
        assert len(page1) == 2
        assert len(page2) == 2
        assert {w["id"] for w in page1}.isdisjoint({w["id"] for w in page2})

    async def test_resolved_filter_partitions_correctly(self, tmp_sqlite_db):
        wid1 = await dd_repo.create_warning("DUPLICATE", "a.pdf", "b.pdf", "x", "y", 0.9)
        wid2 = await dd_repo.create_warning("DUPLICATE", "c.pdf", "d.pdf", "x", "y", 0.9)
        await dd_repo.resolve_warning(wid1)

        unresolved, total_unresolved = await dd_repo.list_warnings(resolved=False)
        resolved, total_resolved = await dd_repo.list_warnings(resolved=True)

        assert total_unresolved == 1
        assert total_resolved == 1
        assert unresolved[0]["id"] == wid2
        assert resolved[0]["id"] == wid1

    async def test_doc_titles_none_when_file_history_empty(self, tmp_sqlite_db):
        await dd_repo.create_warning("DUPLICATE", "new.pdf", "existing.pdf", "x", "y", 0.9)

        warnings, _ = await dd_repo.list_warnings(resolved=False)

        assert warnings[0]["new_doc_title"] is None
        assert warnings[0]["existing_doc_title"] is None

    async def test_doc_titles_joined_from_file_history(self, tmp_sqlite_db, monkeypatch):
        _patch_now(monkeypatch, 3)
        await dd_repo.record_file_event("new.pdf", "New Doc", "uploaded")
        await dd_repo.record_file_event("existing.pdf", "Existing Doc", "uploaded")
        await dd_repo.create_warning("DUPLICATE", "new.pdf", "existing.pdf", "x", "y", 0.9)

        warnings, _ = await dd_repo.list_warnings(resolved=False)

        assert warnings[0]["new_doc_title"] == "New Doc"
        assert warnings[0]["existing_doc_title"] == "Existing Doc"


# ---------------------------------------------------------------------------
# get_warnings_for_report
# ---------------------------------------------------------------------------

class TestGetWarningsForReport:
    async def test_only_unresolved_returned_with_total(self, tmp_sqlite_db):
        wid1 = await dd_repo.create_warning("DUPLICATE", "a.pdf", "b.pdf", "x", "y", 0.9)
        wid2 = await dd_repo.create_warning("STALE", "c.pdf", "d.pdf", "x", "y", 0.8)
        await dd_repo.resolve_warning(wid1)

        warnings, total = await dd_repo.get_warnings_for_report()

        assert total == 1
        assert warnings[0]["id"] == wid2

    async def test_limit_caps_result_but_not_total(self, tmp_sqlite_db, monkeypatch):
        _patch_now(monkeypatch, 3)
        for i in range(3):
            await dd_repo.create_warning("DUPLICATE", f"n{i}.pdf", f"e{i}.pdf", "x", "y", 0.9)

        warnings, total = await dd_repo.get_warnings_for_report(limit=1)

        assert len(warnings) == 1
        assert total == 3
