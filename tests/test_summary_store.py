from rag.dialog.summary_store import (
    delete_doc_summary,
    get_summaries_for_docs,
    upsert_doc_summary,
)


class TestSummaryStore:
    async def test_upsert_then_get(self, tmp_sqlite_db):
        await upsert_doc_summary("a.pdf", "Doc A", "Summary of doc A")

        result = await get_summaries_for_docs(["Doc A"])

        assert result == {"Doc A": "Summary of doc A"}

    async def test_upsert_overwrites_existing_record_by_filename(self, tmp_sqlite_db):
        await upsert_doc_summary("a.pdf", "Doc A", "old summary")
        await upsert_doc_summary("a.pdf", "Doc A v2", "new summary")

        result = await get_summaries_for_docs(["Doc A v2"])

        assert result == {"Doc A v2": "new summary"}
        # The old doc_title no longer resolves -- it was replaced, not duplicated.
        assert await get_summaries_for_docs(["Doc A"]) == {}

    async def test_get_summaries_for_nonexistent_docs_returns_empty_dict(self, tmp_sqlite_db):
        result = await get_summaries_for_docs(["Nonexistent Doc"])
        assert result == {}

    async def test_get_summaries_for_empty_list_returns_empty_dict(self, tmp_sqlite_db):
        assert await get_summaries_for_docs([]) == {}

    async def test_get_summaries_filters_out_requested_but_missing_titles(self, tmp_sqlite_db):
        await upsert_doc_summary("a.pdf", "Doc A", "Summary A")

        result = await get_summaries_for_docs(["Doc A", "Doc B"])

        assert result == {"Doc A": "Summary A"}

    async def test_delete_doc_summary_removes_record(self, tmp_sqlite_db):
        await upsert_doc_summary("a.pdf", "Doc A", "Summary A")
        await delete_doc_summary("a.pdf")

        result = await get_summaries_for_docs(["Doc A"])

        assert result == {}

    async def test_delete_nonexistent_filename_is_a_noop(self, tmp_sqlite_db):
        await delete_doc_summary("does-not-exist.pdf")  # must not raise

    async def test_multiple_docs_round_trip(self, tmp_sqlite_db):
        await upsert_doc_summary("a.pdf", "Doc A", "Summary A")
        await upsert_doc_summary("b.pdf", "Doc B", "Summary B")

        result = await get_summaries_for_docs(["Doc A", "Doc B"])

        assert result == {"Doc A": "Summary A", "Doc B": "Summary B"}
