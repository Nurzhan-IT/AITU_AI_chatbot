import aiosqlite

from bot import faq_repository as repo


# ---------------------------------------------------------------------------
# add_faq / edit_faq / delete_faq / get_all_faq
# ---------------------------------------------------------------------------

class TestFaqCrud:
    async def test_add_then_get_all_ordered_by_id_asc(self, tmp_sqlite_db):
        id1 = await repo.add_faq("Q1", "A1")
        id2 = await repo.add_faq("Q2", "A2")

        rows = await repo.get_all_faq()

        assert [r["id"] for r in rows] == [id1, id2]
        assert rows[0]["question"] == "Q1"
        assert rows[1]["question"] == "Q2"

    async def test_edit_faq_updates_existing_row(self, tmp_sqlite_db):
        faq_id = await repo.add_faq("Q", "A")

        ok = await repo.edit_faq(faq_id, "Q2", "A2")

        assert ok is True
        rows = await repo.get_all_faq()
        assert rows[0]["question"] == "Q2"
        assert rows[0]["answer"] == "A2"

    async def test_edit_faq_nonexistent_id_returns_false(self, tmp_sqlite_db):
        assert await repo.edit_faq(999, "Q", "A") is False

    async def test_delete_faq_removes_existing_row(self, tmp_sqlite_db):
        faq_id = await repo.add_faq("Q", "A")

        assert await repo.delete_faq(faq_id) is True
        assert await repo.get_all_faq() == []

    async def test_delete_faq_nonexistent_id_returns_false(self, tmp_sqlite_db):
        assert await repo.delete_faq(999) is False


# ---------------------------------------------------------------------------
# get_faq_last_updated
# ---------------------------------------------------------------------------

class TestFaqLastUpdated:
    async def test_none_on_empty_table(self, tmp_sqlite_db):
        assert await repo.get_faq_last_updated() is None

    async def test_max_updated_at_after_several_inserts(self, tmp_sqlite_db):
        await repo.add_faq("Q1", "A1")
        faq_id2 = await repo.add_faq("Q2", "A2")
        await repo.edit_faq(faq_id2, "Q2b", "A2b")

        last = await repo.get_faq_last_updated()
        rows = await repo.get_all_faq()

        assert last == max(r["updated_at"] for r in rows)


# ---------------------------------------------------------------------------
# register_user / is_user_notified / mark_user_notified /
# reset_all_user_notifications
# ---------------------------------------------------------------------------

class TestUserNotifications:
    async def test_register_user_is_idempotent(self, tmp_sqlite_db):
        await repo.register_user(42)
        await repo.register_user(42)  # INSERT OR IGNORE -- must not raise or duplicate

        async with aiosqlite.connect(tmp_sqlite_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM users WHERE user_id=?", (42,))
            count = (await cursor.fetchone())[0]
        assert count == 1

    async def test_is_user_notified_true_for_unregistered_user(self, tmp_sqlite_db):
        assert await repo.is_user_notified(9999) is True

    async def test_registered_user_defaults_to_notified(self, tmp_sqlite_db):
        await repo.register_user(1)
        assert await repo.is_user_notified(1) is True

    async def test_reset_then_mark_notified_round_trip(self, tmp_sqlite_db):
        await repo.register_user(1)

        await repo.reset_all_user_notifications()
        assert await repo.is_user_notified(1) is False

        await repo.mark_user_notified(1)
        assert await repo.is_user_notified(1) is True


# ---------------------------------------------------------------------------
# save_document_faq / get_document_faq / clear_document_faq
# ---------------------------------------------------------------------------

class TestDocumentFaq:
    async def test_save_then_get(self, tmp_sqlite_db):
        faqs = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        await repo.save_document_faq("doc.pdf", faqs)

        rows = await repo.get_document_faq("doc.pdf")
        assert [r["question"] for r in rows] == ["Q1", "Q2"]

    async def test_save_replaces_previous_faqs_not_appends(self, tmp_sqlite_db):
        await repo.save_document_faq("doc.pdf", [{"question": "Old", "answer": "A"}])
        await repo.save_document_faq("doc.pdf", [{"question": "New", "answer": "B"}])

        rows = await repo.get_document_faq("doc.pdf")
        assert len(rows) == 1
        assert rows[0]["question"] == "New"

    async def test_save_does_not_touch_other_filenames(self, tmp_sqlite_db):
        await repo.save_document_faq("a.pdf", [{"question": "QA", "answer": "AA"}])
        await repo.save_document_faq("b.pdf", [{"question": "QB", "answer": "AB"}])

        assert len(await repo.get_document_faq("a.pdf")) == 1
        assert len(await repo.get_document_faq("b.pdf")) == 1

    async def test_clear_document_faq(self, tmp_sqlite_db):
        await repo.save_document_faq("doc.pdf", [{"question": "Q", "answer": "A"}])

        await repo.clear_document_faq("doc.pdf")

        assert await repo.get_document_faq("doc.pdf") == []
