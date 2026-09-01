import aiosqlite

from bot.auth import repository as auth_repo
from config import settings


# ---------------------------------------------------------------------------
# save_verification_code -> get_pending_verification round-trip
# ---------------------------------------------------------------------------

class TestVerificationRoundTrip:
    async def test_save_then_get_pending(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "a@b.com", "1234")

        pending = await auth_repo.get_pending_verification(1)

        assert pending is not None
        assert pending["email"] == "a@b.com"
        assert pending["verification_code"] == "1234"
        assert pending["verification_attempts"] == 0

    async def test_pending_none_for_user_not_in_table(self, tmp_sqlite_db):
        assert await auth_repo.get_pending_verification(9999) is None

    async def test_pending_none_when_verification_code_is_null(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "a@b.com", "1234")
        await auth_repo.mark_user_verified(1)  # clears verification_code -> NULL

        assert await auth_repo.get_pending_verification(1) is None


# ---------------------------------------------------------------------------
# ON CONFLICT(user_id) DO UPDATE
# ---------------------------------------------------------------------------

class TestSaveVerificationCodeUpsert:
    async def test_repeated_save_updates_email_code_and_resets_attempts(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "old@b.com", "1111")
        await auth_repo.increment_attempts(1)
        await auth_repo.increment_attempts(1)

        await auth_repo.save_verification_code(1, "new@b.com", "2222")
        pending = await auth_repo.get_pending_verification(1)

        assert pending["email"] == "new@b.com"
        assert pending["verification_code"] == "2222"
        assert pending["verification_attempts"] == 0

    async def test_repeated_save_updates_expires_at(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "a@b.com", "1111")
        first = await auth_repo.get_pending_verification(1)

        await auth_repo.save_verification_code(1, "a@b.com", "2222")
        second = await auth_repo.get_pending_verification(1)

        assert second["verification_expires_at"] != first["verification_expires_at"]

    async def test_repeated_save_does_not_create_second_row(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "a@b.com", "1111")
        await auth_repo.save_verification_code(1, "a@b.com", "2222")

        async with aiosqlite.connect(settings.sqlite_db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM users WHERE user_id=?", (1,))
            count = (await cursor.fetchone())[0]
        assert count == 1


# ---------------------------------------------------------------------------
# increment_attempts
# ---------------------------------------------------------------------------

class TestIncrementAttempts:
    async def test_increments_and_returns_new_value(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "a@b.com", "1234")

        assert await auth_repo.increment_attempts(1) == 1
        assert await auth_repo.increment_attempts(1) == 2
        assert await auth_repo.increment_attempts(1) == 3


# ---------------------------------------------------------------------------
# mark_user_verified
# ---------------------------------------------------------------------------

class TestMarkUserVerified:
    async def test_sets_verified_and_clears_verification_fields(self, tmp_sqlite_db):
        await auth_repo.save_verification_code(1, "a@b.com", "1234")
        await auth_repo.increment_attempts(1)
        assert await auth_repo.is_user_verified(1) is False

        await auth_repo.mark_user_verified(1)

        assert await auth_repo.is_user_verified(1) is True
        assert await auth_repo.get_pending_verification(1) is None

    async def test_is_user_verified_false_for_unknown_user(self, tmp_sqlite_db):
        assert await auth_repo.is_user_verified(9999) is False
