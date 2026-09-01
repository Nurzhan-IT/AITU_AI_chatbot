import duplicate_detection.db as dd_db


class TestInitDbIdempotent:
    async def test_double_init_on_same_path_does_not_raise(self, tmp_path):
        db_path = str(tmp_path / "dd_double_init.db")
        log_path = str(tmp_path / "dd_double_init.log")

        await dd_db.init_db(db_path, log_path)
        await dd_db.init_db(db_path, log_path)  # migrations must be idempotent

        assert dd_db._get_path() == db_path
