import pytest

from config import Settings, _CommaListMixin


def test_settings_importable():
    from config import settings

    assert settings is not None


# ---------------------------------------------------------------------------
# Settings._parse_admin_ids (field_validator, mode="before")
# ---------------------------------------------------------------------------

class TestParseAdminIds:
    def test_comma_separated_string(self):
        assert Settings._parse_admin_ids("1,2,3") == [1, 2, 3]

    def test_comma_separated_string_with_whitespace(self):
        assert Settings._parse_admin_ids(" 1 , 2 ,3") == [1, 2, 3]

    def test_single_int(self):
        assert Settings._parse_admin_ids(5) == [5]

    def test_already_a_list_passthrough(self):
        assert Settings._parse_admin_ids([1, 2, 3]) == [1, 2, 3]

    def test_empty_string(self):
        assert Settings._parse_admin_ids("") == []

    def test_none(self):
        assert Settings._parse_admin_ids(None) == []

    def test_empty_list(self):
        assert Settings._parse_admin_ids([]) == []


# ---------------------------------------------------------------------------
# Settings.is_admin()
# ---------------------------------------------------------------------------

class TestIsAdmin:
    def test_id_in_list(self):
        s = Settings(ADMIN_TELEGRAM_ID=[111, 222])
        assert s.is_admin(111) is True
        assert s.is_admin(222) is True

    def test_id_not_in_list(self):
        s = Settings(ADMIN_TELEGRAM_ID=[111, 222])
        assert s.is_admin(999) is False

    def test_empty_admin_list(self):
        s = Settings(ADMIN_TELEGRAM_ID=[])
        assert s.is_admin(111) is False


# ---------------------------------------------------------------------------
# _CommaListMixin.decode_complex_value
# ---------------------------------------------------------------------------

class TestCommaListMixinDecodeComplexValue:
    def setup_method(self):
        self.mixin = _CommaListMixin()

    def test_bare_comma_list_gets_wrapped_and_parsed(self):
        assert self.mixin.decode_complex_value("x", None, "1,2,3") == [1, 2, 3]

    def test_json_array_passthrough_unwrapped(self):
        assert self.mixin.decode_complex_value("x", None, "[1,2,3]") == [1, 2, 3]

    def test_json_object_passthrough_unwrapped(self):
        assert self.mixin.decode_complex_value("x", None, '{"a": 1}') == {"a": 1}

    def test_degenerate_empty_string(self):
        # "" doesn't start with "[" or "{", so it's wrapped to "[]" before
        # json.loads — must not raise, must yield an empty list.
        assert self.mixin.decode_complex_value("x", None, "") == []


# ---------------------------------------------------------------------------
# Settings.triage_calibrated
# ---------------------------------------------------------------------------

class TestTriageCalibrated:
    def test_false_when_all_none(self):
        s = Settings(
            triage_specific_gap_ratio=None,
            triage_specific_max_entropy=None,
            triage_ambiguous_min_entropy=None,
            triage_ambiguous_doc_spread=None,
        )
        assert s.triage_calibrated is False

    def test_true_when_all_four_set(self):
        s = Settings(
            triage_specific_gap_ratio=0.3,
            triage_specific_max_entropy=0.5,
            triage_ambiguous_min_entropy=0.6,
            triage_ambiguous_doc_spread=3,
        )
        assert s.triage_calibrated is True

    @pytest.mark.parametrize("missing_field", [
        "triage_specific_gap_ratio",
        "triage_specific_max_entropy",
        "triage_ambiguous_min_entropy",
        "triage_ambiguous_doc_spread",
    ])
    def test_false_when_exactly_one_missing(self, missing_field):
        values = {
            "triage_specific_gap_ratio": 0.3,
            "triage_specific_max_entropy": 0.5,
            "triage_ambiguous_min_entropy": 0.6,
            "triage_ambiguous_doc_spread": 3,
        }
        values[missing_field] = None
        s = Settings(**values)
        assert s.triage_calibrated is False
