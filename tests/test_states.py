import pytest

from wiki_census_eval.states import parse_state_filters


def test_parse_state_filters_accepts_commas_and_repeated_values():
    assert parse_state_filters(["AL,GA", "06"]) == {"01", "13", "06"}


def test_parse_state_filters_pads_fips_codes():
    assert parse_state_filters(["1", "8"]) == {"01", "08"}


def test_parse_state_filters_rejects_unknown_state():
    with pytest.raises(ValueError):
        parse_state_filters(["ZZ"])
