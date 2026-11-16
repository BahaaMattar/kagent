from types import MappingProxyType

import pytest

from kagent.adk._request_headers import HEADERS_STATE_KEY, LEGACY_HEADERS_STATE_KEY, headers_from_state


def test_headers_key_is_invocation_scoped():
    # ADK trims "temp:" keys from events before persisting them.
    assert HEADERS_STATE_KEY.startswith("temp:")


def test_reads_temp_headers():
    state = {HEADERS_STATE_KEY: {"authorization": "Bearer current"}}
    assert headers_from_state(state) == {"authorization": "Bearer current"}


def test_falls_back_to_legacy_key_when_temp_key_absent():
    state = {LEGACY_HEADERS_STATE_KEY: {"x-team": "a"}}
    assert headers_from_state(state) == {"x-team": "a"}


def test_temp_key_wins_over_legacy_key():
    state = {
        HEADERS_STATE_KEY: {"authorization": "Bearer current"},
        LEGACY_HEADERS_STATE_KEY: {"authorization": "Bearer stale"},
    }
    assert headers_from_state(state) == {"authorization": "Bearer current"}


def test_empty_temp_headers_do_not_fall_back_to_a_stale_persisted_token():
    # Sessions written by older versions can still hold "headers" in stored state.
    # A request without headers must not pick that token up again.
    state = {
        HEADERS_STATE_KEY: {},
        LEGACY_HEADERS_STATE_KEY: {"authorization": "Bearer stale"},
    }
    assert headers_from_state(state) == {}


@pytest.mark.parametrize("state", [None, {}, {HEADERS_STATE_KEY: "not-a-dict"}, {LEGACY_HEADERS_STATE_KEY: ["x"]}])
def test_missing_or_malformed_headers_return_empty_dict(state):
    assert headers_from_state(state) == {}


def test_accepts_read_only_state_mappings():
    state = MappingProxyType({HEADERS_STATE_KEY: {"x-team": "a"}})
    assert headers_from_state(state) == {"x-team": "a"}
