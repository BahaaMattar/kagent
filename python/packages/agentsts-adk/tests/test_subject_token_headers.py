"""The default subject token comes from the invocation-scoped temp:headers state."""

from agentsts.adk._base import HEADERS_KEY, TEMP_HEADERS_KEY
from agentsts.adk._base import _default_get_subject_token as default_get_subject_token


def test_temp_headers_key_is_invocation_scoped():
    assert TEMP_HEADERS_KEY == "temp:headers"


def test_reads_token_from_temp_headers():
    state = {TEMP_HEADERS_KEY: {"Authorization": "Bearer current-token"}}
    assert default_get_subject_token(state) == "current-token"


def test_falls_back_to_legacy_headers_when_temp_headers_absent():
    state = {HEADERS_KEY: {"Authorization": "Bearer legacy-token"}}
    assert default_get_subject_token(state) == "legacy-token"


def test_temp_headers_win_over_legacy_headers():
    state = {
        TEMP_HEADERS_KEY: {"Authorization": "Bearer current-token"},
        HEADERS_KEY: {"Authorization": "Bearer stale-token"},
    }
    assert default_get_subject_token(state) == "current-token"


def test_request_without_authorization_does_not_reuse_a_stale_persisted_token():
    state = {
        TEMP_HEADERS_KEY: {},
        HEADERS_KEY: {"Authorization": "Bearer stale-token"},
    }
    assert default_get_subject_token(state) is None
