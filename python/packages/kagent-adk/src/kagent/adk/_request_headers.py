"""Where the inbound A2A request headers live in ADK session state.

The executor stores them under ADK's ``temp:`` prefix. ADK applies ``temp:``
keys to the in-memory session for the current invocation, so header providers,
plugins and tools can still read them, and trims them from the event before any
session service persists it. Request headers can carry credentials such as
``Authorization``, so they must not end up in stored session state or event
history.
"""

from typing import Any, Optional

HEADERS_STATE_KEY = "temp:headers"

# Key the headers were stored under before they moved to temp state. It is read
# only when HEADERS_STATE_KEY is absent, so code that seeds ``state["headers"]``
# itself keeps working. The executor always sets HEADERS_STATE_KEY, so a value
# persisted under this key by an older version is never used for a new request.
LEGACY_HEADERS_STATE_KEY = "headers"


def headers_from_state(state: Optional[Any]) -> dict[str, Any]:
    """Return the current request's headers from session state, or ``{}``."""
    if state is None:
        return {}
    headers = state.get(HEADERS_STATE_KEY)
    if headers is None:
        headers = state.get(LEGACY_HEADERS_STATE_KEY)
    return headers if isinstance(headers, dict) else {}
