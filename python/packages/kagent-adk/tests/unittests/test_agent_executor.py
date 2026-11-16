from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from a2a.server.agent_execution.context import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import Artifact, Message, Part, Role, SendMessageRequest, TaskArtifactUpdateEvent
from google.adk.a2a.converters.request_converter import AgentRunRequest
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner, Runner
from google.genai import types
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Value
from kagent.core.a2a import (
    A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL,
    A2A_PART_TYPE_METADATA_KEY,
    A2A_USAGE_METADATA_KEY,
)
from kagent.core.a2a._requests import KAgentUser

import kagent.adk._agent_executor as executor_module
from kagent.adk._agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig
from kagent.adk._bearer_token import bearer_token
from kagent.adk._local_session_service import LocalSessionService
from kagent.adk._request_headers import headers_from_state
from kagent.adk._request_identity import request_user_id


@pytest.fixture(autouse=True)
def _reset_bearer_token():
    token = bearer_token.set(None)
    yield
    bearer_token.reset(token)


def _request_context(*, state: dict | None = None) -> RequestContext:
    message = Message(
        message_id="message-1",
        role=Role.ROLE_USER,
        parts=[Part(text="hello from a2a")],
    )
    return RequestContext(
        ServerCallContext(state=state or {}),
        SendMessageRequest(message=message),
        task_id="task-1",
        context_id="context-1",
    )


@pytest.mark.parametrize(
    ("stream", "expected_mode"),
    [(False, StreamingMode.NONE), (True, StreamingMode.SSE)],
)
def test_request_converter_adds_headers_without_mutating_message_metadata(stream, expected_mode):
    context = _request_context(state={"headers": {"authorization": "Bearer token"}})
    executor = A2aAgentExecutor(runner=lambda: None, config=A2aAgentExecutorConfig(stream=stream))

    run_request = executor._convert_request(context, None)

    assert run_request.run_config.streaming_mode == expected_mode
    assert run_request.state_delta == {"temp:headers": {"authorization": "Bearer token"}}
    assert not context.message.metadata


@pytest.mark.asyncio
async def test_request_headers_are_visible_during_invocation_but_not_persisted(tmp_path):
    """Regression test for #3658: the Authorization header must not be stored."""
    context = _request_context(state={"headers": {"authorization": "Bearer secret-token", "x-team": "a"}})
    executor = A2aAgentExecutor(runner=lambda: None, config=A2aAgentExecutorConfig())
    run_request = executor._convert_request(context, None)

    seen_during_run: list[dict] = []

    class RecordingAgent(BaseAgent):
        async def _run_async_impl(self, ctx):
            seen_during_run.append(headers_from_state(ctx.session.state))
            yield Event(author=self.name, invocation_id=ctx.invocation_id)

    session_service = LocalSessionService(db_url=f"sqlite+aiosqlite:///{(tmp_path / 'sessions.db').as_posix()}")
    runner = Runner(app_name="app", agent=RecordingAgent(name="agent"), session_service=session_service)
    session = await session_service.create_session(app_name="app", user_id="user-1", session_id="context-1")

    async for _ in runner.run_async(
        user_id="user-1",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="hello")]),
        state_delta=run_request.state_delta,
    ):
        pass

    # Header providers, plugins and tools still see the headers during the invocation.
    assert seen_during_run == [{"authorization": "Bearer secret-token", "x-team": "a"}]

    # Nothing about the headers is written to the stored session state or event history.
    stored = await session_service.get_session(app_name="app", user_id="user-1", session_id=session.id)
    assert stored is not None
    assert "temp:headers" not in stored.state
    assert "headers" not in stored.state
    assert stored.events
    for event in stored.events:
        assert "secret-token" not in event.model_dump_json()


def test_convert_request_sets_bearer_token_context_var():
    context = _request_context(state={"headers": {"authorization": "Bearer the-callers-token"}})
    executor = A2aAgentExecutor(runner=lambda: None, config=A2aAgentExecutorConfig())

    executor._convert_request(context, None)

    assert bearer_token.get() == "the-callers-token"


def test_convert_request_clears_bearer_token_when_no_auth_header():
    context = _request_context(state={"headers": {}})
    executor = A2aAgentExecutor(runner=lambda: None, config=A2aAgentExecutorConfig())

    executor._convert_request(context, None)

    assert bearer_token.get() is None


def test_convert_request_understands_canonical_function_call_parts():
    context = _request_context()
    context.message.parts[0].CopyFrom(
        Part(
            data=ParseDict({"id": "call-1", "name": "lookup", "args": {"query": "weather"}}, Value()),
            metadata={A2A_PART_TYPE_METADATA_KEY: A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL},
        )
    )
    executor = A2aAgentExecutor(runner=lambda: None)

    run_request = executor._convert_request(context, executor_module._convert_public_a2a_part_to_genai_part)

    assert run_request.new_message is not None
    assert run_request.new_message.parts[0].function_call is not None
    assert run_request.new_message.parts[0].function_call.name == "lookup"


def test_adk_event_metadata_is_projected_at_adapter_boundary():
    part = Part(
        data=ParseDict({"name": "lookup"}, Value()),
        metadata={"adk_type": "function_call", "adk_thought": True},
    )
    event = TaskArtifactUpdateEvent(
        task_id="task-1",
        context_id="context-1",
        artifact=Artifact(artifact_id="artifact-1", parts=[part]),
        metadata={"adk_usage_metadata": {"total_token_count": 3}, "adk_invocation_id": "private"},
    )

    executor_module._canonicalize_adk_event(event)

    part = event.artifact.parts[0]
    assert part.metadata[A2A_PART_TYPE_METADATA_KEY] == "function_call"
    assert event.metadata[A2A_USAGE_METADATA_KEY]["total_token_count"] == 3
    assert not any(key.startswith("adk_") for key in part.metadata)
    assert not any(key.startswith("adk_") for key in event.metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["", "alice"])
async def test_execute_delegates_to_adk_2_executor_and_closes_request_runner(monkeypatch, caller):
    context = _request_context()
    if caller:
        context.call_context.user = KAgentUser(caller)
    event_queue = object()
    runner = InMemoryRunner(agent=BaseAgent(name="agent"), app_name="app")
    run_request = AgentRunRequest(
        user_id="user-1",
        session_id="context-1",
        run_config=RunConfig(),
    )
    executor = A2aAgentExecutor(runner=lambda: None)
    executor._resolve_runner = AsyncMock(return_value=runner)
    executor._convert_request = lambda request_context, part_converter: run_request
    executor._prepare_session = AsyncMock()
    executor._safe_close_runner = AsyncMock()

    calls = {}

    class FakeUpstreamExecutor:
        def __init__(self, *, runner, config, force_new_version):
            calls.update(runner=runner, config=config, force_new_version=force_new_version)

        async def execute(self, request_context, queue):
            calls.update(context=request_context, event_queue=queue)
            assert request_user_id.get() == caller

    monkeypatch.setattr(executor_module, "UpstreamA2aAgentExecutor", FakeUpstreamExecutor)

    token = request_user_id.set("outer-caller")
    try:
        await executor.execute(context, event_queue)
        assert request_user_id.get() == "outer-caller"
    finally:
        request_user_id.reset(token)

    assert calls["runner"] is runner
    assert calls["force_new_version"] is True
    assert calls["context"] is context
    assert calls["event_queue"]._queue is event_queue
    assert calls["config"].request_converter == executor._convert_request
    assert calls["config"].a2a_part_converter == executor_module._convert_public_a2a_part_to_genai_part
    executor._prepare_session.assert_awaited_once_with(context, run_request, runner)
    executor._safe_close_runner.assert_awaited_once_with(runner)
