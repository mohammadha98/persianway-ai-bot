"""httpx-level contract tests for the SSE chat stream (`GET /api/chat/stream`).

Where `tests/api/test_chat_stream.py` pins *when* the route writes bytes (driving
Starlette's ASGI interface directly), this module drives the real app through
`httpx.AsyncClient` - the same shape a browser uses - and pins the *wire format*:

  * `text/event-stream` with `Cache-Control: no-cache`, `Connection: keep-alive`
    and `X-Accel-Buffering: no`, so no proxy buffers or resets the answer,
  * one `data: <json>` line per event, preceded by a `: ping` comment frame,
  * a terminal `data: [DONE]` frame, so "the answer ended" is never inferred from
    a closed connection,
  * the first byte on the wire arrives in well under two seconds even when the
    answer itself takes longer (production symptom: the browser reported
    `net::ERR_HTTP2_PROTOCOL_ERROR` because nothing at all was written until
    intent detection, retrieval and the first LLM round trip had finished).

Timing note: `httpx.ASGITransport` collects the whole response inside the ASGI call
and only hands the bytes back when the app is done, so the bytes it returns cannot
show *when* something was written. The first-byte assertions therefore route the
request through a recording ASGI app (`_RecordingASGI`) that timestamps every
`http.response.body` message as the app emits it - exactly what the proxies in
front of the service see. The request still goes through `httpx.AsyncClient`.
"""
import asyncio
import json
import time

import httpx
import pytest

from app.api.routes import chat as chat_module
from app.api.routes.chat import stream_chat


def _app():
    """The real FastAPI app (imported lazily: importing it builds the SPA routes)."""
    from main import app

    return app


def _params():
    return {
        "user_id": "user-1",
        "message": "best fertilizer for wheat?",
        "session_id": "session-1",
        "user_email": "user@example.com",
    }


class _FakeConversationService:
    """Minimal stand-in for the route's ConversationService dependency."""

    async def get_conversations_by_session_id(self, session_id):
        return []

    async def store_conversation(self, **kwargs):
        return {"conversation_id": "conv-1", "assistant_message_id": "msg-1"}


def _override_dependencies(chat_service, conversation_service=None):
    """Bind the fake services for the duration of one request."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        from app.services.chat_service import get_chat_service
        from app.services.conversation_service import get_conversation_service

        app = _app()
        conversation = conversation_service or _FakeConversationService()
        app.dependency_overrides[get_chat_service] = lambda: chat_service
        app.dependency_overrides[get_conversation_service] = lambda: conversation
        try:
            yield
        finally:
            app.dependency_overrides.clear()

    return _ctx()


def _sse_frames(body: str):
    """Split an SSE body into frames (frames are separated by a blank line)."""
    return [frame for frame in body.split("\n\n") if frame]


def _data_payloads(frames):
    return [frame[len("data: "):] for frame in frames if frame.startswith("data: ")]


# metadata -> tokens -> the service's authoritative final answer.
_STREAM_EVENTS = [
    {"type": "metadata", "query_analysis": {"confidence_score": 0.9}, "normalized_sources": []},
    {"type": "chunk", "content": "Urea "},
    {"type": "chunk", "content": "is the best nitrogen source."},
    {"type": "done", "answer": "Urea is the best nitrogen source.", "query_analysis": {"confidence_score": 0.9}},
]


class _FakeChatService:
    """Async-generator stand-in for `ChatService.process_message_stream`."""

    def __init__(self, events, delay=0.0):
        self.events = events
        # Latency before every event: models retrieval + the first LLM round trip.
        self.delay = delay

    async def generate_conversation_title(self, message):
        return "Test Title"

    async def process_message_stream(self, user_id, message, conversation_history=None):
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event

    def get_conversation_history(self, user_id):
        return []


class _RecordingASGI:
    """ASGI wrapper that timestamps every `send` the wrapped app performs."""

    def __init__(self, app):
        self.app = app
        self.sent = []  # (elapsed_seconds, message)
        self.started_at = None

    async def __call__(self, scope, receive, send):
        self.started_at = time.perf_counter()

        async def _send(message):
            self.sent.append((time.perf_counter() - self.started_at, message))
            await send(message)

        await self.app(scope, receive, _send)

    @property
    def body_frames(self):
        """Non-empty `http.response.body` messages, with their elapsed time."""
        return [
            (at, message.get("body", b""))
            for at, message in self.sent
            if message["type"] == "http.response.body" and message.get("body")
        ]

@pytest.mark.asyncio
async def test_httpx_client_receives_a_valid_sse_stream():
    """The body is well-formed SSE and ends with an explicit `[DONE]` frame."""
    service = _FakeChatService(_STREAM_EVENTS)

    with _override_dependencies(service):
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            body = ""
            async with client.stream("GET", "/api/chat/stream", params=_params()) as response:
                assert response.status_code == 200

                # Every header the proxies need to stream this answer untouched.
                assert response.headers["content-type"].startswith("text/event-stream")
                assert response.headers["cache-control"] == "no-cache"
                assert response.headers["connection"] == "keep-alive"
                assert response.headers["x-accel-buffering"] == "no"

                async for text in response.aiter_text():
                    body += text

    frames = _sse_frames(body)

    # The very first frame is the keep-alive comment, so the headers are flushed
    # before any slow work runs.
    assert frames[0] == ": ping"

    # Every line is a valid SSE field: a comment or a single `data:` line. A raw
    # JSON blob (the pre-fix behaviour) would fail here.
    for frame in frames:
        for line in frame.split("\n"):
            assert line.startswith((": ", "data: ")), f"invalid SSE line: {line!r}"

    payloads = _data_payloads(frames)
    assert payloads[-1] == "[DONE]"

    events = [json.loads(payload) for payload in payloads[:-1]]
    assert [event["type"] for event in events] == ["metadata", "chunk", "chunk", "done"]
    # Tokens arrive one event each, never as a single blob.
    assert [event["content"] for event in events if event["type"] == "chunk"] == [
        "Urea ",
        "is the best nitrogen source.",
    ]
    assert events[-1]["answer"] == "Urea is the best nitrogen source."
    assert events[-1]["conversation_id"] == "conv-1"
    assert events[-1]["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_error_event_is_closed_with_a_done_marker():
    """A structured error is followed by `[DONE]`: the stream is closed on
    purpose instead of being reset by the connection dropping."""
    events = [
        {"type": "metadata", "normalized_sources": []},
        {"type": "error", "message": "knowledge base unavailable", "code": "vector_store_unavailable"},
    ]

    with _override_dependencies(_FakeChatService(events)):
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/chat/stream", params=_params())

    frames = _sse_frames(response.text)
    payloads = _data_payloads(frames)
    assert payloads[-1] == "[DONE]"

    reported = [json.loads(payload) for payload in payloads[:-1]]
    assert [event["type"] for event in reported] == ["metadata", "error"]
    assert reported[-1]["code"] == "vector_store_unavailable"

@pytest.mark.asyncio
async def test_first_byte_reaches_the_socket_under_two_seconds(monkeypatch):
    """The first byte is written immediately, and heartbeats keep it coming."""
    # A 0.1s heartbeat interval makes the silent phase observable in a fast test.
    monkeypatch.setattr(chat_module, "SSE_HEARTBEAT_INTERVAL_SECONDS", 0.1)

    delay = 0.4  # 4 events x 0.4s: the answer cannot be ready before 0.4s.
    service = _FakeChatService(_STREAM_EVENTS, delay=delay)
    recorder = _RecordingASGI(_app())

    with _override_dependencies(service):
        # The recording app is wrapped by the standard httpx ASGI transport, so the
        # request/response cycle is a real `httpx.AsyncClient` one.
        transport = httpx.ASGITransport(app=recorder)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/chat/stream", params=_params())

    assert response.status_code == 200

    frames = recorder.body_frames
    first_at, first_body = frames[0]
    assert first_body == chat_module.SSE_STREAM_OPEN_FRAME.encode()
    # The requirement: the first byte is on the wire well under two seconds ...
    assert first_at < 2.0
    # ... and in fact before the first event could possibly be produced, i.e. the
    # client is not waiting for retrieval / the first LLM round trip.
    assert first_at < delay

    heartbeat = chat_module.SSE_HEARTBEAT_FRAME.encode()
    first_data_index = next(
        i for i, (_, body) in enumerate(frames) if body.startswith(b"data:")
    )
    # The open frame is byte-identical to the periodic heartbeat, so start looking
    # for the first *periodic* one after it.
    heartbeat_index = next(
        i for i, (_, body) in enumerate(frames[1:], start=1) if body == heartbeat
    )
    # Heartbeats are written while retrieval / the first LLM round trip run, so a
    # proxy idle timeout can never fire during the silent phase of an answer.
    assert 0 < heartbeat_index < first_data_index

    # The whole answer still arrives, ending with the explicit marker.
    assert response.text.endswith(chat_module.SSE_DONE_FRAME)


@pytest.mark.asyncio
async def test_client_disconnect_is_treated_as_cancellation(caplog):
    """A disconnect cancels the stream: no error frame, no leaked exception.

    `httpx.ASGITransport` cannot express a disconnect (it only reports one after
    the response has completed), so the cancellation Starlette performs is raised
    at the generator directly - the same exception, at the same suspension point.
    """
    service = _FakeChatService(_STREAM_EVENTS, delay=0.05)

    with _override_dependencies(service):
        response = await stream_chat(
            user_id="user-1",
            message="best fertilizer for wheat?",
            session_id="session-1",
            user_email="user@example.com",
            chat_service=service,
            conversation_service=_FakeConversationService(),
        )

    iterator = response.body_iterator
    assert await iterator.__anext__() == chat_module.SSE_STREAM_OPEN_FRAME

    with caplog.at_level("INFO"):
        # Cancellation must propagate: swallowing it would leave the service
        # generator running behind a connection nobody is reading.
        with pytest.raises(asyncio.CancelledError):
            await iterator.athrow(asyncio.CancelledError())

    assert "cancelled" in caplog.text.lower()


class _SlowConversationService(_FakeConversationService):
    """Persistence that takes longer than the heartbeat interval.

    Models the real post-processing: the Mongo write plus the conversation-title
    LLM call inside `store_conversation`, both of which run after the last token
    and before the route can emit its `done` event.
    """

    def __init__(self, delay):
        self.delay = delay

    async def store_conversation(self, **kwargs):
        await asyncio.sleep(self.delay)
        return {"conversation_id": "conv-1", "assistant_message_id": "msg-1"}


@pytest.mark.asyncio
async def test_heartbeats_continue_while_the_answer_is_persisted(monkeypatch):
    """The post-processing wait writes bytes too, so `done` is never awaited in
    silence (a silent tail can make a proxy reset an otherwise finished stream)."""
    monkeypatch.setattr(chat_module, "SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    service = _FakeChatService(_STREAM_EVENTS)
    recorder = _RecordingASGI(_app())

    with _override_dependencies(service, _SlowConversationService(0.3)):
        transport = httpx.ASGITransport(app=recorder)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/chat/stream", params=_params())

    assert response.status_code == 200
    frames = [body.decode() for _, body in recorder.body_frames]
    heartbeat = chat_module.SSE_HEARTBEAT_FRAME

    last_token_index = max(
        i for i, frame in enumerate(frames) if '"type": "chunk"' in frame
    )
    done_index = next(
        i for i, frame in enumerate(frames) if '"type": "done"' in frame
    )
    assert done_index > last_token_index

    # At least one keep-alive frame crosses the persistence window.
    assert any(
        frame == heartbeat
        for frame in frames[last_token_index + 1:done_index]
    ), "no heartbeat was written while the conversation was being persisted"
    assert frames[-1] == chat_module.SSE_DONE_FRAME


