"""Regression tests for the SSE chat stream route (`GET/POST /api/chat/stream`).

Production symptom under test: the route answered 200 and then the connection was
reset mid-answer -- `net::ERR_HTTP2_PROTOCOL_ERROR` in the browser, surfaced by the
frontend as the structured `stream_read_error` event (chat.service.ts) -- because
the route wrote nothing at all while intent detection, retrieval and the first LLM
round trip were running. nginx then aborted the upstream read of a request that was
still perfectly healthy ("upstream prematurely closed connection").

These tests pin the behaviours that keep that from happening again:
  * the stream opens with an SSE comment frame, so the response headers are flushed
    before the first slow step,
  * comment frames keep being written while the answer is still being produced,
  * a structured service error / unexpected exception turns into an SSE error frame
    instead of a dead connection,
  * the pre-response title LLM call is bounded and can never delay the headers,
  * closing the stream (client disconnect) cancels the underlying generator.

Comment frames carry no `data:` line, which is why the frontend parser ignores them.
"""
import asyncio
import json

import pytest

from app.api.routes import chat as chat_module
from app.api.routes.chat import stream_chat


class _FakeConversationService:
    """Minimal stand-in for ConversationService used by the route."""

    async def get_conversations_by_session_id(self, session_id):
        return []

    async def store_conversation(self, **kwargs):
        return {"conversation_id": "conv-1", "assistant_message_id": "msg-1"}


class _FakeChatService:
    """Async-generator stand-in for `ChatService.process_message_stream`."""

    def __init__(self, events, delay=0.0, title_sleep=None, title_error=None):
        self.events = events
        # Artificial latency before every event, so the wait is long enough to
        # require keep-alive frames.
        self.delay = delay
        self._title_sleep = title_sleep
        self._title_error = title_error

    async def generate_conversation_title(self, message):
        if self._title_error is not None:
            raise self._title_error
        if self._title_sleep:
            await asyncio.sleep(self._title_sleep)
        return "Test Title"

    async def process_message_stream(self, user_id, message, conversation_history=None):
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event

    def get_conversation_history(self, user_id):
        return []


async def _stream_parts(chat_service):
    """Drive the real route handler and collect the raw SSE chunks it yields."""
    response = await stream_chat(
        user_id="user-1",
        message="بهترین کود برای گندم چیست؟",
        session_id="session-1",
        user_email="user@example.com",
        chat_service=chat_service,
        conversation_service=_FakeConversationService(),
    )
    parts = []
    async for part in response.body_iterator:
        parts.append(part)
    return parts


def _data_frames(parts):
    """Parse the JSON payloads of the `data:` frames.

    Comment frames (`: ping`) carry no payload and the terminal `data: [DONE]`
    marker is deliberately not JSON, so both are skipped here.
    """
    frames = []
    for part in parts:
        for frame in part.split("\n\n"):
            if not frame.startswith("data:"):
                continue
            payload = frame[5:].strip()
            if payload == "[DONE]":
                continue
            frames.append(json.loads(payload))
    return frames


# A representative happy-path sequence: metadata first, then tokens, then the
# service's authoritative final answer.
_STREAM_EVENTS = [
    {"type": "metadata", "query_analysis": {"confidence_score": 0.9}, "normalized_sources": []},
    {"type": "chunk", "content": "بهترین "},
    {"type": "chunk", "content": "کود اوره است."},
    {"type": "done", "answer": "بهترین کود اوره است.", "query_analysis": {"confidence_score": 0.9}},
]


@pytest.mark.asyncio
async def test_stream_opens_with_a_frame_before_any_slow_work():
    """The first bytes on the wire are a comment frame, never silence."""
    parts = await _stream_parts(_FakeChatService(_STREAM_EVENTS, delay=0.05))

    assert parts[0] == chat_module.SSE_STREAM_OPEN_FRAME
    # ... and the stream is closed with the explicit marker rather than by the
    # connection dropping.
    assert parts[-1] == chat_module.SSE_DONE_FRAME

    # The injected open frame must not disturb the event sequence the frontend
    # relies on: metadata -> chunks -> done, with the final answer preserved.
    frames = _data_frames(parts)
    assert [frame["type"] for frame in frames] == ["metadata", "chunk", "chunk", "done"]
    assert frames[-1]["answer"] == "بهترین کود اوره است."
    assert frames[-1]["title"] == "Test Title"
    assert frames[-1]["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_heartbeats_keep_the_connection_alive_while_waiting(monkeypatch):
    """Comment frames are written while the first token is still pending."""
    monkeypatch.setattr(chat_module, "SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    # 4 events x 0.12s: long enough for several heartbeats at a 0.05s interval.
    parts = await _stream_parts(_FakeChatService(_STREAM_EVENTS, delay=0.12))

    assert chat_module.SSE_HEARTBEAT_FRAME in parts
    first_data_index = next(i for i, part in enumerate(parts) if part.startswith("data:"))
    # Skip the open frame: it is the same keep-alive comment as the periodic
    # heartbeat, so the first *periodic* heartbeat is the one that matters here.
    periodic_heartbeat_index = next(
        i for i, part in enumerate(parts[1:], start=1) if part == chat_module.SSE_HEARTBEAT_FRAME
    )
    # At least one heartbeat precedes the first data frame, so a proxy with an
    # idle timeout no longer sees an idle connection during retrieval.
    assert 0 < periodic_heartbeat_index < first_data_index
    # ... and the heartbeat frames stay invisible to the frontend parser.
    assert [frame["type"] for frame in _data_frames(parts)] == [
        "metadata", "chunk", "chunk", "done",
    ]


@pytest.mark.asyncio
async def test_structured_service_error_is_forwarded_once():
    """A structured service error reaches the client instead of a dead stream."""
    events = [
        {"type": "metadata", "normalized_sources": []},
        {
            "type": "error",
            "message": "پایگاه دانش در دسترس نیست",
            "code": "vector_store_unavailable",
        },
    ]

    frames = _data_frames(await _stream_parts(_FakeChatService(events)))

    assert [frame["type"] for frame in frames] == ["metadata", "error"]
    assert frames[-1]["code"] == "vector_store_unavailable"
    assert frames[-1]["message"] == "پایگاه دانش در دسترس نیست"


@pytest.mark.asyncio
async def test_unexpected_service_exception_becomes_a_stream_error_frame():
    """An exception mid-answer is reported as an SSE error frame, not a reset."""

    class _ExplodingChatService(_FakeChatService):
        async def process_message_stream(self, user_id, message, conversation_history=None):
            yield {"type": "chunk", "content": "شروع پاسخ"}
            raise RuntimeError("provider exploded")

    frames = _data_frames(await _stream_parts(_ExplodingChatService([])))

    assert [frame["type"] for frame in frames] == ["chunk", "error"]
    assert frames[-1]["code"] == "stream_route_error"
    # The internal message must never leak to the client.
    assert "provider exploded" not in frames[-1]["message"]


@pytest.mark.asyncio
async def test_title_generation_is_bounded(monkeypatch):
    """A hanging title LLM call cannot hold back the response headers."""
    monkeypatch.setattr(chat_module, "TITLE_GENERATION_TIMEOUT_SECONDS", 0.05)

    parts = await _stream_parts(_FakeChatService(_STREAM_EVENTS, title_sleep=30))

    # The answer still streams, with the documented fallback title.
    assert parts[0] == chat_module.SSE_STREAM_OPEN_FRAME
    frames = _data_frames(parts)
    assert [frame["type"] for frame in frames] == ["metadata", "chunk", "chunk", "done"]
    assert frames[-1]["title"] == "New Conversation"


@pytest.mark.asyncio
async def test_closing_the_stream_cancels_the_underlying_generator():
    """A client disconnect must not leave the service generator running."""
    started = asyncio.Event()
    source_closed = asyncio.Event()

    async def _endless_source():
        try:
            started.set()
            while True:
                await asyncio.sleep(0.01)
                yield {"type": "chunk", "content": "x"}
        finally:
            source_closed.set()

    stream = chat_module._with_sse_heartbeats(_endless_source())
    assert await stream.__anext__() == {"type": "chunk", "content": "x"}
    await started.wait()

    await stream.aclose()

    # The producer task is cancelled, so it stops pulling from the (potentially
    # expensive, still-generating) service generator.
    await asyncio.wait_for(source_closed.wait(), timeout=2.0)


@pytest.mark.asyncio
async def test_heartbeat_wrapper_signals_silence_with_a_sentinel(monkeypatch):
    """`_with_sse_heartbeats` reports silence instead of inventing a fake event."""
    monkeypatch.setattr(chat_module, "SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    async def _slow_source():
        await asyncio.sleep(0.2)
        yield {"type": "done", "answer": "ok"}

    stream = chat_module._with_sse_heartbeats(_slow_source())

    assert await stream.__anext__() is chat_module.SSE_HEARTBEAT_SENTINEL

    # One heartbeat per silent interval, and the real event still arrives
    # untouched once the source produces it.
    remaining = [item async for item in stream]
    assert all(item is chat_module.SSE_HEARTBEAT_SENTINEL for item in remaining[:-1])
    assert remaining[-1] == {"type": "done", "answer": "ok"}


async def _collect_asgi_messages(chat_service):
    """Run the real app through the ASGI interface, recording the `send` calls.

    `TestClient` and httpx's `ASGITransport` both buffer the whole response before
    handing anything back, so they cannot show *when* a byte reached the server.
    Driving the app directly records every `http.response.body` message at the
    moment it is produced -- exactly what the production proxies see.
    """
    import time as _time

    from main import app
    from app.services.chat_service import get_chat_service
    from app.services.conversation_service import get_conversation_service

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/chat/stream",
        "raw_path": b"/api/chat/stream",
        "query_string": b"user_id=user-1&message=hello",
        "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    sent = []
    started_at = _time.perf_counter()
    # Never set: a client that keeps the connection open. Starlette's disconnect
    # listener needs a real suspension point (answering immediately would spin).
    disconnect = asyncio.Event()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append((_time.perf_counter() - started_at, message))

    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_conversation_service] = lambda: _FakeConversationService()
    try:
        await app(scope, receive, send)
    finally:
        app.dependency_overrides.clear()

    return sent


@pytest.mark.asyncio
async def test_http_layer_flushes_bytes_before_the_answer_is_ready(monkeypatch):
    """End to end: real bytes reach the socket while the answer is pending."""
    monkeypatch.setattr(chat_module, "SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)

    # 4 events x 0.2s: the answer cannot possibly be ready before 0.2s.
    sent = await _collect_asgi_messages(_FakeChatService(_STREAM_EVENTS, delay=0.2))

    assert sent[0][1]["type"] == "http.response.start"
    assert sent[0][1]["status"] == 200
    headers = dict(sent[0][1]["headers"])
    assert headers[b"content-type"].startswith(b"text/event-stream")
    # nginx must not buffer the stream, otherwise the heartbeats are useless.
    assert headers[b"x-accel-buffering"] == b"no"

    bodies = [
        (at, message["body"])
        for at, message in sent
        if message["type"] == "http.response.body"
    ]
    # The status line, the headers and the first body chunk all leave together.
    first_at, first_body = bodies[0]
    assert first_body == chat_module.SSE_STREAM_OPEN_FRAME.encode()
    assert first_at < 0.2

    heartbeat = chat_module.SSE_HEARTBEAT_FRAME.encode()
    first_data_index = next(i for i, (_, body) in enumerate(bodies) if body.startswith(b"data:"))
    # Index 0 is the open frame, which is the same comment as a heartbeat: look for
    # the first *periodic* heartbeat after it.
    heartbeat_index = next(
        i for i, (_, body) in enumerate(bodies[1:], start=1) if body == heartbeat
    )
    # Heartbeats arrive before the first real token, so a proxy idle timeout
    # never fires while retrieval + the first LLM round trip are running.
    assert 0 < heartbeat_index < first_data_index


@pytest.mark.asyncio
async def test_status_events_are_forwarded_so_the_ui_can_explain_the_wait():
    """`status` frames survive the route instead of being dropped.

    The keep-alive (`: ping`) is a comment frame the frontend parser ignores, so
    `status` is the only *user-renderable* progress signal. Before this, the route
    had no branch for it and a slow retrieval/generation looked like a bare
    spinner -- the reported "no status message, endless loading" symptom.
    """
    events = [
        {"type": "status", "stage": "retrieval", "message": "در حال جستجو در پایگاه دانش…"},
        {"type": "metadata", "query_analysis": {"confidence_score": 0.9}, "normalized_sources": []},
        {"type": "status", "stage": "generation", "message": "در حال تولید پاسخ…"},
        {"type": "chunk", "content": "پاسخ "},
        {"type": "done", "answer": "پاسخ ", "query_analysis": {"confidence_score": 0.9}},
    ]

    frames = _data_frames(await _stream_parts(_FakeChatService(events)))

    assert [frame["type"] for frame in frames] == [
        "status", "metadata", "status", "chunk", "done",
    ]
    assert frames[0]["stage"] == "retrieval"
    assert frames[0]["message"] == "در حال جستجو در پایگاه دانش…"
    assert frames[2]["stage"] == "generation"
    assert frames[2]["message"] == "در حال تولید پاسخ…"


@pytest.mark.asyncio
async def test_provider_error_event_is_forwarded_with_its_code():
    """A provider-level failure keeps its own code so the UI can name the cause."""
    events = [
        {"type": "status", "stage": "generation", "message": "در حال تولید پاسخ…"},
        {
            "type": "error",
            "message": "سرویس موقتاً شلوغ است؛ لطفاً چند لحظه دیگر دوباره تلاش کنید.",
            "code": "provider_rate_limited",
        },
    ]

    frames = _data_frames(await _stream_parts(_FakeChatService(events)))

    assert [frame["type"] for frame in frames] == ["status", "error"]
    assert frames[-1]["code"] == "provider_rate_limited"
    assert "شلوغ" in frames[-1]["message"]
