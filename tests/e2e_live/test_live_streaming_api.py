"""Live SSE contract tests for `GET /api/chat/stream` on the deployed service.

What this module answers
------------------------
"The chat reaches the user as a *stream*, and it survives the proxies between the
browser and the worker." Every other streaming test in the repository verifies the
route through Starlette's ASGI interface inside the test process, which cannot see
the deployed nginx, its buffering defaults or the parser the shipped frontend
actually runs. These tests talk to the real HTTPS endpoint instead, and their
assertions are written from the *frontend's* point of view
(`frontend/ai-panel/src/app/services/chat.service.ts`):

  1. `fetch()` resolves early -- the route writes a `: ping` comment frame before
     intent detection, retrieval and the first LLM round trip, so `fetch()` gets its
     status line immediately and the UI can start rendering. A request that writes
     nothing until the answer is ready is what produced the production symptom
     `net::ERR_HTTP2_PROTOCOL_ERROR`.
  2. Tokens arrive **one frame at a time** (`type: chunk`), not in a single burst at
     the end: the number of distinct arrival timestamps proves whether an
     intermediate layer buffered the answer.
  3. `status` / `metadata` arrive before the first token, so the typing indicator can
     name the stage instead of spinning silently.
  4. The stream ends with a `data: [DONE]` frame after exactly one `done` event, and
     `done.answer` equals the concatenation of the tokens -- the client never has to
     infer "the answer ended" from a closed connection.
  5. Silence never exceeds the heartbeat interval, so no proxy read-timeout can reset
     a slow-but-healthy generation.
  6. A client that hangs up mid-answer does not wedge the worker: the next request
     still opens immediately and `/health` keeps answering.
  7. Persian text survives the whole path (no `U+FFFD`, no UTF-8-decoded-as-Latin-1
     mojibake).

Cost: one *full* turn for `primary_stream`, one more for the follow-up turn (the
title is already stored, which is what makes the strict "opens immediately"
assertion possible), plus one cheap hanging-up request per cancel-safety test -- see
`docs/LIVE_E2E_STREAMING_TESTS.md`.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import httpx
import pytest

from tests.e2e_live.conftest import LiveSettings, note_buffering, write_report
from tests.e2e_live.live_client import (
    DONE_MARKER,
    StreamRun,
    run_probe,
    sse_shape_problems,
    stream_header_warnings,
    stream_headers_are_proxy_safe,
)

pytestmark = pytest.mark.live

# Persian block + the byte pairs that show up when UTF-8 Persian is decoded as
# Latin-1 ("Ø§", "Ù…", ...): the answer must be Persian text, not mojibake.
PERSIAN_RANGE = re.compile(r"[\u0600-\u06FF]")
MOJIBAKE = re.compile(r"[ØÙÚÃÂ][\u0080-\u00BF\u0600-\u06FF]")


# ==================== reachability (no LLM cost) ====================

def test_health_endpoint_reports_healthy(live_client: httpx.Client, live_settings: LiveSettings):
    """The deployment answers before anything else is probed.

    A failing chat assertion is only meaningful if the service itself was up: this
    separates "the service is down" from "streaming regressed".
    """
    started = time.perf_counter()
    response = live_client.get(f"{live_settings.base_url}/health", timeout=20.0)
    elapsed = time.perf_counter() - started

    assert response.status_code == 200, f"/health returned {response.status_code}"
    payload = response.json()
    assert payload.get("status") == "healthy", payload
    assert elapsed < 10.0, f"/health took {elapsed:.2f}s"


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_stream_endpoint_rejects_a_request_without_a_message(
    live_client: httpx.Client, live_settings: LiveSettings, method: str
):
    """Validation runs before any LLM work, and the GET/POST aliases both exist.

    Two things are checked for free (no tokens spent): FastAPI still rejects the
    malformed request with 422 *and* nothing resembling an SSE stream is returned, so
    a client cannot mistake an error page for a stream that then hangs.
    """
    url = f"{live_settings.base_url}/api/chat/stream"
    response = live_client.request(method, url, params={"user_id": "live-e2e-probe"}, timeout=20.0)

    assert response.status_code == 422, (
        f"{method} {url} without `message` returned {response.status_code}; expected 422"
    )
    assert not response.headers.get("content-type", "").startswith("text/event-stream")
    assert "message" in response.text, response.text[:300]


# ==================== the streamed turn (shared evidence) ====================

def test_url_contract_matches_the_shipped_frontend(primary_stream: StreamRun):
    """The shipped frontend streams over `GET ?user_id=&session_id=&message=&user_email=`.

    `chat.service.ts -> streamChatMessage` builds exactly these parameters and
    `fetch`es the URL; a drift here (path, method or parameter name) breaks the chat
    page while every in-process test keeps passing.
    """
    assert primary_stream.status_code == 200, (
        f"stream returned HTTP {primary_stream.status_code}; "
        f"transport_error={primary_stream.transport_error}"
    )
    assert "/api/chat/stream" in primary_stream.request_url
    for parameter in ("user_id=", "session_id=", "message=", "user_email="):
        assert parameter in primary_stream.request_url, primary_stream.request_url


def test_headers_do_not_let_a_proxy_buffer_the_answer(
    primary_stream: StreamRun, live_settings: LiveSettings
):
    """`text/event-stream; charset=utf-8` + `no-cache`, and the response is not compressed.

    The advisory half (`X-Accel-Buffering`, hop-by-hop `Connection`) is printed instead of
    asserted: a CDN edge in front of the worker is allowed to drop it, and the *arrival
    timing* of the tokens is the real evidence (see the incremental-delivery test).
    """
    problems = stream_headers_are_proxy_safe(primary_stream)
    assert not problems, (
        f"stream headers would let a proxy break the answer "
        f"(base_url={live_settings.base_url}): " + "; ".join(problems)
    )
    for note in stream_header_warnings(primary_stream):
        print(f"[live-e2e] header note: {note}")


def test_stream_opens_with_a_keepalive_comment_frame(primary_stream: StreamRun):
    """The first frame is a `: ping` comment, written before the slow steps.

    That is what makes `fetch()` resolve early and gives every proxy real bytes to
    forward while retrieval and the first LLM round trip run. Comment frames carry no
    `data:` line, so the frontend parser ignores them.
    """
    assert primary_stream.frames, "the stream produced no frames at all"
    first = primary_stream.frames[0]
    assert first.is_comment, (
        f"first frame was {first.kind!r}: {first.raw[:120]!r}; the client cannot start "
        "rendering until retrieval, rewrite and the first LLM token are done"
    )
    assert primary_stream.t_first_byte is not None


def test_first_turn_first_byte_stays_within_the_title_timeout(
    primary_stream: StreamRun, live_settings: LiveSettings
):
    """Even on a brand-new conversation the first byte is bounded.

    The route generates the conversation title (an LLM round trip) *before* the
    `StreamingResponse` exists, so the very first request of a new session cannot
    write anything until that call returns. It is wrapped in
    `TITLE_GENERATION_TIMEOUT_SECONDS` (15s) precisely so an unresponsive provider
    cannot hold the connection open with no bytes at all; this asserts the bound holds
    on the deployment and records the measured value.
    """
    assert primary_stream.t_first_byte is not None, "no byte ever arrived"
    budget = live_settings.title_budget
    if primary_stream.t_first_byte >= budget:
        note_buffering(
            live_settings,
            f"the first turn's first byte arrived only after "
            f"{primary_stream.t_first_byte:.2f}s",
        )
    assert primary_stream.t_first_byte < budget, (
        f"first byte took {primary_stream.t_first_byte:.2f}s (> {budget}s): the title "
        "generation bound is not protecting the first message of a conversation"
    )
    if primary_stream.t_first_byte > live_settings.ttfb_budget:
        # Not a failure (the title call is expected here), but it must be visible: this
        # is the only window where a heartbeat cannot help, because the generator has
        # not started yet.
        print(
            f"[live-e2e] first byte of a new conversation took "
            f"{primary_stream.t_first_byte:.2f}s (title generation happens before the "
            f"stream opens; the budget for follow-up turns is {live_settings.ttfb_budget}s)"
        )


def test_sse_grammar_and_end_of_stream_marker(primary_stream: StreamRun):
    """Frames are `data:` lines or `:` comments, and `data: [DONE]` closes the stream.

    The frontend recognises `[DONE]` *before* JSON-parsing and completes on the `done`
    event: a stream that ends by simply closing the connection is reported to the user
    as `stream_incomplete` ("اتصال پیش از دریافت پاسخ کامل قطع شد").
    """
    problems = sse_shape_problems(primary_stream)
    assert not problems, "; ".join(problems)
    assert primary_stream.last_frame is not None
    assert primary_stream.last_frame.payload == DONE_MARKER


def test_status_and_metadata_precede_the_first_token(primary_stream: StreamRun):
    """The UI is told what is happening before the first answer token.

    `status` frames drive the typing indicator text and `metadata` carries the query
    analysis / sources / response parameters the message bubble renders. If either
    arrived after the first token, the first UI paint would be missing its data.
    """
    first_chunk = primary_stream.first_arrival_of("chunk")
    assert first_chunk is not None, (
        f"no `chunk` frame arrived (event types: {primary_stream.event_types})"
    )
    early = [
        (name, primary_stream.first_arrival_of(name))
        for name in ("status", "metadata")
        if primary_stream.first_arrival_of(name) is not None
    ]
    assert early, (
        "neither `status` nor `metadata` was written before the answer "
        f"(event types: {primary_stream.event_types})"
    )
    for name, arrival in early:
        assert arrival is not None and arrival <= first_chunk, (
            f"`{name}` arrived at {arrival:.2f}s, after the first token at {first_chunk:.2f}s"
        )


def test_tokens_are_delivered_incrementally_not_in_one_burst(
    primary_stream: StreamRun, live_settings: LiveSettings
):
    """The central assertion: the answer reaches the client *while* it is generated.

    A buffering layer (nginx without `X-Accel-Buffering: no`, a compressing proxy, an
    SSE-incompatible gateway) collapses the whole turn into one delivery: every chunk
    frame then shares the same arrival timestamp and the user watches a spinner for the
    entire generation. Requiring several distinct arrival times, spread over real
    wall-clock time, is what makes "it streams" measurable rather than assumed.
    """
    chunks = primary_stream.chunk_frames
    assert chunks, (
        "no token frames at all: `CHAT_STREAMING_ENABLED` looks disabled on the "
        f"deployment (event types: {primary_stream.event_types})"
    )

    arrivals = [frame.arrival for frame, _ in chunks]
    distinct = len(set(round(arrival, 3) for arrival in arrivals))
    spread = arrivals[-1] - arrivals[0]
    if distinct < 2 or spread <= 0.2:
        note_buffering(
            live_settings,
            f"{len(chunks)} token frame(s) arrived within {spread:.3f}s of each other "
            f"(distinct arrival moments: {distinct})",
        )
    assert distinct >= 2, (
        f"{len(chunks)} token frame(s) arrived at a single moment ({arrivals[0]:.3f}s): the "
        "answer was buffered somewhere between the worker and the client"
    )
    assert spread > 0.2, (
        f"the {len(chunks)} token frames span only {spread:.3f}s of wall clock; the client "
        "did not receive the answer progressively"
    )
    done_arrival = primary_stream.first_arrival_of("done")
    assert done_arrival is not None and done_arrival >= arrivals[-1] - 0.001, (
        "the `done` frame arrived before the last token"
    )
    assert primary_stream.total_seconds <= live_settings.answer_timeout, (
        f"the whole turn took {primary_stream.total_seconds:.1f}s, over the "
        f"{live_settings.answer_timeout:.0f}s budget this suite allows a live answer"
    )


def test_answer_is_complete_persian_text_without_mojibake(primary_stream: StreamRun):
    """`done.answer` is authoritative, non-empty, Persian and equals the streamed tokens.

    Two regressions are covered at once: a truncated answer (the `done` frame must
    carry the complete text -- in single-frame mode no token frame exists at all, and in
    token mode the client keeps whatever it accumulated), and an encoding regression (a
    response piped through Latin-1 turns Persian into "Ø§Ù„..." plus `U+FFFD`).
    """
    done = primary_stream.done_event
    assert done is not None, f"no `done` event (error event: {primary_stream.error_event})"
    answer = done.get("answer") or ""

    assert answer.strip(), "the `done` event carried an empty answer"
    assert len(answer.strip()) >= 20, f"answer looks truncated: {answer!r}"
    assert PERSIAN_RANGE.search(answer), f"answer is not Persian text: {answer[:200]!r}"
    assert "\ufffd" not in answer, "answer contains U+FFFD replacement characters"
    assert not MOJIBAKE.search(answer), f"answer looks like mis-decoded UTF-8: {answer[:120]!r}"
    assert primary_stream.streamed_answer == answer, (
        "the streamed tokens differ from the authoritative `done.answer` "
        f"({len(primary_stream.streamed_answer)} vs {len(answer)} chars): the UI would "
        "render a truncated or doubled answer"
    )


def test_done_frame_carries_the_metadata_the_panel_renders(primary_stream: StreamRun):
    """`done` carries the ids and history the chat page needs after the answer.

    `chat.component.ts -> done` writes `message_id` / `conversation_id` onto the
    message, which is what enables the sources and "prompt sent to the LLM" actions and
    what lets a later turn find its history. A `None` id means persistence silently
    failed: the answer still renders, but those features disappear from the panel.
    """
    done = primary_stream.done_event
    assert done is not None
    assert done.get("conversation_id"), "done.conversation_id is missing (persistence failed?)"
    assert done.get("message_id"), "done.message_id is missing (persistence failed?)"
    assert done.get("title"), "done.title is missing"
    assert isinstance(done.get("normalized_sources"), list), done.get("normalized_sources")
    history = done.get("conversation_history")
    assert isinstance(history, list), history
    for message in history:
        assert set(message) >= {"role", "content"}, message
        assert message["role"] in {"user", "assistant"}, message


def test_no_silence_outlives_the_heartbeat_interval(
    primary_stream: StreamRun, live_settings: LiveSettings
):
    """No gap between frames exceeds the heartbeat interval by more than the slack.

    The route writes `: ping` every `SSE_HEARTBEAT_INTERVAL_SECONDS` (10s) while the
    answer is silent, so nginx' 60s `proxy_read_timeout` can never see an idle
    connection. A longer gap means the keep-alive is not reaching the client and a proxy
    may reset a healthy turn.
    """
    gaps = [end - start for start, end in primary_stream.silent_gaps()]
    worst = max(gaps, default=0.0)
    if worst > live_settings.max_silent_gap:
        note_buffering(
            live_settings,
            f"the wire was silent for {worst:.2f}s (nothing flushed by the edge)",
        )
    assert worst <= live_settings.max_silent_gap, (
        f"the wire was silent for {worst:.2f}s (> {live_settings.max_silent_gap}s): the "
        "heartbeat did not reach the client"
    )


# ==================== follow-up turn: the strict "opens immediately" case ====================

def test_followup_turn_opens_immediately(
    primary_stream: StreamRun,
    live_client: httpx.Client,
    live_settings: LiveSettings,
    report_dir: Any,
):
    """A second turn in the same session writes its first byte within the budget.

    Production-critical case and the reason the keep-alive machinery exists: the
    conversation already has a title, so no LLM call runs before the `StreamingResponse`
    is returned and the open frame must reach the client "immediately". If this
    regresses, `fetch()` does not resolve, nothing is on the wire, and nginx resets the
    request (`ERR_HTTP2_PROTOCOL_ERROR` in the browser).

    Also asserts the turn still ends properly, so a fast-open regression cannot hide
    behind a broken stream.
    """
    assert primary_stream.done_event is not None, "the first turn must complete before this"
    started = time.perf_counter()
    run = run_probe(
        live_client,
        live_settings.base_url,
        message=live_settings.probe_message("followup"),
        user_id=live_settings.user_id,
        session_id=live_settings.session_id,
        user_email=live_settings.user_email,
        timeout=live_settings.answer_timeout,
        evidence_dir=report_dir,
        name="followup_stream",
    )
    elapsed = time.perf_counter() - started

    assert run.status_code == 200, f"follow-up stream returned HTTP {run.status_code}"
    assert run.t_first_byte is not None, "no byte arrived on the follow-up turn"
    if run.t_first_byte > live_settings.ttfb_budget:
        note_buffering(
            live_settings,
            f"the follow-up turn's first byte reached the client after "
            f"{run.t_first_byte:.2f}s (the server writes it in well under a second: the "
            "session already has a title, so no LLM call precedes the stream)",
        )
    assert run.t_first_byte <= live_settings.ttfb_budget, (
        f"the follow-up turn wrote its first byte after {run.t_first_byte:.2f}s (budget "
        f"{live_settings.ttfb_budget}s): the client and every proxy waited in silence "
        f"(total turn {elapsed:.1f}s)"
    )
    assert run.frames[0].is_comment, "the follow-up turn did not open with the `: ping` frame"
    problems = sse_shape_problems(run)
    assert not problems, "; ".join(problems)
    assert run.answer.strip(), "the follow-up turn produced an empty answer"


# ==================== cancel safety ====================

def test_client_hangup_does_not_wedge_the_worker(
    live_client: httpx.Client, live_settings: LiveSettings, report_dir: Any
):
    """Hanging up mid-answer is a normal end of request, not a crash.

    A closed tab, a navigation or a proxy reset cancels the request while the answer is
    being produced. The route must re-raise `CancelledError` (so the service generator
    stops instead of running on for nobody) and keep serving other clients -- a wedged
    worker is exactly the "the system becomes unavailable for everyone" failure mode, so
    it is tested on purpose, once.
    """
    run = run_probe(
        live_client,
        live_settings.base_url,
        message=live_settings.probe_message("hangup"),
        user_id=live_settings.user_id,
        session_id=f"{live_settings.session_id}-hangup",
        user_email=live_settings.user_email,
        timeout=60.0,
        evidence_dir=report_dir,
        name="hangup_stream",
        stop_after_frames=1,  # read the open frame, then drop the connection
    )

    assert run.aborted, "the probe did not hang up as intended"
    assert run.status_code == 200, f"hang-up probe got HTTP {run.status_code}"
    assert run.frames and run.frames[0].is_comment, (
        "the connection was dropped before the open frame, so the cancel path was not the "
        "one exercised"
    )

    health = live_client.get(f"{live_settings.base_url}/health", timeout=20.0)
    assert health.status_code == 200, (
        f"/health answered {health.status_code} right after a client hung up mid-answer"
    )


def test_a_new_stream_still_opens_immediately_after_a_hangup(
    live_client: httpx.Client, live_settings: LiveSettings, report_dir: Any
):
    """The next client is served normally right after someone else's connection died.

    Reads only the first frame of a fresh request: it proves the accept path and the
    response start are still healthy without paying for a second full answer.
    """
    run = run_probe(
        live_client,
        live_settings.base_url,
        message=live_settings.probe_message("after-hangup"),
        user_id=live_settings.user_id,
        session_id=f"{live_settings.session_id}-after-hangup",
        user_email=live_settings.user_email,
        timeout=60.0,
        evidence_dir=report_dir,
        name="after_hangup_open_frame",
        stop_after_frames=1,
    )

    assert run.status_code == 200, f"fresh stream returned HTTP {run.status_code}"
    assert run.t_first_byte is not None, "no byte arrived on the fresh stream"
    if run.t_first_byte > live_settings.title_budget:
        note_buffering(
            live_settings,
            f"a fresh stream's first byte reached the client after {run.t_first_byte:.2f}s",
        )
    assert run.t_first_byte <= live_settings.title_budget, (
        f"the fresh stream opened after {run.t_first_byte:.2f}s (> "
        f"{live_settings.title_budget}s): title generation is the only work allowed before "
        "the first byte on a new session"
    )
    if run.t_first_byte > live_settings.ttfb_budget:
        print(
            f"[live-e2e] a brand-new session waited {run.t_first_byte:.2f}s for its first "
            "byte because conversation-title generation runs before the stream opens; the "
            "user sees a silent spinner for that whole time"
        )


# ==================== delivery profile (always green, always informative) ====================

def test_stream_delivery_profile_is_recorded(
    primary_stream: StreamRun, report_dir: Any, live_settings: LiveSettings
):
    """Print and record how the deployment actually delivers a turn -- never fails.

    The timing assertions above turn red when the edge buffers; this test is the report:
    how many flushes a turn was delivered in, when the first byte arrived, how the tokens
    were spread, and where the silent windows were. It is the fastest way to tell "the
    service is slow" apart from "the delivery is collapsed", and its numbers are kept in
    `reports/live_e2e_<ts>/primary_stream.json` for the next investigation.
    """
    flushes: List[Tuple[float, int]] = []
    for frame in primary_stream.frames:
        if flushes and abs(frame.arrival - flushes[-1][0]) < 0.05:
            flushes[-1] = (flushes[-1][0], flushes[-1][1] + 1)
        else:
            flushes.append((frame.arrival, 1))
    gaps = [end - start for start, end in primary_stream.silent_gaps()]
    profile = {
        "base_url": live_settings.base_url,
        "first_byte_s": primary_stream.t_first_byte,
        "turn_seconds": primary_stream.total_seconds,
        "frames": len(primary_stream.frames),
        "token_frames": len(primary_stream.chunk_frames),
        "flush_count": len(flushes),
        "flushes": [{"at_s": round(at, 3), "frames": count} for at, count in flushes],
        "longest_silence_s": max(gaps, default=0.0),
        "event_types": primary_stream.event_types,
        "server": primary_stream.headers.get("server"),
        "x_accel_buffering": primary_stream.headers.get("x-accel-buffering"),
    }
    write_report(report_dir, "delivery_profile", profile)

    print(
        "[live-e2e] delivery profile: "
        f"first byte {primary_stream.t_first_byte:.2f}s, turn {primary_stream.total_seconds:.2f}s, "
        f"{len(primary_stream.chunk_frames)} token frame(s) delivered in {len(flushes)} flush(es), "
        f"longest silence {profile['longest_silence_s']:.2f}s, "
        f"server={profile['server']!r}, x-accel-buffering={profile['x_accel_buffering']!r}"
    )
    assert primary_stream.frames, "the stream produced no frames at all"


# ==================== load: opt-in (E2E_LIVE_HEAVY=1) ====================

@pytest.mark.live_heavy
def test_two_concurrent_turns_both_complete(
    live_client: httpx.Client,
    live_settings: LiveSettings,
    report_dir: Any,
    live_heavy: None,
):
    """Two chats at once: both stream, neither 5xx, neither hangs.

    Off by default (`E2E_LIVE_HEAVY=1`). The service runs a single worker and the
    retrieval path is partially GIL-bound (see `docs/RAG_LATENCY_AUDIT_BM25.md`), so this
    deliberately overlaps two turns to check the *correctness* consequence -- both clients
    get a complete, well-formed stream -- while the latency cost of the overlap is
    recorded in the evidence files next to it.
    """
    def _turn(suffix: str) -> Tuple[str, StreamRun]:
        run = run_probe(
            live_client,
            live_settings.base_url,
            message=live_settings.probe_message(f"concurrent-{suffix}"),
            user_id=live_settings.user_id,
            session_id=f"{live_settings.session_id}-concurrent-{suffix}",
            user_email=live_settings.user_email,
            timeout=live_settings.answer_timeout,
            evidence_dir=report_dir,
            name=f"concurrent_{suffix}",
        )
        return suffix, run

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results: Dict[str, StreamRun] = dict(pool.map(_turn, ["a", "b"]))
    wall = time.perf_counter() - started

    for suffix, run in results.items():
        assert run.status_code == 200, f"turn {suffix} returned HTTP {run.status_code}"
        assert run.transport_error is None, f"turn {suffix}: {run.transport_error}"
        problems = sse_shape_problems(run)
        assert not problems, f"turn {suffix}: " + "; ".join(problems)
        assert run.error_event is None, f"turn {suffix} errored: {run.error_event}"
        assert run.done_event and run.done_event.get("answer"), f"turn {suffix}: empty answer"

    print(
        "[live-e2e] 2 concurrent turns finished in "
        f"{wall:.1f}s (a={results['a'].total_seconds:.1f}s, b={results['b'].total_seconds:.1f}s)"
    )
