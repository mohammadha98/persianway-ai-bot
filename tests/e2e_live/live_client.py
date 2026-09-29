"""Client primitives shared by the live (deployed-service) E2E tests.

`stream_chat` is the important one: it records **when every SSE frame arrived**,
not just what it contained. httpx hands back bytes as the socket delivers them,
so the arrival timestamps are the client-side proof that

  * the open frame (`: ping`) beat the first slow step (a proxy in front of the
    service resets a request that writes nothing until retrieval + the first LLM
    round trip are done -- the production `ERR_HTTP2_PROTOCOL_ERROR` symptom),
  * `chunk` frames really arrive one by one instead of the answer being buffered
    and delivered in a single burst at the end,
  * `status` / `metadata` arrive before the tokens, so the UI can say what it is
    waiting for,
  * the `data: [DONE]` terminator is the last frame on the wire, i.e. the client
    never has to infer "the answer ended" from a closed connection.

Nothing in this module starts the app; every function talks HTTP to a base URL.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

# The frontend parser (`chat.service.ts -> processFrame`) only understands frames
# made of `data:` lines plus `:` comments; anything else on the wire is ignored,
# so the client here applies the same rule.
SSE_DATA_PREFIX = "data:"
SSE_COMMENT_PREFIX = ":"
DONE_MARKER = "[DONE]"

# Markers the chat page needs inside its lazy chunk. Kept next to the parser
# constants because they describe the same code path: the fetch-based SSE reader
# and the exact error codes it surfaces to the user.
FRONTEND_STREAM_MARKERS = (
    "text/event-stream",
    "api/chat/stream",
    "stream_incomplete",
    "stream_read_error",
)
# Dead fallbacks inside the Angular services (`environment.apiUrl || 'http://localhost:8000'`).
# They must never be the value actually used at runtime; the browser test proves
# that by looking at the requests the page really makes.
FRONTEND_DEAD_FALLBACK = "http://localhost:8000"


class SSEShapeError(AssertionError):
    """Raised when the response is not a well-formed SSE stream."""


@dataclass
class SSEFrame:
    """One frame off the wire, with the moment the client completed reading it."""

    arrival: float          # seconds since the request was sent
    kind: str               # "comment" | "data" | "empty" | "invalid"
    payload: str            # for `data:` frames: the joined payload, verbatim
    raw: str                # the frame exactly as it appeared on the wire

    @property
    def is_comment(self) -> bool:
        return self.kind == "comment"

    @property
    def is_data(self) -> bool:
        return self.kind == "data"

    def as_json(self) -> Optional[Dict[str, Any]]:
        """The decoded event, or None when the payload is not JSON (`[DONE]`)."""
        if not self.is_data:
            return None
        try:
            return json.loads(self.payload)
        except json.JSONDecodeError:
            return None

    def to_report(self) -> Dict[str, Any]:
        return {
            "arrival_s": round(self.arrival, 4),
            "kind": self.kind,
            "payload": self.payload if len(self.payload) <= 400 else self.payload[:400] + "...",
        }


@dataclass
class StreamRun:
    """Everything one `GET /api/chat/stream` call produced, timings included."""

    request_url: str
    message: str
    status_code: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    http_version: str = ""
    frames: List[SSEFrame] = field(default_factory=list)
    grammar_violations: List[str] = field(default_factory=list)
    t_first_byte: Optional[float] = None
    total_seconds: float = 0.0
    aborted: bool = False
    abort_reason: str = ""
    transport_error: Optional[str] = None

    # ---------- derived views ----------

    @property
    def data_frames(self) -> List[SSEFrame]:
        return [frame for frame in self.frames if frame.is_data]

    @property
    def comment_frames(self) -> List[SSEFrame]:
        return [frame for frame in self.frames if frame.is_comment]

    @property
    def json_events(self) -> List[Dict[str, Any]]:
        return [event for event in (frame.as_json() for frame in self.data_frames) if event]

    @property
    def event_types(self) -> List[str]:
        return [event.get("type") for event in self.json_events]

    @property
    def chunk_frames(self) -> List[Tuple[SSEFrame, str]]:
        """`(frame, content)` for every token frame, in arrival order."""
        pairs = ((frame, frame.as_json()) for frame in self.data_frames)
        return [(frame, event.get("content", "")) for frame, event in pairs if event
                and event.get("type") == "chunk"]

    @property
    def streamed_answer(self) -> str:
        return "".join(content for _, content in self.chunk_frames)

    @property
    def done_event(self) -> Optional[Dict[str, Any]]:
        for event in self.json_events:
            if event.get("type") == "done":
                return event
        return None

    @property
    def error_event(self) -> Optional[Dict[str, Any]]:
        for event in self.json_events:
            if event.get("type") == "error":
                return event
        return None

    @property
    def status_events(self) -> List[Dict[str, Any]]:
        return [event for event in self.json_events if event.get("type") == "status"]

    @property
    def metadata_event(self) -> Optional[Dict[str, Any]]:
        for event in self.json_events:
            if event.get("type") == "metadata":
                return event
        return None

    @property
    def saw_done_marker(self) -> bool:
        return any(frame.payload == DONE_MARKER for frame in self.data_frames)

    @property
    def last_frame(self) -> Optional[SSEFrame]:
        return self.frames[-1] if self.frames else None

    @property
    def answer(self) -> str:
        """Authoritative answer: the `done` frame's text, else the streamed tokens."""
        done = self.done_event
        if done and done.get("answer"):
            return done["answer"]
        return self.streamed_answer

    def first_arrival_of(self, event_type: str) -> Optional[float]:
        for frame in self.data_frames:
            event = frame.as_json()
            if event and event.get("type") == event_type:
                return frame.arrival
        return None

    def silent_gaps(self) -> List[Tuple[float, float]]:
        """`(gap_start, gap_end)` for every window with no bytes on the wire.

        Measured between consecutive frames, so a long silence that a heartbeat
        was supposed to cover shows up as one pair (e.g. `(2.1, 14.3)`).
        """
        gaps: List[Tuple[float, float]] = []
        previous = 0.0
        for frame in self.frames:
            if frame.arrival - previous > 0.0:
                gaps.append((round(previous, 3), round(frame.arrival, 3)))
            previous = frame.arrival
        return gaps

    def to_report(self) -> Dict[str, Any]:
        done = self.done_event or {}
        return {
            "request_url": self.request_url,
            "message": self.message,
            "status_code": self.status_code,
            "http_version": self.http_version,
            "headers": self.headers,
            "t_first_byte_s": round(self.t_first_byte, 4) if self.t_first_byte is not None else None,
            "total_seconds": round(self.total_seconds, 4),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "transport_error": self.transport_error,
            "frame_count": len(self.frames),
            "comment_frame_count": len(self.comment_frames),
            "event_types": [
                {"type": name, "arrival_s": round(self.first_arrival_of(name) or 0.0, 4)}
                for name in dict.fromkeys(self.event_types)
            ],
            "chunk_count": len(self.chunk_frames),
            "streamed_answer_len": len(self.streamed_answer),
            "answer_len": len(self.answer),
            "done": {
                "conversation_id": done.get("conversation_id"),
                "message_id": done.get("message_id"),
                "title": done.get("title"),
                "normalized_sources": len(done.get("normalized_sources") or []),
                "query_analysis_present": bool(done.get("query_analysis")),
                "conversation_history_len": len(done.get("conversation_history") or []),
            },
            "error_event": self.error_event,
            "grammar_violations": self.grammar_violations,
            "silent_gaps_s": self.silent_gaps(),
            "timeline": [frame.to_report() for frame in self.frames],
        }


def _classify_frame(frame: str) -> Tuple[str, str]:
    """Split one raw frame into `(kind, payload)`, mirroring `chat.service.ts`.

    Comment frames carry no `data:` line and are ignored by the frontend parser;
    they exist so proxies see bytes while the answer is still being produced.
    """
    data_lines: List[str] = []
    for line in frame.split("\n"):
        line = line.rstrip("\r")
        if line.startswith(SSE_DATA_PREFIX):
            data_lines.append(line[len(SSE_DATA_PREFIX):].lstrip())
    if data_lines:
        return "data", "\n".join(data_lines)
    if not frame.strip():
        return "empty", ""
    if frame.strip().startswith(SSE_COMMENT_PREFIX):
        return "comment", ""
    return "invalid", frame


def stream_chat(
    client: httpx.Client,
    base_url: str,
    *,
    message: str,
    user_id: str,
    session_id: str = "",
    user_email: str = "",
    timeout: float = 240.0,
    stop_after_frames: Optional[int] = None,
    stop_after_events: Optional[Sequence[str]] = None,
) -> StreamRun:
    """Drive one real streamed chat turn and record its wire timeline.

    `stop_after_frames` / `stop_after_events` deliberately hang up in the middle
    of a turn -- that is how the cancel-safety test proves a client that goes away
    (closed tab, proxy reset) does not wedge the worker.
    """
    url = f"{base_url.rstrip('/')}/api/chat/stream"
    params = {
        "user_id": user_id,
        "message": message,
        "session_id": session_id,
        "user_email": user_email,
    }
    run = StreamRun(request_url=str(httpx.URL(url, params=params)), message=message)

    started = time.perf_counter()
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    seen_events: List[str] = []

    try:
        with client.stream(
            "GET",
            url,
            params=params,
            timeout=timeout,
            headers={"accept": "text/event-stream"},
        ) as response:
            run.status_code = response.status_code
            run.headers = {key.lower(): value for key, value in response.headers.items()}
            run.http_version = response.http_version

            for raw in response.iter_bytes():
                if run.t_first_byte is None:
                    run.t_first_byte = time.perf_counter() - started
                arrival = time.perf_counter() - started
                buffer += decoder.decode(raw, False)

                separator = buffer.find("\n\n")
                while separator != -1:
                    raw_frame = buffer[:separator]
                    buffer = buffer[separator + 2:]
                    kind, payload = _classify_frame(raw_frame)
                    run.frames.append(
                        SSEFrame(arrival=arrival, kind=kind, payload=payload, raw=raw_frame)
                    )

                    if kind == "invalid":
                        run.grammar_violations.append(
                            f"frame at {arrival:.2f}s has no `data:` line and is not a `:` "
                            f"comment: {raw_frame[:120]!r}"
                        )
                    elif kind == "data":
                        event = run.frames[-1].as_json()
                        if event:
                            seen_events.append(event.get("type", ""))

                    stop = False
                    if stop_after_frames is not None and len(run.frames) >= stop_after_frames:
                        run.aborted = True
                        run.abort_reason = f"client hung up after {len(run.frames)} frame(s)"
                        stop = True
                    if stop_after_events and all(n in seen_events for n in stop_after_events):
                        run.aborted = True
                        run.abort_reason = (
                            f"client hung up after events {sorted(set(stop_after_events))}"
                        )
                        stop = True
                    if stop:
                        break

                    separator = buffer.find("\n\n")
                if run.aborted:
                    break
    except httpx.HTTPError as exc:  # connection reset, read timeout, ...
        run.transport_error = f"{type(exc).__name__}: {exc}"

    run.total_seconds = time.perf_counter() - started
    return run


def stream_headers_are_proxy_safe(run: StreamRun) -> List[str]:
    """Header problems that would let a proxy buffer or break the stream (hard failures).

    `content-type`, `charset=utf-8` and `cache-control` are the ones a client can rely
    on: without them the browser (or a cache) mishandles the response. See
    `stream_header_warnings` for the advisory half.
    """
    problems: List[str] = []
    content_type = run.headers.get("content-type", "")
    if not content_type.startswith("text/event-stream"):
        problems.append(f"content-type is {content_type!r}, not text/event-stream")
    if "charset=utf-8" not in content_type.lower():
        problems.append(
            f"content-type {content_type!r} does not pin UTF-8; Persian answers are multi-byte"
        )
    if run.headers.get("cache-control") != "no-cache":
        problems.append(f"cache-control is {run.headers.get('cache-control')!r}, not 'no-cache'")
    encoding = run.headers.get("content-encoding")
    if encoding and encoding != "identity":
        problems.append(
            f"response is content-encoded ({encoding!r}); a compressing layer can hold the "
            "tokens until the stream ends"
        )
    return problems


def stream_header_warnings(run: StreamRun) -> List[str]:
    """Advisory header notes: real, but not proof that the stream is broken.

    `X-Accel-Buffering: no` is written by the route for the nginx in front of the worker,
    but a CDN edge (this deployment is served through ArvanCloud) is free to strip it --
    as it is free to strip any hop-by-hop-ish header. Its absence therefore cannot fail a
    test; whether the answer is really streamed is decided by the *arrival timing* of the
    chunk frames (`test_tokens_are_delivered_incrementally_not_in_one_burst`), which no
    header can fake.
    """
    warnings: List[str] = []
    accel = run.headers.get("x-accel-buffering")
    if accel is None:
        warnings.append(
            "x-accel-buffering is absent from the response (expected if a CDN edge strips "
            "it; the incremental token arrival test is what proves nothing buffers)"
        )
    elif accel != "no":
        warnings.append(f"x-accel-buffering is {accel!r}, not 'no'")
    if run.headers.get("connection") and run.http_version == "HTTP/1.1":
        warnings.append(
            f"connection: {run.headers.get('connection')!r} "
            f"(keep-alive: {run.headers.get('keep-alive')!r})"
        )
    return warnings


def sse_shape_problems(run: StreamRun) -> List[str]:
    """SSE-grammar problems that the deployed frontend parser would choke on."""
    problems = list(run.grammar_violations)
    if not run.frames:
        problems.append("no frames at all were received")
        return problems
    if not run.frames[0].is_comment:
        problems.append(
            "the stream does not open with a `: ping` comment frame, so nothing is written "
            "before retrieval + the first LLM round trip run"
        )
    if not run.saw_done_marker:
        problems.append("the `data: [DONE]` end-of-stream marker was never written")
    elif run.last_frame is None or run.last_frame.payload != DONE_MARKER:
        problems.append(
            "the stream did not end with `data: [DONE]`; last frame was "
            f"{run.last_frame.raw[:80]!r}"
        )
    done_count = len([e for e in run.json_events if e.get("type") == "done"])
    if run.error_event is None and done_count != 1:
        problems.append(
            f"expected exactly one `done` event (or one `error`), got {run.event_types}"
        )
    if run.error_event is None:
        metadata_arrival = run.first_arrival_of("metadata")
        first_chunk_arrival = run.first_arrival_of("chunk")
        if metadata_arrival is not None and first_chunk_arrival is not None:
            if metadata_arrival > first_chunk_arrival:
                problems.append("`metadata` arrived after the first answer token")
    return problems


# ==================== deployed-frontend inspection ====================

_SCRIPT_SRC = re.compile(r'src="([^"]+\.js)"')
_CHUNK_REF = re.compile(r"chunk-[A-Z0-9]+\.js")
_ABSOLUTE_URL = re.compile(r"https?://[A-Za-z0-9._:-]+")

# Angular nests lazy chunks (the chat page is a chunk reachable only through another
# chunk), so the crawl follows references transitively. Bounded so a cyclic reference
# cannot turn into an infinite fetch loop.
_MAX_CHUNK_DEPTH = 5


@dataclass
class DeployedAsset:
    path: str            # path as referenced by the page, e.g. "main-222NEOAE.js"
    status_code: int
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


@dataclass
class DeployedFrontend:
    index_html: str
    scripts: List[DeployedAsset]
    lazy_chunks: Dict[str, DeployedAsset]
    absolute_urls: List[str]

    def find_asset_with(self, marker: str) -> Optional[str]:
        """Path of the first deployed asset containing `marker` (entry bundles first)."""
        for asset in self.scripts:
            if marker in asset.text:
                return asset.path
        for path, asset in self.lazy_chunks.items():
            if marker in asset.text:
                return path
        return None

    def to_report(self) -> Dict[str, Any]:
        return {
            "scripts": [
                {"path": a.path, "status": a.status_code, "sha256": a.sha256, "bytes": len(a.content)}
                for a in self.scripts
            ],
            "lazy_chunks": {
                path: {"status": a.status_code, "sha256": a.sha256, "bytes": len(a.content)}
                for path, a in self.lazy_chunks.items()
            },
            "absolute_urls": self.absolute_urls,
        }


def fetch_deployed_frontend(client: httpx.Client, base_url: str) -> DeployedFrontend:
    """Fetch the SPA shell, its entry bundles and every lazy chunk reachable from them.

    Follows references the way a browser does: `index.html` -> entry scripts -> the
    `chunk-*.js` names those scripts ask for, then the chunks *those* chunks ask for
    (Angular nests them: the chat page is only referenced two levels down), up to
    `_MAX_CHUNK_DEPTH`. Everything is kept as bytes so hashes are comparable with the
    local build.
    """
    base = base_url.rstrip("/")
    index = client.get(f"{base}/", timeout=30.0)
    index.raise_for_status()
    index_html = index.text

    scripts: List[DeployedAsset] = []
    for src in _SCRIPT_SRC.findall(index_html):
        url = src if src.startswith("http") else f"{base}/{src.lstrip('/')}"
        asset = client.get(url, timeout=60.0)
        scripts.append(DeployedAsset(path=src, status_code=asset.status_code, content=asset.content))

    chunk_names = sorted({name for asset in scripts for name in _CHUNK_REF.findall(asset.text)})
    lazy_chunks: Dict[str, DeployedAsset] = {}
    frontier = list(chunk_names)
    for _depth in range(_MAX_CHUNK_DEPTH):
        pending = [name for name in frontier if name not in lazy_chunks]
        if not pending:
            break
        discovered: set = set()
        for name in pending:
            asset = client.get(f"{base}/{name}", timeout=60.0)
            deployed = DeployedAsset(path=name, status_code=asset.status_code, content=asset.content)
            lazy_chunks[name] = deployed
            discovered.update(_CHUNK_REF.findall(deployed.text))
        frontier = sorted(discovered - set(lazy_chunks))

    texts = [index_html] + [a.text for a in scripts] + [a.text for a in lazy_chunks.values()]
    absolute_urls = sorted({url for text in texts for url in _ABSOLUTE_URL.findall(text)})
    return DeployedFrontend(
        index_html=index_html, scripts=scripts, lazy_chunks=lazy_chunks, absolute_urls=absolute_urls
    )


def local_build_hashes(dist_root: str) -> Dict[str, str]:
    """`{file name: sha256}` for `frontend/ai-panel/dist/ai-panel/browser`."""
    hashes: Dict[str, str] = {}
    if not os.path.isdir(dist_root):
        return hashes
    for name in os.listdir(dist_root):
        full = os.path.join(dist_root, name)
        if os.path.isfile(full):
            with open(full, "rb") as handle:
                hashes[name] = hashlib.sha256(handle.read()).hexdigest()
    return hashes


def login(client: httpx.Client, base_url: str, username: str, password: str) -> Dict[str, Any]:
    """`POST /api/users/login`, exactly as `auth.service.ts` does it."""
    response = client.post(
        f"{base_url.rstrip('/')}/api/users/login",
        json={"username": username, "password": password},
        timeout=30.0,
    )
    if response.status_code != 200:
        raise AssertionError(
            f"login failed with HTTP {response.status_code}: {response.text[:300]}"
        )
    payload = response.json()
    if not payload.get("access_token"):
        raise AssertionError(f"login response has no access_token: {list(payload)}")
    return payload


# ==================== probe + evidence ====================

def write_report(evidence_dir, name: str, payload: Dict[str, Any]):
    """Persist one probe's evidence JSON; a read-only disk must not fail a probe."""
    import pathlib

    target = pathlib.Path(evidence_dir) / f"{name}.json"
    try:
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:  # evidence is not a verdict
        return target
    return target


def run_probe(
    client: httpx.Client,
    base_url: str,
    *,
    message: str,
    user_id: str,
    session_id: str = "",
    user_email: str = "",
    timeout: float = 240.0,
    evidence_dir=None,
    name: str = "probe",
    **kwargs: Any,
) -> StreamRun:
    """One probe turn + its evidence file. The single entry point used by all tests."""
    run = stream_chat(
        client,
        base_url,
        message=message,
        user_id=user_id,
        session_id=session_id,
        user_email=user_email,
        timeout=timeout,
        **kwargs,
    )
    if evidence_dir is not None:
        write_report(evidence_dir, name, run.to_report())
    return run
