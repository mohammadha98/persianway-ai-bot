"""Gate, fixtures and reporting for the live (deployed-service) E2E tests.

Why a separate gate
-------------------
These tests talk to a **running production deployment**. They spend real LLM
tokens, occupy the single worker that serves real users and write a conversation
row, so they must never run as part of `pytest tests`. Everything in this package
therefore requires an explicit opt-in and skips with an actionable message
otherwise:

    E2E_LIVE=1                  # or: pytest tests/e2e_live --live
    E2E_LIVE_HEAVY=1            # additionally enables the concurrent-stream test
    E2E_BASE_URL=...            # default: https://crm.persianway.ir
    E2E_USER_ID=live-e2e-probe  # which user id the probe conversations belong to
    E2E_USER_EMAIL=...          # optional
    E2E_MESSAGE=...             # optional: override the probe question
    E2E_EXPECT_BUFFERED=1       # known edge buffering: timing tests xfail instead of fail
    E2E_USERNAME / E2E_PASSWORD # only needed for the browser test (real login)
    E2E_BROWSER_PATH=...        # optional: Edge/Chrome binary for the browser test

Cost and risk profile per test lives in `docs/LIVE_E2E_STREAMING_TESTS.md`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator

import httpx
import pytest

from tests.e2e_live import live_client as live_api
from tests.e2e_live.live_client import StreamRun

# The deployment under test. Overridable, never guessed at runtime.
DEFAULT_BASE_URL = "https://crm.persianway.ir"

# The probe question: short (cheap), Persian (exercises the UTF-8 path) and
# retrieval-relevant, so a normal turn (intent -> rewrite -> retrieve -> answer)
# really happens.
DEFAULT_MESSAGE = "سلام، به صورت کوتاه بگویید چگونه می‌توانم از خدمات مشاوره استفاده کنم؟"

# The wire contract this suite guards: the open frame must be written before any
# slow work, so the client (and every proxy) sees a byte almost immediately. The
# in-process contract test `tests/api/test_chat_stream_sse.py` pins the same
# budget at the ASGI level; here it is pinned through the real TLS endpoint.
TTFB_BUDGET_SECONDS = 2.0

# One streamed answer is allowed this long before the test fails. Deliberately
# generous: the live service shares the provider rate limit with real users.
ANSWER_TIMEOUT_SECONDS = 240.0

# Heartbeats are written every `SSE_HEARTBEAT_INTERVAL_SECONDS` (10s) while the
# answer is silent; anything longer than this between two frames means the
# keep-alive did not reach the client and a proxy may reset the stream.
MAX_SILENT_GAP_SECONDS = 15.0

# `app/api/routes/chat.py` bounds conversation-title generation with
# `TITLE_GENERATION_TIMEOUT_SECONDS` (15s), and that call happens *before* the
# `StreamingResponse` is returned -- so the first request of a brand-new session
# cannot write a byte sooner than the title call returns. A measured first byte near
# this budget means the title call timed out (the route then falls back to "New
# Conversation") and the user waited ~15s in front of a silent spinner.
TITLE_GENERATION_TIMEOUT_SECONDS = 15.0
TITLE_BUDGET_SECONDS = TITLE_GENERATION_TIMEOUT_SECONDS + 2.0


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else default


def _live_enabled(config: pytest.Config) -> bool:
    return bool(config.getoption("--live")) or _env_flag("E2E_LIVE")


def note_buffering(live_settings: "LiveSettings", detail: str) -> None:
    """Turn a *known deployment-level buffering failure* into an expected failure.

    The live deployment sits behind a CDN edge (ArvanCloud) which does not forward the
    response body as it is written: measured on 2026-09-21, a whole turn arrives in one or
    two bursts (e.g. the open frame and the retrieval notification landing in the same
    flush, then all 81 token frames in a single flush at the end), and the first byte of a
    turn reaches the client after 15-33s. The origin still writes the frames incrementally
    (proved in-process by `tests/api/test_chat_stream_sse.py`), so the assertions below --
    first byte early, tokens one by one, no long silence -- cannot hold until the edge is
    configured to stream.

    Set `E2E_EXPECT_BUFFERED=1` to mark these as expected failures (a green run while the
    deployment issue is open) instead of a red one that everyone learns to ignore.
    """
    if live_settings.expect_buffered:
        pytest.xfail(
            "known deployment buffering (E2E_EXPECT_BUFFERED=1): " + detail
        )


@dataclass
class LiveSettings:
    """Everything the live tests need to know about *where* and *how* to probe."""

    base_url: str
    user_id: str
    user_email: str
    message: str
    username: str
    password: str
    browser_path: str
    heavy: bool
    ttfb_budget: float = TTFB_BUDGET_SECONDS
    title_budget: float = TITLE_BUDGET_SECONDS
    answer_timeout: float = ANSWER_TIMEOUT_SECONDS
    max_silent_gap: float = MAX_SILENT_GAP_SECONDS
    # `E2E_EXPECT_BUFFERED=1`: the deployment is known not to stream through its CDN edge,
    # so the timing assertions become expected failures instead of red tests.
    expect_buffered: bool = False
    # Filled in once per pytest run: probe turns must never join a real thread.
    session_id: str = "live-e2e-session"

    def probe_message(self, suffix: str = "") -> str:
        """The probe question, optionally tagged so it is findable in the panel."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"[live-e2e {self.user_id} {stamp}{(' ' + suffix) if suffix else ''}] "
        return tag + self.message


@pytest.fixture(scope="session")
def live_settings(request: pytest.FixtureRequest) -> LiveSettings:
    """Resolved live-target settings; skips the whole package when not opted in."""
    config = request.config
    if not _live_enabled(config):
        pytest.skip(
            "live tests are off: set E2E_LIVE=1 (or pass --live) to run them against "
            "the deployed service; they spend LLM tokens and touch production data"
        )
    base_url = (config.getoption("--live-base-url") or os.getenv("E2E_BASE_URL")
                or DEFAULT_BASE_URL).rstrip("/")
    settings = LiveSettings(
        base_url=base_url,
        user_id=os.getenv("E2E_USER_ID", "live-e2e-probe"),
        user_email=os.getenv("E2E_USER_EMAIL", ""),
        message=os.getenv("E2E_MESSAGE", DEFAULT_MESSAGE),
        username=os.getenv("E2E_USERNAME", ""),
        password=os.getenv("E2E_PASSWORD", ""),
        browser_path=os.getenv("E2E_BROWSER_PATH", ""),
        heavy=_env_flag("E2E_LIVE_HEAVY") or bool(config.getoption("--live-heavy")),
        expect_buffered=_env_flag("E2E_EXPECT_BUFFERED"),
    )
    # One session id per pytest run, so every probe turn of this run lands in the
    # same (obviously synthetic) conversation instead of a real user's thread.
    settings.session_id = f"live-e2e-{int(time.time())}"
    return settings


@pytest.fixture(scope="session")
def live_client() -> Iterator[httpx.Client]:
    """HTTP client for the deployment, with a browser-like fingerprint.

    Per-request timeouts are passed explicitly where they matter (a streamed
    answer legitimately takes minutes; a health probe must not), so the client
    default stays short enough to fail a hung connection fast.
    """
    with httpx.Client(
        timeout=httpx.Timeout(connect=15.0, read=30.0, write=30.0, pool=30.0),
        follow_redirects=True,
        headers={"user-agent": "persianway-live-e2e/1.0 (+pytest)"},
    ) as client:
        yield client


@pytest.fixture(scope="session")
def report_dir() -> Path:
    """`reports/live_e2e_<timestamp>/`: raw evidence for every probe this run made."""
    directory = _env_path("E2E_REPORT_DIR", Path(__file__).resolve().parents[2] / "reports")
    if directory.name != "reports":
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    run_dir = directory / f"live_e2e_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_report(report_dir: Path, name: str, payload: Dict[str, Any]) -> Any:
    """Persist one probe's evidence (thin re-export of the client helper).

    Namespaced as `live_api`: the module-level name `live_client` in this file is the
    fixture below, so the helper module must not be referenced through it.
    """
    return live_api.write_report(report_dir, name, payload)


def _run_and_record(
    client: httpx.Client,
    settings: LiveSettings,
    report_dir: Path,
    *,
    name: str,
    message: str,
    **kwargs: Any,
) -> StreamRun:
    return live_api.run_probe(
        client,
        settings.base_url,
        message=message,
        user_id=settings.user_id,
        session_id=settings.session_id,
        user_email=settings.user_email,
        timeout=settings.answer_timeout,
        evidence_dir=report_dir,
        name=name,
        **kwargs,
    )


@pytest.fixture(scope="session")
def primary_stream(
    live_client: httpx.Client,
    live_settings: LiveSettings,
    report_dir: Path,
) -> StreamRun:
    """**The one expensive probe**: a real streamed turn, shared by its assertions.

    Every assertion about the wire contract (headers, open frame, ordering,
    incremental delivery, encoding, `done`) is made against this single request
    instead of firing a request per assertion, so a full run costs a handful of
    LLM turns rather than dozens.

    Evidence: `reports/live_e2e_<ts>/primary_stream.json`.
    """
    return _run_and_record(
        live_client,
        live_settings,
        report_dir,
        name="primary_stream",
        message=live_settings.probe_message("primary"),
    )


@pytest.fixture(scope="session")
def live_heavy(live_settings: LiveSettings) -> None:
    """Opt-in guard for the tests that put real load on the production worker."""
    if not live_settings.heavy:
        pytest.skip(
            "heavy live test is off: set E2E_LIVE_HEAVY=1 (or pass --live-heavy). It runs "
            "concurrent chats against the single production worker, which can slow real "
            "users' turns down while it runs"
        )


@pytest.fixture(scope="session")
def browser_credentials(live_settings: LiveSettings) -> Dict[str, str]:
    """Login credentials for the real-browser test, or a skip with instructions."""
    if not (live_settings.username and live_settings.password):
        pytest.skip(
            "browser login is off: set E2E_USERNAME and E2E_PASSWORD (a throwaway panel "
            "account with the Chat permission) to run the real-browser streaming test"
        )
    return {"username": live_settings.username, "password": live_settings.password}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("live-e2e")
    group.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run the tests that talk to the deployed service (same as E2E_LIVE=1)",
    )
    group.addoption(
        "--live-heavy",
        action="store_true",
        default=False,
        help="additionally run the concurrent-stream test (same as E2E_LIVE_HEAVY=1)",
    )
    group.addoption(
        "--live-base-url",
        action="store",
        default=None,
        help=f"deployment under test (default: E2E_BASE_URL or {DEFAULT_BASE_URL})",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: talks to the deployed service (needs E2E_LIVE=1)")
    config.addinivalue_line("markers", "live_llm: spends real LLM tokens on the deployment")
    config.addinivalue_line("markers", "live_heavy: loads the production worker (E2E_LIVE_HEAVY=1)")
    config.addinivalue_line("markers", "live_browser: drives a real browser (needs credentials)")
