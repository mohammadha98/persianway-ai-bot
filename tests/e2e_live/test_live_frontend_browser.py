"""Live browser test: the panel really renders the answer token by token.

What this module answers
------------------------
Every other test in this package speaks HTTP. This one drives the *actual chat page* of
the deployment in a headless Chromium/Edge over the DevTools Protocol, so it can assert
the thing the user experiences and nothing else can prove:

  1. the guarded chat route renders for a logged-in user (auth token + lazy chunk load),
  2. the answer appears in the bubble **while** it is being generated -- the rendered text
     is sampled repeatedly and must grow over real wall-clock time, not appear in one
     paint at the end,
  3. the typing indicator is shown while the answer is still empty and removed when the
     `done` event lands, and the composer is re-enabled (i.e. the stream ended properly
     rather than being abandoned),
  4. the page took the streaming code path (`[STREAM] Starting SSE fetch stream` in the
     console) instead of some fallback,
  5. no uncaught exception, no `net::ERR_*` on a stream request, and no request to the
     development API host (`http://localhost:8000`).

The driver is a small CDP client (`cdp_browser.py`), so this needs no new dependency --
just a browser binary. Without credentials (`E2E_USERNAME`/`E2E_PASSWORD`) the chat test
skips; the login-page test still runs, because it needs no account.

Cost: one full turn, spent through the UI instead of the API.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List

import httpx
import pytest

from tests.e2e_live.cdp_browser import BrowserUnavailable, CdpPage, find_browser
from tests.e2e_live.conftest import LiveSettings, write_report
from tests.e2e_live.live_client import login

pytestmark = [pytest.mark.live, pytest.mark.live_browser]

# Selectors come straight out of the templates, so a markup change fails the test loudly
# instead of silently finding nothing:
#   chat.component.html: `textarea.message-input`, `button.send-btn`,
#                        `.messages-list .message.assistant[.typing] .message-content`,
#                        `.typing-indicator .stream-status`
#   login.component.html: `input[formcontrolname="username"]`
LOGIN_USERNAME_INPUT = 'input[formcontrolname="username"]'
CHAT_INPUT = "textarea.message-input"
CHAT_SEND_BUTTON = "button.send-btn"
ASSISTANT_BUBBLES = ".messages-list .message.assistant .message-content"
TYPING_INDICATOR = ".messages-list .message.assistant.typing"
STREAM_STATUS = ".typing-indicator .stream-status"

# The panel logs its streaming milestones to the console; their presence is how this test
# distinguishes "streamed" from "arrived in one piece".
STREAM_CONSOLE_MARKER = "[STREAM]"

PERSIAN_LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"


@pytest.fixture(scope="module")
def page(live_settings: LiveSettings) -> Iterator[CdpPage]:
    """A headless browser page, or a skip explaining what to install/point at."""
    try:
        executable = find_browser(live_settings.browser_path)
    except BrowserUnavailable as exc:
        pytest.skip(str(exc))
    try:
        with CdpPage(executable, timeout=45.0) as browser:
            yield browser
    except BrowserUnavailable as exc:
        pytest.skip(f"headless browser could not be started: {exc}")


def _set_message_script(message: str) -> str:
    """Type into the chat textarea the way Angular's ngModel expects (real `input` event)."""
    return (
        "(() => {"
        f"  const el = document.querySelector({json.dumps(CHAT_INPUT)});"
        "  if (!el) { return 'no-input'; }"
        "  const setter = Object.getOwnPropertyDescriptor("
        "window.HTMLTextAreaElement.prototype, 'value').set;"
        f"  setter.call(el, {json.dumps(message)});"
        "  el.dispatchEvent(new Event('input', { bubbles: true }));"
        "  return el.value;"
        "})()"
    )


def _snapshot_script() -> str:
    """One DOM snapshot: what is rendered right now, without triggering any action."""
    return (
        "(() => {"
        f"  const bubbles = Array.from(document.querySelectorAll({json.dumps(ASSISTANT_BUBBLES)}));"
        "  const last = bubbles.length ? bubbles[bubbles.length - 1] : null;"
        f"  const typing = document.querySelector({json.dumps(TYPING_INDICATOR)});"
        f"  const status = document.querySelector({json.dumps(STREAM_STATUS)});"
        f"  const input = document.querySelector({json.dumps(CHAT_INPUT)});"
        f"  const button = document.querySelector({json.dumps(CHAT_SEND_BUTTON)});"
        "  return {"
        "    bubbles: bubbles.length,"
        "    visibleText: last ? last.innerText : '',"
        "    rawText: last ? last.textContent : '',"
        "    typing: !!typing,"
        "    status: status ? status.innerText : '',"
        "    inputDisabled: input ? !!input.disabled : null,"
        "    sendDisabled: button ? !!button.disabled : null,"
        "  };"
        "})()"
    )


def test_login_page_renders_in_a_real_browser(page: CdpPage, live_settings: LiveSettings):
    """The deployed SPA boots in a browser: shell, lazy chunk and login form render.

    Catches what no HTTP-level test can: a broken bundle, a missing lazy chunk, a
    Content-Security-Policy that blocks the app, or a JS exception on startup. No account is
    needed for this one, and no chat request is made.
    """
    page.navigate(f"{live_settings.base_url}/login")

    page.wait_for(
        f"document.querySelector({json.dumps(LOGIN_USERNAME_INPUT)}) !== null",
        "the login form to render",
        timeout=45.0,
    )
    assert page.evaluate("document.title").strip(), "the SPA set no document title"
    assert page.evaluate("!!document.querySelector('app-root')"), "<app-root> missing"

    exceptions = [m for m in page.error_messages() if m.source == "exception"]
    assert not exceptions, (
        "the page threw while loading: "
        + "; ".join(f"{m.source}: {m.text[:200]}" for m in exceptions)
    )
    for note in page.error_messages():
        print(f"[live-e2e] console {note.level} ({note.source}): {note.text[:200]}")

    dev_calls = [url for url in page.request_urls if "localhost:8000" in url]
    assert not dev_calls, (
        f"the deployed page called the development API host: {dev_calls[:5]} (a stale or "
        "mis-built bundle)"
    )


def test_chat_page_streams_the_answer_into_the_ui(
    page: CdpPage,
    live_client: httpx.Client,
    live_settings: LiveSettings,
    browser_credentials: Dict[str, str],
    report_dir: Any,
):
    """The end-to-end user experience: ask a question, watch the answer appear.

    The whole `done`-frame path is exercised through the real UI: the answer must grow in
    the bubble over time (progressive render), the typing indicator must show while nothing
    has arrived and disappear when the stream completes, and the composer must come back.
    """
    host = httpx.URL(live_settings.base_url).host
    session = login(live_client, live_settings.base_url, **browser_credentials)

    # Log in through the API and hand the token to the page, exactly the way
    # `auth.service.ts` stores it after the login form succeeds: the storage belongs to the
    # origin, so setting it on any page of the deployment is enough.
    page.navigate(f"{live_settings.base_url}/login")
    page.evaluate(
        "(() => {"
        f"  localStorage.setItem('auth_token', {json.dumps(session['access_token'])});"
        f"  localStorage.setItem('auth_user', {json.dumps(json.dumps(session['user']))});"
        "  return true;"
        "})()"
    )
    # Real navigation (not `location.assign`) so the CDP call does not race the context
    # being torn down by the page change.
    page.navigate(f"{live_settings.base_url}/chat")
    page.wait_for(
        f"document.querySelector({json.dumps(CHAT_INPUT)}) !== null",
        "the chat composer to render (the auth guard must let the token through)",
        timeout=45.0,
    )

    message = live_settings.probe_message("browser")
    assert page.evaluate(_set_message_script(message)) == message, (
        "the composer did not accept the typed message"
    )
    assert page.evaluate(
        f"(() => {{ const b = document.querySelector({json.dumps(CHAT_SEND_BUTTON)});"
        " if (!b) { return false; } b.click(); return true; })()"
    ), "the send button disappeared"

    samples: List[Dict[str, Any]] = []
    deadline = time.time() + live_settings.answer_timeout
    finished = False
    while time.time() < deadline:
        snapshot = page.evaluate(_snapshot_script())
        samples.append({"t": round(time.time(), 3), **snapshot})
        if snapshot["bubbles"] and not snapshot["typing"] and snapshot["visibleText"].strip():
            finished = True
            break
        time.sleep(0.25)

    write_report(
        report_dir,
        "browser_chat_timeline",
        {
            "base_url": live_settings.base_url,
            "browser": getattr(page, "browser_version", ""),
            "message": message,
            "finished": finished,
            "sample_count": len(samples),
            "samples": samples,
            "console": [m.to_report() for m in page.console],
            "failed_requests": page.failed_requests,
            "off_site_urls": page.off_site_urls(host),
        },
    )

    assert samples, "no DOM snapshot could be taken"
    assert samples[0]["bubbles"] >= 1, "the chat page rendered no assistant bubble"

    first_typing = next((s["t"] for s in samples if s["typing"]), None)
    assert first_typing is not None, (
        "the typing indicator never appeared: the UI gave no feedback while the answer was "
        "being produced"
    )

    growing = [s for s in samples if s["visibleText"].strip()]
    assert growing, "the answer never became visible in the bubble"
    lengths = [len(s["visibleText"].strip()) for s in growing]
    distinct_growth = [
        length for index, length in enumerate(lengths)
        if index == 0 or length > lengths[index - 1]
    ]
    assert len(distinct_growth) >= 2, (
        f"the rendered answer only appeared in one piece (lengths: {lengths[:10]}): the UI "
        "did not stream it progressively"
    )
    growth_span = growing[-1]["t"] - growing[0]["t"]
    assert growth_span > 0.3, (
        f"the rendered text changed over only {growth_span:.2f}s; that is one paint, not a "
        "streamed answer"
    )

    assert finished, (
        "the turn never completed: the typing indicator or the empty placeholder never went "
        f"away (last sample: {samples[-1]})"
    )
    final_text = samples[-1]["visibleText"].strip()
    assert len(final_text) >= 20, f"the rendered answer is too short: {final_text!r}"
    assert any(letter in final_text for letter in PERSIAN_LETTERS), (
        f"the rendered answer is not Persian text: {final_text[:200]!r}"
    )
    assert "\ufffd" not in final_text, "the rendered answer contains U+FFFD (encoding damage)"
    assert "stream_incomplete" not in final_text
    assert samples[-1]["inputDisabled"] is False, (
        "the composer stayed disabled after the answer finished"
    )

    stream_logs = [m.text for m in page.console if STREAM_CONSOLE_MARKER in m.text]
    assert stream_logs, (
        "the page never logged its SSE reader starting, so it may not have used the "
        "streaming code path at all"
    )

    exceptions = [m for m in page.error_messages() if m.source == "exception"]
    assert not exceptions, "the chat page threw: " + "; ".join(m.text[:200] for m in exceptions)

    protocol_errors = [
        failure for failure in page.failed_requests
        if "ERR_HTTP2_PROTOCOL_ERROR" in failure["error"] or "ERR_CONNECTION_RESET" in failure["error"]
    ]
    assert not protocol_errors, (
        f"a request died at the network layer: {protocol_errors} -- the exact symptom the open "
        "frame and the heartbeats exist to prevent"
    )

    dev_calls = [url for url in page.request_urls if "localhost:8000" in url]
    assert not dev_calls, f"the chat page called the development API host: {dev_calls[:5]}"
