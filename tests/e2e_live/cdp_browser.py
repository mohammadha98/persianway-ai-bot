"""Minimal Chrome DevTools Protocol driver for the live browser test.

Why not `Playwright`/`Selenium`?
--------------------------------
This repository pins its Python dependencies in `requirements.txt` and has no browser
automation installed; adding one just for this suite would drag a browser download and a
new dependency into a service image whose only job is to answer chat requests. The CDP
itself needs nothing but a browser binary and a websocket client (`websocket-client` is
already a transitive dependency here), so this module speaks the protocol directly:

    launch a headless Chromium/Edge  ->  open a tab via the HTTP endpoint
    ->  connect to the tab's websocket  ->  Page/Runtime/Log/Network domains
    ->  `Runtime.evaluate` for DOM assertions, collected console + network logs.

The two things the live tests need from it:

  * **observe the DOM while the answer streams** (`evaluate` polling the rendered text),
    which is the only way to prove the *frontend* renders token by token rather than in
    one paint at the end;
  * **see what the page really did** -- console errors and request URLs -- which is how a
    stray `http://localhost:8000` call or a `net::ERR_HTTP2_PROTOCOL_ERROR` surfaces.

Only the CDP surface needed by these tests is implemented; unknown events are collected,
not interpreted.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import websocket  # websocket-client


class BrowserUnavailable(RuntimeError):
    """No usable browser binary/websocket stack: the browser test must skip, not fail."""


# Chrome/Edge installations worth trying, in order. `E2E_BROWSER_PATH` wins.
BROWSER_CANDIDATES: List[str] = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Chromium\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

DEFAULT_WINDOW = (1440, 900)


def find_browser(explicit: str = "") -> str:
    """Absolute path of a Chromium-based browser, or raise `BrowserUnavailable`."""
    if explicit and os.path.exists(explicit):
        return explicit
    if explicit:
        raise BrowserUnavailable(f"E2E_BROWSER_PATH={explicit!r} does not exist")
    for candidate in BROWSER_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    for name in ("chrome", "msedge", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise BrowserUnavailable(
        "no Chromium-based browser found: install Chrome/Edge or point E2E_BROWSER_PATH "
        "at the executable to run the browser test"
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@dataclass
class ConsoleMessage:
    level: str
    text: str
    source: str = ""

    def to_report(self) -> Dict[str, str]:
        return {"level": self.level, "text": self.text, "source": self.source}


class CdpPage:
    """One headless page: `with CdpPage(...) as page: page.navigate(url)`."""

    def __init__(
        self,
        executable: str = "",
        *,
        headless: bool = True,
        window: tuple = DEFAULT_WINDOW,
        timeout: float = 30.0,
    ) -> None:
        self.executable = find_browser(executable)
        self.headless = headless
        self.window = window
        self.timeout = timeout

        self.process: Optional[subprocess.Popen] = None
        self.profile_dir: Optional[str] = None
        self.port: int = 0
        self.target: Dict[str, Any] = {}
        self.ws: Optional[websocket.WebSocket] = None
        self._next_id = 1

        self.console: List[ConsoleMessage] = []
        self.failed_requests: List[Dict[str, str]] = []
        self.request_urls: List[str] = []
        self.responses: List[Dict[str, Any]] = []

    # ---------- lifecycle ----------

    def __enter__(self) -> "CdpPage":
        self.launch()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def launch(self) -> None:
        self.profile_dir = tempfile.mkdtemp(prefix="persianway-e2e-browser-")
        self.port = _free_port()
        args = [
            self.executable,
            "--headless=new" if self.headless else "--start-maximized",
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile_dir}",
            f"--window-size={self.window[0]},{self.window[1]}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--disable-sync",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            # The DevTools endpoint is bound to 127.0.0.1 in this process only; Chromium
            # rejects the websocket handshake for any origin it does not know, and this
            # local automation client has one.
            "--remote-allow-origins=*",
            "about:blank",
        ]
        self.process = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        deadline = time.time() + self.timeout
        version: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                version = httpx.get(f"http://127.0.0.1:{self.port}/json/version", timeout=2.0).json()
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        if not version:
            self.close()
            raise BrowserUnavailable(
                f"{os.path.basename(self.executable)} did not expose a DevTools endpoint on "
                f"port {self.port} within {self.timeout}s"
            )
        self.browser_version = version.get("Browser", "unknown")

        self.target = self._open_target()
        ws_url = self.target["webSocketDebuggerUrl"]
        try:
            # `suppress_origin` keeps the handshake independent of Chromium's
            # `--remote-allow-origins` policy (websocket-client sends an Origin otherwise).
            self.ws = websocket.create_connection(
                ws_url, timeout=self.timeout, suppress_origin=True
            )
        except websocket.WebSocketException as exc:
            self.close()
            raise BrowserUnavailable(f"could not attach to the browser DevTools socket: {exc}")
        for domain in ("Page", "Runtime", "Log", "Network"):
            self.send(f"{domain}.enable")

    def _open_target(self) -> Dict[str, Any]:
        """Open a fresh tab (newer Chromium requires PUT for `/json/new`)."""
        base = f"http://127.0.0.1:{self.port}"
        for method in ("PUT", "GET"):
            try:
                response = httpx.request(method, f"{base}/json/new?about:blank", timeout=5.0)
                if response.status_code == 200:
                    return response.json()
            except httpx.HTTPError:
                continue
        # Fall back to whatever tab the browser opened with.
        for target in httpx.get(f"{base}/json/list", timeout=5.0).json():
            if target.get("type") == "page":
                return target
        raise BrowserUnavailable("could not open a page target in the headless browser")

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
        self.ws = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self.profile_dir:
            shutil.rmtree(self.profile_dir, ignore_errors=True)
            self.profile_dir = None

    # ---------- protocol ----------

    def send(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
        """Send one CDP command and return its result, collecting events on the way."""
        if self.ws is None:
            raise BrowserUnavailable("the browser session is not open")
        message_id = self._next_id
        self._next_id += 1
        self.ws.settimeout(timeout or self.timeout)
        self.ws.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))

        deadline = time.time() + (timeout or self.timeout)
        while time.time() < deadline:
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                continue
            message = json.loads(raw)
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})
            self._handle_event(message)
        raise TimeoutError(f"{method} did not answer within {timeout or self.timeout}s")

    def pump(self, quiet_seconds: float = 0.05) -> None:
        """Drain pending events without waiting for a command reply."""
        if self.ws is None:
            return
        self.ws.settimeout(quiet_seconds)
        while True:
            try:
                raw = self.ws.recv()
            except (websocket.WebSocketTimeoutException, OSError):
                return
            if not raw:
                return
            self._handle_event(json.loads(raw))

    def _handle_event(self, message: Dict[str, Any]) -> None:
        method = message.get("method", "")
        params = message.get("params", {}) or {}

        if method == "Runtime.consoleAPICalled":
            self.console.append(
                ConsoleMessage(
                    level=params.get("type", "log"),
                    text=" ".join(
                        str(arg.get("value", arg.get("description", "")))
                        for arg in params.get("args", [])
                    ),
                    source="console",
                )
            )
        elif method == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            self.console.append(
                ConsoleMessage(
                    level="error",
                    text=str(
                        details.get("exception", {}).get("description")
                        or details.get("text", "uncaught exception")
                    ),
                    source="exception",
                )
            )
        elif method == "Log.entryAdded":
            entry = params.get("entry", {})
            if entry.get("level") in {"error", "warning"}:
                self.console.append(
                    ConsoleMessage(
                        level=entry.get("level", "error"),
                        text=f"{entry.get('text', '')} ({entry.get('url', '')})",
                        source="log",
                    )
                )
        elif method == "Network.requestWillBeSent":
            self.request_urls.append(params.get("request", {}).get("url", ""))
        elif method == "Network.responseReceived":
            response = params.get("response", {})
            self.responses.append(
                {
                    "url": response.get("url", ""),
                    "status": response.get("status"),
                    "mime": response.get("mimeType", ""),
                }
            )
        elif method == "Network.loadingFailed":
            self.failed_requests.append(
                {"error": params.get("errorText", ""), "type": params.get("type", "")}
            )

    # ---------- page helpers ----------

    def evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        """Evaluate JS in the page and return its value (JSON-serialisable)."""
        result = self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
            },
        )
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            raise RuntimeError(
                "page threw while evaluating: "
                f"{details.get('text')} {details.get('exception', {}).get('description', '')}"
            )
        return result.get("result", {}).get("value")

    def navigate(self, url: str, *, wait_load: bool = True) -> None:
        """Navigate and (by default) wait until the page finished loading."""
        self.send("Page.navigate", {"url": url})
        if not wait_load:
            return
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if self.evaluate("document.readyState") == "complete":
                return
            time.sleep(0.2)
        raise TimeoutError(f"{url} did not finish loading within {self.timeout}s")

    def wait_for(
        self,
        expression: str,
        description: str,
        *,
        timeout: Optional[float] = None,
        poll: float = 0.25,
    ) -> Any:
        """Poll `expression` until it is truthy, returning its last value."""
        deadline = time.time() + (timeout or self.timeout)
        last: Any = None
        while time.time() < deadline:
            last = self.evaluate(expression)
            if last:
                return last
            time.sleep(poll)
        raise TimeoutError(f"timed out waiting for {description} (last value: {last!r})")

    # ---------- logs ----------

    def error_messages(self) -> List[ConsoleMessage]:
        """Console errors and uncaught exceptions -- a page that broke under the user."""
        return [message for message in self.console if message.level == "error"]

    def http_errors(self) -> List[Dict[str, Any]]:
        """Responses with status >= 400 (probe calls excluded by the caller)."""
        return [
            response for response in self.responses
            if isinstance(response.get("status"), int) and response["status"] >= 400
        ]

    def off_site_urls(self, allowed_host: str) -> List[str]:
        """Requests that do not go to the deployment host (e.g. the dev fallback)."""
        return sorted({
            url for url in self.request_urls
            if url.startswith(("http://", "https://")) and allowed_host not in url
        })
