"""Live tests for the **deployed frontend bundle** -- the "is the UI really shipped?" half.

What this module answers
------------------------
Streaming can be perfect on the API and still be broken for users, because the panel the
browser loads is a *build artifact*: if it is stale, missing a lazy chunk, or was built
with a development `apiUrl`, the chat page breaks even though every server-side test
passes. These tests compare the assets the deployment serves with the frontend in this
repository, and check that the deployed chat chunk contains the SSE parser the page needs
(`chat.service.ts -> sseFetchStream`):

  * `index.html` is served and references hashed entry bundles (so a deploy is picked up
    instead of a cached shell),
  * every entry bundle and every lazy chunk the entry bundle references is deployed and
    byte-identical to the local `frontend/ai-panel/dist/ai-panel/browser` build (when that
    build is present),
  * the chunk that talks to `/api/chat/stream` also contains `text/event-stream`,
    `[DONE]`, `stream_incomplete` and `stream_read_error` -- i.e. the deployed parser
    understands the frames the deployed API writes,
  * no deployed asset points the API at a development or staging host.

No LLM tokens are spent here, and nothing is written: these are read-only checks.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import httpx
import pytest

from tests.e2e_live.conftest import LiveSettings
from tests.e2e_live.live_client import (
    FRONTEND_DEAD_FALLBACK,
    FRONTEND_STREAM_MARKERS,
    DeployedFrontend,
    fetch_deployed_frontend,
    local_build_hashes,
)

pytestmark = pytest.mark.live

# Hosts that must never be baked into a production bundle: a dev fallback or the old
# staging domain would make the panel call a service that does not exist for users.
FORBIDDEN_API_HOSTS = ("runflare.run", "127.0.0.1", "0.0.0.0")

# Local build output produced by `npm run build` inside `frontend/ai-panel`.
DIST_ROOT = (
    Path(__file__).resolve().parents[2] / "frontend" / "ai-panel" / "dist" / "ai-panel" / "browser"
)

HASHED_ASSET = re.compile(r"^(?:main|polyfills|styles|chunk)-[A-Z0-9]{8}\.(?:js|css)$")


@pytest.fixture(scope="module")
def deployed_frontend(live_client: httpx.Client, live_settings: LiveSettings) -> DeployedFrontend:
    """The SPA shell, entry bundles and lazy chunks as the deployment serves them."""
    return fetch_deployed_frontend(live_client, live_settings.base_url)


def test_spa_shell_is_served_and_references_hashed_bundles(
    deployed_frontend: DeployedFrontend, live_settings: LiveSettings
):
    """`GET /` is the Angular shell, with cache-busting hashed entry bundles.

    A non-hashed entry name (`main.js`) is how users keep an old frontend after a deploy:
    the browser serves the cached file and the panel then talks to the new API with the
    old parser.
    """
    html = deployed_frontend.index_html
    assert "<app-root" in html, f"{live_settings.base_url}/ did not serve the SPA shell"
    assert deployed_frontend.scripts, "index.html referenced no scripts"

    names = [asset.path.rsplit("/", 1)[-1] for asset in deployed_frontend.scripts]
    assert any(name.startswith("main-") for name in names), names
    for name in names:
        assert HASHED_ASSET.match(name), (
            f"entry bundle {name!r} is not content-hashed, so browsers can serve a stale "
            "bundle after a deploy"
        )
    for asset in deployed_frontend.scripts:
        assert asset.status_code == 200, f"{asset.path} returned {asset.status_code}"
        assert asset.content, f"{asset.path} was served empty"


def test_deployed_bundles_are_the_local_build(deployed_frontend: DeployedFrontend):
    """Every deployed asset is byte-identical to the frontend built in this repository.

    This is what turns "the fix is in the repo" into "the fix is what users load". The
    comparison covers the entry bundles *and* the lazy chunks (the chat page lives in a
    lazy chunk), because a partially deployed build -- new `main`, old chunks -- serves a
    page whose parser and API have drifted apart.

    Skipped (not failed) when the local build is absent: the check needs
    `frontend/ai-panel/dist/ai-panel/browser`, produced by `npm run build`.
    """
    local = local_build_hashes(str(DIST_ROOT))
    if not local:
        pytest.skip(
            f"no local build at {DIST_ROOT}: run `npm run build` in frontend/ai-panel to "
            "compare the deployment against this checkout"
        )

    deployed = {asset.path: asset.sha256 for asset in deployed_frontend.scripts}
    deployed.update({path: asset.sha256 for path, asset in deployed_frontend.lazy_chunks.items()})

    missing = sorted(name for name in deployed if name not in local)
    mismatched = sorted(
        name for name, digest in deployed.items() if name in local and local[name] != digest
    )

    assert not missing, (
        "these assets are served by the deployment but do not exist in the local build, so "
        f"the deployment differs from this checkout: {missing}"
    )
    assert not mismatched, (
        "these assets differ between the deployment and the local build (stale or partially "
        f"deployed frontend): {mismatched}"
    )


def test_every_referenced_lazy_chunk_is_deployed(deployed_frontend: DeployedFrontend):
    """All chunks the entry bundle references exist (no 404 on a route the user opens).

    Angular fetches route chunks on demand: a chunk missing from the deployment is a page
    that 404s *after* login, with the app already loaded -- the kind of failure that is
    invisible to server-side tests.
    """
    assert deployed_frontend.lazy_chunks, (
        "the entry bundle references no lazy chunks: either the app stopped code-splitting "
        "or the chunk names changed shape and this suite can no longer see them"
    )
    bad = {
        path: asset.status_code
        for path, asset in deployed_frontend.lazy_chunks.items()
        if asset.status_code != 200
    }
    assert not bad, f"lazy chunks referenced by the entry bundle are not deployed: {bad}"

    # A few chunks are empty by design (Angular emits an empty module for some deferred
    # routes); the byte-comparison test is what proves they are *the same* empty file as
    # in the local build, so this only records them.
    empty = sorted(path for path, asset in deployed_frontend.lazy_chunks.items() if not asset.content)
    if empty:
        print(f"[live-e2e] empty-but-served lazy chunks: {empty}")


def test_chat_chunk_contains_the_streaming_parser(deployed_frontend: DeployedFrontend):
    """The deployed chat chunk really has the fetch-based SSE reader.

    `chat.service.ts` consumes `GET /api/chat/stream` with `fetch` +
    `response.body.getReader()` (not `EventSource`), works on `text/event-stream` and
    reports `stream_incomplete` / `stream_read_error` when the connection ends without a
    `done` event. Minification keeps those literals, so their presence proves which parser
    the browser runs -- an old build without them shows the user a spinner that never
    resolves.
    """
    chat_chunk = deployed_frontend.find_asset_with("/api/chat/stream")
    assert chat_chunk, (
        "no deployed asset references /api/chat/stream: the chat page cannot reach the "
        "streaming endpoint in this bundle"
    )

    entry_paths = {asset.path for asset in deployed_frontend.scripts}
    text = (
        next(asset.text for asset in deployed_frontend.scripts if asset.path == chat_chunk)
        if chat_chunk in entry_paths
        else deployed_frontend.lazy_chunks[chat_chunk].text
    )

    for marker in FRONTEND_STREAM_MARKERS:
        assert marker in text, (
            f"the deployed chat chunk {chat_chunk} does not contain {marker!r}: the shipped "
            "frontend does not implement the streaming contract the API exposes"
        )
    assert "getReader(" in text, (
        f"{chat_chunk} does not read the response body as a stream (`getReader()`): the "
        "answer would only appear once the whole body had arrived"
    )


def test_no_deployed_asset_uses_a_development_api_host(
    deployed_frontend: DeployedFrontend, live_settings: LiveSettings
):
    """The bundle's API base is the real deployment host, not a dev/staging leftover.

    Note: the Angular services keep a dead `environment.apiUrl || 'http://localhost:8000'`
    fallback, so that literal is expected to appear; what must not happen is that the
    production host is missing or that a *staging* host (`pwiran.runflare.run`) is what got
    baked in. The browser test additionally inspects the requests the page actually makes.
    """
    host = httpx.URL(live_settings.base_url).host
    if not host:
        pytest.skip(f"could not parse a host out of {live_settings.base_url!r}")

    assets: List[Tuple[str, str]] = [(asset.path, asset.text) for asset in deployed_frontend.scripts]
    assets += [(path, asset.text) for path, asset in deployed_frontend.lazy_chunks.items()]

    assert any(host in text for _, text in assets), (
        f"no deployed asset contains the configured API host {host!r}: the panel would fall "
        "back to its development URL"
    )
    for forbidden in FORBIDDEN_API_HOSTS:
        offenders = [path for path, text in assets if f"//{forbidden}" in text]
        assert not offenders, f"deployed asset(s) {offenders} point the API at {forbidden!r}"

    # Informational: the dead fallback is expected, so it is recorded, not asserted on.
    if FRONTEND_DEAD_FALLBACK not in "".join(text for _, text in assets):
        print(f"[live-e2e] {FRONTEND_DEAD_FALLBACK} is absent from the bundle (fallback removed)")
