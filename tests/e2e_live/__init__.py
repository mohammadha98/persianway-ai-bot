"""Live end-to-end tests: they run against the *deployed* service, not in-process.

Every other test package in this repository starts the app inside the test
process (ASGI transport, fake services, monkeypatched settings), so it can only
prove what the code does when the transport is trusted. This package proves what
the deployment actually puts on the wire, through the same proxies and TLS
endpoint a browser uses:

  * `test_live_streaming_api.py`      -- the SSE contract of `GET /api/chat/stream`
                                         (headers, open frame, heartbeats, chunk
                                         ordering, `[DONE]`, encoding, cancel safety).
  * `test_live_frontend_artifact.py`  -- the *deployed* Angular bundle: hashes vs
                                         the local build, and the presence of the
                                         streaming parser the chat page needs.
  * `test_live_frontend_browser.py`   -- a real headless browser: the chat page
                                         renders the answer token by token, with no
                                         console/network errors.

Nothing here is enabled by accident; see `conftest.py` for the `E2E_LIVE` gate and
`docs/LIVE_E2E_STREAMING_TESTS.md` for the cost/risk profile of every test.
"""
