"""Guards on the chat transport switch (`CHAT_STREAMING_ENABLED`).

`POST/GET /api/chat/stream` serves the answer in one of two ways, chosen by that
single variable:

  * `true`  -- token by token: one `chunk` frame per model token,
  * `false` -- buffered server-side and written once, inside the `done` frame.

`false` is the shipped default. It is a transport choice, not a deletion: the
route, the keep-alive comment frames, the `status` / `metadata` / `error` frames
and the `[DONE]` terminator are shared by both modes, so re-enabling token
streaming is an environment change with no code change.
"""
from app.core.config import Settings


def test_streaming_transport_defaults_to_single_frame():
    """The shipped default is single-frame: token streaming is opt-in.

    The default is read off the model field rather than the live instance, so the
    result is the same in CI and on a machine that happens to export
    `CHAT_STREAMING_ENABLED=true`.
    """
    field = Settings.model_fields["CHAT_STREAMING_ENABLED"]

    assert field.default is False


def test_streaming_transport_is_read_when_the_response_is_built():
    """The flag is read per request, not copied at import time.

    `stream_chat` reads `settings.CHAT_STREAMING_ENABLED` while building the
    response body, which is what makes the transport switchable in tests
    (`monkeypatch.setattr(chat_module.settings, ...)`) and in a deployment that
    reloads `settings` without restarting the worker.
    """
    import inspect

    from app.api.routes import chat as chat_module

    source = inspect.getsource(chat_module.stream_chat)

    assert "settings.CHAT_STREAMING_ENABLED" in source
