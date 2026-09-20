"""Regression tests for provider-level failures and stage status in the chat stream.

Production symptoms under test:

1. *No status message / endless loading.* A rate-limited or stalled OpenRouter
   model kept a turn inside the SDK retry loop with no bound. `get_llm` set
   neither `request_timeout` nor `max_retries`, so the openai defaults applied
   (`httpx.Timeout(timeout=600, connect=5.0)` and `DEFAULT_MAX_RETRIES = 2`, see
   `openai/_constants.py`): ~30 minutes worst case. The only bytes on the wire
   were SSE keep-alive *comment* frames, which the frontend parser drops, so the
   user saw a bare spinner with no explanation.

2. *The failure never named its cause.* `openai.RateLimitError` /
   `openai.APITimeoutError` were caught nowhere in the repository, so the turn
   ended in the generic `stream_internal_error` event.

These tests pin:
  * a bounded request budget on every ChatOpenAI instance built by `get_llm`,
  * a *separate, tighter* budget for the best-effort conversation title, which
    runs before the first byte of a response and must not inherit the answer's
    retry budget,
  * `status` events before retrieval and before generation (the user-renderable
    progress signal the UI needs),
  * `openai.RateLimitError` -> `provider_rate_limited`, `openai.APITimeoutError`
    -> `provider_timeout`, with the configured human-referral message kept,
  * a timeout in the intent classifier (the first LLM call of a turn) is also
    reported as a provider failure instead of a generic internal error.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from app.services import chat_service as chat_module


STATIC_REFERRAL = "پیام ارجاع ثابت به کارشناس"


def _rate_limit_error():
    """The exception the openai SDK raises once its own retries are exhausted."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(
        429, request=request, json={"error": {"message": "rate limited"}}
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def _timeout_error():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.APITimeoutError(request=request)


def _llm_settings():
    settings = MagicMock()
    settings.default_model = "google/gemini-flash"
    settings.temperature = 0.2
    settings.max_tokens = 512
    settings.top_p = 1.0
    settings.preferred_api_provider = "openrouter"
    settings.openrouter_api_key = "test-openrouter-key"
    settings.openrouter_api_base = "https://openrouter.ai/api/v1"
    settings.openai_api_key = "test-openai-key"
    return settings


def _config_service():
    settings = MagicMock()
    settings._load_config = AsyncMock()
    settings.get_llm_settings = AsyncMock(return_value=_llm_settings())
    rag_settings = MagicMock()
    rag_settings.human_referral_message = STATIC_REFERRAL
    rag_settings.knowledge_base_confidence_threshold = 0.7
    rag_settings.system_prompt = "system prompt"
    rag_settings.temperature = 0.1
    settings.get_rag_settings = AsyncMock(return_value=rag_settings)
    return settings


def _intent():
    return {
        "intent": "PRIVATE",
        "is_public": False,
        "explanation": "specialized question",
        "topic_shift": False,
        "context_state": "self_contained",
        "resolved_query": "بهترین کود برای گندم چیست؟",
        "referenced_entities": [],
        "context_decided": True,
    }


def _retrieval(confidence):
    """`_retrieve_context` payload; a high score also needs scored docs."""
    docs_with_scores = [({"page_content": "context"}, confidence)] if confidence >= 0.7 else []
    return {
        "confidence_score": confidence,
        "sources": [],
        "rewritten_query": "بهترین کود برای گندم چیست؟",
        "docs_with_scores": docs_with_scores,
        "normalized_docs": [{"page_content": "context"}] if docs_with_scores else [],
        "source_type": "knowledge_base",
    }


def _make_service(chain=None, general_answer=False):
    service = chat_module.ChatService()
    service.config_service = _config_service()
    service.generalAnswer = general_answer
    service.detect_query_intent = AsyncMock(return_value=_intent())
    service._get_or_create_session = AsyncMock(return_value=chain)
    return service


async def _collect(service, message="بهترین کود برای گندم چیست؟"):
    return [event async for event in service.process_message_stream("user-1", message)]


def _kb_service(confidence, tokens=None, error=None):
    """KB service double: either streams `tokens` or raises `error`."""
    kb = MagicMock()
    kb._retrieve_context = AsyncMock(return_value=_retrieval(confidence))
    kb.build_prompt_snapshot = AsyncMock(return_value={"response_type": "knowledge_base"})

    async def _stream(*args, **kwargs):
        if error is not None:
            raise error
        for token in tokens or []:
            yield token

    kb.stream_answer_from_context = _stream
    return kb


@pytest.mark.asyncio
async def test_get_llm_builds_a_bounded_request_budget():
    """Without this, the SDK defaults are a 600s timeout x 3 attempts."""
    with patch(
        "app.services.config_service.get_config_service",
        new=AsyncMock(return_value=_config_service()),
    ):
        llm = await chat_module.get_llm()

    assert llm.request_timeout is not None, "request_timeout must be explicit (openai default is 600s)"
    assert isinstance(llm.request_timeout, (int, float))
    assert 0 < llm.request_timeout <= 60
    assert llm.max_retries is not None, "max_retries must be explicit (openai default is 2)"
    assert 0 <= llm.max_retries <= 1


@pytest.mark.asyncio
async def test_rate_limited_generation_is_reported_as_a_provider_error():
    """A 429 during KB generation names the cause and keeps the referral path."""
    kb = _kb_service(confidence=0.9, error=_rate_limit_error())
    service = _make_service()

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    assert [event["type"] for event in events] == ["status", "metadata", "status", "error"]
    assert events[0]["stage"] == "retrieval"
    assert events[2]["stage"] == "generation"
    assert events[2]["message"] == "در حال تولید پاسخ…"
    assert events[-1]["code"] == "provider_rate_limited"
    assert chat_module.PROVIDER_BUSY_MESSAGE in events[-1]["message"]
    # The referral text is kept, not replaced.
    assert STATIC_REFERRAL in events[-1]["message"]
    assert events[-1]["query_analysis"]["requires_human_referral"] is True


@pytest.mark.asyncio
async def test_generation_timeout_is_reported_as_a_provider_error():
    """The timeout clause of the provider handler uses its own code/message."""
    kb = _kb_service(confidence=0.9, error=_timeout_error())
    service = _make_service()

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    assert [event["type"] for event in events] == ["status", "metadata", "status", "error"]
    assert events[-1]["code"] == "provider_timeout"
    assert chat_module.PROVIDER_TIMEOUT_MESSAGE in events[-1]["message"]
    assert STATIC_REFERRAL in events[-1]["message"]


@pytest.mark.asyncio
async def test_rate_limited_general_path_is_reported_as_a_provider_error():
    """The general-knowledge (handoff) chain is covered the same way."""

    class _ExplodingChain:
        async def astream(self, inputs):
            raise _rate_limit_error()
            yield  # pragma: no cover - makes this an async generator

    kb = _kb_service(confidence=0.1)
    service = _make_service(chain=_ExplodingChain(), general_answer=True)

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    assert [event["type"] for event in events] == ["status", "metadata", "status", "error"]
    assert events[-1]["code"] == "provider_rate_limited"
    assert STATIC_REFERRAL in events[-1]["message"]


@pytest.mark.asyncio
async def test_intent_classifier_failure_is_reported_as_a_provider_error():
    """Intent detection is the FIRST LLM call of a turn: it must not fail silently."""
    service = _make_service()
    service.detect_query_intent = AsyncMock(side_effect=_timeout_error())

    events = await _collect(service)

    # No retrieval status, no metadata: the failure happens before either.
    assert [event["type"] for event in events] == ["error"]
    assert events[0]["code"] == "provider_timeout"
    assert chat_module.PROVIDER_TIMEOUT_MESSAGE in events[0]["message"]
    assert STATIC_REFERRAL in events[0]["message"]


@pytest.mark.asyncio
async def test_status_events_do_not_disturb_the_successful_sequence():
    """Happy path: status frames are additive, the token stream is unchanged."""
    kb = _kb_service(confidence=0.9, tokens=["بهترین ", "کود اوره است."])
    service = _make_service()

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    assert [event["type"] for event in events] == [
        "status", "metadata", "status", "chunk", "chunk", "done",
    ]
    assert events[0] == {
        "type": "status",
        "stage": "retrieval",
        "message": "در حال جستجو در پایگاه دانش…",
    }
    assert "".join(e["content"] for e in events if e["type"] == "chunk") == "بهترین کود اوره است."
    assert events[-1]["answer"] == "بهترین کود اوره است."


@pytest.mark.asyncio
async def test_get_llm_budget_can_be_overridden_per_caller():
    """`request_timeout` / `max_retries` must pass through to the HTTP client."""
    with patch(
        "app.services.config_service.get_config_service",
        new=AsyncMock(return_value=_config_service()),
    ):
        default_llm = await chat_module.get_llm()
        tight_llm = await chat_module.get_llm(request_timeout=7.5, max_retries=0)

    # No override -> the process-wide budget, unchanged.
    assert default_llm.request_timeout == chat_module.LLM_REQUEST_TIMEOUT_SECONDS
    assert default_llm.max_retries == chat_module.LLM_MAX_RETRIES

    # Override -> the values actually handed to the openai SDK / httpx, not just
    # the pydantic fields (`0` must not be coerced back to a default).
    assert tight_llm.request_timeout == 7.5
    assert tight_llm.max_retries == 0
    assert tight_llm.root_client.timeout == 7.5
    assert tight_llm.root_client.max_retries == 0


@pytest.mark.asyncio
async def test_title_generation_uses_a_tighter_budget_than_the_answer_path():
    """The pre-response title call must fail fast instead of retrying.

    `generate_conversation_title` runs before the first byte of a streamed turn
    (chat.py) and before the POST `/api/chat/` response, so inheriting the answer
    budget would let a stalled provider hold the client for ~92s. Retrying that
    provider for a cosmetic title also adds load to the model the answer is about
    to need.
    """
    captured = {}

    async def _fake_get_llm(**kwargs):
        captured.update(kwargs)
        llm = MagicMock()
        response = MagicMock()
        response.content = "کود مناسب گندم"
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    service = _make_service()

    with patch.object(chat_module, "get_llm", new=_fake_get_llm):
        title = await service.generate_conversation_title("بهترین کود برای گندم چیست؟")

    assert title == "کود مناسب گندم"
    assert captured["request_timeout"] == chat_module.TITLE_LLM_REQUEST_TIMEOUT_SECONDS
    assert captured["max_retries"] == chat_module.TITLE_LLM_MAX_RETRIES
    # The whole point of the override: strictly tighter than the answer budget.
    assert captured["request_timeout"] < chat_module.LLM_REQUEST_TIMEOUT_SECONDS
    assert captured["max_retries"] < chat_module.LLM_MAX_RETRIES


@pytest.mark.asyncio
async def test_title_generation_falls_back_when_the_provider_refuses():
    """With `max_retries=0` a 429 is final: fallback title, never an exception."""
    async def _fake_get_llm(**kwargs):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=_rate_limit_error())
        return llm

    service = _make_service()

    with patch.object(chat_module, "get_llm", new=_fake_get_llm):
        title = await service.generate_conversation_title("سلام")

    assert title == "New Conversation"
