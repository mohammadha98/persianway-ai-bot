"""`/api/config/*` is the credential store, so it must never answer anonymously.

`GET /api/config/llm` used to hand the OpenAI/OpenRouter keys to anyone who could
reach the host, and `/api/config/database` the MongoDB URL; the panel that owns
those screens already sends its JWT (`auth.interceptor.ts`). These tests pin both
halves of the fix:

* no token (or a bad one) is rejected before a handler runs, and
* an authenticated read still answers masked keys only.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.schemas.config import (
    DynamicConfig,
    LLMSettings,
    TavilySearchSettings,
)
from app.services.chat_service import get_chat_service
from app.services.config_service import get_config_service
from app.services.knowledge_base import get_knowledge_base_service
from app.services.user_service import get_user_service

from main import app

client = TestClient(app)

OPENROUTER_KEY = "sk-or-v1-secret-1111"
OPENAI_KEY = "sk-proj-secret-2222"
TAVILY_KEY = "tvly-secret-3333"
MANAGEMENT_KEY = "sk-or-v1-mgmt-4444"

# Every route the panel needs a login for; reads and writes both.
CONFIG_READS = (
    "/api/config/",
    "/api/config/llm",
    "/api/config/llm/credits",
    "/api/config/rag",
    "/api/config/database",
    "/api/config/app",
    "/api/config/tavily",
)


class _FakeConfigService:
    """Serves a document with a stored key in every secret slot."""

    def __init__(self):
        self.config = DynamicConfig(
            llm_settings=LLMSettings(
                openai_api_key=OPENAI_KEY,
                openrouter_api_key=OPENROUTER_KEY,
                openrouter_management_key=MANAGEMENT_KEY,
            ),
            tavily_settings=TavilySearchSettings(tavily_api_key=TAVILY_KEY),
        )

    async def clear_cache(self) -> None:
        return None

    async def get_config(self) -> DynamicConfig:
        return self.config

    async def get_llm_settings(self) -> LLMSettings:
        return self.config.llm_settings


@pytest.fixture(autouse=True)
def stub_services():
    """The KB/chat singletons are expensive to build and are not under test."""
    kb_service = MagicMock()
    kb_service.refresh = AsyncMock()
    chat_service = MagicMock()
    chat_service.refresh = AsyncMock()

    app.dependency_overrides[get_knowledge_base_service] = lambda: kb_service
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_config_service] = lambda: _FakeConfigService()
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_chat_service, None)
        app.dependency_overrides.pop(get_config_service, None)


@pytest.fixture
def user_service():
    """A user service whose token decoding is the only thing that varies."""
    service = MagicMock()
    app.dependency_overrides[get_user_service] = lambda: service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_user_service, None)


def _accept_tokens(user_service) -> None:
    """Make the JWT dependency resolve to an active user."""
    user_service._decode_access_token = MagicMock(return_value={"sub": "user-1"})
    user_service.get_user_by_id = AsyncMock(
        return_value=MagicMock(is_active=True, username="operator")
    )


@pytest.mark.parametrize("path", CONFIG_READS)
def test_reads_reject_anonymous_callers(path):
    """No token, no settings - the keys are not public."""
    response = client.get(path)

    # HTTPBearer answers 403 for a missing header; a present-but-invalid token
    # is a 401 (covered below). Either way the handler never runs.
    assert response.status_code in (401, 403)
    assert OPENROUTER_KEY not in response.text
    assert OPENAI_KEY not in response.text
    assert TAVILY_KEY not in response.text


def test_writes_reject_anonymous_callers():
    """An open write path could point the bot at an attacker's provider."""
    assert client.put("/api/config/llm", json={"default_model": "x"}).status_code in (401, 403)
    assert client.put("/api/config/", json={}).status_code in (401, 403)
    assert client.put("/api/config/tavily", json={"is_enabled": True}).status_code in (401, 403)
    assert client.post("/api/config/reset").status_code in (401, 403)
    assert client.post("/api/config/tavily/test", json={"query": "x"}).status_code in (401, 403)
    assert client.post("/api/config/tavily/reset").status_code in (401, 403)


def test_an_invalid_token_is_a_401(user_service):
    """A forged/expired token must not fall through to the settings."""
    user_service._decode_access_token = MagicMock(side_effect=ValueError("bad signature"))

    response = client.get(
        "/api/config/llm", headers={"Authorization": "Bearer not-a-real-token"}
    )

    assert response.status_code == 401
    assert OPENROUTER_KEY not in response.text


def test_authenticated_read_masks_every_key(user_service):
    """A logged-in panel gets the settings it needs, without the usable secrets."""
    _accept_tokens(user_service)

    response = client.get(
        "/api/config/llm", headers={"Authorization": "Bearer valid-token"}
    )

    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["openrouter_api_key"] == "****1111"
    assert settings["openai_api_key"] == "****2222"
    # The optional management key (credit balance only) is a credential too.
    assert settings["openrouter_management_key"] == "****4444"
    assert OPENROUTER_KEY not in response.text
    assert OPENAI_KEY not in response.text
    assert MANAGEMENT_KEY not in response.text


def test_authenticated_whole_document_masks_the_nested_keys(user_service):
    """`GET /api/config/` answers every section, so masking must be recursive."""
    _accept_tokens(user_service)

    response = client.get("/api/config/", headers={"Authorization": "Bearer valid-token"})

    assert response.status_code == 200
    config = response.json()["config"]
    assert config["llm_settings"]["openrouter_api_key"] == "****1111"
    assert config["llm_settings"]["openai_api_key"] == "****2222"
    assert config["tavily_settings"]["tavily_api_key"] == "****3333"
    assert OPENROUTER_KEY not in response.text
    assert OPENAI_KEY not in response.text
    assert TAVILY_KEY not in response.text


def test_authenticated_read_of_tavily_masks_the_search_key(user_service):
    _accept_tokens(user_service)

    response = client.get(
        "/api/config/tavily", headers={"Authorization": "Bearer valid-token"}
    )

    assert response.status_code == 200
    assert response.json()["settings"]["tavily_api_key"] == "****3333"
    assert TAVILY_KEY not in response.text
