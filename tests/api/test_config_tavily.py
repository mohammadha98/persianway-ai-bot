"""`/api/config/tavily` is the contract behind the admin panel's web-search page.

Covers what the panel depends on: fresh (uncached) reads, partial saves that keep
the untouched fields, input normalisation/validation, the test-search button and
the scoped reset.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.routes import config as config_module
from app.core.config import settings as static_settings
from app.schemas.config import DynamicConfig, TavilySearchSettings
from app.services.chat_service import get_chat_service
from app.services.config_service import get_config_service
from app.services.knowledge_base import get_knowledge_base_service

from main import app

client = TestClient(app)
To a farm tradition, as mine harmonical
TAVILY_URL = "/api/config/tavily"


class _FakeConfigService:
    """In-memory stand-in built with the same partial-update semantics as the real service."""

    def __init__(self, tavily: TavilySearchSettings = None):
        self.config = DynamicConfig(tavily_settings=tavily or TavilySearchSettings())
        self.clear_cache_calls = 0
        self.update_calls = []

    async def clear_cache(self) -> None:
        self.clear_cache_calls += 1

    async def get_config(self) -> DynamicConfig:
        return self.config

    async def update_config(self, updates):
        for section, values in updates.items():
            current = getattr(self.config, section).dict()
            current.update(values)
            setattr(self.config, section, TavilySearchSettings(**current))
        self.update_calls.append(updates)
        return self.config


@pytest.fixture
def fake_config_service():
    service = _FakeConfigService()
    app.dependency_overrides[get_config_service] = lambda: service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_config_service, None)


@pytest.fixture(autouse=True)
def stub_heavy_services():
    """The write routes refresh the KB and the chat session cache; those are not under test."""
    kb_service = MagicMock()
    kb_service.refresh = AsyncMock()
    chat_service = MagicMock()
    chat_service.refresh = AsyncMock()

    app.dependency_overrides[get_knowledge_base_service] = lambda: kb_service
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    try:
        yield kb_service, chat_service
    finally:
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_chat_service, None)


def test_get_tavily_settings_is_never_cached(fake_config_service):
    """Multi-instance deployments need the stored document, not the process cache."""
    response = client.get(TAVILY_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["settings"]["is_enabled"] is True
    assert body["settings"]["search_depth"] == "advanced"
    assert fake_config_service.clear_cache_calls == 1
    assert response.headers["cache-control"].startswith("no-store")
    assert response.headers["pragma"] == "no-cache"

def test_put_saves_a_single_toggle_without_touching_the_rest(fake_config_service):
    """The enable/disable switch is the most used control, so it must stand alone."""
    response = client.put(TAVILY_URL, json={"is_enabled": False})

    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["is_enabled"] is False
    # Untouched fields keep their stored values.
    assert settings["search_depth"] == "advanced"
    assert settings["max_results"] == 5
    assert settings["snippet_length"] == 200


def test_put_normalizes_typed_domain_lists(fake_config_service):
    """Operators paste URLs, mixed case and duplicates; the stored list must be clean."""
    response = client.put(
        TAVILY_URL,
        json={
            "include_domains": ["https://WWW.PersianWay.IR/blog?x=1", "persianway.ir", "  "],
            "exclude_domains": "spam.example, *.ads.example",
        },
    )

    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["include_domains"] == ["www.persianway.ir", "persianway.ir"]
    assert settings["exclude_domains"] == ["spam.example", "ads.example"]


def test_put_rejects_an_unknown_search_depth(fake_config_service):
    """Tavily would answer with a 400 far away from the panel; reject it here instead."""
    response = client.put(TAVILY_URL, json={"search_depth": "deep"})

    assert response.status_code == 400
    assert "search_depth" in response.json()["detail"]
    assert fake_config_service.update_calls == []


def test_put_rejects_out_of_range_numbers(fake_config_service):
    response = client.put(TAVILY_URL, json={"max_results": 99})

    assert response.status_code == 400
    assert "Invalid Tavily settings" in response.json()["detail"]


def test_put_rejects_a_document_without_known_fields(fake_config_service):
    """A misspelled key must fail loudly instead of reporting a fake success."""
    response = client.put(TAVILY_URL, json={"enable_search": True})

    assert response.status_code == 400


def test_test_endpoint_runs_the_real_search_path(fake_config_service, monkeypatch):
    """The button reports the formatted sources the answer would be built from."""
    search_tool = MagicMock()
    search_tool.ainvoke = AsyncMock(return_value="Answer: تست\nSources:\n- [t](u): snippet...")
    monkeypatch.setattr(config_module, "search_persianway", search_tool)

    response = client.post(f"{TAVILY_URL}/test", json={"query": "قیمت محصولات"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["enabled"] is True
    assert "snippet" in body["result"]
    assert body["elapsed_seconds"] >= 0
    search_tool.ainvoke.assert_awaited_once_with({"query": "قیمت محصولات"})


def test_test_endpoint_surfaces_search_failures(fake_config_service, monkeypatch):
    search_tool = MagicMock()
    search_tool.ainvoke = AsyncMock(return_value="Error searching web: invalid api key")
    monkeypatch.setattr(config_module, "search_persianway", search_tool)

    response = client.post(f"{TAVILY_URL}/test", json={"query": "قیمت محصولات"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "invalid api key" in body["message"]


def test_test_endpoint_explains_that_search_is_disabled(fake_config_service, monkeypatch):
    """No request is issued while the panel switch is off, and the panel says why."""
    fake_config_service.config.tavily_settings = TavilySearchSettings(is_enabled=False)
    search_tool = MagicMock()
    search_tool.ainvoke = AsyncMock(side_effect=AssertionError("must not search while disabled"))
    monkeypatch.setattr(config_module, "search_persianway", search_tool)

    response = client.post(f"{TAVILY_URL}/test", json={"query": "قیمت محصولات"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["enabled"] is False
    assert "غیرفعال" in body["message"]
    search_tool.ainvoke.assert_not_awaited()


def test_test_endpoint_requires_a_query(fake_config_service):
    assert client.post(f"{TAVILY_URL}/test", json={"query": "   "}).status_code == 400
    assert client.post(f"{TAVILY_URL}/test", json={}).status_code == 400


def test_reset_restores_only_the_tavily_section(fake_config_service):
    """A search mistake must not wipe the LLM/RAG configuration as `/config/reset` would."""
    fake_config_service.config.tavily_settings = TavilySearchSettings(
        is_enabled=False,
        tavily_api_key="tvly-wrong",
        search_depth="basic",
        max_results=1,
        include_domains=["typo.example"],
        snippet_length=7,
    )

    response = client.post(f"{TAVILY_URL}/reset")

    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["is_enabled"] is True
    assert settings["search_depth"] == "advanced"
    assert settings["max_results"] == 5
    assert settings["include_domains"] == []
    assert settings["snippet_length"] == 200
    assert settings["tavily_api_key"] == static_settings.TAVILY_API_KEY
    # Only the Tavily section was written.
    assert list(fake_config_service.update_calls[-1].keys()) == ["tavily_settings"]
