"""`/api/config/tavily` is the contract behind the admin panel's web-search page.

Covers what the panel depends on: fresh (uncached) reads, partial saves that keep
the untouched fields, input normalisation/validation, the test-search button and
the scoped reset.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.routes import config as config_module
from app.api.routes.users import get_current_user
from app.core.config import settings as static_settings
from app.schemas.config import DynamicConfig, TavilySearchSettings
from app.services.chat_service import get_chat_service
from app.services.config_service import get_config_service
from app.services.knowledge_base import get_knowledge_base_service

from main import app

client = TestClient(app)
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


@pytest.fixture(autouse=True)
def authenticated_panel():
    """`/api/config/*` sits behind the panel's JWT; these tests pin the payload.

    Anonymous access is covered separately in `tests/api/test_config_auth.py`, so
    here the token check is stubbed out and every case exercises the real route.
    """
    app.dependency_overrides[get_current_user] = lambda: MagicMock(username="operator")
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)


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


def test_get_tavily_settings_is_never_cached_masks_the_key(fake_config_service):
    """Multi-instance deployments need the stored document, not the process cache.

    The stored key never leaves the API verbatim: an operator only needs to
    recognise which key is active.
    """
    fake_config_service.config.tavily_settings = TavilySearchSettings(
        tavily_api_key="tvly-secret-abcd"
    )

    response = client.get(TAVILY_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["settings"]["is_enabled"] is True
    assert body["settings"]["search_depth"] == "advanced"
    assert body["settings"]["tavily_api_key"] == "****abcd"
    assert "tvly-secret-abcd" not in response.text
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
    # The locked default: web search goes back to reading only Persian Way.
    assert settings["include_domains"] == ["persianway.ir"]
    assert settings["snippet_length"] == 200
    # The response only carries a masked key; the `.env` value is what got stored.
    assert settings["tavily_api_key"] != static_settings.TAVILY_API_KEY
    stored = fake_config_service.config.tavily_settings.tavily_api_key
    assert stored == static_settings.TAVILY_API_KEY
    if stored:
        assert settings["tavily_api_key"] == f"****{stored[-4:]}"
    # Only the Tavily section was written.
    assert list(fake_config_service.update_calls[-1].keys()) == ["tavily_settings"]


def test_put_stores_a_new_key_masked_in_the_response(fake_config_service):
    """A real key replaces the stored one, and the answer still hides it."""
    fake_config_service.config.tavily_settings = TavilySearchSettings(
        tavily_api_key="tvly-old-key"
    )

    response = client.put(TAVILY_URL, json={"tavily_api_key": "tvly-new-key-6789"})

    assert response.status_code == 200
    assert fake_config_service.config.tavily_settings.tavily_api_key == "tvly-new-key-6789"
    assert response.json()["settings"]["tavily_api_key"] == "****6789"
    assert "tvly-new-key-6789" not in response.text


def test_put_keeps_the_stored_key_when_the_panel_echoes_the_mask(fake_config_service):
    """The panel posts the whole form back; the mask it read must not wipe the key.

    `tools.component.ts` loads the form from the masked `GET` and saves it as-is
    when an operator only flips a switch, which used to be a key-overwriting path.
    """
    fake_config_service.config.tavily_settings = TavilySearchSettings(
        tavily_api_key="tvly-stored-key"
    )

    response = client.put(
        TAVILY_URL, json={"is_enabled": False, "tavily_api_key": "****-key"}
    )

    assert response.status_code == 200
    assert fake_config_service.config.tavily_settings.tavily_api_key == "tvly-stored-key"
    assert fake_config_service.config.tavily_settings.is_enabled is False
    assert "tavily_api_key" not in fake_config_service.update_calls[-1]["tavily_settings"]


def test_put_keeps_the_stored_key_when_the_field_is_cleared(fake_config_service):
    """An empty field means "unchanged", so a key is never blanked by accident."""
    fake_config_service.config.tavily_settings = TavilySearchSettings(
        tavily_api_key="tvly-stored-key", is_enabled=False
    )

    response = client.put(TAVILY_URL, json={"is_enabled": True, "tavily_api_key": "   "})

    assert response.status_code == 200
    assert fake_config_service.config.tavily_settings.tavily_api_key == "tvly-stored-key"
    assert fake_config_service.config.tavily_settings.is_enabled is True


def test_put_rejects_a_document_that_only_carries_a_placeholder(fake_config_service):
    """A placeholder is not a setting: nothing would be written, so say so loudly."""
    response = client.put(TAVILY_URL, json={"tavily_api_key": "****-key"})

    assert response.status_code == 400
    assert fake_config_service.update_calls == []


def test_put_saves_the_full_panel_form(fake_config_service):
    """Every field the panel renders is writable through this one endpoint."""
    response = client.put(
        TAVILY_URL,
        json={
            "tavily_api_key": "tvly-panel-key-1234",
            "is_enabled": True,
            "search_depth": "basic",
            "max_results": 7,
            "include_answer": False,
            "include_domains": ["persianway.ir"],
            "exclude_domains": ["ads.example"],
            "snippet_length": 321,
        },
    )

    assert response.status_code == 200
    stored = fake_config_service.config.tavily_settings
    assert stored.tavily_api_key == "tvly-panel-key-1234"
    assert stored.is_enabled is True
    assert stored.search_depth == "basic"
    assert stored.max_results == 7
    assert stored.include_answer is False
    assert stored.include_domains == ["persianway.ir"]
    assert stored.exclude_domains == ["ads.example"]
    assert stored.snippet_length == 321
    # Only the web search section was written, and the answer stays masked.
    assert list(fake_config_service.update_calls[-1].keys()) == ["tavily_settings"]
    assert response.json()["settings"]["tavily_api_key"] == "****1234"


def test_put_rejects_an_empty_allowed_domains_list(fake_config_service):
    """Clearing the field must not turn the site search into a whole-web search."""
    response = client.put(TAVILY_URL, json={"include_domains": []})

    assert response.status_code == 400
    assert "include_domains" in response.json()["detail"]
    assert fake_config_service.update_calls == []


def test_put_rejects_a_blank_allowed_domains_string(fake_config_service):
    """The panel sends the textarea as a comma separated string; blank stays empty."""
    response = client.put(TAVILY_URL, json={"include_domains": " , "})

    assert response.status_code == 400
    assert fake_config_service.update_calls == []


def test_a_fresh_document_is_locked_to_persianway(fake_config_service):
    """Defaults matter: a wiped database must not open the whole web to Tavily."""
    response = client.get(TAVILY_URL)

    assert response.status_code == 200
    assert response.json()["settings"]["include_domains"] == ["persianway.ir"]
    assert TavilySearchSettings().include_domains == ["persianway.ir"]


def test_test_endpoint_reports_the_domains_the_search_used(fake_config_service, monkeypatch):
    """An empty stored list still searches persianway.ir, and the panel says so."""
    search_tool = MagicMock()
    search_tool.ainvoke = AsyncMock(return_value="Answer: x")
    monkeypatch.setattr(config_module, "search_persianway", search_tool)
    fake_config_service.config.tavily_settings = TavilySearchSettings(include_domains=[])

    response = client.post(f"{TAVILY_URL}/test", json={"query": "price"})

    assert response.status_code == 200
    assert response.json()["include_domains"] == ["persianway.ir"]
