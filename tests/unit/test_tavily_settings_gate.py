"""The panel's web-search switch must actually switch the search off.

`GET/PUT /api/config/tavily` only stores `is_enabled`; these tests pin the other
half of the contract — the retrieval code and the Tavily service honour it, and
the request parameters the panel exposes are the ones sent to Tavily.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.schemas.config import DynamicConfig, TavilySearchSettings
from app.services import knowledge_base, utility


def _fake_config_service(config: DynamicConfig):
    class _FakeConfigService:
        async def get_config(self) -> DynamicConfig:
            return config

    return _FakeConfigService()


def _patch_config(monkeypatch, module, settings: TavilySearchSettings) -> None:
    """Point `get_config_service` in `module` at a config carrying `settings`."""

    async def _get_config_service():
        return _fake_config_service(DynamicConfig(tavily_settings=settings))

    monkeypatch.setattr(module, "get_config_service", _get_config_service)


@pytest.mark.asyncio
async def test_search_is_skipped_when_disabled_in_the_panel(monkeypatch):
    """Disabled wins over an API key: no client is built and no request is made."""
    _patch_config(monkeypatch, utility, TavilySearchSettings(is_enabled=False, tavily_api_key="tvly-test"))

    client_constructor = MagicMock(side_effect=AssertionError("TavilyClient must not be built"))
    monkeypatch.setattr(utility, "TavilyClient", client_constructor)
    # Even a key in `.env` must not resurrect a disabled feature.
    monkeypatch.setattr(utility, "settings", SimpleNamespace(TAVILY_API_KEY="tvly-from-env"))

    service = utility.TavilySearchService()
    result = await service.search("قیمت محصولات")

    assert result["disabled"] is True
    assert result["results"] == []
    assert utility.TAVILY_DISABLED_MESSAGE in result["error"]
    client_constructor.assert_not_called()
    assert service.client is None


@pytest.mark.asyncio
async def test_search_web_reports_disabled_without_returning_content(monkeypatch):
    """The disabled notice keeps the `Error searching web` marker callers filter on."""
    _patch_config(monkeypatch, utility, TavilySearchSettings(is_enabled=False))

    output = await utility.search_web("قیمت محصولات")

    assert output.startswith("Error searching web:")
    assert utility.TAVILY_DISABLED_MESSAGE in output


@pytest.mark.asyncio
async def test_enabled_search_forwards_every_panel_parameter(monkeypatch):
    """The saved depth/results/domains/snippet length are what Tavily receives."""
    panel_settings = TavilySearchSettings(
        is_enabled=True,
        tavily_api_key="tvly-panel",
        search_depth="basic",
        max_results=3,
        include_answer=False,
        include_domains=["persianway.ir"],
        exclude_domains=["spam.example"],
        snippet_length=42,
    )
    _patch_config(monkeypatch, utility, panel_settings)
    # An env key must never shadow the key stored from the panel.
    monkeypatch.setattr(utility, "settings", SimpleNamespace(TAVILY_API_KEY="tvly-from-env"))

    client_instance = MagicMock()
    client_instance.search.return_value = {"answer": "پاسخ", "results": [{"title": "t", "url": "u", "content": "c"}]}
    monkeypatch.setattr(utility, "TavilyClient", MagicMock(return_value=client_instance))

    service = utility.TavilySearchService()
    result = await service.search("قیمت محصولات")

    assert result["answer"] == "پاسخ"
    utility.TavilyClient.assert_called_once_with(api_key="tvly-panel")
    kwargs = client_instance.search.call_args.kwargs
    assert kwargs["query"] == "قیمت محصولات"
    assert kwargs["search_depth"] == "basic"
    assert kwargs["max_results"] == 3
    assert kwargs["include_answer"] is False
    assert kwargs["include_domains"] == ["persianway.ir"]
    assert kwargs["exclude_domains"] == ["spam.example"]

    formatted = await utility.search_web("قیمت محصولات")
    assert "پاسخ" in formatted
    # snippet_length from the panel truncates the source snippet.
    assert "c..." in formatted


@pytest.mark.asyncio
async def test_retrieval_switch_follows_the_panel(monkeypatch):
    """`_web_search_enabled` is the retrieval-side read of the panel switch."""
    _patch_config(monkeypatch, knowledge_base, TavilySearchSettings(is_enabled=False))
    assert await knowledge_base._web_search_enabled() is False

    _patch_config(monkeypatch, knowledge_base, TavilySearchSettings(is_enabled=True))
    assert await knowledge_base._web_search_enabled() is True


@pytest.mark.asyncio
async def test_retrieval_switch_defaults_to_enabled_when_config_is_unreadable(monkeypatch):
    """A transient database failure must not silently disable search."""

    async def _broken_config_service():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(knowledge_base, "get_config_service", _broken_config_service)

    assert await knowledge_base._web_search_enabled() is True
