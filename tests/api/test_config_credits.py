"""`GET /api/config/llm/credits` feeds the panel's top-up notice.

The route lives under `/api/config`, which is behind the panel's JWT, and it is
a monitoring read: an unreachable billing API must come back as
``success: false`` instead of a 5xx, so a provider outage can never take the
chat screen down with it.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.routes import config as config_module
from app.api.routes.users import get_current_user
from app.schemas.config import DynamicConfig, LLMSettings
from app.services.chat_service import get_chat_service
from app.services.config_service import get_config_service
from app.services.credit_monitor import LOW_CREDIT_MESSAGE, CreditStatus
from app.services.knowledge_base import get_knowledge_base_service

from main import app

client = TestClient(app)

CREDITS_URL = "/api/config/llm/credits"


@pytest.fixture
def authenticated_user():
    """The JWT dependency resolves to an active panel operator."""
    app.dependency_overrides[get_current_user] = lambda: MagicMock(
        is_active=True, username="operator", role="admin"
    )
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def _low_status() -> CreditStatus:
    return CreditStatus(
        applicable=True,
        balance_usd=0.8,
        threshold_usd=1.0,
        is_low=True,
        source="account",
        checked_at="2026-10-18T00:00:00+00:00",
    )


def test_anonymous_callers_are_rejected():
    """The balance is internal: no token, no answer."""
    response = client.get(CREDITS_URL)

    assert response.status_code in (401, 403)


def test_a_low_balance_is_answered_for_the_panel(authenticated_user, monkeypatch):
    status = _low_status()
    lookup = AsyncMock(return_value=status)
    monkeypatch.setattr(config_module, "get_credit_status", lookup)

    response = client.get(CREDITS_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["credits"]["is_low"] is True
    assert payload["credits"]["balance_usd"] == 0.8
    assert payload["credits"]["threshold_usd"] == 1.0
    assert payload["credits"]["topup_url"].startswith("https://openrouter.ai/")
    assert payload["credits"]["message"] == LOW_CREDIT_MESSAGE
    # Never cached: the panel needs the current number, not a CDN copy.
    assert "no-store" in response.headers["cache-control"]
    lookup.assert_awaited_once_with(force=False)


def test_force_is_forwarded_to_the_monitor(authenticated_user, monkeypatch):
    lookup = AsyncMock(return_value=_low_status())
    monkeypatch.setattr(config_module, "get_credit_status", lookup)

    response = client.get(f"{CREDITS_URL}?force=true")

    assert response.status_code == 200
    lookup.assert_awaited_once_with(force=True)


def test_a_failed_probe_stays_a_200_and_keeps_the_chat_usable(
    authenticated_user, monkeypatch
):
    """A billing-API outage is reported in the body, not as a 5xx."""
    lookup = AsyncMock(side_effect=RuntimeError("billing exploded"))
    monkeypatch.setattr(config_module, "get_credit_status", lookup)

    response = client.get(CREDITS_URL)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["credits"] is None
    assert "billing exploded" in payload["message"]


MANAGEMENT_KEY = "sk-or-v1-mgmt-4444"


class _RecordingConfigService:
    """A config store that merges updates the way `ConfigService` does."""

    def __init__(self):
        self.config = DynamicConfig(
            llm_settings=LLMSettings(
                openrouter_api_key="sk-or-v1-inference-1111",
                openrouter_management_key=MANAGEMENT_KEY,
                openrouter_credit_alert_enabled=True,
                openrouter_credit_alert_threshold_usd=1.0,
            )
        )
        self.updates = None

    async def clear_cache(self) -> None:
        return None

    async def get_config(self) -> DynamicConfig:
        return self.config

    async def get_llm_settings(self) -> LLMSettings:
        return self.config.llm_settings

    async def update_config(self, updates):
        self.updates = updates
        merged = self.config.dict()
        merged["llm_settings"].update(updates.get("llm_settings", {}))
        self.config = DynamicConfig(**merged)
        return self.config


@pytest.fixture
def recording_config_service():
    """Stub the config store and the two services `PUT /llm` refreshes."""
    service = _RecordingConfigService()
    kb_service = MagicMock()
    kb_service.refresh = AsyncMock()
    chat_service = MagicMock()
    chat_service.refresh = AsyncMock()

    app.dependency_overrides[get_config_service] = lambda: service
    app.dependency_overrides[get_knowledge_base_service] = lambda: kb_service
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    try:
        yield service
    finally:
        app.dependency_overrides.pop(get_config_service, None)
        app.dependency_overrides.pop(get_knowledge_base_service, None)
        app.dependency_overrides.pop(get_chat_service, None)


def test_saving_the_alert_settings_never_overwrites_the_stored_secret(
    authenticated_user, recording_config_service, monkeypatch
):
    """The panel posts the whole form back, including the masks it was served."""
    reset_cache = MagicMock()
    monkeypatch.setattr(config_module, "reset_credit_cache", reset_cache)

    response = client.put(
        "/api/config/llm",
        json={
            "openrouter_management_key": "****4444",
            "openrouter_credit_alert_enabled": False,
            "openrouter_credit_alert_threshold_usd": 2.5,
            "default_model": "openai/gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    stored = recording_config_service.config.llm_settings
    # Only the real values were written; the echoed mask was dropped.
    assert recording_config_service.updates == {
        "llm_settings": {
            "openrouter_credit_alert_enabled": False,
            "openrouter_credit_alert_threshold_usd": 2.5,
            "default_model": "openai/gpt-4o-mini",
        }
    }
    assert stored.openrouter_management_key == MANAGEMENT_KEY
    assert stored.openrouter_credit_alert_enabled is False
    assert stored.openrouter_credit_alert_threshold_usd == 2.5
    # A new key/toggle/threshold must not wait out the 60s credit cache.
    reset_cache.assert_called_once()

    settings = response.json()["settings"]
    assert settings["openrouter_management_key"] == "****4444"
    assert settings["openrouter_credit_alert_enabled"] is False
    assert settings["openrouter_credit_alert_threshold_usd"] == 2.5
    assert MANAGEMENT_KEY not in response.text
