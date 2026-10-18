"""OpenRouter credit monitoring (app/services/credit_monitor.py).

The monitor is the source of the panel's top-up notice, so these tests pin the
decisions that matter:

* the threshold is inclusive (`balance <= threshold`),
* an account balance needs a management key, a per-key limit does not,
* an unreadable balance never raises and never invents a number,
* the snapshot is cached for one TTL window and `force` bypasses it,
* an HTTP 402 is remembered even when no balance endpoint can be read.

Everything is faked at the config-service and HTTP seams: no network, no DB.
"""

from types import SimpleNamespace

import httpx
import pytest

from app.services import credit_monitor


def _llm(**overrides):
    values = dict(
        preferred_api_provider="openrouter",
        openrouter_api_key="sk-or-v1-inference",
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_management_key=None,
        openrouter_credit_alert_enabled=True,
        openrouter_credit_alert_threshold_usd=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def clean_cache():
    credit_monitor.reset_credit_cache()
    yield
    credit_monitor.reset_credit_cache()


def _patch_settings(monkeypatch, llm):
    async def _get_config_service():
        async def _get_llm_settings():
            return llm

        return SimpleNamespace(get_llm_settings=_get_llm_settings)

    monkeypatch.setattr(credit_monitor, "get_config_service", _get_config_service)


def _patch_http(monkeypatch, handler):
    """Route the monitor's client through an httpx.MockTransport."""

    def _client():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(credit_monitor, "_client", _client)


@pytest.mark.asyncio
async def test_a_balance_below_the_threshold_is_reported_as_low(monkeypatch):
    """A management key reads the account balance and the header carries it."""
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"data": {"total_credits": 10, "total_usage": 9.2}}
        )

    _patch_http(monkeypatch, handler)

    status = await credit_monitor.get_credit_status()

    assert len(requests) == 1
    assert requests[0].url.path == "/api/v1/credits"
    assert requests[0].headers["authorization"] == "Bearer sk-or-v1-management"
    assert status.balance_usd == pytest.approx(0.8)
    assert status.applicable is True
    assert status.is_low is True
    assert status.is_exhausted is False
    assert status.source == "account"
    assert status.error is None


@pytest.mark.asyncio
async def test_the_threshold_is_inclusive(monkeypatch):
    """Exactly 1.00 USD must alert: "when it reaches one dollar"."""
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"data": {"total_credits": 10, "total_usage": 9}}
        ),
    )

    status = await credit_monitor.get_credit_status()

    assert status.balance_usd == pytest.approx(1.0)
    assert status.is_low is True
    assert status.is_exhausted is False


@pytest.mark.asyncio
async def test_a_balance_above_the_threshold_stays_quiet(monkeypatch):
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"data": {"total_credits": 10, "total_usage": 8.99}}
        ),
    )

    status = await credit_monitor.get_credit_status()

    assert status.balance_usd == pytest.approx(1.01)
    assert status.is_low is False
    assert status.as_dict()["message"] is None


@pytest.mark.asyncio
async def test_a_zero_balance_is_exhausted_and_low(monkeypatch):
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"data": {"total_credits": 5, "total_usage": 5}}
        ),
    )

    status = await credit_monitor.get_credit_status()

    assert status.balance_usd == pytest.approx(0.0)
    assert status.is_low is True
    assert status.is_exhausted is True
    assert status.as_dict()["message"] == credit_monitor.EXHAUSTED_CREDIT_MESSAGE


@pytest.mark.asyncio
async def test_the_key_limit_is_used_without_a_management_key(monkeypatch):
    """`/key` is called with the inference key; only a limited key has a number."""
    _patch_settings(monkeypatch, _llm())
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"data": {"limit_remaining": 0.4, "usage": 19.6}}
        )

    _patch_http(monkeypatch, handler)

    status = await credit_monitor.get_credit_status()

    assert [request.url.path for request in requests] == ["/api/v1/key"]
    assert requests[0].headers["authorization"] == "Bearer sk-or-v1-inference"
    assert status.balance_usd == pytest.approx(0.4)
    assert status.is_low is True
    assert status.source == "key"
    assert status.error is None


@pytest.mark.asyncio
async def test_an_unlimited_key_leaves_the_balance_unknown(monkeypatch):
    """A null `limit_remaining` must not be read as "zero credit"."""
    _patch_settings(monkeypatch, _llm())
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"data": {"limit_remaining": None, "usage": 3}}
        ),
    )

    status = await credit_monitor.get_credit_status()

    assert status.applicable is True
    assert status.balance_usd is None
    assert status.is_low is False
    assert status.is_exhausted is False
    assert status.source == "unavailable"
    assert "management" in (status.error or "")


@pytest.mark.asyncio
async def test_a_rejected_management_key_falls_back_to_the_key_limit(monkeypatch):
    """A wrong/revoked management key (403) must not hide a readable key limit."""
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/credits":
            return httpx.Response(
                403, json={"error": {"code": 403, "message": "Only management keys"}}
            )
        return httpx.Response(200, json={"data": {"limit_remaining": 0.9}})

    _patch_http(monkeypatch, handler)

    status = await credit_monitor.get_credit_status()

    assert status.source == "key"
    assert status.balance_usd == pytest.approx(0.9)
    assert status.is_low is True
    assert (status.error or "").startswith("credits:")


@pytest.mark.asyncio
async def test_a_failed_lookup_degrades_instead_of_raising(monkeypatch):
    """The chat screen must never break because billing is down."""
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))
    _patch_http(
        monkeypatch,
        lambda request: httpx.Response(500, json={"error": {"code": 500}}),
    )

    status = await credit_monitor.get_credit_status()

    assert status.applicable is True
    assert status.balance_usd is None
    assert status.is_low is False
    assert status.source == "unavailable"
    assert "500" in (status.error or "")


@pytest.mark.asyncio
async def test_a_network_error_never_leaks_the_key(monkeypatch):
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom", request=request)

    _patch_http(monkeypatch, handler)

    status = await credit_monitor.get_credit_status()

    assert status.balance_usd is None
    assert "sk-or-v1-management" not in (status.error or "")
    assert "ConnectTimeout" in (status.error or "")


@pytest.mark.asyncio
async def test_the_snapshot_is_cached_until_a_forced_refresh(monkeypatch):
    """The panel asks after every answer; the provider is called once per TTL."""
    _patch_settings(monkeypatch, _llm(openrouter_management_key="sk-or-v1-management"))
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200, json={"data": {"total_credits": 10, "total_usage": 9.5}}
        )

    _patch_http(monkeypatch, handler)

    first = await credit_monitor.get_credit_status()
    second = await credit_monitor.get_credit_status()

    assert len(calls) == 1
    assert second is first

    forced = await credit_monitor.get_credit_status(force=True)

    assert len(calls) == 2
    assert forced.balance_usd == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_an_http_402_is_remembered_without_any_readable_balance(monkeypatch):
    """The provider itself is the authority: a 402 is enough to warn the panel."""
    _patch_settings(monkeypatch, _llm())  # no management key, no per-key limit

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("the fresh 402 snapshot must not trigger a lookup")

    _patch_http(monkeypatch, handler)

    credit_monitor.note_insufficient_credits()
    status = await credit_monitor.get_credit_status()

    assert status.is_exhausted is True
    assert status.is_low is True
    assert status.balance_usd == pytest.approx(0.0)
    assert status.source == "provider_402"
    assert status.as_dict()["message"] == credit_monitor.EXHAUSTED_CREDIT_MESSAGE


@pytest.mark.asyncio
async def test_alerts_can_be_switched_off_in_the_panel(monkeypatch):
    _patch_settings(
        monkeypatch, _llm(openrouter_credit_alert_enabled=False)
    )

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no lookup when the operator disabled the alerts")

    _patch_http(monkeypatch, handler)

    status = await credit_monitor.get_credit_status()

    assert status.applicable is False
    assert status.is_low is False
    assert status.balance_usd is None


@pytest.mark.asyncio
async def test_an_openai_only_setup_is_not_applicable(monkeypatch):
    """No OpenRouter key -> no balance to watch, and no banner."""
    _patch_settings(
        monkeypatch,
        _llm(preferred_api_provider="openai", openrouter_api_key=None),
    )

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no lookup without an OpenRouter key")

    _patch_http(monkeypatch, handler)

    status = await credit_monitor.get_credit_status()

    assert status.applicable is False
    assert status.error == "OpenRouter is not the active provider"
