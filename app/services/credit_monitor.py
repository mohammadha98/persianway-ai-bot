"""OpenRouter credit monitoring.

Production symptom this exists for: when the OpenRouter account runs out of
credit, every answer dies with an HTTP 402 and the operator finds out only by
opening the OpenRouter dashboard -- in the panel the turn just fails. The
monitor reads the remaining balance and caches it, so the panel can show a
top-up notice as soon as the balance reaches the configured threshold
(`LLMSettings.openrouter_credit_alert_threshold_usd`, 1 USD by default).

Two sources, in order (see OpenRouter's "Limits" documentation):

  * ``GET {base}/credits`` -> ``total_credits - total_usage`` = the account
    balance. Requires a **management key**: OpenRouter answers a normal
    inference key with 403 "Only management keys can perform this operation".
  * ``GET {base}/key`` -> the inference key's own ``limit_remaining``. Only
    meaningful when that key was created with a spend limit; otherwise the
    field is ``null`` and the balance stays unknown.

A third, always-available signal lives in ``chat_service``: an HTTP 402 from the
chat completion itself proves the balance is gone, even when neither endpoint
can be read (``note_insufficient_credits``).

Nothing here may ever break a request: a failed lookup degrades to
``source="unavailable"`` with a short, secret-free error string, and the caller
(the panel) simply hides the numeric notice.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from app.services.config_service import get_config_service

# One upstream call per minute is enough: a balance moves slowly, and the panel
# asks after every answer (chat.component.ts -> refreshCreditAlert).
CACHE_TTL_SECONDS = 60.0

# Billing endpoints are not on the answer path, so a slow one is simply skipped
# instead of being retried: the 402 path remains the guarantee.
REQUEST_TIMEOUT_SECONDS = 5.0

TOPUP_URL = "https://openrouter.ai/settings/credits"

LOW_CREDIT_MESSAGE = (
    "اعتبار حساب OpenRouter رو به پایان است. برای اینکه پاسخ‌دهی به کاربران "
    "قطع نشود، حساب را شارژ کنید."
)
EXHAUSTED_CREDIT_MESSAGE = (
    "اعتبار حساب OpenRouter تمام شده است. تا شارژ نشدن حساب، پاسخ‌دهی به "
    "کاربران امکان‌پذیر نیست."
)


class CreditLookupError(Exception):
    """Internal: the provider's billing API could not be read.

    The message is deliberately short and never contains the API key, so it can
    be logged and handed to the panel as-is.
    """


@dataclass
class CreditStatus:
    """One balance observation, in the shape the panel consumes."""

    provider: str = "openrouter"
    # False = OpenRouter is not the active provider (or alerts are switched off
    # in the panel): the UI must not show anything at all.
    applicable: bool = False
    balance_usd: Optional[float] = None
    threshold_usd: float = 1.0
    is_low: bool = False
    is_exhausted: bool = False
    # account | key | provider_402 | unavailable
    source: str = "unavailable"
    checked_at: str = ""
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        if self.is_exhausted:
            message: Optional[str] = EXHAUSTED_CREDIT_MESSAGE
        elif self.is_low:
            message = LOW_CREDIT_MESSAGE
        else:
            message = None

        return {
            "provider": self.provider,
            "applicable": self.applicable,
            "balance_usd": self.balance_usd,
            "threshold_usd": self.threshold_usd,
            "is_low": self.is_low,
            "is_exhausted": self.is_exhausted,
            "source": self.source,
            "checked_at": self.checked_at,
            "message": message,
            "topup_url": TOPUP_URL,
            "error": self.error,
        }


# Process-local cache. Every gunicorn worker keeps its own copy, which is fine:
# worst case each worker calls OpenRouter once per TTL window.
_snapshot: Optional[CreditStatus] = None
_snapshot_at: float = 0.0

# One lock per running loop: the cache above is reset between tests, and a lock
# bound to a previous event loop would raise on re-use.
_lock: Optional[asyncio.Lock] = None
_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_lock() -> asyncio.Lock:
    global _lock, _lock_loop
    loop = asyncio.get_running_loop()
    if _lock is None or _lock_loop is not loop:
        _lock, _lock_loop = asyncio.Lock(), loop
    return _lock


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_fresh() -> bool:
    return _snapshot is not None and (time.monotonic() - _snapshot_at) < CACHE_TTL_SECONDS


def reset_credit_cache() -> None:
    """Drop the cached snapshot (settings changed, or a test starts)."""
    global _snapshot, _snapshot_at
    _snapshot = None
    _snapshot_at = 0.0


# ==================== Provider requests ====================


def _client() -> httpx.AsyncClient:
    """Client factory, kept separate so tests can inject an httpx.MockTransport."""
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)


async def _request_json(url: str, api_key: str) -> Dict[str, Any]:
    """GET `url` with `api_key`, mapping every failure to `CreditLookupError`."""
    try:
        async with _client() as client:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {api_key}"}
            )
    except httpx.HTTPError as exc:
        # Timeouts and connection errors: name the type only, never the request.
        raise CreditLookupError(f"request failed ({type(exc).__name__})") from exc

    if response.status_code == 401:
        raise CreditLookupError("invalid API key (401)")
    if response.status_code == 403:
        # Exactly what an inference key gets from /credits.
        raise CreditLookupError("key is not allowed to read the balance (403)")
    if response.status_code >= 400:
        raise CreditLookupError(f"HTTP {response.status_code}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise CreditLookupError("response was not valid JSON") from exc

    return payload if isinstance(payload, dict) else {}


def _data_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    """OpenRouter wraps every payload in a `data` object."""
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    """Numbers arrive as int/float/str; anything else means "not reported"."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _fetch_account_balance(
    base_url: str, management_key: str
) -> Tuple[float, Optional[float]]:
    """Account balance from `GET /credits` (management key only)."""
    payload = await _request_json(f"{base_url}/credits", management_key)
    data = _data_section(payload)
    total = _as_float(data.get("total_credits"))
    used = _as_float(data.get("total_usage"))
    if total is None or used is None:
        raise CreditLookupError("credits response is missing total_credits/total_usage")
    return total - used, used


async def _fetch_key_balance(
    base_url: str, api_key: str
) -> Tuple[Optional[float], Optional[float]]:
    """Key balance from `GET /key`; `limit_remaining` is null without a limit."""
    payload = await _request_json(f"{base_url}/key", api_key)
    data = _data_section(payload)
    return _as_float(data.get("limit_remaining")), _as_float(data.get("usage"))


# ==================== Snapshot ====================


async def _refresh() -> CreditStatus:
    """Read the balance once, trying the management key before the key limit."""
    checked_at = _utc_now_iso()

    try:
        config_service = await get_config_service()
        llm = await config_service.get_llm_settings()
    except Exception as exc:  # database/config unavailable
        logger.warning(f"[CREDITS] settings unavailable: {type(exc).__name__}: {exc}")
        return CreditStatus(
            checked_at=checked_at,
            error=f"settings unavailable ({type(exc).__name__})",
        )

    threshold = float(llm.openrouter_credit_alert_threshold_usd or 0.0)

    if not llm.openrouter_credit_alert_enabled:
        return CreditStatus(
            applicable=False,
            threshold_usd=threshold,
            checked_at=checked_at,
            error="credit alerts are disabled in the panel",
        )

    # `auto` also ends up on OpenRouter when its key is the one configured; the
    # selection logic lives in `get_llm` and must not be duplicated here.
    if llm.preferred_api_provider not in ("openrouter", "auto") or not llm.openrouter_api_key:
        return CreditStatus(
            applicable=False,
            threshold_usd=threshold,
            checked_at=checked_at,
            error="OpenRouter is not the active provider",
        )

    base_url = (llm.openrouter_api_base or "https://openrouter.ai/api/v1").rstrip("/")
    management_key = (llm.openrouter_management_key or "").strip()

    errors: List[str] = []
    balance: Optional[float] = None
    usage: Optional[float] = None
    source = "unavailable"

    if management_key:
        try:
            balance, usage = await _fetch_account_balance(base_url, management_key)
            source = "account"
        except CreditLookupError as exc:
            # A wrong/revoked management key must not hide a per-key limit that
            # is perfectly readable: fall through to `/key`.
            errors.append(f"credits: {exc}")

    if balance is None:
        try:
            balance, usage = await _fetch_key_balance(base_url, llm.openrouter_api_key)
            if balance is None:
                errors.append(
                    "key: no management key and the inference key has no spend limit"
                )
            else:
                source = "key"
        except CreditLookupError as exc:
            errors.append(f"key: {exc}")

    status = CreditStatus(
        applicable=True,
        balance_usd=balance,
        threshold_usd=threshold,
        is_low=balance is not None and balance <= threshold,
        is_exhausted=balance is not None and balance <= 0,
        source=source,
        checked_at=checked_at,
        error="; ".join(errors) or None,
    )

    if status.is_low:
        logger.warning(
            "[CREDITS] OpenRouter balance is low: "
            f"{balance:.4f} USD (threshold {threshold:.2f} USD, source={source})"
        )
    elif balance is not None:
        logger.info(f"[CREDITS] OpenRouter balance: {balance:.4f} USD (source={source})")
    else:
        logger.info(f"[CREDITS] OpenRouter balance unknown: {status.error}")

    return status


async def get_credit_status(force: bool = False) -> CreditStatus:
    """Cached balance snapshot; at most one upstream call per TTL window.

    Concurrent callers wait for the in-flight refresh instead of each starting
    their own (the panel asks after every answer, and a burst of answers should
    not become a burst of billing calls).
    """
    global _snapshot, _snapshot_at

    if not force and _is_fresh():
        return _snapshot  # type: ignore[return-value]

    async with _get_lock():
        # Double-checked: another coroutine may have refreshed while we waited.
        if not force and _is_fresh():
            return _snapshot  # type: ignore[return-value]

        _snapshot = await _refresh()
        _snapshot_at = time.monotonic()
        return _snapshot


def note_insufficient_credits(threshold_usd: float = 1.0) -> None:
    """Remember an HTTP 402 from the chat completion itself.

    The provider is the authority here: a 402 proves the balance is gone even
    when neither balance endpoint can be read (no management key, no per-key
    limit, or a network failure). The snapshot stays fresh for one TTL window,
    then the next refresh either confirms it or clears it after a top-up.
    """
    global _snapshot, _snapshot_at

    _snapshot = CreditStatus(
        applicable=True,
        balance_usd=0.0,
        threshold_usd=threshold_usd,
        is_low=True,
        is_exhausted=True,
        source="provider_402",
        checked_at=_utc_now_iso(),
        error="the provider answered HTTP 402 (insufficient credits)",
    )
    _snapshot_at = time.monotonic()
    logger.error("[CREDITS] OpenRouter answered HTTP 402: the account is out of credit")
