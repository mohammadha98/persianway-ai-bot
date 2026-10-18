"""The masking helpers behind `GET /api/config/*`.

`app/api/routes/config.py` used to answer with the stored provider keys verbatim,
which made the API the easiest way to steal them. These tests pin the two halves
of the repair: a stored secret is never echoed back, and the masked value the
panel echoes on save is never written over the real one.
"""

import pytest

from app.api.routes.config import (
    LLM_SECRET_FIELDS,
    MASKED_SECRET_PREFIX,
    TAVILY_SECRET_FIELDS,
    _is_secret_placeholder,
    _mask_config,
    _mask_secret,
    _masked_fields,
    _strip_secret_placeholders,
)
from app.schemas.config import DynamicConfig, LLMSettings, TavilySearchSettings

OPENROUTER_KEY = "sk-or-v1-abcdef123456"
OPENAI_KEY = "sk-proj-abcdef7890"
TAVILY_KEY = "tvly-abcdef4321"
MANAGEMENT_KEY = "sk-or-v1-mgmt-9999"


class TestMaskSecret:
    def test_nothing_stored_stays_empty(self):
        """The panel shows an empty field when no key is configured."""
        assert _mask_secret(None) is None
        assert _mask_secret("") == ""

    def test_real_key_keeps_only_the_last_four_characters(self):
        assert _mask_secret(OPENROUTER_KEY) == f"{MASKED_SECRET_PREFIX}3456"

    def test_a_very_short_key_leaks_no_character(self):
        assert _mask_secret("abc") == MASKED_SECRET_PREFIX
        assert _mask_secret("****") == MASKED_SECRET_PREFIX


class TestMaskedFields:
    def test_only_the_named_fields_are_replaced(self):
        data = {"default_model": "x", "openai_api_key": OPENAI_KEY}

        masked = _masked_fields(data, LLM_SECRET_FIELDS)

        assert masked["default_model"] == "x"
        assert masked["openai_api_key"] == f"{MASKED_SECRET_PREFIX}7890"

    def test_the_source_document_is_not_touched(self):
        """The stored settings object must keep the real key for the LLM client."""
        data = {"openai_api_key": OPENAI_KEY}

        _masked_fields(data, LLM_SECRET_FIELDS)

        assert data["openai_api_key"] == OPENAI_KEY

    def test_absent_fields_stay_absent(self):
        assert _masked_fields({"temperature": 0.5}, TAVILY_SECRET_FIELDS) == {
            "temperature": 0.5
        }


class TestIsSecretPlaceholder:
    @pytest.mark.parametrize(
        "value",
        [f"{MASKED_SECRET_PREFIX}3456", "", "   ", None, 0, ["sk"], f" {MASKED_SECRET_PREFIX}"],
    )
    def test_placeholders_are_recognised(self, value):
        assert _is_secret_placeholder(value) is True

    def test_a_real_key_is_not_a_placeholder(self):
        assert _is_secret_placeholder(OPENROUTER_KEY) is False


class TestStripSecretPlaceholders:
    def test_echoed_mask_keeps_the_stored_key(self):
        """What the panel posts after loading a masked `GET` must not overwrite it."""
        updates = {"openai_api_key": f"{MASKED_SECRET_PREFIX}7890", "max_tokens": 0}

        assert _strip_secret_placeholders(updates, LLM_SECRET_FIELDS) == {"max_tokens": 0}

    def test_a_cleared_field_means_unchanged(self):
        """An operator clearing the password box must not blank a working key."""
        updates = {"tavily_api_key": "  "}

        assert _strip_secret_placeholders(updates, TAVILY_SECRET_FIELDS) == {}

    def test_a_new_key_replaces_the_stored_one(self):
        updates = {"openai_api_key": "sk-proj-brand-new"}

        assert _strip_secret_placeholders(updates, LLM_SECRET_FIELDS) == updates

    def test_non_secret_fields_are_never_dropped(self):
        updates = {"search_depth": "basic", "include_domains": [], "snippet_length": 0}

        assert _strip_secret_placeholders(updates, TAVILY_SECRET_FIELDS) == updates

    def test_empty_values_in_plain_fields_are_kept(self):
        """Only the named fields are secrets; an empty model name is not a key."""
        updates = {"default_model": "x", "availability": ""}

        assert _strip_secret_placeholders(updates, LLM_SECRET_FIELDS) == updates

    def test_the_panel_round_trip_is_lossless_for_secrets(self):
        """Mask -> echo back -> strip leaves the stored value untouched."""
        masked_read = _masked_fields(
            {"openai_api_key": OPENAI_KEY, "max_tokens": 512}, LLM_SECRET_FIELDS
        )

        assert _strip_secret_placeholders(masked_read, LLM_SECRET_FIELDS) == {
            "max_tokens": 512
        }


class TestMaskConfig:
    def test_nested_llm_and_tavily_keys_are_masked(self):
        config = DynamicConfig(
            llm_settings=LLMSettings(
                openai_api_key=OPENAI_KEY,
                openrouter_api_key=OPENROUTER_KEY,
                # Optional, but still a credential: it reads the account balance.
                openrouter_management_key=MANAGEMENT_KEY,
            ),
            tavily_settings=TavilySearchSettings(tavily_api_key=TAVILY_KEY),
        )

        masked = _mask_config(config)

        assert masked.llm_settings.openai_api_key == f"{MASKED_SECRET_PREFIX}7890"
        assert masked.llm_settings.openrouter_api_key == f"{MASKED_SECRET_PREFIX}3456"
        assert masked.llm_settings.openrouter_management_key == f"{MASKED_SECRET_PREFIX}9999"
        assert masked.tavily_settings.tavily_api_key == f"{MASKED_SECRET_PREFIX}4321"

    def test_the_stored_config_keeps_its_keys(self):
        """The chat/KB services read the same object, so it must stay authoritative."""
        config = DynamicConfig(
            llm_settings=LLMSettings(openrouter_api_key=OPENROUTER_KEY)
        )

        _mask_config(config)

        assert config.llm_settings.openrouter_api_key == OPENROUTER_KEY

    def test_everything_else_is_answered_unchanged(self):
        config = DynamicConfig(
            llm_settings=LLMSettings(default_model="openai/gpt-4o", max_tokens=99)
        )

        masked = _mask_config(config)

        assert masked.llm_settings.default_model == "openai/gpt-4o"
        assert masked.llm_settings.max_tokens == 99
