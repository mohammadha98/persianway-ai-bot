import pytest
from unittest.mock import AsyncMock, patch

# Ensure project root is on sys.path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain.schema import AIMessage

from app.services.chat_service import ChatService
from app.schemas.chat import ChatMessage


class DummyLLM:
    """A lightweight fake LLM for intent detection tests."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.captured_messages = None

    async def ainvoke(self, messages):
        self.captured_messages = messages
        return AIMessage(content=self.response_content)


@pytest.mark.asyncio
async def test_detect_public_data_intent_public_query():
    """LLM indicates the message is about public company information."""
    service = ChatService()
    dummy_llm = DummyLLM('{"public_data": true, "explanation": "General question about PersianWay."}')

    with patch("app.services.chat_service.get_llm", new=AsyncMock(return_value=dummy_llm)) as mock_get_llm:
        result = await service.detect_public_data_intent("PersianWay mission statement?", [])

    assert result is True
    mock_get_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_detect_public_data_intent_product_issue():
    """LLM indicates the message is about a private consultation."""
    service = ChatService()
    dummy_llm = DummyLLM('{"public_data": false, "explanation": "User describing a product defect."}')

    with patch("app.services.chat_service.get_llm", new=AsyncMock(return_value=dummy_llm)):
        result = await service.detect_public_data_intent(
            "My irrigation pump from PersianWay broke; how do I fix it?",
            [ChatMessage(role="user", content="Hi, I need help with my product.")]
        )

    assert result is False


@pytest.mark.asyncio
async def test_detect_public_data_intent_includes_history_in_prompt():
    """Ensure conversation history is passed to the LLM for classification."""
    service = ChatService()
    dummy_llm = DummyLLM('{"public_data": true}')
    history = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="How can I help you today?")
    ]

    with patch("app.services.chat_service.get_llm", new=AsyncMock(return_value=dummy_llm)):
        await service.detect_public_data_intent("Tell me about PersianWay's services.", history)

    assert dummy_llm.captured_messages is not None
    assert any("Hello" in msg.content for msg in dummy_llm.captured_messages if hasattr(msg, "content"))


@pytest.mark.asyncio
async def test_detect_public_data_intent_handles_invalid_llm_response():
    """Invalid responses from the LLM should default to False."""
    service = ChatService()
    dummy_llm = DummyLLM("Unexpected response without JSON")

    with patch("app.services.chat_service.get_llm", new=AsyncMock(return_value=dummy_llm)):
        result = await service.detect_public_data_intent("What is PersianWay?", None)

    assert result is False

