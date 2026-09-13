import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from main import app
from app.services.chat_service import ChatService, get_chat_service
from app.services.conversation_service import get_conversation_service


# Create a test client
client = TestClient(app)


@pytest.fixture
def mock_chat_service():
    """Mock the chat service and inject it via FastAPI's dependency override.

    Patching `app.api.routes.chat.get_chat_service` after `Depends()` has been
    evaluated does not work, so we override the dependency itself.
    """
    mock_service = MagicMock(spec=ChatService)

    # `process_message` / `generate_conversation_title` are awaited by the route.
    mock_service.process_message = AsyncMock(return_value={
        "answer": "This is a test response from the AI.",
        "query_analysis": {
            "confidence_score": 0.9,
            "knowledge_source": "knowledge_base",
            "requires_human_referral": False,
            "reasoning": "Test fixture.",
        },
        "response_parameters": {
            "model": "test-model",
            "temperature": 0.2,
            "max_tokens": 128,
            "top_p": 1.0,
        },
    })
    mock_service.generate_conversation_title = AsyncMock(return_value="Test Conversation")

    # `get_conversation_history` is called synchronously.
    mock_service.get_conversation_history.return_value = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "This is a test response from the AI."}
    ]

    app.dependency_overrides[get_chat_service] = lambda: mock_service
    try:
        yield mock_service
    finally:
        app.dependency_overrides.pop(get_chat_service, None)


@pytest.fixture
def mock_conversation_service():
    """Mock the conversation service and inject it via a dependency override."""
    mock_service = MagicMock()

    # `store_conversation` is awaited and must return a dict of IDs.
    mock_service.store_conversation = AsyncMock(return_value={
        "conversation_id": "test_conversation_id",
        "assistant_message_id": "test_message_id",
    })

    app.dependency_overrides[get_conversation_service] = lambda: mock_service
    try:
        yield mock_service
    finally:
        app.dependency_overrides.pop(get_conversation_service, None)


def test_create_chat(mock_chat_service, mock_conversation_service):
    """Test the chat endpoint."""
    # Test data
    test_request = {
        "user_id": "test_user",
        "message": "Test message"
    }

    # Make the request
    response = client.post("/api/chat/", json=test_request)

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test response from the AI."
    assert "conversation_history" in data
    assert data["conversation_id"] == "test_conversation_id"
    assert data["message_id"] == "test_message_id"

    # Verify the service was called correctly
    mock_chat_service.process_message.assert_awaited_once()
    call_kwargs = mock_chat_service.process_message.await_args.kwargs
    assert call_kwargs["user_id"] == "test_user"
    assert call_kwargs["message"] == "Test message"
    mock_chat_service.get_conversation_history.assert_called_once_with("test_user")

    # Verify the conversation service was called
    mock_conversation_service.store_conversation.assert_awaited_once()


def test_chat_error_handling(mock_chat_service, mock_conversation_service):
    """Test error handling in the chat endpoint."""
    # Configure the mock to raise an exception
    mock_chat_service.process_message.side_effect = Exception("Test error")

    # Test data
    test_request = {
        "user_id": "test_user",
        "message": "Test message"
    }

    # Make the request
    response = client.post("/api/chat/", json=test_request)

    # Check the response
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Test error" in data["detail"]

    # Verify the conversation service was not called due to the error
    mock_conversation_service.store_conversation.assert_not_called()
