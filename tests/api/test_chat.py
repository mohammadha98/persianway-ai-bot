import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.services.chat_service import ChatService


# Create a test client
client = TestClient(app)


@pytest.fixture
def mock_chat_service():
    """Mock the chat service for testing."""
    with patch("app.api.routes.chat.get_chat_service") as mock_get_service:
        # Create a mock service
        mock_service = MagicMock(spec=ChatService)
        
        # Configure the mock to return a predefined response
        mock_service.process_message.return_value = "This is a test response from the AI."
        mock_service.get_conversation_history.return_value = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "This is a test response from the AI."}
        ]
        
        # Configure the get_chat_service function to return our mock
        mock_get_service.return_value = mock_service
        
        yield mock_service


def test_create_chat(mock_chat_service):
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
    assert "response" in data
    assert data["response"] == "This is a test response from the AI."
    assert "conversation_history" in data
    
    # Verify the service was called correctly
    mock_chat_service.process_message.assert_called_once_with(
        user_id="test_user",
        message="Test message"
    )
    mock_chat_service.get_conversation_history.assert_called_once_with("test_user")


def test_chat_error_handling(mock_chat_service):
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