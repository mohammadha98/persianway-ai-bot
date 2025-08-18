import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from bson import ObjectId

from app.services.conversation_service import ConversationService
from app.schemas.message import MessageRole


@pytest.fixture
def mock_collection():
    """Mock MongoDB collection for testing."""
    mock = AsyncMock()
    
    # Configure find_one to return None by default (no existing conversation)
    mock.find_one.return_value = None
    
    # Configure insert_one to return a mock result with inserted_id
    insert_result = MagicMock()
    insert_result.inserted_id = ObjectId()
    mock.insert_one.return_value = insert_result
    
    # Configure update_one to return a mock result
    update_result = MagicMock()
    update_result.modified_count = 1
    mock.update_one.return_value = update_result
    
    return mock


@pytest.fixture
def conversation_service(mock_collection):
    """Create a ConversationService instance with mocked dependencies."""
    service = ConversationService()
    service._collection = mock_collection
    return service


@pytest.mark.asyncio
async def test_store_new_conversation(conversation_service, mock_collection):
    """Test storing a new conversation when no existing conversation is found."""
    # Arrange
    user_id = "test_user"
    session_id = "test_session"
    user_question = "Test question"
    system_response = "Test response"
    
    # Act
    result = await conversation_service.store_conversation(
        user_id=user_id,
        user_question=user_question,
        system_response=system_response,
        query_analysis={},
        response_parameters={},
        session_id=session_id
    )
    
    # Assert
    assert result == str(mock_collection.insert_one.return_value.inserted_id)
    mock_collection.find_one.assert_called_once_with({"session_id": session_id})
    mock_collection.insert_one.assert_called_once()
    mock_collection.update_one.assert_not_called()


@pytest.mark.asyncio
async def test_update_existing_conversation(conversation_service, mock_collection):
    """Test updating an existing conversation when one is found with the same session_id."""
    # Arrange
    user_id = "test_user"
    session_id = "test_session"
    user_question = "Test question"
    system_response = "Test response"
    
    # Configure mock to return an existing conversation
    existing_conversation = {
        "_id": ObjectId(),
        "user_id": user_id,
        "session_id": session_id,
        "messages": [
            {
                "role": "user",
                "content": "Previous question",
                "timestamp": datetime.utcnow()
            },
            {
                "role": "assistant",
                "content": "Previous response",
                "timestamp": datetime.utcnow()
            }
        ],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_messages": 2
    }
    mock_collection.find_one.return_value = existing_conversation
    
    # Act
    result = await conversation_service.store_conversation(
        user_id=user_id,
        user_question=user_question,
        system_response=system_response,
        query_analysis={},
        response_parameters={},
        session_id=session_id
    )
    
    # Assert
    assert result == str(existing_conversation["_id"])
    mock_collection.find_one.assert_called_once_with({"session_id": session_id})
    mock_collection.insert_one.assert_not_called()
    mock_collection.update_one.assert_called_once()
    
    # Check that the update includes the new messages
    update_call_args = mock_collection.update_one.call_args[0]
    assert update_call_args[0] == {"_id": existing_conversation["_id"]}
    
    set_data = update_call_args[1]["$set"]
    assert "messages" in set_data
    assert len(set_data["messages"]) == 4  # 2 original + 2 new messages
    assert set_data["total_messages"] == 4


@pytest.mark.asyncio
async def test_update_conversation_with_user_email(conversation_service, mock_collection):
    """Test updating an existing conversation with user_email when it was previously null."""
    # Arrange
    user_id = "test_user"
    session_id = "test_session"
    user_email = "test@example.com"
    user_question = "Test question"
    system_response = "Test response"
    
    # Configure mock to return an existing conversation without user_email
    existing_conversation = {
        "_id": ObjectId(),
        "user_id": user_id,
        "session_id": session_id,
        "user_email": None,  # No email set
        "messages": [
            {
                "role": "user",
                "content": "Previous question",
                "timestamp": datetime.utcnow()
            },
            {
                "role": "assistant",
                "content": "Previous response",
                "timestamp": datetime.utcnow()
            }
        ],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_messages": 2
    }
    mock_collection.find_one.return_value = existing_conversation
    
    # Act
    result = await conversation_service.store_conversation(
        user_id=user_id,
        user_question=user_question,
        system_response=system_response,
        query_analysis={},
        response_parameters={},
        session_id=session_id,
        user_email=user_email
    )
    
    # Assert
    assert result == str(existing_conversation["_id"])
    mock_collection.update_one.assert_called_once()
    
    # Check that the update includes the user_email
    update_call_args = mock_collection.update_one.call_args[0]
    set_data = update_call_args[1]["$set"]
    assert "user_email" in set_data
    assert set_data["user_email"] == user_email