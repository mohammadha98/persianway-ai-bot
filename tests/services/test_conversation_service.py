import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from bson import ObjectId

from app.services.conversation_service import (
    ConversationService,
    _coerce_source_to_string,
    _coerce_sources_to_strings,
    SOURCES_PREVIEW_MAX_LENGTH,
)
from app.schemas.message import MessageRole


# ==================== Unit Tests: Source Coercion ====================

class TestCoerceSourceToString:
    """Unit tests for _coerce_source_to_string helper."""

    def test_string_passthrough(self):
        assert _coerce_source_to_string("source.pdf") == "source.pdf"

    def test_none_returns_empty(self):
        assert _coerce_source_to_string(None) == ""

    def test_dict_with_content(self):
        item = {"content": "This is the content", "source": "doc.pdf", "page": 1}
        assert _coerce_source_to_string(item) == "This is the content"

    def test_dict_with_page_content(self):
        item = {"page_content": "Page content here", "source": "doc.pdf"}
        assert _coerce_source_to_string(item) == "Page content here"

    def test_dict_fallback_to_source_name(self):
        item = {"source": "doc.pdf", "page": 1}
        assert _coerce_source_to_string(item) == "doc.pdf"

    def test_dict_str_fallback(self):
        item = {"title": "Some Title"}
        result = _coerce_source_to_string(item)
        assert "Some Title" in result

    def test_truncation_at_200_chars(self):
        long_content = "x" * 500
        item = {"content": long_content}
        result = _coerce_source_to_string(item)
        assert len(result) == SOURCES_PREVIEW_MAX_LENGTH + 3  # +3 for "..."
        assert result.endswith("...")

    def test_non_dict_non_str(self):
        result = _coerce_source_to_string(12345)
        assert result == "12345"


class TestCoerceSourcesToStrings:
    """Unit tests for _coerce_sources_to_strings helper."""

    def test_none_returns_none(self):
        assert _coerce_sources_to_strings(None) is None

    def test_empty_list_returns_none(self):
        assert _coerce_sources_to_strings([]) is None

    def test_list_of_strings(self):
        sources = ["source1.pdf", "source2.pdf"]
        result = _coerce_sources_to_strings(sources)
        assert result == ["source1.pdf", "source2.pdf"]

    def test_list_of_dicts(self):
        """Regression test: dict sources (from normalized_sources) must coerce to strings."""
        sources = [
            {"content": "Content A", "source": "a.pdf", "page": 1},
            {"content": "Content B", "source": "b.pdf", "page": 2},
        ]
        result = _coerce_sources_to_strings(sources)
        assert result == ["Content A", "Content B"]
        for item in result:
            assert isinstance(item, str)

    def test_mixed_list(self):
        sources = ["plain_string.pdf", {"content": "Dict content"}]
        result = _coerce_sources_to_strings(sources)
        assert result == ["plain_string.pdf", "Dict content"]

    def test_long_content_truncated(self):
        """Regression test: long dict content must be truncated to 200 chars."""
        sources = [{"content": "y" * 1000, "source": "big.pdf"}]
        result = _coerce_sources_to_strings(sources)
        assert len(result) == 1
        assert len(result[0]) == SOURCES_PREVIEW_MAX_LENGTH + 3
        assert result[0].endswith("...")

    def test_realistic_normalized_source_format(self):
        """Test with the exact format produced by knowledge_base._retrieve_context."""
        sources = [{
            "content": "Title: Test\n\nContent: Some long content here...",
            "source": "Unknown",
            "page": 1,
            "source_type": "qa_contribution",
            "is_public": True,
            "title": "Test",
            "question": "Test question",
            "answer": "Test answer",
            "meta_tags": "tag1,tag2"
        }]
        result = _coerce_sources_to_strings(sources)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert result[0].startswith("Title: Test")


# ==================== Integration Tests: store_conversation ====================

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
    with patch("app.services.conversation_service.ChatService") as mock_cs:
        mock_cs.return_value.generate_conversation_title = AsyncMock(return_value="Test Title")
        
        result = await conversation_service.store_conversation(
            user_id="test_user",
            user_question="Test question",
            system_response="Test response",
            query_analysis={},
            response_parameters={},
            session_id="test_session"
        )
    
    assert isinstance(result, dict)
    assert result["conversation_id"].startswith("conv_")
    assert result["user_message_id"].startswith("msg_")
    assert result["assistant_message_id"].startswith("msg_")
    assert result.get("persistence_failed") is not True
    mock_collection.insert_one.assert_called_once()


@pytest.mark.asyncio
async def test_store_conversation_with_dict_sources(conversation_service, mock_collection):
    """REGRESSION TEST: store_conversation with dict sources_used must succeed.
    
    This is the direct regression test for the bug:
    'sources_used.0 Input should be a valid string [input_value={'content': ...}]'
    """
    dict_sources = [
        {"content": "Title: Test Doc\n\nContent: ...", "source": "doc.pdf", "page": 1,
         "source_type": "qa_contribution", "is_public": True},
        {"content": "x" * 500, "source": "big.pdf", "page": 2},
    ]
    
    with patch("app.services.conversation_service.ChatService") as mock_cs:
        mock_cs.return_value.generate_conversation_title = AsyncMock(return_value="Test")
        
        result = await conversation_service.store_conversation(
            user_id="test_user",
            user_question="Question",
            system_response="Answer",
            query_analysis={"confidence_score": 0.9},
            response_parameters={"model": "gpt-4"},
            sources_used=dict_sources,
            session_id="sess_1"
        )
    
    # Must NOT fail persistence
    assert result.get("persistence_failed") is not True
    assert result["assistant_message_id"] is not None
    
    # Verify the stored document has string sources
    insert_call = mock_collection.insert_one.call_args[0][0]
    assistant_msg = [m for m in insert_call["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg["sources_used"] is not None
    for src in assistant_msg["sources_used"]:
        assert isinstance(src, str), f"Expected str, got {type(src)}: {src}"
        assert len(src) <= SOURCES_PREVIEW_MAX_LENGTH + 3


@pytest.mark.asyncio
async def test_store_conversation_with_string_sources(conversation_service, mock_collection):
    """Test that plain string sources still work as before."""
    string_sources = ["source1.pdf", "source2.pdf"]
    
    with patch("app.services.conversation_service.ChatService") as mock_cs:
        mock_cs.return_value.generate_conversation_title = AsyncMock(return_value="Test")
        
        result = await conversation_service.store_conversation(
            user_id="test_user",
            user_question="Question",
            system_response="Answer",
            query_analysis={},
            response_parameters={},
            sources_used=string_sources,
            session_id="sess_2"
        )
    
    assert result.get("persistence_failed") is not True
    insert_call = mock_collection.insert_one.call_args[0][0]
    assistant_msg = [m for m in insert_call["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg["sources_used"] == ["source1.pdf", "source2.pdf"]


@pytest.mark.asyncio
async def test_store_conversation_with_prompt_snapshot(conversation_service, mock_collection):
    """Test that prompt_snapshot is stored in the assistant message (not top-level)."""
    snapshot = {
        "system_prompt": "You are a helpful assistant",
        "user_query": "Test query",
        "retrieved_context": "Some context",
        "model_parameters": {"temperature": 0.1},
        "full_prompt": "Full prompt text",
        "response_type": "rag",
    }
    
    with patch("app.services.conversation_service.ChatService") as mock_cs:
        mock_cs.return_value.generate_conversation_title = AsyncMock(return_value="Test")
        
        result = await conversation_service.store_conversation(
            user_id="test_user",
            user_question="Question",
            system_response="Answer",
            query_analysis={},
            response_parameters={},
            prompt_snapshot=snapshot,
            session_id="sess_3"
        )
    
    assert result.get("persistence_failed") is not True
    insert_call = mock_collection.insert_one.call_args[0][0]
    
    # prompt_snapshot must be in the assistant message, NOT at top level
    assert "prompt_snapshot" not in insert_call or insert_call.get("prompt_snapshot") is None
    assistant_msg = [m for m in insert_call["messages"] if m["role"] == "assistant"][0]
    assert assistant_msg["prompt_snapshot"] == snapshot


@pytest.mark.asyncio
async def test_persistence_failure_returns_none_ids(conversation_service, mock_collection):
    """Test that persistence failure returns None IDs instead of breaking the chat."""
    mock_collection.insert_one.side_effect = Exception("DB connection lost")
    
    with patch("app.services.conversation_service.ChatService") as mock_cs:
        mock_cs.return_value.generate_conversation_title = AsyncMock(return_value="Test")
        
        result = await conversation_service.store_conversation(
            user_id="test_user",
            user_question="Question",
            system_response="Answer",
            query_analysis={},
            response_parameters={},
            session_id="sess_fail"
        )
    
    # Must return dict with None IDs and persistence_failed flag
    assert isinstance(result, dict)
    assert result["conversation_id"] is None
    assert result["user_message_id"] is None
    assert result["assistant_message_id"] is None
    assert result.get("persistence_failed") is True


@pytest.mark.asyncio
async def test_update_existing_conversation(conversation_service, mock_collection):
    """Test updating an existing conversation when one is found with the same session_id."""
    existing_conversation = {
        "_id": ObjectId(),
        "conversation_id": "conv_existing123",
        "user_id": "test_user",
        "session_id": "test_session",
        "messages": [
            {"role": "user", "content": "Previous question", "timestamp": datetime.utcnow()},
            {"role": "assistant", "content": "Previous response", "timestamp": datetime.utcnow()}
        ],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_messages": 2
    }
    mock_collection.find_one.return_value = existing_conversation
    
    result = await conversation_service.store_conversation(
        user_id="test_user",
        user_question="Test question",
        system_response="Test response",
        query_analysis={},
        response_parameters={},
        session_id="test_session"
    )
    
    assert isinstance(result, dict)
    assert result["conversation_id"] == "conv_existing123"
    assert result.get("persistence_failed") is not True
    mock_collection.insert_one.assert_not_called()
    mock_collection.update_one.assert_called_once()
    
    update_call_args = mock_collection.update_one.call_args[0]
    set_data = update_call_args[1]["$set"]
    assert len(set_data["messages"]) == 4
    assert set_data["total_messages"] == 4


@pytest.mark.asyncio
async def test_update_conversation_with_user_email(conversation_service, mock_collection):
    """Test updating user_email when it was previously null."""
    existing_conversation = {
        "_id": ObjectId(),
        "conversation_id": "conv_456",
        "user_id": "test_user",
        "session_id": "test_session",
        "user_email": None,
        "messages": [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "total_messages": 0
    }
    mock_collection.find_one.return_value = existing_conversation
    
    result = await conversation_service.store_conversation(
        user_id="test_user",
        user_question="Q",
        system_response="A",
        query_analysis={},
        response_parameters={},
        session_id="test_session",
        user_email="test@example.com"
    )
    
    assert result.get("persistence_failed") is not True
    update_call_args = mock_collection.update_one.call_args[0]
    set_data = update_call_args[1]["$set"]
    assert set_data["user_email"] == "test@example.com"


@pytest.mark.asyncio
async def test_get_message_by_id_with_prompt_snapshot(conversation_service, mock_collection):
    """Test that get_message_by_id returns prompt_snapshot for the context endpoint."""
    snapshot = {"system_prompt": "test", "full_prompt": "full"}
    mock_collection.find_one.return_value = {
        "_id": ObjectId(),
        "conversation_id": "conv_789",
        "messages": [
            {"message_id": "msg_target", "role": "assistant", "content": "Answer",
             "timestamp": datetime.utcnow(), "prompt_snapshot": snapshot}
        ]
    }
    
    result = await conversation_service.get_message_by_id("msg_target")
    
    assert result is not None
    assert result["message_id"] == "msg_target"
    assert result["role"] == "assistant"
    assert result["prompt_snapshot"] == snapshot


@pytest.mark.asyncio
async def test_get_message_by_id_not_found(conversation_service, mock_collection):
    """Test that get_message_by_id returns None for missing messages."""
    mock_collection.find_one.return_value = None
    result = await conversation_service.get_message_by_id("msg_missing")
    assert result is None