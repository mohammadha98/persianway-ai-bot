import pytest
import sys, os
from unittest.mock import patch, AsyncMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
with patch('app.services.document_processor.get_document_processor'):
    with patch('app.services.excel_processor.get_excel_qa_processor'):
        from app.services.knowledge_base import KnowledgeBaseService


class MockAsyncResponse:
    def __init__(self, content):
        self.choices = [type('x', (), {'message': type('y', (), {'content': content})()})]


@pytest.mark.asyncio
async def test_query_expansion_error_handling():
    kb = KnowledgeBaseService()
    with patch('openai.AsyncOpenAI') as mock_openai:
        client = AsyncMock()
        client.chat.completions.create.return_value = MockAsyncResponse("not a json")
        mock_openai.return_value = client
        result = await kb.expand_query_with_context("سوال تستی")
        assert result["rewritten_query"] == "سوال تستی"
        assert result["expanded_queries"] == []
