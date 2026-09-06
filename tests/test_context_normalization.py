import pytest
import sys, os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.documents import Document

with patch('app.services.chat_service.get_llm'):
    with patch('app.services.document_processor.get_document_processor'):
        with patch('app.services.excel_processor.get_excel_qa_processor'):
            from app.services.knowledge_base import KnowledgeBaseService


def test_context_normalization():
    kb_service = KnowledgeBaseService()
    large_docs = [
        Document(page_content=("محتوای بسیار طولانی " * 200), metadata={"source": "doc1.pdf", "page": 1, "extra": "unnecessary"}),
        Document(page_content=("متن دیگری که خیلی طولانی است " * 150), metadata={"source": "doc2.pdf", "page": 2, "extra": "data"}),
        Document(page_content=("سومین سند " * 100), metadata={"source": "doc3.pdf", "page": 1}),
    ]
    normalized = kb_service._normalize_documents_for_context(large_docs, max_total_tokens=3000, max_chars_per_doc=1200)
    total_chars = sum(len(doc.page_content) for doc in normalized)
    estimated_tokens = total_chars // 4
    assert len(normalized) <= len(large_docs)
    assert total_chars <= 3000 * 4
    assert estimated_tokens <= 3000
    for doc in normalized:
        assert "extra" not in doc.metadata
        assert "source" in doc.metadata
