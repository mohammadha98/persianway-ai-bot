import sys, os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
with patch('app.services.chat_service.get_llm'):
    with patch('app.services.document_processor.get_document_processor'):
        with patch('app.services.excel_processor.get_excel_qa_processor'):
            from app.services.knowledge_base import KnowledgeBaseService


def test_history_filtering():
    kb_service = KnowledgeBaseService()
    cases = [
        ("سلام چطوری؟", "سلام   چطوری  ؟", True),
        ("پرشین وی چیست", "پرشین وی چیست.", True),
        ("محصولات شما", "محصولات شما", True),
        ("سوال اول", "سوال دوم", False),
    ]
    for query, history_msg, should_match in cases:
        norm_query = kb_service._normalize_text_for_comparison(query)
        norm_history = kb_service._normalize_text_for_comparison(history_msg)
        assert (norm_query == norm_history) == should_match
