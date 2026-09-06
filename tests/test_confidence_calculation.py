import pytest
import sys, os
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
with patch('app.services.chat_service.get_llm'):
    with patch('app.services.document_processor.get_document_processor'):
        with patch('app.services.excel_processor.get_excel_qa_processor'):
            from app.services.knowledge_base import KnowledgeBaseService


def test_confidence_score_calculation():
    kb_service = KnowledgeBaseService()

    test_cases = [
        (0.3, 0.90, 1.0, "Very relevant document"),
        (0.8, 0.70, 0.85, "Relevant document"),
        (1.5, 0.45, 0.65, "Moderately relevant"),
        (2.2, 0.15, 0.35, "Weakly relevant"),
        (3.0, 0.0, 0.2, "Irrelevant document"),
    ]

    for distance, min_conf, max_conf, desc in test_cases:
        confidence = kb_service._similarity_to_confidence(distance)
        assert min_conf <= confidence <= max_conf, (
            f"Distance {distance} ({desc}): confidence {confidence:.3f} not in [{min_conf}, {max_conf}]"
        )
