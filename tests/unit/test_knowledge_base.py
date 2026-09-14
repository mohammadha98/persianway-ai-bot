"""Unit tests for KnowledgeBaseService.

MIGRATION NOTE (query-expansion + retrieval contracts)
------------------------------------------------------
This file was written against two APIs the service no longer exposes, so every
test failed at attribute level and guarded nothing:

1. `KnowledgeBaseService.expand_query` was removed. Expansion now lives in
   `expand_query_with_context(query, conversation_history=None, max_history=4,
   skip_rewrite=False)`. Note the context parameter is `conversation_history`
   (a list of {"role", "content"} dicts), NOT `context`.

2. The expansion LLM is no longer `openai.AsyncOpenAI` called directly. The
   service resolves it through `get_llm(...)` and awaits `llm.ainvoke(messages)`,
   so the old `patch('openai.AsyncOpenAI')` had no effect.

3. `expanded_queries` is intentionally hardcoded to `[]` by the service: the
   single rewrite/expansion LLM call returns only a `rewritten_query`, and the
   search set is surfaced as `all_queries` (rewrite + original, deduped). The
   old `len(result["expanded_queries"]) == 3` assertions therefore contradicted
   the live contract and are now asserted against `all_queries`.

4. `query_knowledge_base` retrieves through
   `_get_hybrid_service().hybrid_retrieve(query, is_public)` and generates with
   `_get_document_chain()`. The old `similarity_search_with_score` /
   `_get_qa_chain` / `_is_content_relevant` mocks referenced attributes that no
   longer exist, and the public-filter test asserted a `filter={"is_public": ...}`
   kwarg that is now passed positionally to `hybrid_retrieve`.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Mock the problematic imports to avoid circular import
with patch('app.services.chat_service.get_llm'):
    with patch('app.services.document_processor.get_document_processor'):
        with patch('app.services.excel_processor.get_excel_qa_processor'):
            from app.services.knowledge_base import KnowledgeBaseService
            from app.services import knowledge_base as kb_module
            from langchain_core.documents import Document


@pytest.fixture
def knowledge_base_service():
    """Create a KnowledgeBaseService instance for testing."""
    return KnowledgeBaseService()


def _make_llm(content):
    """Build a stand-in for the chat model returned by `get_llm`.

    `expand_query_with_context` awaits `llm.ainvoke([...])` and reads
    `getattr(response, "content", "")`, so only `ainvoke`/`content` matter.
    """
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return llm


def _rag_settings(
    threshold=0.7,
    top_k_results=5,
    human_referral_message="لطفاً با پشتیبانی تماس بگیرید.",
):
    """Build the RAG settings object consumed by `query_knowledge_base`."""
    settings = MagicMock()
    settings.knowledge_base_confidence_threshold = threshold
    settings.top_k_results = top_k_results
    settings.human_referral_message = human_referral_message
    return settings


def _expansion_result(query, all_queries=None, expanded_queries=None, rewritten_query=None):
    """Shape of the dict returned by `expand_query_with_context`."""
    return {
        "original_query": query,
        "rewritten_query": rewritten_query if rewritten_query is not None else query,
        "is_self_contained": True,
        "uses_history": False,
        "expanded_queries": expanded_queries or [],
        "all_queries": all_queries if all_queries is not None else [query],
    }


def _doc(content, source, page=1, source_type="pdf", is_public=False, hybrid_score=None):
    """Build a real Document (context normalization needs `.page_content`/`.metadata`)."""
    metadata = {
        "source": source,
        "page": page,
        "source_type": source_type,
        "is_public": is_public,
    }
    if hybrid_score is not None:
        metadata["hybrid_score"] = hybrid_score
    return Document(page_content=content, metadata=metadata)


@pytest.mark.asyncio
async def test_expand_query_success(knowledge_base_service):
    """Test successful query rewriting against the live expansion contract."""
    # Arrange
    test_query = "How to make Persian tea?"
    rewritten = "بهترین روش دم کردن چای ایرانی"
    llm_payload = {
        "is_self_contained": True,
        "uses_history": False,
        "rewritten_query": rewritten,
    }
    mock_llm = _make_llm(json.dumps(llm_payload))

    with patch.object(kb_module, 'get_llm', new=AsyncMock(return_value=mock_llm)) as mock_get_llm:
        # Act
        result = await knowledge_base_service.expand_query_with_context(test_query)

        # Assert - rewrite contract
        assert result["original_query"] == test_query
        assert result["rewritten_query"] == rewritten
        assert result["is_self_contained"] is True
        assert result["uses_history"] is False
        # The service intentionally emits no expanded queries; the search set is
        # `all_queries` = deduped(rewrite + original).
        assert result["expanded_queries"] == []
        assert result["all_queries"] == [rewritten, test_query]

        # Assert - the expansion LLM is resolved through get_llm, not AsyncOpenAI
        mock_get_llm.assert_awaited_once()
        assert mock_get_llm.await_args.kwargs["model_name"] == "qwen/qwen3-32b"
        assert mock_get_llm.await_args.kwargs["temperature"] == 0.0
        assert mock_get_llm.await_args.kwargs["max_tokens"] == 800

        # Assert - one LLM call carrying exactly the system + user messages
        mock_llm.ainvoke.assert_awaited_once()
        messages = mock_llm.ainvoke.await_args.args[0]
        assert len(messages) == 2
        assert test_query in messages[1].content


@pytest.mark.asyncio
async def test_expand_query_malformed_response(knowledge_base_service):
    """Test handling of malformed JSON response."""
    # Arrange
    test_query = "How to make Persian tea?"
    mock_llm = _make_llm("This is not valid JSON")

    with patch.object(kb_module, 'get_llm', new=AsyncMock(return_value=mock_llm)):
        # Act
        result = await knowledge_base_service.expand_query_with_context(test_query)

        # Assert - falls back to the original query as the only search term
        assert result["original_query"] == test_query
        assert result["rewritten_query"] == test_query
        assert result["expanded_queries"] == []
        assert result["all_queries"] == [test_query]


@pytest.mark.asyncio
async def test_expand_query_api_error(knowledge_base_service):
    """Test handling of API errors."""
    # Arrange
    test_query = "How to make Persian tea?"
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API Error"))

    with patch.object(kb_module, 'get_llm', new=AsyncMock(return_value=mock_llm)):
        # Mock the logging to verify the error is logged
        with patch('logging.error') as mock_logging:
            # Act
            result = await knowledge_base_service.expand_query_with_context(test_query)

            # Assert
            assert result["original_query"] == test_query
            assert result["rewritten_query"] == test_query
            assert result["expanded_queries"] == []
            assert result["all_queries"] == [test_query]
            mock_logging.assert_called_once()
            assert "API Error" in mock_logging.call_args[0][0]


@pytest.mark.asyncio
async def test_expand_query_empty_query(knowledge_base_service):
    """Test expansion with empty query."""
    # Arrange
    test_query = ""
    mock_llm = _make_llm(json.dumps({
        "is_self_contained": True,
        "uses_history": False,
        "rewritten_query": "",
    }))

    with patch.object(kb_module, 'get_llm', new=AsyncMock(return_value=mock_llm)):
        # Act
        result = await knowledge_base_service.expand_query_with_context(test_query)

        # Assert - an empty query contributes no searchable terms
        assert result["original_query"] == ""
        assert result["rewritten_query"] == ""
        assert result["expanded_queries"] == []
        assert result["all_queries"] == []


@pytest.mark.asyncio
async def test_expand_query_skip_rewrite_uses_query_verbatim(knowledge_base_service):
    """skip_rewrite=True must bypass the LLM entirely (context decided upstream)."""
    test_query = "مرحله بعد چیه؟"

    with patch.object(kb_module, 'get_llm', new=AsyncMock()) as mock_get_llm:
        result = await knowledge_base_service.expand_query_with_context(
            test_query, skip_rewrite=True
        )

        mock_get_llm.assert_not_awaited()
        assert result["rewritten_query"] == test_query
        assert result["all_queries"] == [test_query]
        assert result["is_self_contained"] is True
        assert result["uses_history"] is False


# ==================== query_knowledge_base ====================


def _mock_retrieval(kb, docs, answer="پاسخ تولید شده", confidence=0.9, expansion=None, query="q"):
    """Wire the live retrieval contract onto `kb` and return the patched mocks.

    `query_knowledge_base` now retrieves via
    `_get_hybrid_service().hybrid_retrieve(query, is_public)` and generates via
    `_get_document_chain().invoke(...)`, so those are the seams under test.
    """
    hybrid = MagicMock()
    hybrid.hybrid_retrieve = AsyncMock(return_value=list(docs))
    chain = MagicMock()
    chain.invoke = MagicMock(return_value=answer)

    patches = [
        patch.object(kb, 'expand_query_with_context',
                     new=AsyncMock(return_value=expansion or _expansion_result(query))),
        patch.object(kb, '_get_hybrid_service', return_value=hybrid),
        patch.object(kb, '_get_document_chain', new=AsyncMock(return_value=chain)),
        patch.object(kb, '_calculate_confidence_score', return_value=confidence),
        patch.object(kb.document_processor, 'get_vector_store', return_value=MagicMock()),
        patch.object(kb.config_service, 'get_rag_settings',
                     new=AsyncMock(return_value=_rag_settings())),
        patch.object(kb.config_service, '_load_config', new=AsyncMock()),
        patch.object(kb, '_log_human_referral', new=MagicMock()),
    ]
    started = [p.start() for p in patches]
    mocks = {
        "hybrid": hybrid,
        "chain": chain,
        "expand": started[0],
        "confidence": started[3],
    }
    return mocks, patches


@pytest.mark.asyncio
async def test_query_knowledge_base_high_confidence_qa_match(knowledge_base_service):
    """A strong QA hit yields the generated answer, its metadata and no referral."""
    # Arrange
    test_query = "How to make Persian tea?"
    answer_text = "To make Persian tea, boil water, add tea leaves, and steep for 5 minutes."
    qa_doc = _doc(
        "Persian tea preparation instructions",
        source="tea_guide.xlsx",
        source_type="excel_qa",
        hybrid_score=0.8,
    )
    qa_doc.metadata.update({
        "title": "How to make Persian tea?",
        "question": "How to make Persian tea?",
        "answer": answer_text,
    })

    expansion = _expansion_result(test_query)
    mocks, patches = _mock_retrieval(
        knowledge_base_service, [qa_doc], answer=answer_text, confidence=0.9,
        expansion=expansion, query=test_query,
    )

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert
        assert result["answer"] == answer_text
        assert result["confidence_score"] == 0.9
        assert result["source_type"] == "excel_qa"
        assert result["requires_human_support"] is False
        assert result["query_id"] is None
        assert len(result["sources"]) == 1
        assert result["sources"][0]["question"] == "How to make Persian tea?"
        assert result["sources"][0]["answer"] == answer_text
        assert result["sources"][0]["title"] == "How to make Persian tea?"
        assert result["rewritten_query"] == test_query
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_query_knowledge_base_pdf_fallback(knowledge_base_service):
    """PDF-sourced retrieval flows through with its own source_type."""
    # Arrange
    test_query = "What is Persian culture?"
    answer_text = "Persian culture encompasses art, literature, and traditions."
    pdf_doc = _doc(
        "Persian culture is rich and diverse with a long history...",
        source="culture_guide.pdf",
        page=15,
        hybrid_score=0.7,
    )

    mocks, patches = _mock_retrieval(
        knowledge_base_service, [pdf_doc], answer=answer_text, confidence=0.75,
        query=test_query,
    )

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert
        assert answer_text in result["answer"]
        assert result["confidence_score"] == 0.75
        assert result["source_type"] == "pdf"
        assert result["requires_human_support"] is False
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "culture_guide.pdf"
        assert result["sources"][0]["page"] == 15
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_query_knowledge_base_public_filter_applied(knowledge_base_service):
    """is_public must be forwarded to hybrid_retrieve as the positional filter arg."""
    test_query = "What does PersianWay do?"
    public_doc = _doc(
        "PersianWay provides public services.",
        source="public_overview.pdf",
        page=2,
        is_public=True,
        hybrid_score=0.7,
    )

    mocks, patches = _mock_retrieval(
        knowledge_base_service, [public_doc], answer="PersianWay is a public company.",
        confidence=0.9, query=test_query,
    )

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query, is_public=True)

        # Assert - the public flag reaches retrieval (positional `is_public` arg)
        assert result["confidence_score"] == 0.9
        assert result["requires_human_support"] is False
        mocks["hybrid"].hybrid_retrieve.assert_awaited()
        for call in mocks["hybrid"].hybrid_retrieve.await_args_list:
            assert call.args[1] is True
        assert all(s["is_public"] is True for s in result["sources"])
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_query_knowledge_base_low_confidence_human_referral(knowledge_base_service):
    """Below-threshold confidence suppresses the answer and opens a referral."""
    # Arrange
    test_query = "Very specific technical question"
    weak_doc = _doc("Some general information...", source="general.pdf", hybrid_score=0.1)

    mocks, patches = _mock_retrieval(
        knowledge_base_service, [weak_doc],
        answer="I'm not sure about this specific question.",
        confidence=0.4, query=test_query,
    )

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert
        assert result["confidence_score"] == 0.4
        assert result["requires_human_support"] is True
        assert result["query_id"] is not None
        assert len(result["query_id"]) > 0  # UUID should be generated
        # The weak generated answer must be replaced by the referral message.
        assert result["answer"] == "لطفاً با پشتیبانی تماس بگیرید."
        mocks["confidence"].assert_called()
    finally:
        for p in patches:
            p.stop()

@pytest.mark.asyncio
async def test_query_knowledge_base_vector_store_unavailable(knowledge_base_service):
    """A missing vector store must produce a system referral, not a crash."""
    # Arrange
    test_query = "Any question"
    referral = "متأسفانه، سیستم در حال حاضر در دسترس نیست."

    patches = [
        patch.object(knowledge_base_service, 'expand_query_with_context',
                     new=AsyncMock(return_value=_expansion_result(test_query))),
        patch.object(knowledge_base_service.document_processor, 'get_vector_store',
                     return_value=None),
        patch.object(knowledge_base_service.config_service, 'get_rag_settings',
                     new=AsyncMock(return_value=_rag_settings(human_referral_message=referral))),
        patch.object(knowledge_base_service.config_service, '_load_config', new=AsyncMock()),
    ]
    for p in patches:
        p.start()

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert
        assert referral in result["answer"]
        assert result["confidence_score"] == 0.0
        assert result["source_type"] == "system"
        assert result["requires_human_support"] is True
        assert result["query_id"] is not None
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_query_knowledge_base_document_chain_unavailable(knowledge_base_service):
    """When the generation chain is unavailable, fall back to the referral message."""
    # Arrange
    test_query = "Question when QA chain fails"
    referral = "لطفاً با پشتیبانی تماس بگیرید."
    pdf_doc = _doc("Some content", source="test.pdf", hybrid_score=0.5)

    hybrid = MagicMock()
    hybrid.hybrid_retrieve = AsyncMock(return_value=[pdf_doc])

    patches = [
        patch.object(knowledge_base_service, 'expand_query_with_context',
                     new=AsyncMock(return_value=_expansion_result(test_query))),
        patch.object(knowledge_base_service, '_get_hybrid_service', return_value=hybrid),
        patch.object(knowledge_base_service, '_get_document_chain',
                     new=AsyncMock(return_value=None)),
        patch.object(knowledge_base_service.document_processor, 'get_vector_store',
                     return_value=MagicMock()),
        patch.object(knowledge_base_service.config_service, 'get_rag_settings',
                     new=AsyncMock(return_value=_rag_settings(human_referral_message=referral))),
        patch.object(knowledge_base_service.config_service, '_load_config', new=AsyncMock()),
    ]
    for p in patches:
        p.start()

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert
        assert referral in result["answer"]
        assert result["confidence_score"] == 0.0
        assert result["source_type"] == "system"
        assert result["requires_human_support"] is True
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_query_knowledge_base_exception_handling(knowledge_base_service):
    """A retrieval exception must be caught and turned into a system referral."""
    # Arrange
    test_query = "Question that causes exception"
    referral = "خطایی رخ داده است."

    patches = [
        patch.object(knowledge_base_service, 'expand_query_with_context',
                     new=AsyncMock(side_effect=Exception("Test exception"))),
        patch.object(knowledge_base_service.config_service, 'get_rag_settings',
                     new=AsyncMock(return_value=_rag_settings(human_referral_message=referral))),
        patch.object(knowledge_base_service.config_service, '_load_config', new=AsyncMock()),
    ]
    for p in patches:
        p.start()
    log_patch = patch('logging.error')
    mock_logging = log_patch.start()

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert
        assert referral in result["answer"]
        assert result["confidence_score"] == 0.0
        assert result["source_type"] == "system"
        assert result["requires_human_support"] is True
        assert result["query_id"] is not None
        assert mock_logging.called
    finally:
        log_patch.stop()
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_query_knowledge_base_query_expansion_integration(knowledge_base_service):
    """Every search query in all_queries must drive its own hybrid retrieval call."""
    # Arrange
    test_query = "Persian tea"
    rewritten = "چای ایرانی"
    all_queries = [rewritten, test_query]
    doc1 = _doc("Persian tea content 1", source="tea1.pdf", page=1, hybrid_score=0.8)
    doc2 = _doc("Persian tea content 2", source="tea2.pdf", page=2, hybrid_score=0.7)

    def _side_effect(q, is_public=False):
        # Different docs per query, mirroring the multi-query retrieval fan-out.
        return [doc1] if q == rewritten else [doc2]

    hybrid = MagicMock()
    hybrid.hybrid_retrieve = AsyncMock(side_effect=_side_effect)
    chain = MagicMock()
    chain.invoke = MagicMock(return_value="Persian tea is a traditional beverage.")

    expansion = _expansion_result(test_query, all_queries=all_queries, rewritten_query=rewritten)
    patches = [
        patch.object(knowledge_base_service, 'expand_query_with_context',
                     new=AsyncMock(return_value=expansion)),
        patch.object(knowledge_base_service, '_get_hybrid_service', return_value=hybrid),
        patch.object(knowledge_base_service, '_get_document_chain',
                     new=AsyncMock(return_value=chain)),
        patch.object(knowledge_base_service, '_calculate_confidence_score', return_value=0.8),
        patch.object(knowledge_base_service.document_processor, 'get_vector_store',
                     return_value=MagicMock()),
        patch.object(knowledge_base_service.config_service, 'get_rag_settings',
                     new=AsyncMock(return_value=_rag_settings())),
        patch.object(knowledge_base_service.config_service, '_load_config', new=AsyncMock()),
        patch.object(knowledge_base_service, '_log_human_referral', new=MagicMock()),
    ]
    for p in patches:
        p.start()

    try:
        # Act
        result = await knowledge_base_service.query_knowledge_base(test_query)

        # Assert - one retrieval call per query in all_queries, in order
        assert hybrid.hybrid_retrieve.await_count == len(all_queries)
        assert [c.args[0] for c in hybrid.hybrid_retrieve.await_args_list] == all_queries
        assert result["confidence_score"] == 0.8
        assert result["requires_human_support"] is False
        # Deduplication keeps both distinct documents.
        assert len(result["sources"]) == 2
        assert result["rewritten_query"] == rewritten
    finally:
        for p in patches:
            p.stop()

