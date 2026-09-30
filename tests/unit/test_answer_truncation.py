"""Regression tests for "the answer stops mid-sentence and nobody notices".

Production symptom: a long, otherwise successful answer ended mid-word -- in the
bubble the user was reading *and* in the answer stored in the conversation --
while the API answered 200, wrote a normal `done` frame, closed the stream with
`data: [DONE]` and logged nothing. The provider had reported
`finish_reason="length"` (the configured `max_tokens` ceiling was reached), and
that value was being discarded in two separate places:

1. `create_stuff_documents_chain` ends in `StrOutputParser`, so
   `KnowledgeBaseService.stream_answer_from_context` only ever sees `str`
   tokens: `response_metadata` -- and `finish_reason` with it -- is already gone
   by the time the caller's loop can look at a chunk.
2. `chat_service.process_message_stream` never looked for it anyway, so the
   `done` event was byte-for-byte identical for a complete and a truncated
   answer.

These tests pin the fixes:
  * the KB stream reads `finish_reason` off the `on_llm_new_token` chunk (the
    only place it survives), reports it through `completion_state` and still
    yields exactly the same tokens -- including when no state dict is passed,
    which is how the benchmark and the E2E harness call it,
  * a `length` finish becomes `truncated=True` + `truncation_notice` in the
    service's `done` event and `query_analysis["answer_truncated"] = True`, so
    the fact is persisted with the conversation,
  * a normal completion (`stop`) is NOT flagged,
  * the general-knowledge and model-generated-handoff streams are covered the
    same way,
  * the SSE route forwards the flag verbatim: see `tests/api/test_chat_stream.py`.
"""
import os
import sys
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompts import ChatPromptTemplate

# Add the project root to the Python path (same as the other unit test modules).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# `chat_service` first: it is what `knowledge_base` imports `FINISH_REASON_LENGTH`
# and `get_llm` from, so importing it first keeps the circular import resolvable.
from app.services import chat_service as chat_module  # noqa: E402
from app.services import knowledge_base as kb_module  # noqa: E402
from app.services.knowledge_base import KnowledgeBaseService  # noqa: E402


STATIC_REFERRAL = "پیام ارجاع ثابت به کارشناس"
TRUNCATED_ANSWER = "سپس با خاک "


class _TruncatingChatModel(BaseChatModel):
    """Chat-model double that streams `finish_reason` exactly like the openai client.

    `_stream` yields `ChatGenerationChunk(..., generation_info={"finish_reason": ...})`,
    which is what `langchain_openai` does; `langchain-core` then merges that
    `generation_info` into `message.response_metadata` and hands the chunk to
    `on_llm_new_token`. A bespoke double is needed because the fakes shipped in
    `langchain_core` build their chunks without any generation metadata at all.
    """

    chunks: List[str] = []
    finish_reason: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "truncating-fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("".join(self.chunks)))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for token in self.chunks[:-1]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=self.chunks[-1]),
            generation_info={"finish_reason": self.finish_reason},
        )


def _document_chain(finish_reason: str) -> Any:
    """The REAL `create_stuff_documents_chain`, i.e. the chain the service builds.

    Using the real chain is the point: the `StrOutputParser` at its end is what
    hides `finish_reason` from the caller, so a hand-rolled double would not
    reproduce the defect under test.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Context:\n{context}"),
        ("human", "{input}"),
    ])
    llm = _TruncatingChatModel(
        chunks=[TRUNCATED_ANSWER, "پوشانده شود"], finish_reason=finish_reason
    )
    return create_stuff_documents_chain(llm, prompt)


def _kb_service() -> KnowledgeBaseService:
    """A real service whose heavy collaborators (vector store, reranker) are doubles."""
    with patch.object(kb_module, "get_document_processor", return_value=MagicMock()), \
            patch.object(kb_module, "get_excel_qa_processor", return_value=MagicMock()), \
            patch.object(kb_module, "ConfigService", return_value=MagicMock()):
        return KnowledgeBaseService()


async def _drain_kb_stream(kb: KnowledgeBaseService, **kwargs) -> List[str]:
    """Collect the tokens of `stream_answer_from_context` (the caller-side contract)."""
    return [
        token
        async for token in kb.stream_answer_from_context(
            "سوال", [Document(page_content="ground truth")], **kwargs
        )
    ]


# ==================== knowledge_base: the finish reason survives the chain ====================

@pytest.mark.asyncio
async def test_kb_stream_reports_a_length_finish_through_completion_state():
    """`finish_reason` reaches the caller even though the chain yields strings."""
    kb = _kb_service()
    state: Dict[str, Any] = {}

    with patch.object(kb, "_get_document_chain",
                      new=AsyncMock(return_value=_document_chain("length"))):
        tokens = await _drain_kb_stream(kb, completion_state=state)

    # The tokens themselves are untouched (the partial answer is still the best
    # answer we have): the fix only adds the completion status alongside them.
    assert "".join(tokens) == TRUNCATED_ANSWER + "پوشانده شود"
    assert state["finish_reason"] == "length"
    assert state["truncated"] is True


@pytest.mark.asyncio
async def test_kb_stream_reports_a_normal_completion_without_flagging_it():
    """`finish_reason="stop"` must not be reported as truncation."""
    kb = _kb_service()
    state: Dict[str, Any] = {}

    with patch.object(kb, "_get_document_chain",
                      new=AsyncMock(return_value=_document_chain("stop"))):
        await _drain_kb_stream(kb, completion_state=state)

    assert state["finish_reason"] == "stop"
    assert state["truncated"] is False


@pytest.mark.asyncio
async def test_kb_stream_without_completion_state_stays_backward_compatible():
    """Callers that pass no state dict (benchmark, E2E harness) keep working."""
    kb = _kb_service()

    with patch.object(kb, "_get_document_chain",
                      new=AsyncMock(return_value=_document_chain("length"))):
        tokens = await _drain_kb_stream(kb)

    assert "".join(tokens) == TRUNCATED_ANSWER + "پوشانده شود"


# ==================== chat_service: the flag reaches the `done` event ====================

def _llm_settings() -> MagicMock:
    settings = MagicMock()
    settings.default_model = "google/gemini-flash"
    settings.temperature = 0.2
    settings.max_tokens = 4096
    settings.top_p = 1.0
    settings.preferred_api_provider = "openrouter"
    settings.openrouter_api_key = "test-openrouter-key"
    settings.openrouter_api_base = "https://openrouter.ai/api/v1"
    settings.openai_api_key = "test-openai-key"
    return settings


def _config_service() -> MagicMock:
    settings = MagicMock()
    settings._load_config = AsyncMock()
    settings.get_llm_settings = AsyncMock(return_value=_llm_settings())
    rag_settings = MagicMock()
    rag_settings.human_referral_message = STATIC_REFERRAL
    rag_settings.knowledge_base_confidence_threshold = 0.7
    rag_settings.system_prompt = "system prompt"
    rag_settings.temperature = 0.1
    settings.get_rag_settings = AsyncMock(return_value=rag_settings)
    return settings


def _intent() -> Dict[str, Any]:
    return {
        "intent": "PRIVATE",
        "is_public": False,
        "explanation": "specialized question",
        "topic_shift": False,
        "context_state": "self_contained",
        "resolved_query": "بهترین کود برای گندم چیست؟",
        "referenced_entities": [],
        "context_decided": True,
    }


def _retrieval(confidence: float) -> Dict[str, Any]:
    docs_with_scores = [({"page_content": "context"}, confidence)] if confidence >= 0.7 else []
    return {
        "confidence_score": confidence,
        "sources": [],
        "rewritten_query": "بهترین کود برای گندم چیست؟",
        "docs_with_scores": docs_with_scores,
        "normalized_docs": [{"page_content": "context"}] if docs_with_scores else [],
        "source_type": "knowledge_base",
    }


def _make_service(chain: Any = None, general_answer: bool = False) -> Any:
    service = chat_module.ChatService()
    service.config_service = _config_service()
    service.generalAnswer = general_answer
    service.detect_query_intent = AsyncMock(return_value=_intent())
    service._get_or_create_session = AsyncMock(return_value=chain)
    service._build_general_snapshot = AsyncMock(return_value={"response_type": "general"})
    return service


async def _collect(service: Any) -> List[Dict[str, Any]]:
    return [
        event
        async for event in service.process_message_stream(
            "user-1", "بهترین کود برای گندم چیست؟"
        )
    ]


def _kb_double(confidence: float, tokens: List[str], finish_reason: Optional[str]) -> MagicMock:
    """KB service double whose stream fills `completion_state` like the real one."""
    kb = MagicMock()
    kb._retrieve_context = AsyncMock(return_value=_retrieval(confidence))
    kb.build_prompt_snapshot = AsyncMock(return_value={"response_type": "knowledge_base"})

    async def _stream(query, normalized_docs, external_context=None, completion_state=None):
        for token in tokens:
            yield token
        if completion_state is not None:
            completion_state["finish_reason"] = finish_reason
            completion_state["truncated"] = finish_reason == chat_module.FINISH_REASON_LENGTH
            completion_state["usage_metadata"] = {"output_tokens": 4096}

    kb.stream_answer_from_context = _stream
    return kb


def _done(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return next(event for event in events if event["type"] == "done")



@pytest.mark.asyncio
async def test_length_terminated_kb_answer_is_flagged_in_the_done_event():
    """The user-visible contract: `truncated` + `truncation_notice` in `done`."""
    kb = _kb_double(
        confidence=0.9, tokens=[TRUNCATED_ANSWER, "پوشانده شود"], finish_reason="length"
    )
    service = _make_service()

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    done = _done(events)
    assert done["answer"] == TRUNCATED_ANSWER + "پوشانده شود"
    assert done["truncated"] is True
    assert done["finish_reason"] == "length"
    assert done["truncation_notice"] == chat_module.ANSWER_TRUNCATED_NOTICE
    # The final analysis travels with the same event, so the UI can show the
    # reason in its analysis panel next to the partial answer.
    assert done["query_analysis"]["answer_truncated"] is True


@pytest.mark.asyncio
async def test_completed_kb_answer_is_not_flagged_in_the_done_event():
    """A `stop`-terminated answer carries the key set to False, and no notice."""
    kb = _kb_double(confidence=0.9, tokens=["بهترین کود اوره است."], finish_reason="stop")
    service = _make_service()

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    done = _done(events)
    assert done["truncated"] is False
    assert done["finish_reason"] == "stop"
    assert "truncation_notice" not in done
    assert "answer_truncated" not in done["query_analysis"]


@pytest.mark.asyncio
async def test_length_terminated_general_answer_is_flagged_in_the_done_event():
    """The general-knowledge stream (`prompt | llm`) is covered too."""

    class _ChunkChain:
        """`prompt | llm` shape: message chunks carry `response_metadata` directly."""

        async def astream(self, inputs):
            yield AIMessageChunk(content="اطلاعات ", response_metadata={"finish_reason": "stop"})
            yield AIMessageChunk(content="ناقص", response_metadata={"finish_reason": "length"})

    kb = _kb_double(confidence=0.1, tokens=[], finish_reason=None)
    service = _make_service(chain=_ChunkChain(), general_answer=True)

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    done = _done(events)
    assert done["answer"] == "اطلاعات ناقص"
    assert done["truncated"] is True
    assert done["truncation_notice"] == chat_module.ANSWER_TRUNCATED_NOTICE
    assert done["query_analysis"]["answer_truncated"] is True


@pytest.mark.asyncio
async def test_static_referral_fallback_is_never_flagged_as_truncated():
    """A model handoff that produced nothing is replaced by static text.

    Static text cannot be cut off by a token ceiling, so the finish reason the
    (empty) model call may have reported must not mark the referral as truncated.
    """

    class _EmptyChain:
        async def astream(self, inputs):
            yield AIMessageChunk(content="", response_metadata={"finish_reason": "length"})

    kb = _kb_double(confidence=0.1, tokens=[], finish_reason=None)
    service = _make_service(chain=_EmptyChain(), general_answer=False)

    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        events = await _collect(service)

    done = _done(events)
    assert done["answer"] == STATIC_REFERRAL
    assert done["truncated"] is False
    assert "truncation_notice" not in done


@pytest.mark.asyncio
async def test_static_intents_carry_the_completed_flag():
    """Greeting/farewell/small-talk frames keep the key so clients never guess."""
    service = _make_service()
    service.detect_query_intent = AsyncMock(return_value={**_intent(), "intent": "GREETING"})

    events = await _collect(service)

    done = _done(events)
    assert done["truncated"] is False
    assert "truncation_notice" not in done


def test_truncation_fields_only_flag_the_token_ceiling():
    """`_truncation_fields` is the single place that decides what truncated means."""
    assert chat_module._truncation_fields("length") == {
        "truncated": True,
        "finish_reason": "length",
        "truncation_notice": chat_module.ANSWER_TRUNCATED_NOTICE,
    }
    for reason in ("stop", None):
        assert chat_module._truncation_fields(reason) == {
            "truncated": False,
            "finish_reason": reason,
        }


def test_extract_finish_reason_reads_both_metadata_shapes():
    """`response_metadata` for chat models, `generation_info` for legacy LLMs."""
    assert chat_module.extract_finish_reason(
        AIMessageChunk(content="x", response_metadata={"finish_reason": "length"})
    ) == "length"
    generation = ChatGenerationChunk(
        message=AIMessageChunk(content="x"), generation_info={"finish_reason": "length"}
    )
    assert chat_module.extract_finish_reason(generation) == "length"
    # Strings (what the KB chain yields) and empty metadata are simply unknown.
    assert chat_module.extract_finish_reason("token") is None
    assert chat_module.extract_finish_reason(AIMessageChunk(content="x")) is None


@pytest.mark.asyncio
async def test_tracker_reads_both_metadata_shapes():
    """The tracker accepts `response_metadata` and `generation_info`, and invents nothing."""
    from_metadata = kb_module._FinishReasonTracker()
    assert from_metadata.finish_reason is None
    await from_metadata.on_llm_new_token(
        "x",
        chunk=ChatGenerationChunk(
            message=AIMessageChunk(content="x", response_metadata={"finish_reason": "length"})
        ),
    )
    assert from_metadata.finish_reason == "length"
    # Set once and never overwritten by a later chunk that carries no reason.
    await from_metadata.on_llm_new_token("y", chunk=ChatGenerationChunk(
        message=AIMessageChunk(content="y")
    ))
    assert from_metadata.finish_reason == "length"

    from_generation_info = kb_module._FinishReasonTracker()
    await from_generation_info.on_llm_new_token(
        "x",
        chunk=ChatGenerationChunk(
            message=AIMessageChunk(content="x", response_metadata={"finish_reason": None}),
            generation_info={"finish_reason": "length"},
        ),
    )
    assert from_generation_info.finish_reason == "length"

