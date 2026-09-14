"""Offline verification of the model-generated handoff change (no API calls).

When generalAnswer is disabled and KB confidence is below threshold, the user
must now receive the model's OWN response (system prompt + history via the
general chain) instead of the static HUMAN_REFERRAL_MESSAGE — in both
process_message and process_message_stream.

Run: python scripts/verify_model_generated_handoff.py
"""
import asyncio
import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

STATIC_REFERRAL = "پیام ارجاع ثابت به کارشناس"
MODEL_HANDOFF = "پاسخ تولیدشده توسط مدل برای ارجاع"
STREAM_CHUNKS = ["پاسخ ", "تولیدشده ", "توسط مدل برای ارجاع"]


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChain:
    """Stands in for the general conversation runnable (prompt | llm)."""

    def __init__(self, content=None, chunks=None, raise_on_invoke=False, raise_on_stream=False):
        self.content = content
        self.chunks = chunks
        self.raise_on_invoke = raise_on_invoke
        self.raise_on_stream = raise_on_stream
        self.invoked_with = None

    async def ainvoke(self, inputs):
        self.invoked_with = inputs
        if self.raise_on_invoke:
            raise RuntimeError("invoke boom")
        return FakeMessage(self.content)

    async def astream(self, inputs):
        self.invoked_with = inputs
        if self.raise_on_stream:
            raise RuntimeError("stream boom")
            yield  # pragma: no cover — makes this an async generator
        for c in self.chunks or []:
            yield FakeMessage(c)


def make_config_service():
    cs = MagicMock()
    cs._load_config = AsyncMock()
    cfg = MagicMock()
    cfg.updated_at = "t1"
    cfg.created_at = "t1"
    cs.get_config = AsyncMock(return_value=cfg)

    llm_settings = MagicMock()
    llm_settings.default_model = "gpt-4o-mini"
    llm_settings.temperature = 0.2
    llm_settings.max_tokens = 512
    llm_settings.top_p = 1.0
    cs.get_llm_settings = AsyncMock(return_value=llm_settings)

    rag_settings = MagicMock()
    rag_settings.human_referral_message = STATIC_REFERRAL
    rag_settings.knowledge_base_confidence_threshold = 0.7
    rag_settings.system_prompt = "شما دستیار پرشین وی هستید."
    cs.get_rag_settings = AsyncMock(return_value=rag_settings)
    return cs


def low_confidence_intent():
    return {
        "intent": "PRIVATE",
        "is_public": False,
        "explanation": "specialized question",
        "topic_shift": False,
        "context_state": "self_contained",
        "resolved_query": "قیمت انسولین چقدر است؟",
        "referenced_entities": [],
        "context_decided": True,
    }


def make_kb_service(confidence=0.1):
    kb = MagicMock()
    kb.query_knowledge_base = AsyncMock(return_value={
        "answer": "unused",
        "confidence_score": confidence,
        "sources": [],
        "rewritten_query": "q",
        "normalized_docs": [],
        "source_type": "knowledge_base",
    })
    kb._retrieve_context = AsyncMock(return_value={
        "confidence_score": confidence,
        "sources": [],
        "rewritten_query": "q",
        "docs_with_scores": [],
        "normalized_docs": [],
        "source_type": "knowledge_base",
    })
    return kb


def make_service(chain):
    from app.services.chat_service import ChatService
    svc = ChatService()
    svc.config_service = make_config_service()
    svc.generalAnswer = False  # general answers disabled -> handoff path
    svc.detect_query_intent = AsyncMock(return_value=low_confidence_intent())
    svc._get_or_create_session = AsyncMock(return_value=chain)
    return svc


async def main():
    import app.services.chat_service as chat_module
    from langchain_core.messages import HumanMessage, AIMessage

    passed = 0

    # ---------- Case 1: non-streaming, model-generated handoff ----------
    chain = FakeChain(content=MODEL_HANDOFF)
    svc = make_service(chain)
    kb = make_kb_service()
    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        result = await svc.process_message("u1", "قیمت انسولین چقدر است؟")
    assert result["answer"] == MODEL_HANDOFF, \
        f"case1 expected model-generated handoff, got: {result['answer']!r}"
    assert result["query_analysis"]["requires_human_referral"] is True
    assert result["query_analysis"]["knowledge_source"] == "none"
    assert result["prompt_snapshot"]["response_type"] == "general"
    hist = svc._message_history["u1"]
    assert isinstance(hist[-1], AIMessage) and hist[-1].content == MODEL_HANDOFF
    assert chain.invoked_with["input"] == "قیمت انسولین چقدر است؟"
    print("CASE1 OK: non-streaming handoff returns model's own response + stores it in history")
    passed += 1

    # ---------- Case 2: non-streaming, LLM failure -> static referral ----------
    chain = FakeChain(raise_on_invoke=True)
    svc = make_service(chain)
    kb = make_kb_service()
    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        result = await svc.process_message("u2", "قیمت انسولین چقدر است؟")
    assert result["answer"] == STATIC_REFERRAL, \
        f"case2 expected static referral fallback, got: {result['answer']!r}"
    assert result["query_analysis"]["requires_human_referral"] is True
    hist = svc._message_history["u2"]
    assert hist[-1].content == STATIC_REFERRAL
    print("CASE2 OK: non-streaming LLM failure falls back to static referral message")
    passed += 1

    # ---------- Case 3: streaming, model-generated handoff streamed token-by-token ----------
    chain = FakeChain(chunks=STREAM_CHUNKS)
    svc = make_service(chain)
    kb = make_kb_service()
    events = []
    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        async for ev in svc.process_message_stream("u3", "قیمت انسولین چقدر است؟"):
            events.append(ev)
    chunk_events = [e for e in events if e["type"] == "chunk"]
    streamed = "".join(e["content"] for e in chunk_events)
    done = [e for e in events if e["type"] == "done"][0]
    assert streamed == MODEL_HANDOFF, f"case3 streamed text mismatch: {streamed!r}"
    assert done["answer"] == MODEL_HANDOFF
    assert done["query_analysis"]["requires_human_referral"] is True
    assert done["query_analysis"]["knowledge_source"] == "none"
    assert done["prompt_snapshot"]["response_type"] == "general"
    hist = svc._message_history["u3"]
    assert isinstance(hist[-1], AIMessage) and hist[-1].content == MODEL_HANDOFF
    print("CASE3 OK: streaming handoff streams the model's own answer and reports it in done")
    passed += 1

    # ---------- Case 4: streaming, empty stream -> static referral fallback ----------
    chain = FakeChain(chunks=[])
    svc = make_service(chain)
    kb = make_kb_service()
    events = []
    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        async for ev in svc.process_message_stream("u4", "قیمت انسولین چقدر است؟"):
            events.append(ev)
    done = [e for e in events if e["type"] == "done"][0]
    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert done["answer"] == STATIC_REFERRAL, f"case4 expected static fallback, got {done['answer']!r}"
    assert any(e["content"] == STATIC_REFERRAL for e in chunk_events), \
        "case4 static referral must be yielded as a chunk"
    hist = svc._message_history["u4"]
    assert hist[-1].content == STATIC_REFERRAL
    print("CASE4 OK: empty model stream falls back to static referral (chunk + done)")
    passed += 1

    # ---------- Case 5: streaming, stream raises -> static referral fallback ----------
    chain = FakeChain(raise_on_stream=True)
    svc = make_service(chain)
    kb = make_kb_service()
    events = []
    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        async for ev in svc.process_message_stream("u5", "قیمت انسولین چقدر است؟"):
            events.append(ev)
    done = [e for e in events if e["type"] == "done"][0]
    assert done["answer"] == STATIC_REFERRAL, f"case5 expected static fallback, got {done['answer']!r}"
    assert done["query_analysis"]["requires_human_referral"] is True
    hist = svc._message_history["u5"]
    assert hist[-1].content == STATIC_REFERRAL
    print("CASE5 OK: stream failure falls back to static referral without emitting an error event")
    passed += 1

    # ---------- Case 6: topic_shift isolation applies to handoff history too ----------
    chain = FakeChain(chunks=STREAM_CHUNKS)
    svc = make_service(chain)
    svc.detect_query_intent = AsyncMock(return_value={
        **low_confidence_intent(),
        "context_state": "topic_shift",
        "topic_shift": True,
    })
    # Pre-seed internal history with 6 messages (3 old-subject exchanges)
    svc._message_history["u6"] = [
        HumanMessage(content="قدیم۱"), AIMessage(content="قدیم۲"),
        HumanMessage(content="قدیم۳"), AIMessage(content="قدیم۴"),
        HumanMessage(content="قدیم۵"), AIMessage(content="قدیم۶"),
    ]
    kb = make_kb_service()
    with patch.object(chat_module, "get_knowledge_base_service", return_value=kb):
        async for ev in svc.process_message_stream("u6", "سوال جدید"):
            if ev["type"] == "done":
                break
    hist_seen = chain.invoked_with["history"]
    assert len(hist_seen) == 2 and hist_seen[0].content == "قدیم۵", \
        f"case6 topic_shift isolation failed, history seen: {[m.content for m in hist_seen]}"
    print("CASE6 OK: topic_shift prunes history to last exchange for the handoff chain")
    passed += 1

    print(f"\nALL {passed}/6 CASES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
