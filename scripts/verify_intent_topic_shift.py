"""Offline verification of detect_query_intent topic_shift changes (no API calls).

Run: python scripts/verify_intent_topic_shift.py
"""
import asyncio
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response_text):
        self._response_text = response_text
        self.last_prompt = None       # SystemMessage (classifier prompt)
        self.last_human_msg = None    # HumanMessage content

    async def ainvoke(self, messages):
        self.last_prompt = messages[0].content  # SystemMessage holds classifier prompt
        if len(messages) > 1:
            self.last_human_msg = messages[1].content  # HumanMessage
        return FakeMessage(self._response_text)


async def main():
    from app.services.chat_service import ChatService

    svc = ChatService()

    history = [
        {"role": "user", "content": "برای یبوست چه مکملی خوبه؟"},
        {"role": "assistant", "content": "برای یبوست فیبر و ..."},
        {"role": "user", "content": "برای ترک قلیان چی پیشنهاد میدی؟"},
        {"role": "assistant", "content": "..."},
    ]

    # Case 1: topic_shift returned as string "true" (LLM string coercion path)
    llm1 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "topic_shift": "true",
        "explanation": "Subject switched from constipation to quitting smoking.",
    }))
    r1 = await svc.detect_query_intent("کشت دوم از طریق برش ساقه چطور انجام میشه؟", history, llm=llm1)
    assert r1["topic_shift"] is True, f"case1 topic_shift expected True, got {r1['topic_shift']!r}"
    assert r1["intent"] == "PRIVATE"
    print("CASE1 OK: string 'true' -> topic_shift=True, intent=PRIVATE")
    assert "کشت دوم از طریق برش ساقه" in llm1.last_prompt, "agriculture example missing from prompt"
    assert "برش ساقه" in llm1.last_prompt, "propagation bullet missing from prompt"
    assert "topic_shift" in llm1.last_prompt, "topic_shift field missing from schema in prompt"
    print("CASE1 OK: prompt contains agriculture example, propagation bullet, topic_shift schema")
    # Verify JSON schema in prompt is valid (commas fixed)
    schema_start = llm1.last_prompt.index("Respond with valid JSON only")
    schema_text = llm1.last_prompt[schema_start:]
    braces = schema_text[schema_text.index("{"):schema_text.rindex("}") + 1]
    # Strip the "key": "A" | "B" | "C" union types which are not valid JSON
    import re as _re
    cleaned = _re.sub(r':\s*"[^"]*"(\s*\|\s*"[^"]*")+', ': "x"', braces)
    cleaned = _re.sub(r':\s*0\.0-1\.0', ': 0.5', cleaned)
    cleaned = _re.sub(r':\s*true\s*\|\s*false', ': true', cleaned)
    json.loads(cleaned)
    print("CASE1 OK: JSON schema in prompt is parseable (commas fixed)")

    # Case 2: topic_shift as bool false
    llm2 = FakeLLM(json.dumps({
        "intent": "PUBLIC", "category": "product_info", "confidence": 0.95,
        "topic_shift": False, "explanation": "Same product topic continues.",
    }))
    r2 = await svc.detect_query_intent("کامبوچا چیه؟", history, llm=llm2)
    assert r2["topic_shift"] is False and r2["is_public"] is True
    print("CASE2 OK: bool False -> topic_shift=False, is_public=True")

    # Case 3: missing topic_shift key -> default False
    llm3 = FakeLLM(json.dumps({"intent": "OFF_TOPIC", "category": "unrelated",
                               "confidence": 0.8, "explanation": "Football question."}))
    r3 = await svc.detect_query_intent("بهترین تیم فوتبال؟", history, llm=llm3)
    assert r3["topic_shift"] is False and r3["intent"] == "OFF_TOPIC"
    print("CASE3 OK: missing key -> topic_shift=False default")

    # Case 4: empty message path returns topic_shift
    r4 = await svc.detect_query_intent("   ", None, llm=FakeLLM(""))
    assert r4["intent"] == "NEEDS_CLARIFICATION" and r4["topic_shift"] is False
    print("CASE4 OK: empty message -> NEEDS_CLARIFICATION with topic_shift=False")

    # Case 5: unparseable response -> fallback path returns topic_shift
    llm5 = FakeLLM("not json at all")
    r5 = await svc.detect_query_intent("سلام", None, llm=llm5)
    assert r5["intent"] == "PRIVATE" and r5["topic_shift"] is False
    print("CASE5 OK: unparseable response -> fallback PRIVATE with topic_shift=False")

    # Case 6: case-insensitive role filter — "Human:"/"User:" capitalized roles
    # must still be picked up as recent user turns.
    capped_history = [
        {"role": "Human", "content": "برای یبوست چه مکملی خوبه؟"},
        {"role": "AI", "content": "برای یبوست فیبر و ..."},
        {"role": "User", "content": "برای ترک قلیان چی پیشنهاد میدی؟"},
        {"role": "Assistant", "content": "..."},
    ]
    llm6 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "topic_shift": True, "explanation": "Different subject.",
    }))
    r6 = await svc.detect_query_intent("چطور گوجه بکارم؟", capped_history, llm=llm6)
    # With the case-insensitive fix, capped roles appear BOTH in the history block
    # AND in the "Recent user turns" block; without the fix they'd appear only once.
    assert llm6.last_prompt.count("Human: برای یبوست") >= 2, \
        "capitalized 'Human:' role not picked up as recent user turn"
    assert llm6.last_prompt.count("User: برای ترک قلیان") >= 2, \
        "capitalized 'User:' role not picked up as recent user turn"
    assert r6["topic_shift"] is True
    print("CASE6 OK: case-insensitive role filter picks up 'Human:'/'User:' roles")

    # Case 7: history deduplication — conversation history lives ONLY in the
    # system prompt; the HumanMessage must contain just the latest message.
    llm7 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "topic_shift": False, "explanation": "Same subject.",
    }))
    await svc.detect_query_intent("سوال جدید", history, llm=llm7)
    assert "Conversation History" in llm7.last_prompt, "history missing from system prompt"
    assert "برای یبوست" in llm7.last_prompt, "history content missing from system prompt"
    hm = llm7.last_human_msg or ""
    assert "برای یبوست" not in hm, "history duplicated in HumanMessage (should be system-prompt only)"
    assert "Conversation history" not in hm.lower(), "history block duplicated in HumanMessage"
    assert "سوال جدید" in hm, "latest message missing from HumanMessage"
    print("CASE7 OK: history in system prompt only; HumanMessage carries just the latest message")

    # Case 8: _prune_history_for_topic_shift keeps only the last exchange,
    # for plain message lists, ConversationResponse-like objects, and
    # lists of ConversationResponse-like objects.
    pruned = svc._prune_history_for_topic_shift(history)
    assert isinstance(pruned, list) and len(pruned) == 2
    assert pruned[-1]["content"] == "...", "pruning must keep the most recent exchange"

    class FakeConvResponse:
        def __init__(self, msgs):
            self.messages = msgs

    conv = FakeConvResponse(history)
    pruned_conv = svc._prune_history_for_topic_shift(conv)
    assert len(pruned_conv) == 2 and pruned_conv[-1]["content"] == "..."

    pruned_list_of_conv = svc._prune_history_for_topic_shift([FakeConvResponse(history)])
    assert len(pruned_list_of_conv) == 2 and pruned_list_of_conv[-1]["content"] == "..."

    assert svc._prune_history_for_topic_shift(None) is None
    assert svc._prune_history_for_topic_shift([]) == []
    print("CASE8 OK: _prune_history_for_topic_shift prunes all history shapes correctly")

    # Case 9: source-level checks — the context decision is actually CONSUMED
    # downstream (context isolation, feedback B1) in both the streaming and
    # non-streaming pipelines, and the classifier role filter is case-insensitive.
    src_path = os.path.join(os.path.dirname(__file__), "..", "app", "services", "chat_service.py")
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert src.count('_prune_history_for_topic_shift(conversation_history)') >= 2, \
        "context_state/topic_shift must prune history before retrieval in BOTH process_message and process_message_stream"
    assert src.count('ContextState.TOPIC_SHIFT.value') >= 5, \
        "context_state must gate history pruning in KB path + general path of both pipelines (plus the derived boolean in detect_query_intent)"
    assert src.count('ContextState.FOLLOW_UP.value') >= 2, \
        "follow_up must be consumed (resolved_query used for retrieval) in both pipelines"
    assert 'skip_rewrite=context_decided' in src, \
        "skip_rewrite must be forwarded to retrieval so the rewrite decision is made ONCE"
    assert src.count('resolved_query") or message') >= 2, \
        "both pipelines must fall back to the raw message when resolved_query is missing"
    assert '[CONTEXT] Topic shift detected' in src, "pruning log line missing"
    assert 'm.lower().startswith("human:")' in src and 'm.lower().startswith("user:")' in src, \
        "role filter must be case-insensitive"
    assert src.count("Conversation history:\\n{history_block}") == 0, \
        "history must not be duplicated in the classifier HumanMessage"
    print("CASE9 OK: context_state consumed downstream (stream + non-stream); follow_up + skip_rewrite wired; no dup history; case-insensitive filter")

    # Case 10: _snapshot_history must handle LangChain messages (HumanMessage/AIMessage
    # from self._message_history) — they have no `.role`, only `.type`. Before the fix
    # this silently returned [] and emptied history_before_append in the general path.
    from langchain_core.messages import HumanMessage as _HM, AIMessage as _AM
    lc_history = [_HM(content="برای یبوست چه مکملی خوبه؟"), _AM(content="فیبر و...")]
    snap = svc._snapshot_history(lc_history)
    assert len(snap) == 2, f"LangChain history snapshot empty/wrong: {snap!r}"
    assert snap[0] == {"role": "user", "content": "برای یبوست چه مکملی خوبه؟"}
    assert snap[1] == {"role": "assistant", "content": "فیبر و..."}

    # Pruning LangChain history also works and snapshots correctly afterwards
    pruned_lc = svc._prune_history_for_topic_shift(lc_history + [_HM(content="ترک قلیان؟"), _AM(content="...")])
    snap_pruned = svc._snapshot_history(pruned_lc)
    assert len(snap_pruned) == 2 and snap_pruned[0]["role"] == "user" and "قلیان" in snap_pruned[0]["content"], \
        f"pruned LangChain history snapshot wrong: {snap_pruned!r}"

    # Dict history still snapshots as before (regression check)
    snap_dicts = svc._snapshot_history(history)
    assert len(snap_dicts) == 4 and snap_dicts[0]["role"] == "user"
    print("CASE10 OK: _snapshot_history handles LangChain messages (role mapping via .type)")

    # Case 11: context_state returned by the classifier is parsed and passed through
    llm11 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "context_state": "follow_up",
        "resolved_query": "مرحله دوم کوددهی پنبه",
        "referenced_entities": ["کوددهی پنبه"],
        "explanation": "Follow-up with pronoun resolution.",
    }))
    r11 = await svc.detect_query_intent("مرحله بعد چیه؟", history, llm=llm11)
    assert r11["context_state"] == "follow_up", f"case11 context_state: {r11['context_state']!r}"
    assert r11["topic_shift"] is False, "follow_up must derive topic_shift=False"
    assert r11["resolved_query"] == "مرحله دوم کوددهی پنبه"
    assert r11["referenced_entities"] == ["کوددهی پنبه"]
    assert r11["context_decided"] is True
    print("CASE11 OK: context_state=follow_up parsed; resolved_query/entities passed through; context_decided=True")

    # Case 12: topic_shift context_state derives the backward-compat boolean
    llm12 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "agriculture", "confidence": 0.9,
        "context_state": "topic_shift",
        "resolved_query": "کشت دوم از طریق برش ساقه",
        "referenced_entities": [],
        "explanation": "New subject.",
    }))
    r12 = await svc.detect_query_intent("کشت دوم از طریق برش ساقه چطور انجام میشه؟", history, llm=llm12)
    assert r12["context_state"] == "topic_shift" and r12["topic_shift"] is True
    assert r12["resolved_query"] == "کشت دوم از طریق برش ساقه", \
        "in-bounds resolved_query passes validation (only blank/runaway rewrites are rejected)"
    print("CASE12 OK: context_state=topic_shift derives topic_shift=True; in-bounds resolved_query passed through")

    # Case 13: invalid context_state falls back to topic_shift derivation (legacy LLMs)
    llm13 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "context_state": "banana",
        "topic_shift": "true",
        "explanation": "Legacy boolean style.",
    }))
    r13 = await svc.detect_query_intent("قیمت لپتاپ چنده؟", history, llm=llm13)
    assert r13["context_state"] == "topic_shift" and r13["topic_shift"] is True, \
        f"case13: {r13['context_state']!r} / {r13['topic_shift']!r}"
    print("CASE13 OK: invalid context_state derived from legacy topic_shift=true")

    # Case 14: resolved_query validation — missing / absurd-length rewrites rejected
    llm14 = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "context_state": "follow_up",
        "resolved_query": "   ",
        "referenced_entities": ["x"],
        "explanation": "Blank resolution.",
    }))
    r14 = await svc.detect_query_intent("مرحله بعد چیه؟", history, llm=llm14)
    assert r14["resolved_query"] == "مرحله بعد چیه؟", "blank resolved_query must fall back to raw message"
    assert r14["referenced_entities"] == [], "entities must be dropped with the invalid rewrite"

    llm14b = FakeLLM(json.dumps({
        "intent": "PRIVATE", "category": "health", "confidence": 0.9,
        "context_state": "follow_up",
        "resolved_query": "طولانی " * 40,  # far beyond 3x the raw message length
        "referenced_entities": ["y"],
        "explanation": "Runaway rewrite.",
    }))
    r14b = await svc.detect_query_intent("مرحله بعد چیه؟", history, llm=llm14b)
    assert r14b["resolved_query"] == "مرحله بعد چیه؟", "out-of-bounds resolved_query must fall back to raw message"
    print("CASE14 OK: blank and runaway resolved_query rejected; raw message kept")

    # Case 15: source-level checks in knowledge_base.py — single rewrite decision,
    # strict gating, and answer suppression
    kb_path = os.path.join(os.path.dirname(__file__), "..", "app", "services", "knowledge_base.py")
    with open(kb_path, "r", encoding="utf-8") as fh:
        kb_src = fh.read()
    assert "skip_rewrite: bool = False" in kb_src, "skip_rewrite parameter missing"
    assert kb_src.count("skip_rewrite=skip_rewrite") >= 2, \
        "skip_rewrite must be forwarded by _retrieve_context AND query_knowledge_base"
    assert "STRICT GATING" in kb_src and "keeping top" not in kb_src, \
        "query_knowledge_base must use strict gating (no keep-anyway fallback)"
    assert "human_referral_message" in kb_src.split("STRICT GATING")[1][:2000], \
        "strict gating must return the human referral message"
    assert "ANSWER SUPPRESSION" in kb_src, "requires_human must suppress the weak generated answer"
    # skip_rewrite fast path returns the query verbatim with no LLM call
    assert '"all_queries": [query]' in kb_src, "skip_rewrite fast path must use the query verbatim"
    print("CASE15 OK: knowledge_base strict gating + answer suppression + skip_rewrite wired")

    print("\nALL 15 CASES PASSED")


if __name__ == "__main__":
    asyncio.run(main())
