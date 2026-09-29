"""Tests for the stage-timing instrumentation (`app/core/perf_timing.py`).

Production motivation: the reported 38-158 s chat latency had no per-stage numbers
behind it, so `timed_stage` / the stage decorators exist to attribute the seconds to
a stage (BM25 index build, BM25 search, dense search, merge, rerank, rewrite, prompt
build, LLM). These tests pin down:

  * the flag: with `PERF_TIMING_LOG=false` nothing is timed and nothing is logged,
  * the log contract: one `[PERF_STAGE] stage=<name> elapsed=<s>` line, with the
    extra fields a call site adds inside the block and `ok=False` on failure,
  * the exception contract: a stage never swallows or rewraps an error,
  * the decorators: coroutine, async generator (timed until exhausted) and the
    guard against decorating a non-generator,
  * that the retrieval pipeline is actually instrumented at the stages the audit
    relies on (so the tooling cannot be dropped silently).
"""
import asyncio
import inspect
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core import perf_timing  # noqa: E402


class _ListHandler(logging.Handler):
    """Captures the module's own handler output (the logger does not propagate)."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def stage_messages():
    handler = _ListHandler()
    perf_timing.logger.addHandler(handler)
    perf_timing.logger.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        perf_timing.logger.removeHandler(handler)
        perf_timing.set_perf_timing_enabled(True)


@pytest.fixture
def timing_on():
    perf_timing.set_perf_timing_enabled(True)
    yield
    perf_timing.set_perf_timing_enabled(True)


def test_disabled_flag_times_and_logs_nothing(stage_messages):
    """`PERF_TIMING_LOG=false` removes the whole cost, not only the log line."""
    perf_timing.set_perf_timing_enabled(False)
    try:
        with perf_timing.timed_stage("bm25_search", docs=3) as data:
            assert data == {}, "the flag-off path must not allocate/measure"

        perf_timing.log_stage("bm25_search", 1.23, {"docs": 3})
    finally:
        perf_timing.set_perf_timing_enabled(True)

    assert stage_messages == []


def test_reload_reads_the_environment_flag(monkeypatch):
    monkeypatch.setenv(perf_timing.PERF_TIMING_ENV_FLAG, "false")
    assert perf_timing.reload_perf_timing_flag() is False
    monkeypatch.setenv(perf_timing.PERF_TIMING_ENV_FLAG, "true")
    assert perf_timing.reload_perf_timing_flag() is True
    monkeypatch.delenv(perf_timing.PERF_TIMING_ENV_FLAG)
    assert perf_timing.reload_perf_timing_flag() is True  # default: on


def test_stage_line_carries_the_step_key_and_counts(timing_on, stage_messages):
    with perf_timing.timed_stage("bm25_index_build", branch="contrib") as extra:
        extra["docs"] = 8123

    assert len(stage_messages) == 1
    line = stage_messages[0]
    assert line.startswith(perf_timing.PERF_TIMING_PREFIX)
    assert "stage=bm25_index_build" in line
    assert "elapsed=" in line
    assert "branch=contrib" in line
    assert "docs=8123" in line
    assert "ok=True" in line


def test_failure_is_logged_and_the_exception_is_reraised(timing_on, stage_messages):
    with pytest.raises(ValueError):
        with perf_timing.timed_stage("dense_search"):
            raise ValueError("chroma down")

    assert len(stage_messages) == 1
    assert "stage=dense_search" in stage_messages[0]
    assert "ok=False" in stage_messages[0]
    assert "ValueError: chroma down" in stage_messages[0]


def test_sync_decorator(timing_on, stage_messages):
    @perf_timing.timed_sync_stage("merge_normalize")
    def merge():
        return 42

    assert merge() == 42
    assert "stage=merge_normalize" in stage_messages[0]


@pytest.mark.asyncio
async def test_async_decorator(timing_on, stage_messages):
    @perf_timing.timed_async_stage("hybrid_retrieve_total", mode="test")
    async def retrieve():
        await asyncio.sleep(0)
        return "docs"

    assert await retrieve() == "docs"
    assert "stage=hybrid_retrieve_total" in stage_messages[0]
    assert "mode=test" in stage_messages[0]


@pytest.mark.asyncio
async def test_async_generator_decorator_times_until_exhaustion(timing_on, stage_messages):
    @perf_timing.timed_async_generator_stage("chat_request_total")
    async def stream():
        yield "a"
        yield "b"

    chunks = [chunk async for chunk in stream()]

    assert chunks == ["a", "b"]
    assert "stage=chat_request_total" in stage_messages[0]
    assert "ok=True" in stage_messages[0]


def test_async_generator_decorator_rejects_a_coroutine_function():
    with pytest.raises(TypeError):

        @perf_timing.timed_async_generator_stage("nope")
        async def not_a_generator():
            return 1


def test_retrieval_pipeline_is_instrumented_at_the_audited_stages():
    """The audit's stages must exist in the pipeline, not only in this module."""
    # `chat_service` first: it imports `get_knowledge_base_service` from
    # `knowledge_base` at module level, so importing `knowledge_base` alone is a
    # circular import (a pre-existing quirk of the two services).
    from app.services import chat_service, hybrid_retrieval, knowledge_base  # noqa: F401

    hybrid_source = inspect.getsource(hybrid_retrieval)
    kb_source = inspect.getsource(knowledge_base)

    for stage in (
        "bm25_index_build",
        "bm25_search",
        "chroma_docs_fetch",
        "embed_query",
        "dense_search",
        "merge_normalize",
        "rerank",
    ):
        assert f'"{stage}"' in hybrid_source, f"hybrid_retrieval lost the {stage} stage"

    assert '@timed_async_stage("hybrid_retrieve_total")' in hybrid_source

    for stage in ("expand_query_llm", "vector_store_load"):
        assert stage in kb_source, f"knowledge_base lost the {stage} stage"

    assert '@timed_async_stage("retrieval_total")' in kb_source
    assert '@timed_async_generator_stage("llm_answer_stream")' in kb_source
