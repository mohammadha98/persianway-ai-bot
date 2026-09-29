"""Standalone retrieval-latency benchmark: BM25 vs vector search vs LLM.

Question this script answers
----------------------------
Chat answers were reported at 38-158 s. Before optimising anything, this script
produces the numbers: how much of a turn is BM25, how much is the Chroma vector
search, how much is the LLM, and how much is everything else (query rewrite,
dedup, prompt build).

It never starts HTTP: it drives the same services the API calls
(`KnowledgeBaseService._retrieve_context`, `HybridRetrievalService.hybrid_retrieve`,
`KnowledgeBaseService.stream_answer_from_context`), so the numbers are the layer's
own numbers rather than a client-side stopwatch around a socket.

What is measured (phases, all optional)
---------------------------------------
  A. corpus      -- real collection size: chunks, per-branch chunks, sampled average
                    chunk length in characters/tokens, Chroma fetch cost per branch.
  B. bm25        -- first run (cold: docs fetch + pure-Python BM25 index build) and
                    warm runs (index served from cache).
  C. hybrid      -- embed_query + dense search (3 branches) + BM25 + merge + rerank.
  D. retrieval   -- `_retrieve_context` end to end (adds rewrite/expand, dedup) plus
                    the prompt-build (template rendering) cost.
  E. llm         -- `--with-llm`: query rewrite (`expand_query_llm`) and the answer
                    stream (time to first token + total generation).
  F. contention  -- `--concurrency N`: N concurrent retrievals plus an event-loop lag
                    probe, to separate "raw duration" from "queueing behind blocking
                    work on the event loop".

Cold vs warm
------------
The FIRST execution of a stage is reported separately (`cold` column) instead of
being averaged into the warm runs: the BM25 index build and the Chroma collection
fetch happen once per process, and mixing them into the warm statistics hides both.

Usage
-----
    python scripts/bench_retrieval_latency.py                     # offline phases A-D
    python scripts/bench_retrieval_latency.py --with-llm          # also phase E
    python scripts/bench_retrieval_latency.py --no-dense          # BM25-only, fully local
    python scripts/bench_retrieval_latency.py --concurrency 4      # phase F

Output: a printed table, plus `reports/bench_retrieval_<timestamp>/summary.json`
and `stages.csv` for later analysis. Nothing is written into the vector store, the
database or the configuration.

Stage timings come from `app/core/perf_timing.py` (the `[PERF_STAGE]` lines the app
emits when `PERF_TIMING_LOG` is on). The recorder below replaces that module's
`log_stage` with an in-process collector, so this script captures the exact same
stage boundaries the production code measures -- with no log parsing and no HTTP.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# The queries and the documents are Persian: never die on a cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - old/odd consoles
    pass

#: The feedback dump the retrieval queries are taken from (sibling of this repo).
DEFAULT_DATASET = os.path.join(
    os.path.dirname(PROJECT_ROOT), "persianway-rag-db.ai_feedback.json"
)
#: The panel configuration dump, used for production-like RAG settings/templates.
DEFAULT_CONFIG_DUMP = os.path.join(
    os.path.dirname(PROJECT_ROOT), "persianway-rag-db.config.json"
)

#: Stages this script understands (reported in `stages.csv`/summary as they appear).
#: Not a filter: any stage the app records is reported, this is only documentation of
#: the expected set and its order inside a turn: vector_store_load ->
#: collection_count_probe -> (chroma_docs_fetch -> bm25_index_build once per process)
#: -> expand_query_llm -> bm25_search / embed_query / dense_search -> merge_normalize
#: -> rerank -> dedup_filter -> prompt_build -> llm_first_chunk -> llm_answer_stream.


def safe_print(text: str = "") -> None:
    """print() that cannot raise UnicodeEncodeError on a legacy console."""
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile (no interpolation); None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def load_queries(path: str, limit: int) -> List[Dict[str, Any]]:
    """Pick real Persian questions out of the feedback dump.

    The dump holds one record per rated answer (`question`, `feedback`,
    `category`). Records are de-duplicated by question text and then sampled
    evenly across the file so the sample is not "the first N of one day".
    """
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    seen = set()
    unique: List[Dict[str, Any]] = []
    for rec in records:
        question = (rec.get("question") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        unique.append(
            {
                "id": rec.get("feedback_id") or rec.get("message_id") or "",
                "question": question,
                "feedback": rec.get("feedback"),
                "category": rec.get("category"),
            }
        )
    if limit and limit < len(unique):
        step = len(unique) / float(limit)
        unique = [unique[min(len(unique) - 1, int(i * step))] for i in range(limit)]
    return unique


class StageRecorder:
    """Collects the application's own stage timings, in process.

    `app/core/perf_timing.py` resolves `log_stage` from its module namespace on
    every call, so replacing it here captures exactly the stages the production
    code measures -- without parsing log lines and without a logger per stage.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def install(self) -> None:
        import app.core.perf_timing as perf_timing

        perf_timing.set_perf_timing_enabled(True)
        perf_timing.log_stage = self._record  # type: ignore[assignment]
        # `knowledge_base` imported the name directly (`from ... import log_stage`),
        # so its binding has to be patched too. `chat_service` is imported first
        # because it imports `get_knowledge_base_service` from `knowledge_base` at
        # module level (importing knowledge_base alone is a circular import).
        try:
            import app.services.chat_service  # noqa: F401  (imported for its side effects)
            import app.services.knowledge_base as kb_module

            kb_module.log_stage = self._record  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover - only if imports change
            safe_print(
                f"[bench] could not patch knowledge_base.log_stage ({exc}); "
                "stages it records directly (llm_first_chunk) will be missing"
            )

    def _record(
        self,
        stage: str,
        elapsed: float,
        fields: Optional[Dict[str, Any]] = None,
        *,
        ok: bool = True,
        error: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        self.records.append(
            {
                "stage": stage,
                "elapsed": float(elapsed),
                "fields": dict(fields or {}),
                "ok": bool(ok),
                "error": error,
            }
        )

    def drain(self) -> List[Dict[str, Any]]:
        drained, self.records = self.records, []
        return drained

    def record(self, stage: str, elapsed: float, **fields: Any) -> None:
        """Add a stage the app does not instrument itself (e.g. prompt build)."""
        self._record(stage, elapsed, fields)


class OfflineConfigService:
    """Config-service stand-in so the benchmark never depends on MongoDB.

    When the panel dump sits next to this repo it is used, so `top_k_results`,
    `prompt_template` and the LLM settings match the deployment; otherwise the
    static defaults are used. Read-only: nothing is written back.
    """

    def __init__(self, dump_path: Optional[str]) -> None:
        from app.schemas.config import LLMSettings, RAGSettings

        self.source = "static defaults"
        self.rag_settings = RAGSettings()
        self.llm_settings = LLMSettings()
        if dump_path and os.path.exists(dump_path):
            try:
                with open(dump_path, encoding="utf-8") as fh:
                    dump = json.load(fh)
                if isinstance(dump, list):
                    dump = dump[0] if dump else {}
                self.rag_settings = RAGSettings(**dump.get("rag_settings", {}))
                self.llm_settings = LLMSettings(**dump.get("llm_settings", {}))
                self.source = f"panel dump ({os.path.basename(dump_path)})"
            except Exception as exc:
                safe_print(f"[bench] could not read the config dump ({exc}); using defaults")

    async def _load_config(self) -> None:  # called by `KnowledgeBaseService`
        return None

    async def get_config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            rag_settings=self.rag_settings, llm_settings=self.llm_settings
        )

    async def get_rag_settings(self):
        return self.rag_settings

    async def get_llm_settings(self):
        return self.llm_settings


async def collect_corpus_stats(
    kb: Any, hrs: Any, rec: StageRecorder, token_sample: int = 200
) -> Dict[str, Any]:
    """Real size of the collection that every retrieval reads.

    Reports the total chunk count, the chunk count of each BM25 branch (the unit
    the index is actually built from), how long Chroma takes to hand that branch
    over, and the average chunk length in characters and tokens (sampled:
    tokenising all chunks is part of the cold cost measured in phase B, so it is
    not repeated here).
    """
    from app.services.hybrid_retrieval import _tokenize

    stats: Dict[str, Any] = {"branches": {}, "total_chunks": None, "token_sample": token_sample}
    vector_store = await kb._get_vector_store_async()
    collection = getattr(vector_store, "_collection", None) if vector_store else None
    if collection is None:
        stats["error"] = "no collection available (embeddings/vector store unavailable)"
        return stats

    t0 = time.perf_counter()
    stats["total_chunks"] = int(collection.count())
    stats["count_seconds"] = time.perf_counter() - t0

    for is_public in (False, True):
        for key, filt in hrs._bm25_branch_filters(is_public).items():
            label = f"{key}{'_public' if is_public else ''}"
            try:
                t_fetch = time.perf_counter()
                data = collection.get(where=filt)
                fetch_seconds = time.perf_counter() - t_fetch
                docs = data.get("documents") or []
                chars = [len(d or "") for d in docs]
                if docs and token_sample:
                    step = max(1, len(docs) // token_sample)
                    sample = docs[::step][:token_sample]
                    sampled_chars = sum(len(d or "") for d in sample)
                    sampled_tokens = sum(len(_tokenize(d or "")) for d in sample)
                    avg_tokens = sampled_tokens / len(sample)
                    avg_chars = sampled_chars / len(sample)
                else:
                    avg_tokens = avg_chars = None
                stats["branches"][label] = {
                    "chunks": len(docs),
                    "fetch_seconds": fetch_seconds,
                    "avg_chunk_chars": avg_chars,
                    "avg_chunk_tokens": avg_tokens,
                    "total_chars": sum(chars) if chars else 0,
                }
            except Exception as exc:
                stats["branches"][label] = {"error": f"{type(exc).__name__}: {exc}"}
    return stats


async def capture(rec: StageRecorder, coro) -> Dict[str, Any]:
    """Await `coro`, returning wall time + the stages it recorded."""
    rec.drain()
    t0 = time.perf_counter()
    outcome: Dict[str, Any] = {"ok": True, "error": None}
    try:
        outcome["value"] = await coro
    except Exception as exc:
        outcome["ok"] = False
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    outcome["wall"] = time.perf_counter() - t0
    outcome["stages"] = rec.drain()
    return outcome


class Phase:
    """One measured phase: a list of runs, each with its own stage timings."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.runs: List[Dict[str, Any]] = []

    def add(self, outcome: Dict[str, Any], label: str) -> None:
        # A stage that runs more than once inside one measurement (one Chroma fetch
        # per BM25 branch, one rewrite per expanded query) is SUMMED, so the cold
        # column of e.g. `chroma_docs_fetch` reads as the whole cost of that
        # measurement rather than "whichever branch finished last".
        stages: Dict[str, float] = {}
        for record in outcome["stages"]:
            stages[record["stage"]] = stages.get(record["stage"], 0.0) + record["elapsed"]
        self.runs.append(
            {
                "label": label,
                "wall": outcome["wall"],
                "ok": outcome["ok"],
                "error": outcome["error"],
                "stages": stages,
                "failures": [
                    {"stage": r["stage"], "error": r["error"]}
                    for r in outcome["stages"]
                    if not r["ok"]
                ],
            }
        )

    @property
    def wall_times(self) -> List[float]:
        return [run["wall"] for run in self.runs]

    def stage_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        """Per stage: the cold sample, and the median/p95 of the warm samples."""
        stats: Dict[str, Dict[str, Optional[float]]] = {}
        names = set()
        for run in self.runs:
            names.update(run["stages"])
        for stage in sorted(names):
            values = [run["stages"][stage] for run in self.runs if stage in run["stages"]]
            if not values:
                continue
            cold, warm = values[0], values[1:]
            stats[stage] = {
                "n": len(values),
                "cold": cold,
                "warm_n": len(warm),
                "median": median(warm),
                "p95": percentile(warm, 95),
                "min": min(warm) if warm else None,
                "max": max(warm) if warm else None,
            }
        return stats

    def summary(self) -> Dict[str, Any]:
        return {
            "phase": self.name,
            "runs": len(self.runs),
            "ok_runs": sum(1 for run in self.runs if run["ok"]),
            "wall_median": median(self.wall_times),
            "wall_p95": percentile(self.wall_times, 95),
            "wall_min": min(self.wall_times) if self.wall_times else None,
            "wall_max": max(self.wall_times) if self.wall_times else None,
            "stages": self.stage_stats(),
            "run_detail": self.runs,
        }


async def run_bm25_phase(hrs: Any, queries: List[Dict[str, Any]], reps: int, k: int, rec: StageRecorder) -> Phase:
    """Phase B: the BM25 path alone (fetches Chroma docs, builds/reuses the index).

    The caches are dropped first on purpose: the point of this phase is to separate
    the one-off index build (cold) from the search itself (warm).
    """
    phase = Phase("B_bm25_only")
    hrs.invalidate_caches()
    for qi, query in enumerate(queries):
        for rep in range(reps):
            outcome = await capture(rec, hrs._bm25_parallel_async(query["question"], k, False))
            phase.add(outcome, label=f"q{qi + 1}#{rep + 1}")
    return phase


async def run_hybrid_phase(hrs: Any, queries: List[Dict[str, Any]], reps: int, rec: StageRecorder) -> Phase:
    """Phase C: dense + BM25 + merge + rerank (`HybridRetrievalService.hybrid_retrieve`)."""
    phase = Phase("C_hybrid_retrieve")
    for qi, query in enumerate(queries):
        for rep in range(reps):
            outcome = await capture(rec, hrs.hybrid_retrieve(query["question"], False))
            phase.add(outcome, label=f"q{qi + 1}#{rep + 1}")
    return phase


async def measure_rewrite(kb: Any, query: str) -> Dict[str, Any]:
    """Time one query rewrite/expansion (an LLM round trip).

    The duration is returned rather than pushed into the recorder: the recorder is
    drained per `capture()`, so a record written here would be discarded by the
    next measurement. `run_retrieval_phase` turns it into a stage of its own phase.
    """
    start = time.perf_counter()
    try:
        result = await kb.expand_query_with_context(
            query=query, conversation_history=None, skip_rewrite=False
        )
        elapsed = time.perf_counter() - start
        return {"ok": True, "rewritten_query": result.get("rewritten_query") or query, "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {"ok": False, "rewritten_query": query, "elapsed": elapsed, "error": str(exc)}


async def run_retrieval_phase(
    kb: Any,
    queries: List[Dict[str, Any]],
    reps: int,
    rec: StageRecorder,
    rewrite_mode: str,
    with_prompt: bool,
) -> Tuple[List[Phase], List[Dict[str, Any]]]:
    """Phase D: `_retrieve_context` end to end, plus the prompt-build cost.

    `rewrite_mode`:
      * "auto" -- measure the rewrite once per query (recorded as `expand_query_llm`)
                  and then run the retrieval reps with `skip_rewrite=True` on the
                  rewritten query, which is what production does after intent
                  detection has already decided the context;
      * "on"   -- every rep runs its own rewrite inside `_retrieve_context`;
      * "off"  -- no rewrite at all (cheapest, and the BM25/vector shares are then
                  directly comparable to the raw query).
    """
    phase = Phase("D_retrieve_context")
    prompt_phase = Phase("D2_prompt_build")
    rewrite_phase = Phase("D1_query_rewrite")
    rewrites: List[Dict[str, Any]] = []

    for qi, query in enumerate(queries):
        effective_query = query["question"]
        if rewrite_mode == "auto":
            rewrite = await measure_rewrite(kb, query["question"])
            rewrites.append({"id": query["id"], **{k: v for k, v in rewrite.items()}})
            effective_query = rewrite["rewritten_query"]
            rewrite_phase.add(
                {
                    "ok": rewrite["ok"],
                    "error": rewrite.get("error"),
                    "wall": rewrite["elapsed"],
                    "stages": [
                        {
                            "stage": "expand_query_llm",
                            "elapsed": rewrite["elapsed"],
                            "ok": rewrite["ok"],
                            "error": rewrite.get("error"),
                        }
                    ],
                },
                label=f"rewrite q{qi + 1}",
            )
        for rep in range(reps):
            outcome = await capture(
                rec,
                kb._retrieve_context(
                    effective_query,
                    None,
                    False,
                    include_web_search=False,
                    skip_rewrite=rewrite_mode != "on",
                ),
            )
            phase.add(
                outcome,
                label=f"q{qi + 1}#{rep + 1}{'' if rewrite_mode != 'on' else ' (rewrite per rep)'}",
            )
            if with_prompt and outcome["ok"] and outcome.get("value"):
                retrieval = outcome["value"]
                start = time.perf_counter()
                try:
                    await kb.build_prompt_snapshot(
                        effective_query, retrieval.get("normalized_docs") or []
                    )
                    elapsed = time.perf_counter() - start
                    rec.record("prompt_build", elapsed, docs=len(retrieval.get("normalized_docs") or []))
                    # Its own phase, so the (tiny) prompt builds neither dilute the
                    # retrieval wall statistics nor hide the first-call cost of
                    # building the chain/template.
                    prompt_phase.add(
                        {
                            "ok": True,
                            "error": None,
                            "wall": elapsed,
                            "stages": [
                                {"stage": "prompt_build", "elapsed": elapsed, "ok": True, "error": None}
                            ],
                        },
                        label=f"prompt_build q{qi + 1}#{rep + 1}",
                    )
                except Exception as exc:
                    rec.record("prompt_build", time.perf_counter() - start, error=str(exc))

    phases = [phase] + ([prompt_phase] if prompt_phase.runs else [])
    if rewrite_phase.runs:
        phases.insert(0, rewrite_phase)
    return phases, rewrites


async def run_llm_phase(
    kb: Any, queries: List[Dict[str, Any]], reps: int, rec: StageRecorder
) -> Phase:
    """Phase E: the LLM share of a turn (`--with-llm` only).

    Measured per query: the answer stream (`llm_answer_stream`) and its time to the
    first token (`llm_first_chunk`, which includes prompt rendering + provider
    TTFB). The retrieval that feeds the stream is re-run here on purpose, so the
    numbers in this phase describe a complete turn.
    """
    phase = Phase("E_llm_answer")
    for qi, query in enumerate(queries):
        for rep in range(reps):
            retrieval_outcome = await capture(
                rec,
                kb._retrieve_context(
                    query["question"], None, False, include_web_search=False, skip_rewrite=True
                ),
            )
            if not retrieval_outcome["ok"]:
                phase.add(retrieval_outcome, label=f"q{qi + 1}#{rep + 1} (retrieval failed)")
                continue
            docs = retrieval_outcome["value"].get("normalized_docs") or []

            async def _drain_stream():
                chars = 0
                async for chunk in kb.stream_answer_from_context(query["question"], docs):
                    chars += len(chunk or "")
                return chars

            outcome = await capture(rec, _drain_stream())
            outcome["wall"] = retrieval_outcome["wall"] + outcome["wall"]
            # Keep the retrieval stages too: the turn total is the sum of both.
            outcome["stages"] = retrieval_outcome["stages"] + outcome["stages"]
            phase.add(outcome, label=f"q{qi + 1}#{rep + 1}")
    return phase


async def run_contention_phase(
    kb: Any,
    queries: List[Dict[str, Any]],
    concurrency: int,
    rec: StageRecorder,
    probe_interval: float = 0.01,
) -> Dict[str, Any]:
    """Phase F: does the blocking work queue requests behind the event loop?

    Runs `concurrency` retrievals at once while a probe task measures how long a
    10 ms sleep actually takes. A blocking call on the loop thread shows up as a
    lag spike *and* as per-request wall times that grow with concurrency -- which
    is the difference between "this stage is slow" and "this stage blocks everyone".
    """
    stop = asyncio.Event()
    lags: List[float] = []

    async def _lag_probe() -> None:
        while not stop.is_set():
            started = time.perf_counter()
            await asyncio.sleep(probe_interval)
            lags.append(time.perf_counter() - started - probe_interval)

    probe_task = asyncio.create_task(_lag_probe())
    try:
        jobs = [
            capture(rec, kb._retrieve_context(q["question"], None, False, include_web_search=False, skip_rewrite=True))
            for q in (queries * concurrency)[:concurrency]
        ]
        wall_start = time.perf_counter()
        outcomes = await asyncio.gather(*jobs)
        wave_seconds = time.perf_counter() - wall_start
    finally:
        stop.set()
        await probe_task

    walls = [outcome["wall"] for outcome in outcomes]
    return {
        "concurrency": concurrency,
        "wave_seconds": wave_seconds,
        "request_median": median(walls),
        "request_p95": percentile(walls, 95),
        "request_max": max(walls) if walls else None,
        "requests_ok": sum(1 for outcome in outcomes if outcome["ok"]),
        "event_loop_lag_probe_interval": probe_interval,
        "event_loop_lag_median_ms": (median(lags) or 0.0) * 1000.0,
        "event_loop_lag_p95_ms": (percentile(lags, 95) or 0.0) * 1000.0,
        "event_loop_lag_max_ms": (max(lags) if lags else 0.0) * 1000.0,
        "note": (
            "lag = how much longer than the requested 10 ms sleep the loop actually "
            "took; spikes mean a blocking call was holding the loop thread"
        ),
    }


def print_phase_table(phase: Phase) -> None:
    """One table per phase: the cold sample next to warm median / p95."""
    safe_print()
    safe_print(f"--- {phase.name} ({len(phase.runs)} runs, "
               f"{sum(1 for run in phase.runs if run['ok'])} ok) ---")
    if not phase.runs:
        safe_print("  (no runs)")
        return
    safe_print(f"  wall clock: median={fmt(median(phase.wall_times))}s "
               f"p95={fmt(percentile(phase.wall_times, 95))}s "
               f"min={fmt(min(phase.wall_times))}s max={fmt(max(phase.wall_times))}s")
    stats = phase.stage_stats()
    if not stats:
        safe_print("  (no stages recorded)")
        return
    safe_print(f"  {'stage':<26} {'cold':>9} {'median':>9} {'p95':>9} {'min':>9} {'max':>9}  n")
    for stage in sorted(stats, key=lambda name: -(stats[name]["median"] or stats[name]["cold"] or 0.0)):
        row = stats[stage]
        safe_print(
            f"  {stage:<26} {fmt(row['cold']):>9} {fmt(row['median']):>9} "
            f"{fmt(row['p95']):>9} {fmt(row['min']):>9} {fmt(row['max']):>9}  {row['n']}"
        )
    failures = [f for run in phase.runs for f in run["failures"]]
    for failure in failures[:5]:
        safe_print(f"  ! {failure['stage']} failed: {failure['error']}")
    errors = [run["error"] for run in phase.runs if run["error"]]
    for error in errors[:3]:
        safe_print(f"  ! run failed: {error}")


def _stage_value(phase: Optional[Phase], stage: str) -> Optional[float]:
    if phase is None:
        return None
    stats = phase.stage_stats().get(stage)
    if not stats:
        return None
    return stats["median"] if stats["median"] is not None else stats["cold"]


def print_decomposition(
    phase_d: Optional[Phase],
    phase_e: Optional[Phase],
    phase_r: Optional[Phase] = None,
) -> Dict[str, Any]:
    """Split one measured turn into BM25 / vector / LLM / other, with percentages."""
    rewrite_value = _stage_value(phase_r, "expand_query_llm")
    if rewrite_value is None:
        rewrite_value = _stage_value(phase_d, "expand_query_llm")
    parts: Dict[str, Optional[float]] = {
        "query_rewrite (LLM)": rewrite_value,

        "bm25_search": _stage_value(phase_d, "bm25_search"),
        "vector_search (embed+dense)": _stage_value(phase_d, "dense_branch_total"),
        "merge_normalize": _stage_value(phase_d, "merge_normalize"),
        "rerank": _stage_value(phase_d, "rerank"),
        "dedup_filter": _stage_value(phase_d, "dedup_filter"),
        "prompt_build": _stage_value(phase_d, "prompt_build"),
        "answer_llm_stream": _stage_value(phase_e, "llm_answer_stream"),
    }
    known = sum(value for value in parts.values() if value)
    retrieval_total = _stage_value(phase_d, "retrieval_total")
    turn_total = None
    if phase_e is not None:
        walls = [run["wall"] for run in phase_e.runs]
        turn_total = median(walls)

    safe_print()
    safe_print("--- latency decomposition (medians of the measured runs) ---")
    safe_print(f"  {'component':<30} {'seconds':>9} {'% of sum':>9}")
    for name, value in parts.items():
        share = f"{(100.0 * value / known):.1f}%" if (value and known) else "-"
        safe_print(f"  {name:<30} {fmt(value):>9} {share:>9}")
    safe_print(f"  {'sum of components':<30} {fmt(known):>9}")
    safe_print(f"  {'retrieval_total (app timer)':<30} {fmt(retrieval_total):>9}")
    if turn_total is not None:
        safe_print(f"  {'full turn wall (retrieval+LLM)':<30} {fmt(turn_total):>9}")
    return {"components": parts, "components_sum": known,
            "retrieval_total": retrieval_total, "turn_median": turn_total}


def write_reports(out_dir: str, payload: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "stages.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["phase", "label", "wall_seconds", "stage", "elapsed_seconds", "ok"])
        for phase in payload.get("phases", []):
            for run in phase["run_detail"]:
                if run["stages"]:
                    for stage, elapsed in run["stages"].items():
                        writer.writerow([phase["phase"], run["label"], f"{run['wall']:.6f}", stage, f"{elapsed:.6f}", run["ok"]])
                else:
                    writer.writerow([phase["phase"], run["label"], f"{run['wall']:.6f}", "", "", run["ok"]])
    safe_print()
    safe_print(f"written: {os.path.join(out_dir, 'summary.json')}")
    safe_print(f"written: {os.path.join(out_dir, 'stages.csv')}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure where a chat turn's seconds go (BM25 / vector / LLM) without HTTP."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="feedback dump with real questions")
    parser.add_argument("--config", default=DEFAULT_CONFIG_DUMP, help="panel config dump (rag/llm settings)")
    parser.add_argument("--queries", type=int, default=8, help="how many real queries to use (5-10 recommended)")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per query (first run = cold)")
    parser.add_argument("--k", type=int, default=15, help="BM25 k (matches hybrid_retrieve)")
    parser.add_argument("--rewrite-mode", choices=["auto", "on", "off"], default="auto",
                        help="auto: rewrite once per query, then retrieve with skip_rewrite")
    parser.add_argument("--no-dense", action="store_true", help="BM25 only (also disables the reranker)")
    parser.add_argument("--no-prompt-build", action="store_true", help="skip the prompt-build stage")
    parser.add_argument("--skip-warmup", action="store_true", help="do not run the boot warm-up first")
    parser.add_argument("--with-llm", action="store_true", help="also measure the LLM answer (costs tokens)")
    parser.add_argument("--llm-queries", type=int, default=2, help="queries used for the LLM phase")
    parser.add_argument("--llm-reps", type=int, default=1, help="repetitions for the LLM phase")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="if >1: run that many retrievals at once plus an event-loop lag probe")
    parser.add_argument("--reports-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    parser.add_argument("--json", action="store_true", help="print the JSON summary instead of tables")
    parser.add_argument("--verbose", action="store_true", help="keep the application's own INFO logs")
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not verbose:
        for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "openai", "langchain", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.ERROR)


async def main_async(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    rec = StageRecorder()
    rec.install()

    import app.services.config_service as config_module

    # Import order matters: `app/services/chat_service.py` imports
    # `get_knowledge_base_service` from `knowledge_base` at module level, so
    # importing `knowledge_base` first raises a circular-import error. The app
    # never notices because the routes import `chat_service` first. Load it here
    # so this script can drive both services.
    import app.services.chat_service  # noqa: F401  (imported for its side effects)
    from app.services.knowledge_base import KnowledgeBaseService

    kb = KnowledgeBaseService()
    offline = OfflineConfigService(args.config)
    kb.config_service = offline  # the benchmark never talks to MongoDB

    async def _get_config_service():
        return offline

    config_module.get_config_service = _get_config_service  # used by `chat_service.get_llm`
    try:
        import app.services.knowledge_base as kb_module

        kb_module.get_config_service = _get_config_service  # used by `_web_search_enabled`
    except Exception:  # pragma: no cover
        pass

    queries = load_queries(args.dataset, args.queries)
    if not queries:
        safe_print(f"[bench] no queries found in {args.dataset}")
        return 1

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": PROJECT_ROOT,
        "dataset": args.dataset,
        "config_source": offline.source,
        "rag_settings": {
            "top_k_results": offline.rag_settings.top_k_results,
            "knowledge_base_confidence_threshold": offline.rag_settings.knowledge_base_confidence_threshold,
            "temperature": offline.rag_settings.temperature,
        },
        "args": vars(args),
        "queries": [
            {
                "id": q["id"],
                "feedback": q["feedback"],
                "category": q["category"],
                "preview": q["question"][:120],
            }
            for q in queries
        ],
    }

    safe_print(
        f"[bench] queries: {len(queries)} (from {os.path.basename(args.dataset)}), "
        f"reps={args.reps}, k={args.k}, rewrite={args.rewrite_mode}, "
        f"dense={'off' if args.no_dense else 'on'}, config={offline.source}"
    )

    phases: List[Phase] = []
    if not args.skip_warmup:
        boot = Phase("A0_boot_warmup (lifespan)")
        boot.add(await capture(rec, kb.warm_up()), label="warm_up")
        phases.append(boot)

    hrs = await kb._get_hybrid_service_async()
    if args.no_dense:
        # Timing-only switch, applied to the instance under test (the app keeps its
        # dense branch untouched): the dense search and the reranker are replaced by
        # no-ops, so phases B/D measure the local BM25 path in isolation. The vector
        # store is KEPT - it is also the source of the documents BM25 indexes.
        async def _no_dense(*_args, **_kwargs):
            return []

        async def _no_rerank(_query, doc_score_pairs, top_k):
            return [doc for doc, _ in (doc_score_pairs or [])[:top_k]]

        hrs._dense_parallel = _no_dense          # type: ignore[assignment]
        hrs._rerank_async = _no_rerank           # type: ignore[assignment]

    payload["corpus"] = await collect_corpus_stats(kb, hrs, rec)
    safe_print(f"[bench] collection chunks: {payload['corpus'].get('total_chunks')}")
    for name, branch in (payload["corpus"].get("branches") or {}).items():
        if "error" in branch:
            safe_print(f"    {name}: {branch['error']}")
        else:
            safe_print(
                f"    {name}: {branch['chunks']} chunks, fetch {fmt(branch['fetch_seconds'])}s, "
                f"~{fmt(branch['avg_chunk_chars'], 0)} chars / ~{fmt(branch['avg_chunk_tokens'], 0)} tokens"
            )

    phases.append(await run_bm25_phase(hrs, queries, args.reps, args.k, rec))
    if not args.no_dense:
        phases.append(await run_hybrid_phase(hrs, queries, args.reps, rec))

    phase_d_list, rewrites = await run_retrieval_phase(
        kb, queries, args.reps, rec, args.rewrite_mode, with_prompt=not args.no_prompt_build
    )
    phases.extend(phase_d_list)
    phase_d = phase_d_list[0]
    payload["rewrites"] = [
        {
            "id": item["id"],
            "ok": item["ok"],
            "seconds": item["elapsed"],
            "rewritten_preview": (item.get("rewritten_query") or "")[:120],
            "error": item.get("error"),
        }
        for item in rewrites
    ]

    phase_e = None
    if args.with_llm:
        llm_queries = queries[: max(1, args.llm_queries)]
        phase_e = await run_llm_phase(kb, llm_queries, args.llm_reps, rec)
        phases.append(phase_e)

    contention = None
    if args.concurrency and args.concurrency > 1:
        contention = await run_contention_phase(kb, queries, args.concurrency, rec)
        payload["contention"] = contention

    payload["phases"] = [phase.summary() for phase in phases]
    phase_r = next((p for p in phase_d_list if p.name.startswith("D1_")), None)
    payload["decomposition"] = print_decomposition(phase_d, phase_e, phase_r)

    if not args.json:
        for phase in phases:
            print_phase_table(phase)
        if contention:
            safe_print()
            safe_print(f"--- F_contention (concurrency={contention['concurrency']}) ---")
            safe_print(
                f"  whole wave: {fmt(contention['wave_seconds'])}s, "
                f"per-request median={fmt(contention['request_median'])}s "
                f"p95={fmt(contention['request_p95'])}s max={fmt(contention['request_max'])}s"
            )
            safe_print(
                f"  event-loop lag: median={fmt(contention['event_loop_lag_median_ms'], 1)}ms "
                f"p95={fmt(contention['event_loop_lag_p95_ms'], 1)}ms "
                f"max={fmt(contention['event_loop_lag_max_ms'], 1)}ms"
            )

    out_dir = os.path.join(
        args.reports_dir, f"bench_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    write_reports(out_dir, payload)

    if args.json:
        safe_print(json.dumps(payload, ensure_ascii=False, indent=1))

    if not [phase for phase in phases if phase.runs]:
        safe_print("[bench] no phase produced a measurement")
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
