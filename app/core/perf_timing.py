"""Stage-level latency instrumentation for the chat / retrieval path.

Why this module exists
----------------------
The reported chat latency (38-158 s) had no per-stage numbers behind it, so it was
impossible to tell whether BM25, the Chroma vector search or the LLM round trip
dominated a slow answer. This module gives every stage of the request pipeline one
uniform way to be timed with `time.perf_counter()` and logged at INFO with a
`stage=` key, so a single log line answers "where did the seconds go?".

Usage
-----
    from app.core.perf_timing import timed_stage

    with timed_stage("bm25_index_build", branch=key, docs=len(docs)) as t:
        retriever = BM25Retriever.from_documents(docs, preprocess_func=_tokenize)
        t["terms"] = 12345                                   # extra field, logged too

    @timed_async_stage("hybrid_retrieve")
    async def hybrid_retrieve(...): ...

    @timed_async_generator_stage("chat_request_total", mode="stream")
    async def process_message_stream(...): ...

Every call site is emitted as one line:

    [PERF_STAGE] stage=bm25_index_build elapsed=3.412s branch=contrib docs=8123 ok=True

Removal / switch
----------------
The whole instrumentation is behind ONE environment flag:

    PERF_TIMING_LOG=true   (default)  -> stages are timed and logged
    PERF_TIMING_LOG=false            -> `timed_stage` yields an empty dict, nothing is
                                        timed, nothing is logged (one branch check per
                                        stage; no `perf_counter` call, no log record)

`set_perf_timing_enabled()` exists for tests and for embedders that want to turn the
timing off at runtime. Deleting the instrumentation is equally mechanical: drop the
`with timed_stage(...)` / decorators, and this module becomes unused.

Log destination
---------------
`logging.info(...)` from `app/` never reaches stdout in production because
`app/core/logging.py::setup_logging()` is not called (the root logger has no INFO
handler, so only library warnings show up). This module therefore attaches its own
`StreamHandler` to its logger *when it is enabled*, so `[PERF_STAGE]` lines appear in
the pod logs without touching the global logging configuration, and appear exactly
once (the logger does not propagate). With `PERF_TIMING_LOG=false` no handler and no
records are created.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, Iterator

logger = logging.getLogger("app.core.perf_timing")

#: Environment flag that switches the whole instrumentation on/off.
PERF_TIMING_ENV_FLAG = "PERF_TIMING_LOG"

#: Every emitted line starts with this prefix; grep for it to extract the timings.
PERF_TIMING_PREFIX = "[PERF_STAGE]"

_TRUTHY = {"1", "true", "yes", "on"}

_enabled: bool = True
_handler_installed = False


def _read_env_flag(default: bool = True) -> bool:
    """Read `PERF_TIMING_ENV_FLAG`; unset (or unparsable) means `default`."""
    raw = os.getenv(PERF_TIMING_ENV_FLAG)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _install_handler() -> None:
    """Give this module's logger its own INFO handler (once per process).

    Deliberately self-contained: `propagate = False`, so a stage is logged exactly
    once whether or not the application (or uvicorn/gunicorn) configured logging.
    """
    global _handler_installed
    if _handler_installed:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _handler_installed = True


def perf_timing_enabled() -> bool:
    """Whether stages are currently timed and logged."""
    return _enabled


def set_perf_timing_enabled(enabled: bool) -> None:
    """Switch the instrumentation on/off at runtime (tests, embedders)."""
    global _enabled
    _enabled = bool(enabled)
    if _enabled:
        _install_handler()


def reload_perf_timing_flag() -> bool:
    """Re-read `PERF_TIMING_LOG` (useful when the env changes after import)."""
    set_perf_timing_enabled(_read_env_flag())
    return _enabled


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, str):
        return value if len(value) <= 60 else value[:57] + "..."
    return str(value)


def log_stage(
    stage: str,
    elapsed: float,
    fields: Dict[str, Any] | None = None,
    *,
    ok: bool = True,
    error: str | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit one `[PERF_STAGE]` line (no-op while the instrumentation is off)."""
    if not _enabled:
        return
    _install_handler()
    parts = [f"stage={stage}", f"elapsed={elapsed:.3f}s"]
    for key in sorted(fields or {}):
        value = (fields or {})[key]
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    parts.append(f"ok={ok}")
    if error:
        parts.append(f"error={_format_value(error)}")
    logger.log(level, f"{PERF_TIMING_PREFIX} " + " ".join(parts))


@contextlib.contextmanager
def timed_stage(
    stage: str, *, level: int = logging.INFO, **fields: Any
) -> Iterator[Dict[str, Any]]:
    """Time the enclosed block and log it as one `[PERF_STAGE]` line.

    Yields a mutable dict: any key assigned inside the block is added to the log
    line, which is how call sites report counts (`docs=`, `chunks=`, ...) that are
    only known after the work ran. Exceptions are re-raised unchanged (the stage is
    still logged, with `ok=False` and the exception text).
    """
    if not _enabled:
        # Same call sites, zero work: no clock read, no log record.
        yield {}
        return

    extra: Dict[str, Any] = dict(fields)
    start = time.perf_counter()
    ok = True
    error: str | None = None
    try:
        yield extra
    except BaseException as exc:  # noqa: BLE001 - measured, not handled, re-raised
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log_stage(
            stage, time.perf_counter() - start, extra, ok=ok, error=error, level=level
        )


def timed_sync_stage(stage: str, *, level: int = logging.INFO, **fields: Any) -> Callable:
    """Decorator for a synchronous callable (see `timed_stage`)."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with timed_stage(stage, level=level, **fields):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timed_async_stage(stage: str, *, level: int = logging.INFO, **fields: Any) -> Callable:
    """Decorator for an async coroutine function (see `timed_stage`)."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            with timed_stage(stage, level=level, **fields):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def timed_async_generator_stage(
    stage: str, *, level: int = logging.INFO, **fields: Any
) -> Callable:
    """Decorator for an async generator: times it until exhausted or closed.

    The stage is logged from a `finally`, so a consumer that abandons a stream
    (client disconnect) still produces a (partial) measurement instead of a leak.
    """

    def decorator(func: Callable) -> Callable:
        if not inspect.isasyncgenfunction(func):  # catch a wrong decorator early
            raise TypeError(
                f"{stage}: timed_async_generator_stage expects an async generator, "
                f"got {type(func).__name__}"
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any):
            with timed_stage(stage, level=level, **fields):
                async for item in func(*args, **kwargs):
                    yield item

        return wrapper

    return decorator


# Honour the flag at import time as well, so the very first stages (app boot and
# the retrieval warm-up) are already measured.
_enabled = _read_env_flag()
if _enabled:
    _install_handler()
