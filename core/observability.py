"""可观测性基础设施：trace_id、结构化日志、耗时埋点。

不引入第三方依赖（不依赖 structlog / loguru），只用标准库：
- contextvars 传递 trace_id，async / thread 都能拿到
- JSON Lines 输出，方便直接进 ELK / Loki
- span() 上下文管理器记录 duration_ms 与异常

用法：
    with bind_trace(new_trace_id()):
        with span("retrieve"):
            ...
        log("answer_ready", chars=len(answer))
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_span_stack: ContextVar[tuple[str, ...]] = ContextVar("span_stack", default=())

_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 走标准 logging，方便被 uvicorn / 采集器接管；handler 只挂一次
_logger = logging.getLogger("medical_agent")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(_LEVEL)
    _logger.propagate = False


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def current_trace_id() -> str:
    return _trace_id.get()


@contextmanager
def bind_trace(trace_id: str | None = None) -> Iterator[str]:
    """绑定 trace_id 到当前上下文，退出时还原（支持嵌套）。"""
    tid = trace_id or new_trace_id()
    token = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


def current_span() -> str:
    stack = _span_stack.get()
    return ".".join(stack)


@contextmanager
def span(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """给一段逻辑计时，结束时打一条日志；异常也会记录。"""
    stack = _span_stack.get()
    token = _span_stack.set(stack + (name,))
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    try:
        yield payload
        elapsed = (time.perf_counter() - started) * 1000
        log("span_ok", span=name, dur_ms=round(elapsed, 2), **{**fields, **payload})
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        log(
            "span_error",
            level="ERROR",
            span=name,
            dur_ms=round(elapsed, 2),
            error=type(exc).__name__,
            message=str(exc)[:300],
            **fields,
        )
        raise
    finally:
        _span_stack.reset(token)


def log(event: str, level: str = "INFO", **fields: Any) -> None:
    """输出一行 JSON 日志。字段名统一用 snake_case。"""
    record: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "level": level,
        "event": event,
        "trace_id": current_trace_id(),
    }
    span_name = current_span()
    if span_name:
        record["span"] = span_name
    record.update(fields)

    line = json.dumps(record, ensure_ascii=False, default=str)
    if level == "ERROR":
        _logger.error(line)
    elif level == "WARNING":
        _logger.warning(line)
    elif level == "DEBUG":
        _logger.debug(line)
    else:
        _logger.info(line)
