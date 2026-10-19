"""重试与退避。

设计约束：
- sleep 可注入，单测里不真的等
- 只重试显式声明 retryable=True 的异常（见 core.errors.is_retryable）
- 指数退避 + 可选抖动，避免下游同时被打
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Sequence, TypeVar

from core.errors import is_retryable
from core.observability import log

T = TypeVar("T")


class RetryExhausted(Exception):
    def __init__(self, attempts: int, last: BaseException) -> None:
        super().__init__(f"重试 {attempts} 次仍失败: {last}")
        self.attempts = attempts
        self.last = last


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    retries: int = 2,
    base_delay: float = 0.4,
    max_delay: float = 4.0,
    jitter: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    retry_on: Sequence[type[BaseException]] | None = None,
    op_name: str = "call",
    **kwargs: Any,
) -> T:
    """执行 fn，失败且可重试时按指数退避重试。

    retries=2 表示最多执行 3 次。
    """
    attempts = retries + 1
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - 需要按契约判断
            last_exc = exc
            allowed = retry_on is not None and isinstance(exc, tuple(retry_on))
            if not (allowed or is_retryable(exc)) or attempt == attempts:
                raise

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if jitter:
                delay += random.uniform(0, jitter)
            log(
                "retry",
                level="WARNING",
                op=op_name,
                attempt=attempt,
                max_attempts=attempts,
                delay_ms=round(delay * 1000),
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            sleep(delay)

    # 理论上不可达：循环内要么 return 要么 raise
    assert last_exc is not None
    raise RetryExhausted(attempts, last_exc)


def with_retry(**options: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """装饰器版本，参数透传给 call_with_retry。"""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(fn, *args, op_name=fn.__name__, **options, **kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator
