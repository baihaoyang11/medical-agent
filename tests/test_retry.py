"""重试策略测试。

用注入的 sleep 保证测试不真的等待。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.errors import UpstreamError, ValidationError, is_retryable  # noqa: E402
from core.retry import call_with_retry  # noqa: E402


def test_success_first_try():
    calls = []
    result = call_with_retry(lambda: calls.append(1) or "ok", sleep=lambda d: None)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_retryable_error_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise UpstreamError("连接抖动")
        return "ok"

    sleeps: list[float] = []
    assert call_with_retry(flaky, retries=2, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_backoff_is_increasing():
    calls = []

    def always_fail():
        calls.append(1)
        raise UpstreamError("失败")

    sleeps: list[float] = []
    with pytest.raises(UpstreamError):
        call_with_retry(always_fail, retries=2, base_delay=0.1, jitter=0, sleep=sleeps.append)

    assert sleeps == [0.1, 0.2]


def test_does_not_retry_validation_error():
    """输入错误重试三次也还是错，只会把 400 拖成超时。"""
    calls = []

    def bad():
        calls.append(1)
        raise ValidationError("问题为空")

    with pytest.raises(ValidationError):
        call_with_retry(bad, retries=3, sleep=lambda d: None)

    assert len(calls) == 1


def test_max_delay_is_capped():
    calls = []

    def always_fail():
        calls.append(1)
        raise UpstreamError("失败")

    sleeps: list[float] = []
    with pytest.raises(UpstreamError):
        call_with_retry(
            always_fail,
            retries=4,
            base_delay=1.0,
            max_delay=2.0,
            jitter=0,
            sleep=sleeps.append,
        )

    assert sleeps == [1.0, 2.0, 2.0, 2.0]


def test_can_force_retry_on_specific_types():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("临时错误")
        return 42

    assert call_with_retry(flaky, retries=1, sleep=lambda d: None, retry_on=(ValueError,)) == 42
    assert len(calls) == 2


def test_is_retryable_contract():
    assert is_retryable(UpstreamError("x")) is True
    assert is_retryable(ValidationError("x")) is False
    assert is_retryable(RuntimeError("x")) is False
