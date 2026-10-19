"""统一异常体系。

设计要点：
1. 每个异常都带 code / http_status / retryable，让上层不用猜要不要重试。
2. retryable 是"契约"，由抛出处声明，而不是让调用方用 isinstance 猜。
3. 不继承 HTTPException，保持 core 层与 Web 框架解耦。
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """所有业务异常的基类。"""

    code: str = "internal_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ValidationError(AppError):
    """输入不合法，重试没有意义。"""

    code = "invalid_input"
    http_status = 400


class UpstreamError(AppError):
    """下游依赖（LLM / Embedding / Milvus）出错，通常可重试。"""

    code = "upstream_error"
    http_status = 503
    retryable = True


class UpstreamTimeout(UpstreamError):
    code = "upstream_timeout"
    http_status = 504


class RateLimited(UpstreamError):
    code = "rate_limited"
    http_status = 429


class RetrievalError(UpstreamError):
    """检索链路失败。检索挂了可以降级为无上下文生成，所以可重试。"""

    code = "retrieval_failed"


class GenerationError(UpstreamError):
    """答案生成失败。"""

    code = "generation_failed"


class ConfigurationError(AppError):
    """配置缺失，启动期就该炸。"""

    code = "configuration_error"
    http_status = 500


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    只有显式声明 retryable=True 的 AppError 才重试；
    其余（包括编程错误、校验失败）一律不重试，避免把 bug 重试成超时。
    """
    return bool(getattr(exc, "retryable", False))
