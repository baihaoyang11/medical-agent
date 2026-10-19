"""检索质量判定。

为什么不用"相似度分数阈值"：
稠密路是 IP 距离、BM25 路是词频分数，RRF 只保留排名不保留可比尺度，
所以拿 top_score 跟一个绝对阈值比是没有意义的（这也正是原实现只看
字符数的原因，只是那个信号太弱）。

这里改用三个与尺度无关的信号：
1. 上下文长度是否达到下限（防"什么都没召回"）
2. query 的字符二元组覆盖率（防"召回了一堆不相关但很长的段落"）
3. 有效文档条数

三者任一不满足即判定需要二次检索。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

DEFAULT_MIN_CHARS = 120
DEFAULT_MIN_COVERAGE = 0.30
DEFAULT_MIN_DOCS = 1


def _bigrams(text: str) -> set[str]:
    """字符二元组，不依赖分词器，对短 query 也稳定。"""
    cleaned = "".join(ch for ch in text if ch.isalnum())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def keyword_coverage(query: str, context: str) -> float:
    """query 的二元组有多大比例出现在 context 里。"""
    query_grams = _bigrams(query)
    if not query_grams:
        return 0.0
    context_grams = _bigrams(context)
    if not context_grams:
        return 0.0
    return len(query_grams & context_grams) / len(query_grams)


@dataclass(frozen=True)
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    context_chars: int = 0
    doc_count: int = 0
    coverage: float = 0.0

    def to_trace(self) -> dict[str, Any]:
        return {
            "quality_passed": self.passed,
            "quality_reasons": self.reasons,
            "context_chars": self.context_chars,
            "doc_count": self.doc_count,
            "coverage": round(self.coverage, 3),
        }


def assess(
    query: str,
    docs: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    context: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_docs: int = DEFAULT_MIN_DOCS,
) -> QualityReport:
    doc_list = list(docs or [])
    chars = len((context or "").strip())
    coverage = keyword_coverage(query, context or "")

    reasons: list[str] = []
    if len(doc_list) < min_docs:
        reasons.append(f"召回文档不足({len(doc_list)}<{min_docs})")
    if chars < min_chars:
        reasons.append(f"上下文过短({chars}<{min_chars})")
    if coverage < min_coverage:
        reasons.append(f"关键词覆盖不足({coverage:.0%}<{min_coverage:.0%})")

    return QualityReport(
        passed=not reasons,
        reasons=reasons,
        context_chars=chars,
        doc_count=len(doc_list),
        coverage=coverage,
    )
