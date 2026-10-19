"""结果重排 — 可插拔、可降级。

为什么需要这一层：

    RRF 只用了「排名」信息，两个检索器给分尺度完全不同（IP 距离 0~1，
    BM25 分数可能几十），直接比分数没意义。融合后拿到的是一份"大致正确"
    的候选列表，但用户真正的意图词可能只命中列表第 7 条。

    Cross-Encoder 把 (query, doc) 拼起来过一遍模型，能拿到细粒度相关性，
    但代价是每对一次前向，只能对少量候选做。所以标准链路是
    「召回 20~30 → 重排 → 取前 3~5」。

设计取舍：

    1. 重排器是策略对象，不是硬编码调用。默认 keyword（零依赖、离线可跑），
       装了 cross-encoder 就换 cross_encoder，没装自动退回 keyword。
    2. 降级对上层透明：FallbackReranker 捕获异常、记一次日志、标记 degraded，
       调用方永远拿到一个排好序的列表。
    3. 不做"重排后过滤"——阈值是不可移植的，不同模型分段不同，
       卡阈值等于把模型版本绑死到业务代码里。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from core.observability import log
from core.quality import keyword_coverage

DEFAULT_WEIGHT = 0.6


@runtime_checkable
class Reranker(Protocol):
    """重排器协议。"""

    name: str
    degraded: bool

    def rerank(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
        *,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]: ...


# ────────────────────────────────────────────────────────────────
# 1. 空实现：保持原顺序，用于消融对比 / 关闭重排
# ────────────────────────────────────────────────────────────────
class NoOpReranker:
    name = "noop"
    degraded = False

    def rerank(self, query, docs, *, top_n=None):
        out = [dict(d) for d in docs]
        return out[:top_n] if top_n is not None else out


# ────────────────────────────────────────────────────────────────
# 2. 关键词重排：零依赖兜底
# ────────────────────────────────────────────────────────────────
class KeywordReranker:
    """按 query 与正文的字符二元组重合度重排。

    只用词面重合当然比不上语义模型，但它有两个工程价值：
    - 离线可跑：CI 里没有 GPU、没有 2GB 模型也能验证链路通
    - 可解释：分数就是覆盖率，用户问"为什么这条排前面"能答上来

    最终分数 = weight * 覆盖率 + (1 - weight) * 原排名分。
    保留原排名分是为了不把融合层的信号全丢掉——纯覆盖率排序会让
    长文档天然占优（词多，碰上的概率大）。
    """

    name = "keyword"
    degraded = False

    def __init__(self, weight: float = DEFAULT_WEIGHT) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight 必须在 0~1 之间")
        self.weight = weight

    def rerank(self, query, docs, *, top_n=None):
        docs = list(docs)
        if not docs:
            return []

        total = len(docs)
        scored = []
        for idx, doc in enumerate(docs):
            coverage = keyword_coverage(query, str(doc.get("content") or ""))
            rank_score = 1.0 - idx / total  # 原顺序越靠前越接近 1
            score = self.weight * coverage + (1.0 - self.weight) * rank_score
            enriched = dict(doc)
            enriched["rerank_score"] = round(score, 4)
            scored.append((score, idx, enriched))

        # 同分时按原顺序，保证排序稳定（否则去重/引用编号会抖）
        scored.sort(key=lambda item: (-item[0], item[1]))
        out = [item[2] for item in scored]
        return out[:top_n] if top_n is not None else out


# ────────────────────────────────────────────────────────────────
# 3. Cross-Encoder 重排：需要模型文件
# ────────────────────────────────────────────────────────────────
class RerankUnavailable(RuntimeError):
    """模型或依赖不可用。由 FallbackReranker 捕获，不让它冒到业务层。"""


class CrossEncoderReranker:
    """用 cross-encoder 对 (query, doc) 打相关性分。

    模型懒加载：进程启动时不碰磁盘，第一次真正需要重排才加载。
    这样即使模型路径配错，服务也能起来（只是退化成关键词重排），
    而不是在 import 阶段就挂掉——这是旧版 reranker.py 的问题。
    """

    name = "cross_encoder"
    degraded = False

    def __init__(
        self,
        model_path: Optional[str],
        *,
        batch_size: int = 8,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path:
                raise RerankUnavailable("未配置 RERANK_MODEL_PATH")
            try:
                from sentence_transformers import CrossEncoder  # 重型可选依赖
            except ImportError as exc:  # pragma: no cover - 依赖缺失分支
                raise RerankUnavailable(
                    "sentence-transformers 未安装，重排降级为 keyword"
                ) from exc
            try:
                self._model = CrossEncoder(
                    self.model_path, device=self.device, max_length=self.max_length
                )
            except Exception as exc:  # noqa: BLE001 - 模型损坏/路径错误
                raise RerankUnavailable(f"加载重排模型失败: {exc}") from exc

    def rerank(self, query, docs, *, top_n=None):
        docs = list(docs)
        if not docs:
            return []
        self._ensure_model()

        pairs = [(query, str(d.get("content") or "")) for d in docs]
        scores = self._model.predict(  # type: ignore[union-attr]
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        scored = []
        for idx, (doc, raw) in enumerate(zip(docs, scores)):
            enriched = dict(doc)
            enriched["rerank_score"] = round(float(raw), 4)
            scored.append((float(raw), idx, enriched))
        scored.sort(key=lambda item: (-item[0], item[1]))
        out = [item[2] for item in scored]
        return out[:top_n] if top_n is not None else out


# ────────────────────────────────────────────────────────────────
# 4. 降级包装：主路失败自动走备路
# ────────────────────────────────────────────────────────────────
class FallbackReranker:
    """主重排器抛错时切到备选，并把自己标记为 degraded。"""

    def __init__(self, primary: Reranker, fallback: Optional[Reranker] = None) -> None:
        self.primary = primary
        self.fallback = fallback or KeywordReranker()
        self.name = primary.name
        self.degraded = False
        self._warned = False

    def rerank(self, query, docs, *, top_n=None):
        if self.degraded:
            return self.fallback.rerank(query, docs, top_n=top_n)

        try:
            return self.primary.rerank(query, docs, top_n=top_n)
        except Exception as exc:  # noqa: BLE001 - 降级点，任何异常都接受
            self.degraded = True
            if not self._warned:
                self._warned = True
                log(
                    "rerank.fallback",
                    level="WARNING",
                    strategy=self.primary.name,
                    fallback=self.fallback.name,
                    error=str(exc),
                )
            return self.fallback.rerank(query, docs, top_n=top_n)


# ────────────────────────────────────────────────────────────────
# 5. 工厂
# ────────────────────────────────────────────────────────────────
STRATEGIES = ("noop", "keyword", "cross_encoder")


def build_reranker(
    strategy: str = "keyword",
    *,
    model_path: Optional[str] = None,
    weight: float = DEFAULT_WEIGHT,
    device: str = "cpu",
) -> Reranker:
    """按配置构造重排器；未知策略或模型缺失时不报错，降级即可。"""
    strategy = (strategy or "keyword").strip().lower()

    if strategy == "noop":
        return NoOpReranker()
    if strategy == "keyword":
        return KeywordReranker(weight=weight)
    if strategy == "cross_encoder":
        return FallbackReranker(
            CrossEncoderReranker(model_path, device=device),
            KeywordReranker(weight=weight),
        )

    log("rerank.unknown_strategy", level="WARNING", strategy=strategy)
    return KeywordReranker(weight=weight)
