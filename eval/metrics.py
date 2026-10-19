"""检索指标。

为什么是这几个
--------------
- **recall@k**：金标文档有没有进前 k。检索层的核心指标，和"上下文里有没有答案"直接对应。
- **MRR**：第一个正确结果排多靠前。recall 看不出"排第1还是排第5"的区别，MRR 能。
- **nDCG@k**：对位置打折。比 recall 更严，因为排第5的正确文档对生成的价值低于排第1。

三个一起看的原因：只看 recall 容易自欺 —— 把 20 条全召回，recall=1.0，
但正确的那条排第19，上下文预算早截掉了。
"""

import math
from typing import Iterable, Sequence


def _hit_ranks(ranked: Sequence[str], relevant: Iterable[str]) -> list[int]:
    """返回每个相关文档的名次（从1开始），未召回的记 0。"""
    rel = set(relevant)
    return [i + 1 if doc_id in rel else 0 for i, doc_id in enumerate(ranked)]


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    top = set(ranked[:k])
    return len(top & rel) / len(rel)


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 k 条里是否至少命中一个。对"用户能不能看到答案"最直观。"""
    rel = set(relevant)
    if not rel:
        return float("nan")
    return 1.0 if set(ranked[:k]) & rel else 0.0


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """Mean Reciprocal Rank 的单条版本。"""
    rel = set(relevant)
    if not rel:
        return float("nan")
    for i, doc_id in enumerate(ranked):
        if doc_id in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """二值相关性的 nDCG@k。"""
    rel = set(relevant)
    if not rel:
        return float("nan")

    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in rel:
            dcg += 1.0 / math.log2(i + 2)

    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else float("nan")


def mean(values: Iterable[float]) -> float:
    """忽略 NaN 的均值 —— 急症 query 没有金标，不参与检索指标。"""
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")
