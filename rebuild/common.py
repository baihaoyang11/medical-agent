"""七步重建的公共部分：语料、评测、打印。

只做三件事：加载数据、算 recall、把结果排成表。
**检索逻辑一律不放这里** —— 每一步都自己实现，才看得出差异在哪。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

# 让 `python rebuild/step1_literal.py` 也能跑（不只是 -m）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.filler import FILLERS  # noqa: E402
from eval.knowledge import CONFUSABLE_DOCS, POINTS, build_queries  # noqa: E402

K = 5


def load() -> tuple[List[dict], List[dict]]:
    """返回 (文档, 查询)。

    文档 136 段，分三类：
    - 48 段核心知识点：每个 query 的正确答案都在这批里
    - 20 段易混干扰：格列美脲/格列齐特、空腹/餐后/糖化这种只差一点的表述
    - 68 段无关干扰：别的科室的内容（眼科、口腔、皮肤…）

    **干扰是必须的。** 只有 48 段的话，取前 5 条就覆盖 10%，
    闭着眼睛都能蒙中。真实场景里你要在一大堆"看起来相关"的段落里找，
    加干扰之后难度才对得上。

    查询 96 条 = 48 个书面问法 + 48 个口语问法，每条都标了正确答案是哪段。
    """
    docs = [
        {"key": p.key, "text": p.content, "topic": p.topic, "facet": p.facet}
        for p in POINTS
    ]
    docs += [
        {"key": cid, "text": content, "topic": topic, "facet": "confusable"}
        for cid, content, topic in CONFUSABLE_DOCS
    ]
    docs += [
        {"key": f.key, "text": f.content, "topic": "filler", "facet": "filler"}
        for f in FILLERS
    ]
    queries = [
        {"query": q, "gold": key, "style": style, "qtype": qtype}
        for q, key, qtype, style in build_queries()
    ]
    return docs, queries


def recall_at(ranked: Sequence[str], gold: str, k: int = K) -> int:
    """正确答案有没有进前 k。返回 1 或 0。

    这就是整个评测的全部数学。没有更复杂的东西 —— 如果你只记一个指标，
    记这个：用户能不能在前 k 条里看到正确的那段。
    """
    return 1 if gold in list(ranked)[:k] else 0


def first_rank(ranked: Sequence[str], gold: str) -> int:
    """正确答案排第几。没进列表就算 0。

    recall 看不出"排第 1 还是第 5"。但只要上下文预算有限，
    排第 5 的那条很可能在拼上下文时就被截掉了 —— 所以还要看排名。
    """
    for i, key in enumerate(ranked, 1):
        if key == gold:
            return i
    return 0


def evaluate(search: Callable[[str, int], List[str]], queries: List[dict], name: str = "") -> dict:
    """跑一遍，返回总体和分问法的 recall。

    `search(query, k)` 是每一步自己要实现的检索函数。
    """
    t0 = time.time()
    hits = {"all": 0}
    total = {"all": 0}
    rr = 0.0
    for q in queries:
        ranked = search(q["query"], K)
        h = recall_at(ranked, q["gold"])
        hits["all"] += h
        total["all"] += 1
        r = first_rank(ranked, q["gold"])
        if r:
            rr += 1.0 / r
        style = q["style"]
        hits[style] = hits.get(style, 0) + h
        total[style] = total.get(style, 0) + 1

    return {
        "name": name,
        "recall": {k: hits[k] / total[k] for k in hits if total[k]},
        "mrr": rr / total["all"],
        "n": total["all"],
        "elapsed": round(time.time() - t0, 2),
    }


def bar(x: float, width: int = 20) -> str:
    filled = int(round(x * width))
    return "█" * filled + "·" * (width - filled)


def print_result(r: dict, prev: dict | None = None) -> None:
    """打印一行结果。有上一步就同时打出变化量。"""
    line = (
        f"{r['name']:<28} R@5 = {r['recall']['all']:.3f}  "
        f"MRR = {r['mrr']:.3f}  {bar(r['recall']['all'])}"
    )
    if prev is not None:
        line += f"  ({r['recall']['all'] - prev['recall']['all']:+.3f})"
    print(line)


def print_detail(r: dict) -> None:
    parts = [f"{s} {r['recall'][s]:.3f}" for s in ("formal", "colloquial") if s in r["recall"]]
    print(f"{'':28} " + "  |  ".join(parts))


def header(title: str, subtitle: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print(f"  {subtitle}")
    print("=" * 78)
