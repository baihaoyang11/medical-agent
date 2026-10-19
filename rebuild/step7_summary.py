"""第 7 步：把前面六步放在同一张表上。

单独看每一步，都能得到"这一步有用"的错觉。放一起才知道：
哪几层真的在起作用，哪几层是安慰剂。

跑法：
    python -m rebuild.step7_summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import jieba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedders import BgeEmbedder  # noqa: E402
from rebuild.common import K, evaluate, header, load  # noqa: E402
from rebuild.step1_literal import grams  # noqa: E402
from rebuild.step3_hybrid import BM25, rrf  # noqa: E402
from rebuild.step4_rerank import keyword_rerank  # noqa: E402

jieba.setLogLevel(20)
DOCS, QUERIES = load()
KEYS = [d["key"] for d in DOCS]


def main() -> None:
    header("第 7 步：汇总", "六层放在一起，看哪层真的在起作用")

    emb = BgeEmbedder()
    doc_mat = emb.encode([d["text"] for d in DOCS])
    qvec = {q["query"]: v for q, v in zip(QUERIES, emb.encode([q["query"] for q in QUERIES]))}
    bm25 = BM25(DOCS)

    def literal(query: str, k: int = K) -> list[str]:
        qg = grams(query)
        if not qg:
            return []
        scored = [
            (len(qg & grams(d["text"])) / len(qg), d["key"]) for d in DOCS
        ]
        scored.sort(reverse=True)
        return [key for _, key in scored[:k]]

    def dense(query: str, k: int = K) -> list[str]:
        sims = doc_mat @ qvec[query]
        return [KEYS[i] for i in np.argsort(-sims)[:k]]

    def sparse(query: str, k: int = K) -> list[str]:
        s = bm25.scores(query)
        return [KEYS[i] for i in np.argsort(-s)[:k] if s[i] > 0]

    POOL = K * 3

    def hybrid(w: float):
        def fn(query: str, k: int = K) -> list[str]:
            return rrf(dense(query, POOL), sparse(query, POOL), w_a=w, w_b=1 - w)[:k]
        return fn

    def rerank(query: str, k: int = K) -> list[str]:
        return keyword_rerank(query, dense(query, POOL), w_orig=0.5)[:k]

    rows = [
        ("1. 字面重叠", literal, "只数字面重合，不懂语义"),
        ("2. 纯向量", dense, "能理解近义表达 —— 最大的一次跃迁"),
        ("3a. 混合（等权）", hybrid(0.5), "加了 BM25，反而拖累"),
        ("3b. 混合（偏向量 0.8）", hybrid(0.8), "调权重后追平，仍不如纯向量"),
        ("4. 向量 + 重排", rerank, "排序微调，基本持平"),
    ]

    results = []
    for name, fn, _note in rows:
        results.append((name, evaluate(fn, QUERIES, name)))

    print()
    print(f"  {'配置':<24}{'R@5':>8}{'MRR':>8}{' vs 纯向量':>12}")
    print("  " + "-" * 54)
    base = results[1][1]
    for name, r in results:
        d = r["recall"]["all"] - base["recall"]["all"]
        mark = "" if name.startswith("2.") else f"{d:+.3f}"
        print(f"  {name:<24}{r['recall']['all']:>8.3f}{r['mrr']:>8.3f}{mark:>12}")

    print()
    print("  一句话总结这张表：")
    print("  **从「数字面」到「懂语义」是唯一一次真正的跃迁（+0.23）。**")
    print("  后面的混合、调权重、重排，在这套数据上全都没有超过纯向量。")
    print()

    print("  但这不是说那几层没意义。它们各自的结论是：")
    for name, _fn, note in rows[2:]:
        print(f"    {name:<24}{note}")
    print()

    print("  ── 有两层不在上表里，因为它们不改召回 ──")
    print()
    print("  5. 路由：召回一样，但上下文少 40%")
    print("     实测：统一给 8 条 vs 按类型给 2~8 条，R@8 都是 1.000，")
    print("     平均段数从 8.0 降到 4.8。价值在成本，不在准确率。")
    print("     但分类器只有 34% 一致率、57% 兜底 —— 理论收益拿不到，这就是它默认关闭的原因。")
    print()
    print("  6. 质量门：不改召回，改的是「不该答的时候别答」")
    print("     能答的相似度 0.52~0.84，答不了的 0.39~0.65 —— 中间是重叠的。")
    print("     单一阈值能把 99% 该答的留下、83% 不该答的挡住，但样本只有 6 条，")
    print("     而且这个阈值换模型就失效。")
    print()

    print("  ── 这套重建最该带走的三句话 ──")
    print()
    print("  1. **先建评测，再谈优化。** 没有金标，你连「变好了」都说不出口。")
    print("  2. **不是每一层都会提升。** 这套数据上 4 层里有 3 层是平的或负的，")
    print("     而发现这件事本身，比多提 2 个点更有价值。")
    print("  3. **一个方案打不过它的简化版，就不该上线。** 混合打不过纯向量、")
    print("     路由打不过直接给 8 条 —— 这些结论只能靠测出来。")
    print()
    print("  完整项目里这六层都在，还多了引用、回退、拒答、服务层和 111 个测试。")
    print("  但骨架就是这七步：字面 → 向量 → 混合 → 重排 → 路由 → 质量门 → 评测。")


if __name__ == "__main__":
    main()
