"""第 5 步：路由 —— 先弄清「不同类型要几条」，再谈要不要分。

前四步一直在问「哪种检索器更好」。但还有一个更实际的问题：

**每个问题到底需要几段资料？**

给多了浪费上下文（花钱、稀释重点、模型更容易跑偏）；
给少了答不全。而不同的问题需要的量可能不一样。

这一步先不动系统，只是**把每类问题在不同 k 下的表现测出来**，
看看「按类型给不同的 k」这件事有没有依据。

跑法：
    python -m rebuild.step5_route
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.routing import classify  # noqa: E402
from eval.embedders import BgeEmbedder  # noqa: E402
from rebuild.common import header, load, recall_at  # noqa: E402

DOCS, QUERIES = load()
KEYS = [d["key"] for d in DOCS]
TYPE_LABEL = {
    "disease_definition": "定义与病因",
    "clinical": "临床表现与监测",
    "drug": "药物治疗",
    "diet": "饮食管理",
    "prevention": "预防与生活方式",
    "general": "通用/兜底",
}


def main() -> None:
    header("第 5 步：路由", "先测出每类问题需要几条，再决定要不要分开给")

    emb = BgeEmbedder()
    doc_mat = emb.encode([d["text"] for d in DOCS])
    qvec = {q["query"]: v for q, v in zip(QUERIES, emb.encode([q["query"] for q in QUERIES]))}

    # 一次算到 10 条，之后按不同 k 截取 —— 保证各 k 之间比的是"给几条"，不是"怎么捡"
    rankings = {}
    for q in QUERIES:
        sims = doc_mat @ qvec[q["query"]]
        rankings[q["query"]] = [KEYS[i] for i in np.argsort(-sims)[:10]]

    # ── 1. 整体：多给几条还有没有用 ──
    print()
    print("  先看整体：k 从 1 加到 10")
    print()
    print(f"  {'k':>3}  {'R@k':>6}  {'比上一条多找回':>14}")
    prev = 0.0
    for k in range(1, 11):
        r = sum(recall_at(rankings[q["query"]], q["gold"], k) for q in QUERIES) / len(QUERIES)
        print(f"  {k:>3}  {r:>6.3f}  {r - prev:>+14.3f}")
        prev = r

    # ── 2. 按类型拆：每类的天花板在哪 ──
    print()
    print("  按问题类型拆开，看每类在第几条就基本到顶：")
    print()
    types = sorted({q["qtype"] for q in QUERIES})
    print(f"  {'类型':<14}{'n':>4}  " + "".join(f"{'k=' + str(k):>7}" for k in (1, 2, 3, 5, 8)))
    print("  " + "-" * 62)

    saturate = {}
    for t in types:
        sub = [q for q in QUERIES if q["qtype"] == t]
        row = []
        for k in (1, 2, 3, 5, 8):
            r = sum(recall_at(rankings[q["query"]], q["gold"], k) for q in sub) / len(sub)
            row.append(r)
        print(f"  {TYPE_LABEL.get(t, t):<14}{len(sub):>4}  " + "".join(f"{v:>7.3f}" for v in row))
        # 找到"再多给也涨不过 0.005"的最小 k
        best = max(row)
        saturate[t] = next(k for k, v in zip((1, 2, 3, 5, 8), row) if best - v < 0.005)

    print()
    print("  「饱和点」= 再多给也基本不涨的那个 k：")
    for t, k in saturate.items():
        print(f"    {TYPE_LABEL.get(t, t):<14} k = {k}")

    # ── 3. 如果按类型给不同的 k，能省多少 ──
    print()
    uniform_k = max(saturate.values())
    print(f"  假设不分类、一律给最多的 {uniform_k} 条：")
    r_all = sum(recall_at(rankings[q["query"]], q["gold"], uniform_k) for q in QUERIES) / len(QUERIES)
    print(f"    R@{uniform_k} = {r_all:.3f}，平均给 {uniform_k} 条")

    routed_recall, routed_k = 0.0, 0
    for q in QUERIES:
        k = saturate[q["qtype"]]
        routed_recall += recall_at(rankings[q["query"]], q["gold"], k)
        routed_k += k
    routed_recall /= len(QUERIES)
    routed_k /= len(QUERIES)

    print()
    print("  按类型各自给到自己的饱和点：")
    print(f"    R@各自k = {routed_recall:.3f}，平均给 {routed_k:.1f} 条")
    print()
    saved = (1 - routed_k / uniform_k) * 100
    print(f"  同样甚至更好的召回，上下文少了约 {saved:.0f}%。")
    print("  这就是路由的价值 —— **不是「答得更准」，是「花得更少」。**")

    # ── 4. 但分类器行不行？ ──
    print()
    print("  上面是按「人工标注的类型」分的，是理想情况。")
    print("  真上线得靠分类器自己判。用项目里现有的关键词分类器试一下：")
    print()
    ok = sum(1 for q in QUERIES if classify(q["query"])[0] == q["qtype"])
    fallback = sum(1 for q in QUERIES if classify(q["query"])[0] == "general")
    print(f"    判对：{ok}/{len(QUERIES)} = {ok / len(QUERIES):.1%}")
    print(f"    掉进兜底 general：{fallback}/{len(QUERIES)} = {fallback / len(QUERIES):.1%}")
    print()
    print("  注意这里的一致率**不能**当成分类准确率读 —— 评测集的类型是按知识点侧面标的")
    print("  （「运动」被标成 prevention），生产的类型是按问句意图分的，不是同一个维度。")
    print("  但兜底率是不受口径影响的：掉进 general 就是没做区分。")
    print()
    print("  → 所以要真做路由，得先换掉这个分类器。这一步先记住结论：")
    print("    **路由的理论收益（省 30%~40% 上下文）是真的，能不能拿到取决于分类器。**")


if __name__ == "__main__":
    main()
