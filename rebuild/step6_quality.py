"""第 6 步：质量门 —— 什么时候该说「我不知道」。

前五步都在优化「怎么找得更准」。但医疗场景有一个比准确更要紧的问题：

**找不到的时候，会不会硬答？**

检索永远能返回前 5 条，哪怕全是无关的。如果直接把这 5 条塞给模型，
它一样会编出一段听起来很像的答案。这比答错更危险 —— 因为答错容易被发现，
编得像就没人怀疑。

所以需要一个门：进来之前先判断「这批资料到底靠不靠谱」。

跑法：
    python -m rebuild.step6_quality
"""

from __future__ import annotations

import sys
from pathlib import Path

import jieba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedders import BgeEmbedder  # noqa: E402
from rebuild.common import header, load  # noqa: E402

jieba.setLogLevel(20)
DOCS, QUERIES = load()

# 语料里**没有**答案的问题。真实系统里这类问题占相当一部分：
# 用户会问具体药价、手术决策、个人化验单，这些都不是教材能回答的。
OUT_OF_DOMAIN = [
    "布洛芬一天最多能吃几片",
    "阿莫西林和头孢可以一起吃吗",
    "甲状腺结节 3 类需要手术吗",
    "这个药医保能报销多少",
    "今天天气怎么样",
    "帮我写一段 Python 代码",
]


def max_sim(vec: np.ndarray, mat: np.ndarray) -> float:
    return float((mat @ vec).max())


def keyword_hit(query: str, text: str) -> float:
    """query 的实词有多大比例出现在这段里。"""
    q = [w for w in jieba.cut(query) if len(w) >= 2]
    if not q:
        return 0.0
    return sum(1 for w in q if w in text) / len(q)


def main() -> None:
    header("第 6 步：质量门", "找不到的时候，怎么知道该闭嘴")

    emb = BgeEmbedder()
    doc_mat = emb.encode([d["text"] for d in DOCS])

    # 每条 query 只看 top1 的三个信号
    def signals(query: str) -> dict:
        v = emb.encode([query])[0]
        sims = doc_mat @ v
        top = int(np.argmax(sims))
        text = DOCS[top]["text"]
        return {
            "sim": float(sims[top]),
            "kw": keyword_hit(query, text),
            "len": len(text),
        }

    print()
    print("  先看两类问题的分数分布：")
    print()

    inside = [signals(q["query"]) for q in QUERIES]
    outside = [signals(q) for q in OUT_OF_DOMAIN]

    def stat(rows, key):
        vals = [r[key] for r in rows]
        return min(vals), sum(vals) / len(vals), max(vals)

    print(f"  {'':<10}{'相似度 最低':>12}{'平均':>9}{'最高':>9}{'   |  ':>6}"
          f"{'词命中 平均':>12}")
    for label, rows in (("能答的", inside), ("答不了的", outside)):
        lo, avg, hi = stat(rows, "sim")
        _, kavg, _ = stat(rows, "kw")
        print(f"  {label:<10}{lo:>12.3f}{avg:>9.3f}{hi:>9.3f}{'':>6}{kavg:>12.3f}")

    print()
    print("  答不了的样本明细：")
    for q, r in zip(OUT_OF_DOMAIN, outside):
        print(f"    {q:<24} 相似度 {r['sim']:.3f}  词命中 {r['kw']:.2f}")

    # 找一个能把两类分开的阈值
    in_sims = sorted(r["sim"] for r in inside)
    out_sims = sorted(r["sim"] for r in outside)
    best_thr, best_gap = 0.0, -1.0
    for thr in [i / 100 for i in range(30, 90)]:
        keep = sum(1 for s in in_sims if s >= thr) / len(in_sims)   # 该答的留下了多少
        block = sum(1 for s in out_sims if s < thr) / len(out_sims)  # 不该答的挡住了多少
        gap = keep + block  # 两者都高才好
        if gap > best_gap:
            best_gap, best_thr = gap, thr

    keep = sum(1 for s in in_sims if s >= best_thr) / len(in_sims)
    block = sum(1 for s in out_sims if s < best_thr) / len(out_sims)
    print()
    print(f"  扫遍阈值，最好的是 {best_thr:.2f}：")
    print(f"    该答的留下 {keep:.1%}，不该答的挡住 {block:.1%}")
    print()

    if keep > 0.9 and block > 0.8:
        print("  看起来还行。但**别急着定这个值**：")
    else:
        print("  **分不太开** —— 这正是真实系统的处境。")

    print("  1. 这只是 6 条「答不了」的样本，样本量根本不够定阈值。")
    print("  2. 相似度分数依赖模型。换一个 embedding 模型，或者换个领域的语料，")
    print("     这个 0.55 立刻就不成立了。**阈值不可跨模型移植。**")
    print("  3. 真正能分开的两类，相似度本来就有明显间隙；分不开的时候，")
    print("     靠单一阈值就是在赌。")
    print()
    print("  项目里 `core/quality.py` 没用单一阈值，而是三个信号一起看：")
    print("    ① 召回的东西彼此像不像（像 = 可能都在讲同一件事）")
    print("    ② query 的实词在上下文里的覆盖率")
    print("    ③ 上下文够不够长")
    print("  三个都不过，才判为「没找到」。这比调一个阈值难调，但换个模型不会失效。")
    print()
    print("  这一步最该记住的一句话：")
    print("  **医疗场景里，「答不出来」是一个合法输出，不是失败。**")
    print("  → 下一步：这么多结论，怎么证明它们不是我自己编的？")


if __name__ == "__main__":
    main()
