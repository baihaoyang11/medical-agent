"""第 4 步：重排 —— 让对的那条排到前面。

前几步关心的是「正确答案有没有进前 5」。但进前 5 不等于够用：
拼上下文是有预算的，排第 5 的那条很可能在预算用完后被丢掉。

所以还要看**排名**：正确答案排第几。这个指标的学名叫 MRR
（Mean Reciprocal Rank，第 1 名得 1 分、第 2 名得 0.5、第 3 名得 0.33，取平均）。

重排做的事：先宽松地召回一批（比如 15 条），再用更精细的标准重新打分。

跑法：
    python -m rebuild.step4_rerank
"""

from __future__ import annotations

import sys
from pathlib import Path

import jieba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedders import BgeEmbedder  # noqa: E402
from rebuild.common import K, evaluate, header, load, print_detail, print_result  # noqa: E402

jieba.setLogLevel(20)
DOCS, QUERIES = load()
KEYS = [d["key"] for d in DOCS]
TEXT = {d["key"]: d["text"] for d in DOCS}


def content_words(text: str) -> list[str]:
    """只留两字以上的词 —— 单字（的/是/了）对判断相关性没帮助。"""
    return [w for w in jieba.cut(text) if len(w) >= 2]


def keyword_rerank(query: str, cands: list[str], w_orig: float = 0.0) -> list[str]:
    """用一个比「向量相似度」更挑剔的标准重新打分。

    两个信号：
    1. query 里的实词，有多少在这段里出现过（覆盖率）
    2. 这些词出现在**开头**的比例 —— 医学段落通常第一句就是结论

    `w_orig` 决定保留多少原来的排名。这个参数很关键，见下面的实验：
    完全听新标准的（w_orig=0）会出事。

    为什么不用更强的模型（cross-encoder）：那要额外下一个模型。
    这一步想说明「重排的机制」，不是「哪个重排器最好」。
    """
    q = content_words(query)
    if not q:
        return cands

    scored = []
    for rank, key in enumerate(cands):
        text = TEXT[key]
        head = text[: max(1, len(text) // 3)]
        cover = sum(1 for w in q if w in text) / len(q)
        head_cover = sum(1 for w in q if w in head) / len(q)
        kw_score = 0.6 * cover + 0.4 * head_cover
        orig_score = 1.0 / (rank + 1)          # 原来排第几，折成 0~1 的分
        scored.append((w_orig * orig_score + (1 - w_orig) * kw_score, key))
    scored.sort(reverse=True)
    return [k for _, k in scored]


def main() -> None:
    header("第 4 步：重排", "先宽松召回一批，再用更挑剔的标准重新排序")

    emb = BgeEmbedder()
    doc_mat = emb.encode([d["text"] for d in DOCS])
    qvec = {q["query"]: v for q, v in zip(QUERIES, emb.encode([q["query"] for q in QUERIES]))}

    def dense(query: str, k: int = K) -> list[str]:
        sims = doc_mat @ qvec[query]
        return [KEYS[i] for i in np.argsort(-sims)[:k]]

    DEPTH = K * 3  # 先召回 15 条

    def make_rerank(w_orig: float):
        def fn(query: str, k: int = K) -> list[str]:
            return keyword_rerank(query, dense(query, DEPTH), w_orig)[:k]
        return fn

    r_base = evaluate(dense, QUERIES, "2. 纯向量（不重排）")
    r_rr = evaluate(make_rerank(0.0), QUERIES, "4a. 重排（完全听新标准）")
    r_con = evaluate(make_rerank(0.5), QUERIES, "4b. 重排（保留一半原排序）")

    print()
    print_result(r_base)
    print_detail(r_base)
    print_result(r_rr, r_base)
    print_detail(r_rr)
    print_result(r_con, r_base)
    print_detail(r_con)

    print()
    print("  **完全听新标准的那个，掉得比不重排还惨。** 这不是 bug，是个很实用的教训：")
    print("  你用的重排器是「数词」，而它替换掉的向量排序是「懂语义」的。")
    print("  用一个更笨的标准去改一个更聪明的结果，等于开倒车。")
    print()
    print("  把原排序留一半，就不掉了 —— 因为新标准只在「原来也差不多」的候选之间做微调。")
    print()
    print("  扫一遍这个权重，看得更清楚：")
    print()
    print(f"  {'保留原排序':>10}  {'R@5':>6}  {'MRR':>6}")
    for w in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        r = evaluate(make_rerank(w), QUERIES, "")
        print(f"  {w:>10.1f}  {r['recall']['all']:>6.3f}  {r['mrr']:>6.3f}")

    print()
    print("  w=1.0 就是完全不重排。曲线说明：在这批数据上，重排要么持平要么拖累。")
    print("  **为什么还留着它？** 因为在真实项目里它确实涨了 —— hybrid 0.905 → 0.914。")
    print("  差别在于那里是在「混合检索」的结果上重排，两路本来都偏词面，")
    print("  新标准和它同类，不会打架。")
    print()
    print("  这就是 `config.RERANK_WEIGHT` 这个参数存在的理由：")
    print("  它调的正是「新标准和原排序各占多少」。默认 0.6 偏向原排序，不是随便定的。")

    print()
    print("  顺带注意一个容易说错的点：")
    print("  重排不能提高召回的上限。如果正确的压根没进前 15，重排再怎么排也变不出来。")
    print("  所以「召回」和「排序」是两件事，要分开测、分开说。")
    print()
    print("  → 下一步：既然不同问法适合不同检索器，那就让系统自己判断该用哪个。")


if __name__ == "__main__":
    main()
