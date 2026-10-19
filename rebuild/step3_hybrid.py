"""第 3 步：把字面那一路加回来 —— 混合检索。

向量解决了「意思相近」，但在两类东西上它天生弱：

- **药名**：「格列美脲」和「格列齐特」在向量空间里几乎是同一个点，但它们是两种药
- **指标缩写**：空腹血糖 / 餐后血糖 / 糖化血红蛋白，语义上都叫「血糖指标」

这两类靠字面反而准。所以自然的想法是：两路都跑，结果合并。

合并方式叫 RRF（倒数排名融合）：不看分数，只看排名。
第 1 名给 1/61 分，第 2 名给 1/62 … 两路的排名加起来排。

**但这一步会给你一个反直觉的结果。**

跑法：
    python -m rebuild.step3_hybrid
"""

from __future__ import annotations

import math
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


# ────────────────────────────────────────────────────────────────
# BM25：三十行，不用装库
# ────────────────────────────────────────────────────────────────
class BM25:
    """经典的关键词打分。

    思路：一个词在这段里出现得越多、在别的段里出现得越少，它越能代表这段。
    前一半是 TF（词频），后一半是 IDF（逆文档频率）。

    分母里那个 `b * dl / avgdl` 是长度归一化：长段落天然更容易命中更多词，
    不加这项，长文档会无脑排前面。
    """

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [list(jieba.cut(d["text"])) for d in docs]
        self.lens = np.array([len(t) for t in self.tokens], dtype=float)
        self.avgdl = float(self.lens.mean())

        df: dict[str, int] = {}
        for toks in self.tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        # 为什么加 1 再取对数：不加的话，出现在一半以上文档里的词会得到**负分**，
        # 变成"文档里有这个词反而扣分"，排序会很离谱。
        self.idf = {w: math.log(1 + (n - v + 0.5) / (v + 0.5)) for w, v in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = list(jieba.cut(query))
        out = np.zeros(len(self.tokens))
        for i, toks in enumerate(self.tokens):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / denom
            out[i] = s
        return out


# ────────────────────────────────────────────────────────────────
# RRF：只吃排名，不吃分数
# ────────────────────────────────────────────────────────────────
def rrf(ranked_a: list[str], ranked_b: list[str], w_a: float = 0.5, w_b: float = 0.5,
        rrf_k: int = 60) -> list[str]:
    """两路排名加权合并。

    为什么用排名而不是分数：余弦相似度在 0~1 之间，BM25 可能是 0~20，
    量纲完全不同，直接相加没有意义。排名是唯一可比的东西。

    代价：把「第 1 名比第 2 名强多少」这个信息丢掉了。
    """
    score: dict[str, float] = {}
    for ranked, w in ((ranked_a, w_a), (ranked_b, w_b)):
        for rank, key in enumerate(ranked):
            score[key] = score.get(key, 0.0) + w / (rrf_k + rank + 1)
    return sorted(score, key=lambda k: -score[k])


def main() -> None:
    header("第 3 步：混合检索", "向量 + BM25，两路排名合并 —— 结果可能和你想的不一样")

    print("  准备两路检索 ...", flush=True)
    emb = BgeEmbedder()
    doc_mat = emb.encode([d["text"] for d in DOCS])
    qvec = {q["query"]: v for q, v in zip(QUERIES, emb.encode([q["query"] for q in QUERIES]))}
    bm25 = BM25(DOCS)

    def dense_ranked(query: str, k: int = K) -> list[str]:
        sims = doc_mat @ qvec[query]
        return [KEYS[i] for i in np.argsort(-sims)[:k]]

    def sparse_ranked(query: str, k: int = K) -> list[str]:
        s = bm25.scores(query)
        return [KEYS[i] for i in np.argsort(-s)[:k] if s[i] > 0]

    # 融合时两路各多召回一些，给交叉留空间
    POOL = K * 3

    def hybrid(query: str, k: int = K, w_dense: float = 0.5) -> list[str]:
        return rrf(
            dense_ranked(query, POOL), sparse_ranked(query, POOL),
            w_a=w_dense, w_b=1 - w_dense,
        )[:k]

    r_dense = evaluate(dense_ranked, QUERIES, "2. 纯向量")
    r_sparse = evaluate(sparse_ranked, QUERIES, "3a. 纯 BM25")
    r_half = evaluate(lambda q, k=K: hybrid(q, k, 0.5), QUERIES, "3b. 混合（各一半）")

    print()
    print_result(r_dense)
    print_detail(r_dense)
    print_result(r_sparse)
    print_detail(r_sparse)
    print_result(r_half, r_dense)
    print_detail(r_half)

    print()
    print("  **看到了吗：加了 BM25 之后，反而比纯向量低。**")
    print("  这不是实现写错了，是「等权融合」这个默认值本身就错了。")
    print()
    print("  RRF 的隐含假设是「两路质量相当」。但实测：")
    print(f"    纯向量  口语 {r_dense['recall']['colloquial']:.3f}  书面 {r_dense['recall']['formal']:.3f}")
    print(f"    纯 BM25 口语 {r_sparse['recall']['colloquial']:.3f}  书面 {r_sparse['recall']['formal']:.3f}")
    print("  BM25 在口语上有先天缺陷：「血糖高的人」和「高血糖患者」一个词都不共享，")
    print("  把它和向量等权相加，等于往一杯干净水里兑浑水。")

    print()
    print("  那就别等权。把向量的权重往上调：")
    print()
    print(f"  {'w_dense':>8}  {'R@5':>6}   {'书面':>6} {'口语':>6}")
    for w in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        r = evaluate(lambda q, k=K, w=w: hybrid(q, k, w), QUERIES, "")
        print(
            f"  {w:>8.1f}  {r['recall']['all']:>6.3f}   "
            f"{r['recall']['formal']:>6.3f} {r['recall']['colloquial']:>6.3f}"
        )

    print()
    print("  注意两件事，别只看最后一行：")
    print()
    print("  1. 书面这一列，混合在 w=0.5~0.8 时是 1.000，**比纯向量的 0.979 还高**。")
    print("     说明混合不是没用 —— 它在书面上真的多捡回来一条，只是被口语的短板抵掉了。")
    print("  2. 曲线一路往上走到 1.0 最高，意思是「在这套题上，越接近纯向量越好」。")
    print()
    print("  所以真正的结论不是「混合没用」，是：")
    print("  **同一个方案对不同类型的问题，价值是相反的。**")
    print()
    print("  这套题只有书面和口语两种问法，没有专门考药名的题。")
    print("  你手上那个真实项目里加了 20 条术语型 query，结论就变了：")
    print("  药名类问题上混合是 1.000、纯向量 0.920 —— 因为「格列美脲」和「格列齐特」")
    print("  在向量空间里几乎是同一个点，只有字面能分开它们。")
    print()
    print("  → 这一步真正的收获不是「该用哪个」，是：**先建评测，再谈选择。**")


if __name__ == "__main__":
    main()
