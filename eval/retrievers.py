"""检索器：四种配置，逐步加组件，用来做消融对比。

    dense       纯向量
    sparse      纯 BM25
    hybrid      向量 + BM25，用**生产的** core.rrf.rrf_merge 融合
    hybrid+rr   混合召回后，用**生产的** core.rerank.KeywordReranker 重排

后两个直接 import 项目的 core/，不是复刻一份 ——
评测和生产跑同一段逻辑，测出来的才作数。如果评测自己实现一套 RRF，
那测的是评测脚本，不是系统。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from core.rerank import KeywordReranker, NoOpReranker
from core.rrf import rrf_merge


@dataclass(frozen=True)
class Doc:
    doc_id: str
    content: str
    topic: str
    facet: str
    kind: str  # core / filler


class Corpus:
    def __init__(self, docs: Sequence[Doc]) -> None:
        self.docs: List[Doc] = list(docs)
        self._by_id: Dict[str, Doc] = {d.doc_id: d for d in self.docs}

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, doc_id: str) -> Doc:
        return self._by_id[doc_id]


# ────────────────────────────────────────────────────────────────
# BM25：自己实现而不是拉 rank_bm25
# ────────────────────────────────────────────────────────────────
# 理由是可复现性。第三方库的 IDF 变体、负值处理会轻微影响分数，
# 而 BM25 只是 40 行标准公式。自己写能保证换机器结果一致。
#
# 分词**必须和生产一致**：生产链路（skills/hybrid_search.py）用 jieba。
# 评测如果用字符 bigram 代替，测的就是一个不存在的系统 ——
# 这个偏差会系统性地低估稀疏检索，进而得出"混合检索没用"的错误结论。
_JIEBA = None
_JIEBA_TRIED = False
_VALID_CHAR = None


def _get_jieba():
    global _JIEBA, _JIEBA_TRIED
    if not _JIEBA_TRIED:
        try:
            import jieba

            _JIEBA = jieba
        except ImportError:
            _JIEBA = None
        _JIEBA_TRIED = True
    return _JIEBA


def tokenizer_name() -> str:
    return "jieba" if _get_jieba() is not None else "char-bigram"


def _tokenize(text: str) -> List[str]:
    global _VALID_CHAR
    if _VALID_CHAR is None:
        import re

        _VALID_CHAR = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

    jb = _get_jieba()
    if jb is not None:
        return [w for w in jb.lcut(text) if _VALID_CHAR.search(w)]

    # 没有 jieba 时的回退：字符二元组。没有未登录词问题，但粒度更粗。
    clean = "".join(ch for ch in text if not ch.isspace())
    if len(clean) < 2:
        return [clean] if clean else []
    return [clean[i : i + 2] for i in range(len(clean) - 1)]


class BM25:
    def __init__(self, corpus: Sequence[Doc], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = [d.doc_id for d in corpus]
        self.tokens = [_tokenize(d.content) for d in corpus]
        self.doc_len = np.array([len(t) for t in self.tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0

        self.tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for toks in self.tokens:
            counter: Dict[str, int] = {}
            for t in toks:
                counter[t] = counter.get(t, 0) + 1
            self.tf.append(counter)
            for t in counter:
                df[t] = df.get(t, 0) + 1

        n = len(corpus)
        # Lucene 风格的 IDF，加 1 保证非负 —— 否则高频词会得到负权重，
        # 出现"文档里有这个词反而扣分"的反直觉排序
        self.idf = {t: math.log(1 + (n - v + 0.5) / (v + 0.5)) for t, v in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        out = np.zeros(len(self.doc_ids), dtype=np.float32)
        for i, counter in enumerate(self.tf):
            dl = self.doc_len[i]
            s = 0.0
            for t in q_tokens:
                f = counter.get(t, 0)
                if f == 0:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                s += idf * (f * (self.k1 + 1)) / denom
            out[i] = s
        return out


# ────────────────────────────────────────────────────────────────
# 检索器
# ────────────────────────────────────────────────────────────────
class BaseRetriever:
    name = "base"

    def search(self, query: str, k: int) -> List[str]:
        raise NotImplementedError


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(self, corpus: Corpus, embedder) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.doc_mat = embedder.encode([d.content for d in corpus.docs])
        # query 向量缓存：权重扫描要对同一批 query 跑十几轮，
        # 每轮重新编码在 CPU 上要几十秒，缓存后是常数开销
        self._qcache: Dict[str, np.ndarray] = {}

    def _encode(self, query: str) -> np.ndarray:
        vec = self._qcache.get(query)
        if vec is None:
            vec = self.embedder.encode([query])[0]
            self._qcache[query] = vec
        return vec

    def search(self, query: str, k: int) -> List[str]:
        qv = self._encode(query)
        sims = self.doc_mat @ qv  # 都已归一化，点积即余弦
        order = np.argsort(-sims)[:k]
        return [self.corpus.docs[i].doc_id for i in order]


class SparseRetriever(BaseRetriever):
    name = "sparse"

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.bm25 = BM25(corpus.docs)

    def search(self, query: str, k: int) -> List[str]:
        sims = self.bm25.scores(query)
        order = np.argsort(-sims)[:k]
        return [self.corpus.docs[i].doc_id for i in order]


class HybridRetriever(BaseRetriever):
    """RRF 融合。融合函数直接调 core.rrf.rrf_merge。"""

    name = "hybrid"

    def __init__(self, corpus: Corpus, dense: BaseRetriever, sparse: BaseRetriever,
                 rrf_k: int = 60, pool_factor: int = 3,
                 weights: tuple[float, float] | None = None) -> None:
        self.corpus = corpus
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.pool_factor = pool_factor
        self.weights = weights

    def _as_results(self, ids: Sequence[str]) -> List[dict]:
        return [
            {"doc_id": i, "content": self.corpus.get(i).content, "score": 0.0}
            for i in ids
        ]

    def search(self, query: str, k: int) -> List[str]:
        pool = k * self.pool_factor
        merged = rrf_merge(
            self._as_results(self.dense.search(query, pool)),
            self._as_results(self.sparse.search(query, pool)),
            rrf_k=self.rrf_k,
            weights=self.weights,
        )
        return [r["doc_id"] for r in merged[:k]]


class RerankRetriever(BaseRetriever):
    """混合召回 → 重排。重排器直接调 core.rerank.KeywordReranker。"""

    name = "hybrid+rerank"

    def __init__(self, corpus: Corpus, base: BaseRetriever,
                 reranker=None, depth: int = 3) -> None:
        self.corpus = corpus
        self.base = base
        self.reranker = reranker or KeywordReranker()
        self.depth = depth

    def search(self, query: str, k: int) -> List[str]:
        cand_ids = self.base.search(query, k * self.depth)
        docs = [
            {"doc_id": i, "content": self.corpus.get(i).content, "score": 0.0}
            for i in cand_ids
        ]
        reranked = self.reranker.rerank(query, docs)
        return [d["doc_id"] for d in reranked[:k]]


class OracleRerankRetriever(BaseRetriever):
    """上限参考：把金标文档强行提到第一位。

    它的存在是为了回答"重排最多能带来多少提升"这个天花板问题。
    如果 KeywordReranker 和 oracle 之间差距很大，说明重排还有优化空间；
    如果差距很小，说明瓶颈在召回，换重排器没有意义。
    """

    name = "oracle"

    def __init__(self, corpus: Corpus, base: BaseRetriever, gold_lookup) -> None:
        self.corpus = corpus
        self.base = base
        self.gold_lookup = gold_lookup

    def search(self, query: str, k: int) -> List[str]:
        ids = self.base.search(query, k)
        rel = set(self.gold_lookup.get(query, ()))
        hit = [i for i in ids if i in rel]
        rest = [i for i in ids if i not in rel]
        return (hit + rest)[:k]


def build_reranker(kind: str = "keyword"):
    if kind in ("none", "noop"):
        return NoOpReranker()
    if kind == "keyword":
        return KeywordReranker()
    raise ValueError(f"未知重排器: {kind}")
