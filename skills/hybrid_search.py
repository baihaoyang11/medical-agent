"""
混合检索 Skill — 稠密向量语义检索 + BM25关键词检索 + RRF融合排序
"""
from __future__ import annotations

import threading

import jieba
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from rank_bm25 import BM25Okapi
from core.base_skill import BaseSkill, SkillResult
from core.errors import UpstreamError
from core.observability import span
from core.retry import call_with_retry
from core.rrf import rrf_merge

# pymilvus 只用在类型注解上，运行时不需要这个符号。
# 放到 TYPE_CHECKING 里，重型依赖就不在 import 路径上了 ——
# 单测可以只装最小依赖集跑真实代码，而不是被一个用不到的类型拖住。
if TYPE_CHECKING:
    from pymilvus import MilvusClient

jieba.setLogLevel(20)

# 与 core/routing.py 的 RetrieverMode 保持一致
_VALID_MODES = ("dense", "sparse", "hybrid")


class HybridSearchSkill(BaseSkill):
    """混合检索：稠密语义 + BM25关键词 + RRF融合"""

    def __init__(
        self,
        milvus_client: MilvusClient,
        embedding_model,
        collection_name: str = "LangChainCollection",
    ):
        super().__init__()
        self._client = milvus_client
        self._embeddings = embedding_model
        self._collection = collection_name
        self._name = "hybrid_search"
        self._description = (
            "从医学QA问答对数据库中执行混合检索（稠密语义 + BM25关键词 + RRF融合）。"
            "语义检索理解近义表达，BM25确保专业术语精确匹配。"
            "适用于疾病定义、药物信息、症状描述等医学知识查询。"
        )
        self._parameters = {
            "query": {"type": "string", "description": "用户的医学问题"},
            "k": {"type": "integer", "description": "返回的文档数量，默认10条"},
        }
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict] = []
        # BM25 索引是懒加载的：并发首请求会重复构建（几万篇文档要好几秒），
        # 用锁做双检，保证只建一次。
        self._bm25_lock = threading.Lock()

    # ────────── 外部调用封装 ──────────

    def _embed(self, query: str) -> List[float]:
        """Embedding 调用：任何异常都包装成可重试的 UpstreamError。"""

        def _call() -> List[float]:
            try:
                return self._embeddings.embed_query(query)
            except Exception as exc:  # noqa: BLE001
                raise UpstreamError(f"embedding 调用失败: {exc}") from exc

        return call_with_retry(_call, op_name="embed_query")

    # ────────── BM25 索引 ──────────

    def _load_texts(self) -> List[Dict]:
        docs = []
        offset = 0
        while True:
            results = self._client.query(
                self._collection, limit=1000, offset=offset,
                output_fields=["pk", "text", "doc_id"])
            if not results:
                break
            for r in results:
                docs.append({
                    "pk": r["pk"],
                    "doc_id": str(r.get("doc_id", r["pk"])),
                    "text": r.get("text", ""),
                })
            offset += len(results)
            if len(results) < 1000:
                break
        return docs

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        with self._bm25_lock:
            if self._bm25 is not None:
                return
            print("  [HybridSearch] 构建 BM25 索引...")
            with span("bm25_build", collection=self._collection):
                docs = self._load_texts()
                if not docs:
                    self._bm25 = BM25Okapi([])
                    return
                corpus = [list(jieba.cut(d["text"])) for d in docs]
                self._bm25 = BM25Okapi(corpus)
                self._bm25_docs = docs
            print(f"  [HybridSearch] BM25 索引完成: {len(docs)} 篇")

    # ────────── 稠密检索 ──────────

    def _dense_search(self, query: str, k: int) -> List[Dict]:
        vec = self._embed(query)
        results = self._client.search(
            self._collection, [vec], anns_field="dense",
            limit=k * 2, output_fields=["pk", "text", "doc_id"])
        docs = []
        if results and results[0]:
            for hit in results[0]:
                e = hit.get("entity", {})
                docs.append({
                    "doc_id": str(e.get("doc_id", hit["id"])),
                    "content": e.get("text", ""),
                    "score": float(hit.get("distance", 0)),
                })
        return docs

    # ────────── BM25 检索 ──────────

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        self._ensure_bm25()
        if not self._bm25 or not self._bm25_docs:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k * 2]
        docs = []
        for idx in top:
            if scores[idx] <= 0:
                continue
            d = self._bm25_docs[idx]
            docs.append({"doc_id": d["doc_id"], "content": d["text"], "score": float(scores[idx])})
        return docs

    # ────────── 主入口 ──────────

    def execute(self, query: str = "", k: int = 10, mode: str = "hybrid", **kwargs) -> Dict[str, Any]:
        """mode 决定走哪一路：

        dense  只跑向量（语义强、术语弱）
        sparse 只跑 BM25（术语强、口语弱）
        hybrid 两路都跑再 RRF 融合（默认）

        mode 由路由层给（见 core/routing.py 的 RETRIEVERS），不是调用方随手传的。
        非法值回退 hybrid 而不是报错 —— 检索配置写错不该让整个请求 500。
        """
        if not query:
            return SkillResult.fail("query 不能为空")
        if mode not in _VALID_MODES:
            mode = "hybrid"
        try:
            dense = self._dense_search(query, k) if mode in ("dense", "hybrid") else []
            bm25 = self._bm25_search(query, k) if mode in ("sparse", "hybrid") else []

            if dense and bm25:
                final = rrf_merge(dense, bm25, top_n=k)
                method = "hybrid_dense+bm25+rrf"
            elif dense:
                final = dense[:k]
                # requested 与 method 并列，才能区分"配置就要 dense"和"融合降级成 dense"
                method = "dense_only"
            elif bm25:
                final = bm25[:k]
                method = "bm25_only"
            else:
                return SkillResult.ok(data=[], count=0, method="no_results",
                                      requested=mode, query=query)

            return SkillResult.ok(
                data=[{"content": r["content"], "score": r.get("score", 0), "doc_id": r.get("doc_id", "")} for r in final],
                count=len(final), method=method, requested=mode, query=query,
                dense_count=len(dense), bm25_count=len(bm25))
        except Exception as e:
            # 关键：把异常向上抛，让 Agent 层决定降级；不要再 print 吞掉
            return SkillResult.fail(f"混合检索失败: {e}", query=query, error_type=type(e).__name__)
