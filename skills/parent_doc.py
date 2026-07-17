"""
PDF教材检索 Skill — 混合检索 + 父子文档重建
"""
import jieba
import numpy as np
from typing import Any, Dict, List, Optional
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from core.base_skill import BaseSkill, SkillResult
from core.rrf import rrf_merge

jieba.setLogLevel(20)


class ParentDocSearchSkill(BaseSkill):
    """PDF教材混合检索 + 父子文档重建"""

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
        self._name = "parent_doc_search"
        self._description = (
            "从PDF医学教材（生物化学、生理学、医学微生物学）中执行混合检索。"
            "先对子文档块（200字）进行稠密语义+BM25关键词混合检索，"
            "再按doc_id聚合重建完整父文档段落（约1000字），保证上下文完整。"
            "适用于需要深入医学原理、机制解释、详细病理的场景。"
        )
        self._parameters = {
            "query": {"type": "string", "description": "医学概念或问题关键词"},
            "k": {"type": "integer", "description": "返回的段落数量，默认3条"},
        }
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict] = []
        self._doc_id_map: Dict[str, List[Dict]] = {}

    # ────────── 数据加载 ──────────

    def _load_texts(self) -> List[Dict]:
        docs = []
        offset = 0
        while True:
            results = self._client.query(
                self._collection, limit=1000, offset=offset,
                output_fields=["pk", "text", "doc_id", "source", "page"])
            if not results:
                break
            for r in results:
                did = str(r.get("doc_id", r["pk"]))
                docs.append({
                    "pk": r["pk"], "doc_id": did,
                    "text": r.get("text", ""),
                    "source": r.get("source", ""),
                    "page": r.get("page", 0),
                })
            offset += len(results)
            if len(results) < 1000:
                break
        return docs

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        print("  [ParentDoc] 构建 BM25 索引...")
        docs = self._load_texts()
        if not docs:
            self._bm25 = BM25Okapi([])
            return
        corpus = [list(jieba.cut(d["text"])) for d in docs]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_docs = docs
        # 构建 doc_id → 子文档列表的映射，用于重建父文档
        for d in docs:
            self._doc_id_map.setdefault(d["doc_id"], []).append(d)
        print(f"  [ParentDoc] BM25 索引完成: {len(docs)} 个子文档, {len(self._doc_id_map)} 个父文档")

    # ────────── 检索 ──────────

    def _dense_search(self, query: str, k: int) -> List[Dict]:
        vec = self._embeddings.embed_query(query)
        results = self._client.search(
            self._collection, [vec], anns_field="dense",
            limit=k * 3, output_fields=["pk", "text", "doc_id", "source"])
        docs = []
        if results and results[0]:
            for hit in results[0]:
                e = hit.get("entity", {})
                docs.append({
                    "doc_id": str(e.get("doc_id", hit["id"])),
                    "content": e.get("text", ""),
                    "score": float(hit.get("distance", 0)),
                    "source": e.get("source", ""),
                })
        return docs

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        self._ensure_bm25()
        if not self._bm25 or not self._bm25_docs:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k * 3]
        docs = []
        for idx in top:
            if scores[idx] <= 0:
                continue
            d = self._bm25_docs[idx]
            docs.append({
                "doc_id": d["doc_id"], "content": d["text"],
                "score": float(scores[idx]), "source": d.get("source", ""),
            })
        return docs

    # ────────── 父文档重建 ──────────

    def _reconstruct_parent_docs(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """按 doc_id 聚合所有子文档，合并为完整父文档"""
        # 按 doc_id 去重，保留第一个出现的顺序
        seen_order = []
        seen_set = set()
        for r in candidates:
            did = r["doc_id"]
            if did not in seen_set:
                seen_set.add(did)
                seen_order.append(did)

        result = []
        for did in seen_order[:top_k]:
            siblings = self._doc_id_map.get(did, [])
            if siblings:
                siblings.sort(key=lambda x: x["pk"])
                full_text = "\n".join(s["text"] for s in siblings if s["text"])
                first = siblings[0]
                result.append({
                    "content": full_text,
                    "doc_id": did,
                    "source": first.get("source", ""),
                    "page": first.get("page", 0),
                    "chunk_count": len(siblings),
                    "char_count": len(full_text),
                })
            else:
                # doc_id 不在映射中，返回候选内容本身
                result.append({
                    "content": candidates[0].get("content", ""),
                    "doc_id": did,
                    "source": candidates[0].get("source", ""),
                    "page": 0, "chunk_count": 1, "char_count": len(candidates[0].get("content", "")),
                })
        return result

    # ────────── 主入口 ──────────

    def execute(self, query: str = "", k: int = 3, **kwargs) -> Dict[str, Any]:
        if not query:
            return SkillResult.fail("query 不能为空")
        try:
            dense = self._dense_search(query, k)
            bm25 = self._bm25_search(query, k)

            if dense and bm25:
                merged = rrf_merge(dense, bm25, top_n=k * 2)
                method = "hybrid_dense+bm25+rrf+parent_doc"
            elif dense:
                merged = dense[:k * 2]
                method = "dense_only+parent_doc"
            elif bm25:
                merged = bm25[:k * 2]
                method = "bm25_only+parent_doc"
            else:
                return SkillResult.ok(data=[], count=0, method="no_results", query=query)

            # 重建父文档
            parent_docs = self._reconstruct_parent_docs(merged, top_k=k)

            return SkillResult.ok(
                data=[{
                    "content": d["content"],
                    "doc_id": d["doc_id"],
                    "source": d.get("source", ""),
                    "page": d.get("page", 0),
                } for d in parent_docs],
                count=len(parent_docs), method=method, query=query,
                dense_count=len(dense), bm25_count=len(bm25))
        except Exception as e:
            print(f"  [ParentDoc] 错误: {e}")
            return SkillResult.fail(f"PDF教材检索失败: {e}", query=query)
