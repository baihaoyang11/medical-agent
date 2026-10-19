"""
RRF (Reciprocal Rank Fusion) — 混合检索结果融合工具

将稠密向量检索和稀疏(BM25)检索的结果按倒数排序加权合并，
取两种检索方式的交集优势。
"""

from typing import List, Dict, Any, Optional


def rrf_merge(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_n: Optional[int] = None,
    weights: Optional[tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion 融合排序

    RRF 分数 = Σ w_i / (k + rank_i(doc))

    其中 rank 是文档在某个检索方式中的排名（从1开始），
    k 是常数（通常取60），防止排名过高主导。

    Args:
        dense_results: 稠密检索结果 [{doc_id, content, score, ...}]
        sparse_results: 稀疏检索结果 [{doc_id, content, score, ...}]
        rrf_k: RRF 常数，默认 60
        top_n: 返回 Top-N 条，默认 None（全部返回）
        weights: (稠密权重, 稀疏权重)，默认 None 即等权 (1.0, 1.0)

    weights 为什么需要
    ------------------
    等权 RRF 隐含一个假设：两路检索的质量相当。这个假设经常不成立。
    实测中稠密向量在口语化 query 上明显强于 BM25，等权融合会把 BM25 的
    噪声一起拉进来，结果两路都不如纯向量。给弱的一路降权是必要的，
    但也不能降到 0 —— BM25 在药名、指标这类精确匹配上有不可替代性。

    注意：权重必须在**独立验证集**上确定。在同一批 query 上扫出最优权重
    再报告该权重的分数，是过拟合，分数不可信。
    """
    w_dense, w_sparse = weights if weights is not None else (1.0, 1.0)

    # 用 dict 聚合：key=doc_id -> {doc_data, rrf_score}
    merged: Dict[str, Any] = {}

    def add_results(results: List[Dict[str, Any]], source: str, weight: float):
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("doc_id", doc.get("pk", f"{source}_{rank}")))
            if doc_id not in merged:
                merged[doc_id] = {
                    "doc_id": doc_id,
                    "content": doc.get("content", doc.get("text", "")),
                    "score": doc.get("score", 0),
                    "source": source,
                    **{k: v for k, v in doc.items() if k not in ("doc_id", "content", "score")},
                }
                merged[doc_id]["_rrf_score"] = 0.0
            # 累加带权的 RRF 分数
            merged[doc_id]["_rrf_score"] += weight / (rrf_k + rank + 1)

    add_results(dense_results, "dense", w_dense)
    add_results(sparse_results, "sparse", w_sparse)

    # 按 RRF 分数降序排列
    sorted_results = sorted(
        merged.values(),
        key=lambda x: x["_rrf_score"],
        reverse=True,
    )

    # 移除内部使用的 _rrf_score
    for r in sorted_results:
        del r["_rrf_score"]

    if top_n is not None:
        sorted_results = sorted_results[:top_n]

    return sorted_results


def format_search_results(
    results: List[Dict[str, Any]],
    query: str,
    method: str = "hybrid",
) -> Dict[str, Any]:
    """
    将检索结果格式化为统一的 SkillResult 风格字典
    """
    return {
        "success": True,
        "data": [
            {
                "content": r["content"],
                "score": r.get("score", 0),
                "doc_id": r.get("doc_id", ""),
                "source": r.get("source", ""),
            }
            for r in results
        ],
        "error": None,
        "metadata": {
            "count": len(results),
            "method": method,
            "query": query,
        },
    }
