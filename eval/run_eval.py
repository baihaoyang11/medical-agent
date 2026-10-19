"""检索评测入口。

用法：
    python -m eval.run_eval --embedder bge        # 真实语义向量
    python -m eval.run_eval --embedder hashing    # 零依赖兜底，CI 用

输出：
    eval/report.md     人类可读的对比报告
    eval/results.json  机器可读的原始结果

设计上刻意做的两件事：
1. **复用生产代码**：RRF 用 core.rrf.rrf_merge，重排用 core.rerank.KeywordReranker。
   评测里自己复刻一套是最常见的自欺 —— 测的是脚本，不是系统。
2. **分桶统计**：按"书面/口语"和"问题类型"分组。总体分数会掩盖结构性差异，
   而这里的结构性差异恰恰是混合检索存在的理由。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

from eval import metrics as M
from eval.embedders import get_embedder
from eval.filler import FILLERS
from eval.knowledge import (
    CONFUSABLE_DOCS,
    EXTRA_QUERIES,
    POINTS,
    TERM_QUERIES,
    TOPICS,
    build_queries,
)
from eval.retrievers import (
    Corpus,
    Doc,
    DenseRetriever,
    HybridRetriever,
    OracleRerankRetriever,
    RerankRetriever,
    SparseRetriever,
    build_reranker,
    tokenizer_name,
)

OUT_DIR = Path(__file__).parent
K_VALUES = (1, 3, 5, 10)
MAIN_K = 5


def build_corpus() -> Corpus:
    docs: List[Doc] = []
    for p in POINTS:
        docs.append(Doc(p.key, p.content, p.topic, p.facet, "core"))
    for doc_id, content, topic in CONFUSABLE_DOCS:
        docs.append(Doc(doc_id, content, topic, "confusable", "confusable"))
    for f in FILLERS:
        docs.append(Doc(f.key, f.content, "filler", "filler", "filler"))
    return Corpus(docs)


def build_query_set() -> List[dict]:
    rows = []
    for query, key, qtype, style in build_queries():
        rows.append({"query": query, "gold": [key], "qtype": qtype, "style": style})
    for query, key, qtype in TERM_QUERIES:
        rows.append({"query": query, "gold": [key], "qtype": qtype, "style": "term"})
    for query, key, qtype, style in EXTRA_QUERIES:
        rows.append(
            {"query": query, "gold": [key] if key else [], "qtype": qtype, "style": style}
        )
    return rows


def gold_lookup(queries: Sequence[dict]) -> Dict[str, List[str]]:
    return {q["query"]: q["gold"] for q in queries}


def evaluate(retriever, queries: Sequence[dict], depth: int = 10) -> dict:
    """跑一个检索器，返回逐条结果与汇总。"""
    rows = []
    for q in queries:
        ranked = retriever.search(q["query"], depth)
        gold = q["gold"]
        rows.append(
            {
                "query": q["query"],
                "qtype": q["qtype"],
                "style": q["style"],
                "gold": gold,
                "ranked": ranked,
                "rank": next((i + 1 for i, d in enumerate(ranked) if d in set(gold)), 0),
                "recall": {k: M.recall_at_k(ranked, gold, k) for k in K_VALUES},
                "mrr": M.mrr(ranked, gold),
                "ndcg5": M.ndcg_at_k(ranked, gold, MAIN_K),
            }
        )

    def agg(subset: Sequence[dict]) -> dict:
        return {
            f"recall@{k}": M.mean(r["recall"][k] for r in subset) for k in K_VALUES
        } | {
            "mrr": M.mean(r["mrr"] for r in subset),
            "ndcg@5": M.mean(r["ndcg5"] for r in subset),
            "n": sum(1 for r in subset if r["gold"]),
        }

    style_buckets = {
        s: agg([r for r in rows if r["style"] == s])
        for s in ("formal", "colloquial", "term")
    }
    type_buckets = {
        t: agg([r for r in rows if r["qtype"] == t])
        for t in sorted({r["qtype"] for r in rows})
    }
    return {"overall": agg(rows), "by_style": style_buckets, "by_type": type_buckets, "rows": rows}


def sweep_weights(corpus, dense, sparse, queries, depth: int = 10) -> List[dict]:
    """扫稠密权重 0.0 → 1.0，看融合权重对结果的影响。

    这是**诊断**，不是调参。产出的是整条曲线而不是单个最优点——
    如果最优区间很窄，说明这个参数对数据分布敏感、不可靠；
    如果是一段平台，说明结论稳健。只在同一批 query 上扫出最优点然后
    报告那个点的分数，是过拟合，那种数字不该写进报告。
    """
    rows = []
    for i in range(11):
        w = round(i / 10, 1)
        r = HybridRetriever(corpus, dense, sparse, weights=(w, 1.0 - w))
        res = evaluate(r, queries, depth=depth)
        st = res["by_style"]
        rows.append(
            {
                "w_dense": w,
                "overall": res["overall"]["recall@5"],
                "mrr": res["overall"]["mrr"],
                "formal": st.get("formal", {}).get("recall@5", float("nan")),
                "term": st.get("term", {}).get("recall@5", float("nan")),
                "colloquial": st.get("colloquial", {}).get("recall@5", float("nan")),
            }
        )
    return rows


def fmt(x: float, digits: int = 3) -> str:
    return "—" if x != x else f"{x:.{digits}f}"  # NaN 显示为破折号


def render_report(results: dict, meta: dict, sweep: List[dict] | None = None) -> str:
    names = list(results.keys())
    lines: List[str] = []

    lines.append("# 检索评测报告\n")
    lines.append("> 本文件由 `python -m eval.run_eval` 自动生成，请勿手改。\n")

    lines.append("## 设置\n")
    lines.append(f"- **语料**：{meta['n_docs']} 个 chunk（{meta['n_core']} 核心知识点 + {meta.get('n_confusable', 0)} 易混干扰 + {meta['n_filler']} 无关干扰）")
    lines.append(f"- **query**：{meta['n_queries']} 条（{meta['n_queries'] - meta['n_emergency']} 条可检索 + {meta['n_emergency']} 条急症，急症不计入 recall）")
    lines.append(f"- **embedding**：{meta['embedder']}（dim={meta['dim']}，来源 `{meta.get('embedder_source', '?')}`）")
    lines.append(f"- **BM25 分词**：{meta.get('tokenizer', '?')}（与生产链路 `skills/hybrid_search.py` 保持一致）")
    lines.append(f"- **检索深度**：top-{meta['depth']}；主指标 recall@{MAIN_K}")
    lines.append(f"- **耗时**：{meta['elapsed']:.1f}s\n")
    lines.append(f"- **语料性质**：合成语料。真实教材受著作权保护不能随仓库分发（见 README 已知边界第 7 条），")
    lines.append(f"  因此这里的分数衡量的是**各检索配置在受控难度下的相对表现**，不等于真实业务效果。\n")

    # 总表
    lines.append("## 总体对比\n")
    header = "| 检索配置 | " + " | ".join(f"R@{k}" for k in K_VALUES) + " | MRR | nDCG@5 |"
    lines.append(header)
    lines.append("|" + "---|" * (len(K_VALUES) + 3))
    for name in names:
        o = results[name]["overall"]
        cells = " | ".join(fmt(o[f"recall@{k}"]) for k in K_VALUES)
        lines.append(f"| `{name}` | {cells} | {fmt(o['mrr'])} | {fmt(o['ndcg@5'])} |")
    lines.append("")

    # 按问法
    lines.append("## 按问法拆开\n")
    lines.append("三种问法考的是不同能力：**书面**共享词汇，**术语**依赖精确匹配（药名、指标、数值），")
    lines.append("**口语**几乎不共享词、只能靠语义。看最后两列比看平均分有用——")
    lines.append("平均分高可能是被强项拉起来的，**最差列**才反映用户会遇到的坏情况。\n")
    lines.append(f"| 检索配置 | 书面 R@{MAIN_K} | 术语 R@{MAIN_K} | 口语 R@{MAIN_K} | 最差 |")
    lines.append("|---|---|---|---|---|")
    for name in names:
        st = results[name]["by_style"]
        vals = [st.get(s, {}).get(f"recall@{MAIN_K}", float("nan")) for s in ("formal", "term", "colloquial")]
        worst = min(v for v in vals if v == v) if any(v == v for v in vals) else float("nan")
        lines.append(
            f"| `{name}` | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | **{fmt(worst)}** |"
        )
    lines.append("")

    # 按类型
    lines.append("## 按问题类型拆开\n")
    types = sorted({t for name in names for t in results[name]["by_type"]})
    lines.append("| 类型 | " + " | ".join(f"`{n}`" for n in names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for t in types:
        cells = " | ".join(fmt(results[n]["by_type"].get(t, {}).get(f"recall@{MAIN_K}", float("nan"))) for n in names)
        label = TOPICS.get(t, t)
        lines.append(f"| {label} `{t}` | {cells} |")
    lines.append("")

    # 权重扫描
    if sweep:
        lines.append("## 融合权重扫描\n")
        lines.append("`w_dense` 是稠密一路的权重，稀疏一路取 `1 - w`。w=1.0 等价纯向量，w=0.0 等价纯 BM25。")
        lines.append("**这是诊断曲线，不是调参结果。** 看的是平台宽度而不是峰值——")
        lines.append("曲线平缓说明这个参数不敏感、结论稳健；尖峰说明它依赖这批数据，不可外推。\n")
        lines.append(f"| w_dense | 总体 R@{MAIN_K} | MRR | 书面 | 术语 | 口语 |")
        lines.append("|---|---|---|---|---|---|")
        for row in sweep:
            mark = ""
            if row["w_dense"] == 1.0:
                mark = " ← 纯向量"
            elif row["w_dense"] == 0.0:
                mark = " ← 纯 BM25"
            elif row["w_dense"] == 0.5:
                mark = " ← 等权（默认）"
            lines.append(
                f"| {row['w_dense']:.1f}{mark} | {fmt(row['overall'])} | {fmt(row['mrr'])} | "
                f"{fmt(row['formal'])} | {fmt(row['term'])} | {fmt(row['colloquial'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge", choices=["bge", "hashing"])
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--no-oracle", action="store_true", help="跳过上限参考")
    ap.add_argument("--no-sweep", action="store_true", help="跳过融合权重扫描")
    args = ap.parse_args()

    t0 = time.time()
    corpus = build_corpus()
    queries = build_query_set()

    print(f"[语料] {len(corpus)} chunk（核心 {len(POINTS)} + 干扰 {len(FILLERS)}）")
    print(f"[query] {len(queries)} 条")
    print(f"[embedding] 加载 {args.embedder} ...")

    emb = get_embedder(args.embedder)
    print(f"[embedding] 就绪，dim={emb.dim}")

    dense = DenseRetriever(corpus, emb)
    sparse = SparseRetriever(corpus)
    hybrid = HybridRetriever(corpus, dense, sparse)
    hybrid_rr = RerankRetriever(corpus, hybrid, reranker=build_reranker("keyword"))

    retrievers = {
        "dense": dense,
        "sparse": sparse,
        "hybrid": hybrid,
        "hybrid+rerank": hybrid_rr,
    }
    if not args.no_oracle:
        retrievers["oracle(上限)"] = OracleRerankRetriever(corpus, hybrid, gold_lookup(queries))

    results = {}
    for name, r in retrievers.items():
        print(f"[评测] {name} ...", end=" ", flush=True)
        results[name] = evaluate(r, queries, depth=args.depth)
        o = results[name]["overall"]
        print(f"R@5={fmt(o['recall@5'])}  MRR={fmt(o['mrr'])}")

    sweep = []
    if not args.no_sweep:
        print("[扫描] 融合权重 0.0 → 1.0 ...")
        sweep = sweep_weights(corpus, dense, sparse, queries, depth=args.depth)
        for row in sweep:
            if row["w_dense"] in (0.0, 0.5, 1.0):
                print(
                    f"       w={row['w_dense']:.1f}  R@5={fmt(row['overall'])}  "
                    f"术语={fmt(row['term'])}  口语={fmt(row['colloquial'])}"
                )

    meta = {
        "n_docs": len(corpus),
        "n_core": len(POINTS),
        "n_confusable": len(CONFUSABLE_DOCS),
        "n_filler": len(FILLERS),
        "n_queries": len(queries),
        "n_emergency": sum(1 for q in queries if not q["gold"]),
        "embedder": emb.name,
        "embedder_source": getattr(emb, "source", "builtin"),
        "tokenizer": tokenizer_name(),
        "dim": emb.dim,
        "depth": args.depth,
        "elapsed": time.time() - t0,
    }

    (OUT_DIR / "results.json").write_text(
        json.dumps({"meta": meta, "results": results, "sweep": sweep}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = render_report(results, meta, sweep)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(f"\n[完成] {meta['elapsed']:.1f}s -> eval/report.md  eval/results.json\n")
    print(report)


if __name__ == "__main__":
    main()
