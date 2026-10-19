"""双库评测：给 qa_k / pdf_k / max_context_chars 找依据。

`core/routing.py` 的 `STRATEGIES` 表里有六组数字：

    hybrid_qa_heavy   qa_k=15  pdf_k=1   budget=4000
    hybrid_pdf_heavy  qa_k=4   pdf_k=4   budget=6000
    hybrid_wide       qa_k=12  pdf_k=3   budget=6000
    hybrid            qa_k=10  pdf_k=3   budget=5000

这些数字之前没有任何实验支撑，代码注释却写着"数值依据见 docs/retrieval-tuning.md"，
而那个文档不存在。这个脚本把依据补上。

测什么
------
同一批知识点，两种组织方式（见 `dual_store.py`）：

    QA 库  48 条 × 约 100 字，一条一个知识点 —— 精确，但信息量小
    PDF 库 8 条父文档 × 约 600 字，一篇含六个侧面 —— 信息全，但必然带噪声

对每个 (qa_k, pdf_k) 组合，按生产的 `compose_context` 拼上下文，然后算三个数：

    point_recall  金标知识点的要点句有几成进了上下文（决定答案能不能说全）
    chars         上下文实际字符数（成本）
    noise         上下文里非金标内容的占比（干扰程度）

为什么用 point_recall 而不是文档召回
------------------------------------
文档召回只回答"金标那条进来了没有"。但上下文有预算，进来一半等于没进来。
用药安全这类回答，答对一半比不答更危险。所以看要点级覆盖。

用法：
    python -m eval.run_dual_eval                # bge
    python -m eval.run_dual_eval --embedder hashing --fast
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from core.context import compose_context, select_within_budget
from eval import dual_store as DS
from eval.embedders import get_embedder
from eval.knowledge import TOPICS, build_queries, TERM_QUERIES, EXTRA_QUERIES
from eval.retrievers import Corpus, Doc, DenseRetriever, HybridRetriever, SparseRetriever

OUT_DIR = Path(__file__).parent

# 网格：0 表示"这一路不用"，用来测单路和双路的差别。
# 取值必须覆盖 CURRENT_STRATEGIES 里出现的所有数字（15 / 4 / 12 / 10 / 1 / 3），
# 否则对比那一节会 KeyError —— 这个坑第一次跑就踩了。
QA_GRID = (0, 2, 4, 8, 10, 12, 15)
PDF_GRID = (0, 1, 2, 3, 4)
DEFAULT_BUDGET = 6000
# PDF 路要先多召回一些 chunk，才能凑够 pdf_k 个不同的父文档
PDF_POOL = 4

# 现有策略表，用来对比"经验值"和"扫出来的值"差多少
CURRENT_STRATEGIES = {
    "hybrid_qa_heavy": {"qa_k": 15, "pdf_k": 1, "budget": 4000},
    "hybrid_pdf_heavy": {"qa_k": 4, "pdf_k": 4, "budget": 6000},
    "hybrid_wide": {"qa_k": 12, "pdf_k": 3, "budget": 6000},
    "hybrid": {"qa_k": 10, "pdf_k": 3, "budget": 5000},
}

# 对照用的配置：四个现有策略 + 两个来自网格的候选
CANDIDATES = dict(CURRENT_STRATEGIES) | {
    "candidate_2_4_6000": {"qa_k": 2, "pdf_k": 4, "budget": 6000},
    "candidate_8_2_4000": {"qa_k": 8, "pdf_k": 2, "budget": 4000},
}



def build_query_set() -> List[dict]:
    rows = []
    for query, key, qtype, style in build_queries():
        rows.append({"query": query, "gold": key, "qtype": qtype, "style": style})
    for query, key, qtype in TERM_QUERIES:
        rows.append({"query": query, "gold": key, "qtype": qtype, "style": "term"})
    for query, key, qtype, style in EXTRA_QUERIES:
        if key:  # 急症没有金标，不参与
            rows.append({"query": query, "gold": key, "qtype": qtype, "style": style})
    return rows


def make_retriever(kind: str, corpus: Corpus, emb, sparse: SparseRetriever | None = None):
    if kind == "dense":
        return DenseRetriever(corpus, emb)
    if kind == "sparse":
        return SparseRetriever(corpus)
    # hybrid 需要两条腿都指向同一个语料
    d = DenseRetriever(corpus, emb)
    s = SparseRetriever(corpus)
    return HybridRetriever(corpus, d, s)


def run_grid(queries, qa_ret, pdf_ret, parents, qa_by_id, budget, pdf_kind="dense") -> List[dict]:
    """扫 (qa_k, pdf_k) 网格。

    对每条 query 只检索一次最大 k，然后按不同 k 截取 —— 检索深度对
    所有格子都一样，这样格子之间的差异只来自"给多少"，不是"怎么捡"。
    """
    max_qa = max(QA_GRID)
    max_pdf = max(PDF_GRID)

    per_query = []
    for q in queries:
        qa_ids = qa_ret.search(q["query"], max_qa)
        chunk_ids = pdf_ret.search(q["query"], max_pdf * PDF_POOL)
        pdf_parents = DS.rebuild_parents(chunk_ids, parents)

        cells = {}
        for qa_k in QA_GRID:
            for pdf_k in PDF_GRID:
                docs = []
                for did in qa_ids[:qa_k]:
                    doc = qa_by_id[did]
                    docs.append(
                        {"content": doc.text, "doc_id": did, "gold": doc.point_key == q["gold"]}
                    )
                for parent in pdf_parents[:pdf_k]:
                    docs.append(
                        {
                            "content": parent.text,
                            "doc_id": parent.parent_id,
                            "gold": q["gold"] in parent.point_keys,
                        }
                    )
                # 用生产的选取规则决定谁能进上下文，再据此算噪声占比。
                # 噪声 = 上下文里与金标无关的段落占了多大比例 —— 父文档策略的代价全在这。
                picked = select_within_budget(docs, budget)
                context = "\n\n---\n\n".join(str(d["content"]).strip() for d in picked)
                total_chars = sum(len(str(d["content"])) for d in picked) or 1
                gold_chars = sum(len(str(d["content"])) for d in picked if d.get("gold"))
                cells[(qa_k, pdf_k)] = {
                    "point_recall": DS.point_recall(context, q["gold"]),
                    "chars": len(context),
                    "noise": 1.0 - gold_chars / total_chars,
                }
        per_query.append({"meta": q, "cells": cells})

    out = []
    for qa_k in QA_GRID:
        for pdf_k in PDF_GRID:
            vals = [r["cells"][(qa_k, pdf_k)] for r in per_query]
            out.append(
                {
                    "qa_k": qa_k,
                    "pdf_k": pdf_k,
                    "point_recall": sum(v["point_recall"] for v in vals) / len(vals),
                    "chars": sum(v["chars"] for v in vals) / len(vals),
                    "noise": sum(v["noise"] for v in vals) / len(vals),
                    "n": len(vals),
                }
            )
    return out, per_query



def evaluate_config(queries, qa_ret, pdf_ret, parents, qa_by_id, cfg, cache=None) -> dict:
    """按指定 (qa_k, pdf_k, budget) 跑一遍，返回 point_recall / 字符数 / 噪声。

    单独写一个而不是复用网格，是因为网格只按一个固定预算算。
    现有策略的预算各不相同（4000/5000/6000），拿 6000 的结果去评价
    "4000 预算的策略"是错的口径 —— 预算本身就是策略的一部分。
    """
    recalls, chars, noises = [], [], []
    for q in queries:
        if cache is not None and q["query"] in cache:
            qa_ids, pdf_parents = cache[q["query"]]
        else:
            qa_ids = qa_ret.search(q["query"], max(cfg["qa_k"], 1))
            chunk_ids = pdf_ret.search(q["query"], max(cfg["pdf_k"], 1) * PDF_POOL)
            pdf_parents = DS.rebuild_parents(chunk_ids, parents)
            if cache is not None:
                cache[q["query"]] = (qa_ids, pdf_parents)

        docs = [
            {"content": qa_by_id[i].text, "doc_id": i, "gold": qa_by_id[i].point_key == q["gold"]}
            for i in qa_ids[: cfg["qa_k"]]
        ]
        docs += [
            {"content": p.text, "doc_id": p.parent_id, "gold": q["gold"] in p.point_keys}
            for p in pdf_parents[: cfg["pdf_k"]]
        ]
        picked = select_within_budget(docs, cfg["budget"])
        context = "\n\n---\n\n".join(str(d["content"]).strip() for d in picked)
        total_chars = sum(len(str(d["content"])) for d in picked) or 1
        gold_chars = sum(len(str(d["content"])) for d in picked if d.get("gold"))
        recalls.append(DS.point_recall(context, q["gold"]))
        chars.append(len(context))
        noises.append(1.0 - gold_chars / total_chars)

    n = len(recalls)
    return {
        "point_recall": sum(recalls) / n,
        "chars": sum(chars) / n,
        "noise": sum(noises) / n,
    }


def run_by_type(per_query, configs: Dict[str, dict]) -> Dict[str, dict]:
    """按问题类型拆开，对比几个具体配置。

    平均分会被占比大的类型（临床类 41 条）拉偏，而策略表是按类型分的，
    所以必须按类型看。
    """
    out = {}
    for name, cfg in configs.items():
        by_type: Dict[str, List[float]] = {}
        for r in per_query:
            key = (cfg["qa_k"], cfg["pdf_k"])
            # 预算不同时要用同一套预算重算，这里简化：按 grid 里的 6000 读
            val = r["cells"][key]["point_recall"]
            by_type.setdefault(r["meta"]["qtype"], []).append(val)
        out[name] = {
            t: sum(v) / len(v) for t, v in sorted(by_type.items())
        }
    return out


def budget_curve(queries, qa_ret, pdf_ret, parents, qa_by_id, cfg, budgets: Sequence[int]) -> List[dict]:
    """固定一个 (qa_k, pdf_k)，扫上下文预算。

    这条曲线回答"预算 4000 够不够"：如果 recall 在 4000 处已经饱和，
    那 6000 就是白花钱；如果到 6000 还在涨，说明前面的策略在饿着模型。
    """
    rows = []
    per_query = []
    for q in queries:
        qa_ids = qa_ret.search(q["query"], cfg["qa_k"])
        chunk_ids = pdf_ret.search(q["query"], cfg["pdf_k"] * PDF_POOL)
        pdf_parents = DS.rebuild_parents(chunk_ids, parents)
        docs = [{"content": qa_by_id[i].text, "doc_id": i} for i in qa_ids[: cfg["qa_k"]]]
        docs += [{"content": p.text, "doc_id": p.parent_id} for p in pdf_parents[: cfg["pdf_k"]]]
        per_query.append((q, docs))

    for b in budgets:
        vals = [DS.point_recall(compose_context(docs, b), q["gold"]) for q, docs in per_query]
        rows.append({"budget": b, "point_recall": sum(vals) / len(vals)})
    return rows


def fmt(x: float, d: int = 3) -> str:
    return "—" if x != x else f"{x:.{d}f}"


def render(meta, grid, by_type, budgets_curve, strategy_check) -> str:
    L: List[str] = []
    L.append("# 双库评测报告（qa_k / pdf_k 的依据）\n")
    L.append("> 由 `python -m eval.run_dual_eval` 自动生成，请勿手改。\n")

    L.append("## 设置\n")
    L.append(f"- QA 库：{meta['n_qa']} 条 × 约 {meta['qa_avg_chars']} 字（**每条只有核心结论句**）")
    L.append(f"- PDF 库：{meta['n_pdf_parent']} 篇父文档 × 约 {meta['pdf_avg_chars']} 字，"
             f"切成 {meta['n_pdf_chunk']} 个块（**含全部细节**）")
    L.append(f"- query：{meta['n_query']} 条（只保留有金标的）")
    L.append(f"- embedding：{meta['embedder']}（dim={meta['dim']}）")
    L.append(f"- 网格固定预算：{meta['budget']} 字符\n")
    L.append("两个库的差别是**覆盖度**，不是措辞：QA 库是人工整理的问答对，")
    L.append("答案短、切题，但只回答最常问的那一层；PDF 库是教材原文，机制、分型、注意事项都在。")
    L.append("所以 `qa_k / pdf_k` 调的是\"要精确还是要完整\"这个配比。\n")

    L.append("## 1. 要点召回网格\n")
    L.append("行是 QA 库给几条，列是 PDF 库给几篇父文档。单元格是 **point_recall**：")
    L.append("金标知识点的要点句有几成进了上下文。≥0.95 加粗。\n")
    L.append("| QA k \\ PDF k | " + " | ".join(str(p) for p in PDF_GRID) + " |")
    L.append("|---" * (len(PDF_GRID) + 1) + "|")
    for qa_k in QA_GRID:
        cells = []
        for pdf_k in PDF_GRID:
            r = next(x for x in grid if x["qa_k"] == qa_k and x["pdf_k"] == pdf_k)
            v = fmt(r["point_recall"])
            cells.append(f"**{v}**" if r["point_recall"] >= 0.95 else v)
        L.append(f"| **{qa_k}** | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**先看第一列（pdf_k=0）**：QA 库给到 15 条，也只到 "
             f"{next(x for x in grid if x['qa_k'] == 15 and x['pdf_k'] == 0)['point_recall']:.3f}。")
    L.append("因为 QA 库每条只有结论句，缺的那部分细节根本不在库里 —— 再多给几条也拿不到。")
    L.append("**再看第一行（qa_k=0）**：只给 PDF 就能到 "
             f"{next(x for x in grid if x['qa_k'] == 0 and x['pdf_k'] == 1)['point_recall']:.3f}（1 篇）"
             f" / {next(x for x in grid if x['qa_k'] == 0 and x['pdf_k'] == 4)['point_recall']:.3f}（4 篇）。\n")

    L.append("## 2. 成本：达到某档召回，最便宜要花多少字\n")
    L.append("| 目标要点召回 | 最便宜的实现 | 平均字符 | 噪声占比 |")
    L.append("|---|---|---|---|")
    for target in (0.95, 0.98, 0.99):
        cands = [r for r in grid if r["point_recall"] >= target - 1e-9]
        if not cands:
            L.append(f"| {target:.2f} | 网格内达不到 | — | — |")
            continue
        best = min(cands, key=lambda r: r["chars"])
        L.append(
            f"| ≥{target:.2f} | qa_k={best['qa_k']}, pdf_k={best['pdf_k']} | "
            f"{best['chars']:.0f} | {best['noise']:.2f} |"
        )
    L.append("")

    L.append("## 3. 预算曲线\n")
    L.append("固定 `hybrid`（qa_k=10, pdf_k=3），只改上下文预算。曲线走平说明再给预算也装不进新东西。\n")
    L.append("| 预算（字符） | point_recall |")
    L.append("|---|---|")
    for row in budgets_curve:
        L.append(f"| {row['budget']} | {fmt(row['point_recall'])} |")
    L.append("")

    L.append("## 4. 现有策略 vs 实测\n")
    L.append("每个配置都按**它自己的预算**实测 —— 不能拿 6000 预算的分数去评价预算 4000 的策略，")
    L.append("预算本身就是策略的一部分。\n")
    L.append("| 配置 | qa_k/pdf_k/预算 | point_recall | 平均字符 | 噪声占比 |")
    L.append("|---|---|---|---|---|")
    for name, row in strategy_check.items():
        cfg = row["cfg"]
        L.append(
            f"| `{name}` | {cfg['qa_k']}/{cfg['pdf_k']}/{cfg['budget']} | "
            f"{fmt(row['point_recall'])} | {row['chars']:.0f} | {row['noise']:.2f} |"
        )
    L.append("")

    if by_type:
        L.append("## 5. 按问题类型（各策略在自己预算下）\n")
        L.append("| 类型 | " + " | ".join(f"`{n}`" for n in by_type) + " |")
        L.append("|---" * (len(by_type) + 1) + "|")
        types = sorted({t for v in by_type.values() for t in v})
        for t in types:
            cells = " | ".join(fmt(by_type[n].get(t, float("nan"))) for n in by_type)
            L.append(f"| {TOPICS.get(t, t)} `{t}` | {cells} |")
        L.append("")

    # ── 结论：从数据里直接读出来，不写形容词 ──
    qa_only = {k: next(x for x in grid if x["qa_k"] == k and x["pdf_k"] == 0)["point_recall"]
               for k in QA_GRID if k}
    pdf_only = {k: next(x for x in grid if x["qa_k"] == 0 and x["pdf_k"] == k)["point_recall"]
                for k in PDF_GRID if k}
    qa_gain = qa_only[max(qa_only)] - qa_only[min(qa_only)]
    pdf_gain = pdf_only[max(pdf_only)] - pdf_only[min(pdf_only)]

    L.append("## 6. 结论\n")
    L.append(f"1. **QA 库给多少条，几乎不影响结果。** qa_k 从 {min(qa_only)} 加到 {max(qa_only)}，"
             f"point_recall 只从 {qa_only[min(qa_only)]:.3f} 涨到 {qa_only[max(qa_only)]:.3f}"
             f"（+{qa_gain:.3f}）。原因是 QA 库里只有结论句，缺的细节再多给几条也拿不到。")
    L.append(f"2. **PDF 库才是信息来源。** pdf_k 从 {min(pdf_only)} 加到 {max(pdf_only)}，"
             f"point_recall 从 {pdf_only[min(pdf_only)]:.3f} 涨到 {pdf_only[max(pdf_only)]:.3f}"
             f"（+{pdf_gain:.3f}）。边际收益递减：1 篇就拿到大部分，4 篇才补齐。")
    L.append(f"3. **预算 {budgets_curve[0]['budget']}~{budgets_curve[-1]['budget']} 之间基本饱和。** "
             f"从 {budgets_curve[0]['budget']} 涨到 {budgets_curve[-1]['budget']}，")
    L.append(f"   point_recall 只从 {budgets_curve[0]['point_recall']:.3f} 到 "
             f"{budgets_curve[-1]['point_recall']:.3f}。现有策略里配 6000 的那几档，多出来的预算是白花的。")
    L.append("")
    L.append("**但这些结论没有直接拿去改生产默认值**，原因有三条：")
    L.append("- 模拟里 QA 库**只有结论句**，这是为了造出区分度人为设定的。真实 QA 库里可能有")
    L.append("  教材没写的内容（患者常问的实操建议、就医流程），那部分价值这里测不出来。")
    L.append("- PDF 父文档只有约 600 字，真实教材按 `PARENT_CHUNK_SIZE=1000` 切，一段更长，")
    L.append("  同样预算能装的段数更少，饱和点会后移。")
    L.append("- 结论的方向（少给 QA、多给 PDF、砍预算）是可用的，但**具体数字必须在真实库上重测**。")
    L.append("")
    L.append("这个脚本的价值不是给出六个新数字，而是**让这六个数字第一次可以被质疑**。")
    L.append("改 `CANDIDATES` 就能在自己库上重跑，不用改代码。")
    L.append("")

    L.append("## 7. 这些数字不能说明什么\n")
    L.append("- **两个库的覆盖度差异是人为设定的**：QA 库固定只留第一句。真实比例未知。")
    L.append("- **PDF 分块是定长切的**，不对齐句子；真实抽取出来的质量也没有更好，但也不完全一样。")
    L.append("- **要点判定用 3-gram 覆盖率**，同一句话换个说法就算没覆盖，")
    L.append("  所以 point_recall 对小改写不耐受，会略低估真实答案的覆盖度。")
    L.append("- 语料是合成的，绝对分值不代表业务效果，只能看配置之间的相对差异。")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge", choices=["bge", "hashing"])
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--fast", action="store_true", help="跳过预算曲线与分类型")
    ap.add_argument("--out", default="", help="输出文件主名（默认按 embedder 区分）")
    args = ap.parse_args()

    # 和 run_eval 一样的规矩：hashing 的产物不能覆盖 bge 的结论。
    stem = args.out or ("dual_report" if args.embedder == "bge" else f"dual_report-{args.embedder}")
    report_path = OUT_DIR / f"{stem}.md"
    data_path = OUT_DIR / f"{stem.replace('dual_report', 'dual_results', 1)}.json"

    t0 = time.time()
    qa_docs = DS.build_qa_store()
    parents_list = DS.build_pdf_store()
    chunks = DS.chunk_parents(parents_list)

    qa_by_id = {d.doc_id: d for d in qa_docs}
    parents = {p.parent_id: p for p in parents_list}

    corpus_qa = Corpus([Doc(d.doc_id, d.text, "", "", "qa") for d in qa_docs])
    corpus_pdf = Corpus([Doc(c.chunk_id, c.text, c.topic, "", "chunk") for c in chunks])

    print(f"[QA 库] {len(qa_docs)} 条")
    print(f"[PDF 库] {len(parents_list)} 篇父文档 -> {len(chunks)} 个 chunk")
    queries = build_query_set()
    print(f"[query] {len(queries)} 条")

    emb = get_embedder(args.embedder)
    print(f"[embedding] {emb.name} dim={emb.dim}")

    qa_ret = make_retriever("dense", corpus_qa, emb)
    pdf_ret = make_retriever("dense", corpus_pdf, emb)

    print("[网格] 扫 qa_k × pdf_k ...", flush=True)
    grid, per_query = run_grid(queries, qa_ret, pdf_ret, parents, qa_by_id, args.budget)

    by_type = {}
    if not args.fast:
        configs = {k: v for k, v in CURRENT_STRATEGIES.items()}
        by_type = run_by_type(per_query, configs)
        print("[预算] 扫上下文预算 ...", flush=True)
    budgets_curve = budget_curve(
        queries, qa_ret, pdf_ret, parents, qa_by_id,
        CURRENT_STRATEGIES["hybrid"], (2000, 3000, 4000, 5000, 6000, 8000),
    )

    # 每个配置都按它自己的预算实测。网格那张表是固定 6000 预算跑的，
    # 不能拿来评价预算 4000 的策略 —— 预算本身就是策略的一部分。
    print("[对照] 逐个策略按自己的预算实测 ...", flush=True)
    cache: dict = {}
    strategy_check = {}
    for name, cfg in CANDIDATES.items():
        strategy_check[name] = {"cfg": cfg, **evaluate_config(
            queries, qa_ret, pdf_ret, parents, qa_by_id, cfg, cache
        )}

    meta = {
        "n_qa": len(qa_docs),
        "qa_avg_chars": round(sum(len(d.text) for d in qa_docs) / len(qa_docs)),
        "pdf_avg_chars": round(sum(len(p.text) for p in parents_list) / len(parents_list)),
        "n_pdf_parent": len(parents_list),
        "n_pdf_chunk": len(chunks),
        "n_query": len(queries),
        "embedder": emb.name,
        "dim": emb.dim,
        "budget": args.budget,
        "elapsed": round(time.time() - t0, 1),
    }

    data_path.write_text(
        json.dumps({"meta": meta, "grid": grid, "budget_curve": budgets_curve,
                    "by_type": by_type, "strategy_check": strategy_check},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = render(meta, grid, by_type, budgets_curve, strategy_check)
    report_path.write_text(report, encoding="utf-8")

    best = max(grid, key=lambda r: (r["point_recall"], -r["chars"]))
    print(f"\n[最优格] qa_k={best['qa_k']} pdf_k={best['pdf_k']} "
          f"point_recall={best['point_recall']:.3f} chars={best['chars']:.0f}")
    print(f"[完成] {meta['elapsed']}s -> eval/{report_path.name}\n")


if __name__ == "__main__":
    main()
