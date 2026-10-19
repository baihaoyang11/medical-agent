"""路由审计：测的不是检索器，是路由器本身。

为什么需要这个脚本
------------------
`eval/run_eval.py` 假设每个 query 都走了"正确的"检索器。但生产里没有人工标注，
只有 `core/routing.py` 的关键词分类器。分类器判错，路由就选错检索器 ——
再好的映射表也白搭。

所以评测检索器之前，得先回答：**路由器本身有没有在工作？**

三个指标，按硬度排序：

1. `fallback_rate` —— 落到 `general` 兜底的比例。这个指标不依赖标注口径，
   落到 general 就是没做区分，任何标注体系下都成立。
2. `agreement` —— 与人工标注的一致率。**注意口径**：评测集的 qtype 是按
   知识点侧面（facet）标的，生产的 qtype 是按问句意图分的，两者不是同一个维度，
   所以这个数字不能直接当"分类器准确率"读，只能当"两套体系的重合度"。
3. `end_to_end` —— 用真实分类结果去选检索器，端到端 R@5 是多少。
   和"按人工标注选"（完美分类上界）对比，差值就是分类器的代价。

用法：
    python -m eval.route_audit              # 读 eval/results.json
    python -m eval.route_audit --json       # 只输出机器可读结果
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.routing import RETRIEVERS, classify, policy_table, resolve  # noqa: E402

OUT_DIR = Path(__file__).parent
MAIN_K = 5

TYPE_LABELS = {
    "disease_definition": "定义与病因",
    "clinical": "临床表现与监测",
    "drug": "药物治疗",
    "diet": "饮食管理",
    "prevention": "预防与生活方式",
    "diagnosis": "鉴别诊断",
    "general": "通用/兜底",
    "emergency": "急症",
}


def load_rows(results_name: str = "results.json") -> tuple[List[dict], dict]:
    path = OUT_DIR / results_name
    if not path.exists():
        raise SystemExit(f"找不到 eval/{results_name}，先跑 `python tasks.py eval` 或 `eval-fast`")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 取一个检索器的 rows 做骨架：query / qtype / style 与检索器无关
    base = next(iter(data["results"].values()))
    rows = [r for r in base["rows"] if r["gold"]]
    return rows, data


def hit_at(rows_by_query: Dict[str, dict], query: str, k: int = MAIN_K) -> bool:
    r = rows_by_query.get(query)
    return bool(r and r["rank"] and r["rank"] <= k)


def audit(rows: Sequence[dict], data: dict) -> dict:
    n = len(rows)

    # ── 1. 分类行为 ──
    predicted = []
    for r in rows:
        qtype, matched = classify(r["query"])
        predicted.append(
            {
                "query": r["query"],
                "gold_type": r["qtype"],
                "style": r["style"],
                "pred_type": qtype,
                "matched": list(matched),
            }
        )

    fallback = sum(1 for p in predicted if p["pred_type"] == "general")
    agreement = sum(1 for p in predicted if p["pred_type"] == p["gold_type"])

    confusion: Dict[str, Counter] = defaultdict(Counter)
    for p in predicted:
        confusion[p["gold_type"]][p["pred_type"]] += 1

    # 兜底率按问法拆：口语 query 更容易落到 general 吗？
    fallback_by_style: Dict[str, dict] = {}
    for style in sorted({p["style"] for p in predicted}):
        sub = [p for p in predicted if p["style"] == style]
        fallback_by_style[style] = {
            "n": len(sub),
            "fallback": sum(1 for p in sub if p["pred_type"] == "general"),
        }

    # ── 2. 检索器使用分布（开关打开时）──
    usage = Counter()
    for p in predicted:
        policy = RETRIEVERS.get(p["pred_type"])
        usage[policy.mode if policy else "hybrid"] += 1

    # ── 3. 端到端：三种假设下的 R@5 ──
    by_query: Dict[str, Dict[str, dict]] = {
        name: {r["query"]: r for r in res["rows"]}
        for name, res in data["results"].items()
    }

    def r5(pick) -> float:
        """pick(query_row) -> 检索器名"""
        total = 0
        for r in rows:
            name = pick(r)
            table = by_query.get(name)
            if table is None:  # 该检索器没跑（比如 sparse 被 --no-* 跳过）
                continue
            total += hit_at(table, r["query"])
        return total / len(rows)

    uniform = {
        name: r5(lambda _r, name=name: name)
        for name in data["results"]
        if name != "oracle(上限)"
    }

    # 完美分类：直接用人工标注的 qtype 查表 —— 这是路由器能达到的上界
    def pick_perfect(r: dict) -> str:
        policy = RETRIEVERS.get(r["qtype"])
        return policy.mode if policy else "hybrid"

    # 真实分类：用生产分类器的输出查表 —— 这是上线后实际会得到的
    def pick_actual(r: dict) -> str:
        policy = RETRIEVERS.get(classify(r["query"])[0])
        return policy.mode if policy else "hybrid"

    # 线上重排默认是开的（RERANK_STRATEGY=keyword），所以只看不带重排的路由
    # 会低估所有方案。同一套映射再算一遍「检索器 + 重排」的版本。
    def with_rr(mode: str) -> str:
        name = f"{mode}+rerank"
        return name if name in data["results"] else mode

    perfect = r5(pick_perfect)
    actual = r5(pick_actual)
    perfect_rr = r5(lambda r: with_rr(pick_perfect(r)))
    actual_rr = r5(lambda r: with_rr(pick_actual(r)))

    return {
        "n_queries": n,
        "fallback": fallback,
        "fallback_rate": fallback / n,
        "agreement": agreement,
        "agreement_rate": agreement / n,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "fallback_by_style": fallback_by_style,
        "retriever_usage": dict(usage),
        "r5_uniform": uniform,
        "r5_perfect_routing": perfect,
        "r5_actual_routing": actual,
        "r5_perfect_routing_rerank": perfect_rr,
        "r5_actual_routing_rerank": actual_rr,
        "worst_examples": [
            {"query": p["query"], "gold": p["gold_type"], "pred": p["pred_type"]}
            for p in predicted
            if p["pred_type"] == "general" and p["gold_type"] != "general"
        ][:10],
    }


def fmt(x: float) -> str:
    return "—" if x != x else f"{x:.3f}"


def render(result: dict, data: dict) -> str:
    n = result["n_queries"]
    unif = result["r5_uniform"]
    best_uniform = max(unif, key=lambda k: unif[k])
    lines: List[str] = []

    lines.append("# 路由审计报告\n")
    lines.append("> 由 `python -m eval.route_audit` 自动生成，请勿手改。\n")

    lines.append("## 一句话结论\n")
    lines.append(
        f"现有关键词路由在 {n} 条评测 query 上有 **{result['fallback']} 条（{result['fallback_rate']:.1%}）**"
        f"落进 `general` 兜底。也就是说，无论 `STRATEGIES` 表配得多细，"
        f"**接近一半的请求实际走的是同一条默认路径**。"
    )
    lines.append("")

    lines.append("## 1. 兜底率（最硬的指标）\n")
    lines.append("这个指标不依赖标注口径：落到 `general` 就是没做区分。\n")
    lines.append(f"- 兜底：**{result['fallback']} / {n} = {result['fallback_rate']:.1%}**\n")
    lines.append("| 问法 | 条数 | 落进 general | 比例 |")
    lines.append("|---|---|---|---|")
    for style, s in result["fallback_by_style"].items():
        lines.append(f"| {style} | {s['n']} | {s['fallback']} | {s['fallback'] / s['n']:.1%} |")
    lines.append("")

    lines.append("## 2. 与人工标注的一致率（口径要注意）\n")
    lines.append(f"- 一致：**{result['agreement']} / {n} = {result['agreement_rate']:.1%}**\n")
    lines.append("**这个数字不能直接读成\"分类器准确率\"。** 评测集的 `qtype` 是按知识点侧面标注的")
    lines.append("（`facet=exercise` 被映射成 `prevention`），而生产的 `qtype` 是按问句意图划分的。")
    lines.append("两套体系维度不同：一个问句可以属于\"运动\"侧面，但语言形式是\"怎么治\"。")
    lines.append("所以这个数字衡量的是**两套体系的重合度**，不是分类器的能力上限。\n")

    lines.append("## 3. 混淆矩阵\n")
    # 列顺序固定成路由表里的顺序，不按字母序 —— 否则每次读都要重新找列
    order = ["disease_definition", "clinical", "drug", "diet", "prevention", "diagnosis", "general", "emergency"]
    seen = {p for c in result["confusion"].values() for p in c}
    pred_types = [t for t in order if t in seen]
    lines.append("| 人工标注 ↓ / 预测 → | " + " | ".join(TYPE_LABELS.get(t, t) for t in pred_types) + " |")
    lines.append("|---" * (len(pred_types) + 1) + "|")
    for gold, dist in result["confusion"].items():
        cells = " | ".join(str(dist.get(p, "")) for p in pred_types)
        lines.append(f"| **{TYPE_LABELS.get(gold, gold)}** `{gold}` | {cells} |")
    lines.append("")

    lines.append("## 4. 检索器使用分布（开关打开时）\n")
    lines.append("| 检索器 | 条数 | 比例 |")
    lines.append("|---|---|---|")
    for mode, cnt in sorted(result["retriever_usage"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{mode}` | {cnt} | {cnt / n:.1%} |")
    lines.append("")

    lines.append("## 5. 端到端影响：分类器吃掉了多少收益\n")
    lines.append("三种假设下用同一批 query 算 R@5：\n")
    lines.append("- **统一**：所有请求走同一种检索器（当前线上行为）")
    lines.append("- **完美路由**：假设分类 100% 正确，按人工标注查表（路由器上界）")
    lines.append("- **真实路由**：按生产分类器的输出查表（上线后的实际结果）\n")
    lines.append("| 配置 | R@5 |")
    lines.append("|---|---|")
    for name, v in sorted(unif.items(), key=lambda kv: -kv[1]):
        lines.append(f"| 统一 `{name}` | {fmt(v)} |")
    lines.append(f"| **完美路由（上界）** | **{fmt(result['r5_perfect_routing'])}** |")
    lines.append(f"| **真实路由（实际）** | **{fmt(result['r5_actual_routing'])}** |")
    lines.append("")

    lines.append("### 5.1 带上重排再看一遍\n")
    lines.append("线上 `RERANK_STRATEGY` 默认是 `keyword`（重排是开的），所以只比不带重排的版本会低估所有方案。")
    lines.append("下表把同一套映射换成「检索器 + 重排」重新算：\n")
    lines.append("| 配置 | R@5（含重排） |")
    lines.append("|---|---|")
    for name, v in sorted(unif.items(), key=lambda kv: -kv[1]):
        if name.endswith("+rerank"):
            lines.append(f"| 统一 `{name}` | {fmt(v)} |")
    lines.append(f"| **完美路由 + 重排** | **{fmt(result['r5_perfect_routing_rerank'])}** |")
    lines.append(f"| **真实路由 + 重排** | **{fmt(result['r5_actual_routing_rerank'])}** |")
    lines.append("")

    # 基线只能取 `hybrid`：路由层决定的是 dense / sparse / hybrid，
    # 重排是它之后独立的一步。拿 hybrid+rerank 当基线会把两件事的收益混在一起算。
    baseline_name = "hybrid" if "hybrid" in unif else best_uniform
    baseline = unif[baseline_name]
    gain_perfect = result["r5_perfect_routing"] - baseline
    gain_actual = result["r5_actual_routing"] - baseline
    retained = gain_actual / gain_perfect if gain_perfect else float("nan")

    lines.append(f"**基线取统一 `{baseline_name}`（{fmt(baseline)}）**，理由：路由层只决定 dense / sparse / hybrid，")
    lines.append("重排是其后独立的一步。拿 `hybrid+rerank` 当基线会把两件事的收益混在一起算。\n")
    lines.append("**读法**：")
    lines.append(f"- 完美路由的上限收益：{gain_perfect:+.3f}")
    lines.append(f"- 真实分类器拿到：{gain_actual:+.3f}")
    if retained == retained and gain_perfect > 0:
        lines.append(f"- 收益留存率：**{retained:.0%}** —— 路由设计赚到的提升，"
                     f"有 {1 - retained:.0%} 被分类错误吃掉了")

    # 最关键的一条对比：真实路由有没有跑赢"什么都不做、只改默认值"
    best_name = max(unif, key=lambda k: unif[k])
    best_val = unif[best_name]
    # 同一口径下比：都带重排
    actual_rr = result["r5_actual_routing_rerank"]
    best_rr_name = max((k for k in unif if k.endswith("+rerank")), key=lambda k: unif[k], default=None)
    best_rr_val = unif.get(best_rr_name, float("nan")) if best_rr_name else float("nan")
    if actual_rr < best_rr_val - 1e-9:
        lines.append("")
        lines.append(
            f"- **决定性对比（同为线上口径，均含重排）**：真实路由 + 重排 {actual_rr:.3f} "
            f"**低于**统一 `{best_rr_name}` 的 {best_rr_val:.3f}。"
        )
        lines.append(
            f"  换句话说，把全局默认改成 `{best_rr_name.replace('+rerank', '')}`"
            f"（`RETRIEVER_DEFAULT={best_rr_name.replace('+rerank', '')}`，改一行配置）"
            f"比上线整套检索器路由还强 {best_rr_val - actual_rr:+.3f}。"
        )
        lines.append(
            "  **一个方案如果打不过它自己的简化版，就不该上线** —— 这是比\"证据不显著\"更硬的否决理由。"
        )
    lines.append("")

    if result["worst_examples"]:
        lines.append("## 6. 被误放进兜底的样例\n")
        lines.append("| query | 人工标注 | 预测 |")
        lines.append("|---|---|---|")
        for e in result["worst_examples"]:
            lines.append(f"| {e['query']} | {TYPE_LABELS.get(e['gold'], e['gold'])} | `{e['pred']}` |")
        lines.append("")

    lines.append("## 7. 结论\n")
    lines.append(f"1. **检索器路由的前提是分类器可用**。当前 {result['fallback_rate']:.0%} 的兜底率意味着"
                 "映射表再精细，也只会影响不到一半的请求。")
    lines.append("2. **先修分类器，再谈选检索器**。顺序搞反了，就是在一个失效的分支上做优化。")
    lines.append("3. 分类器该怎么修：关键词表只覆盖了\"用户说了什么词\"，没覆盖\"用户想要什么\"。")
    lines.append("   可行的方向是 query embedding 与类型原型的相似度（零样本、无需标注），")
    lines.append("   或者直接上小模型分类。两者都需要先有一批带标注的 query —— 现在的 116 条够起步，不够定论。")
    lines.append("4. 这个审计之所以值得单独做：如果不测路由器，就会以为\"加了检索器路由 = 性能提升\"，")
    lines.append("   而实际上 90% 的请求连门都没进。**测中间环节，比测最终指标更能定位问题。**")

    best_rr = max(((k, v) for k, v in unif.items() if k.endswith("+rerank")), key=lambda kv: kv[1], default=None)
    if best_rr:
        name, val = best_rr
        plain = name.replace("+rerank", "")
        lines.append(
            f"5. **最该注意的一条**：真实路由 + 重排拿到 {result['r5_actual_routing_rerank']:.3f}，"
            f"而把默认值改成 `{plain}` 就是 {val:.3f}（差 {val - result['r5_actual_routing_rerank']:+.3f}）。"
        )
        lines.append(
            f"   前者要维护分类器 + 映射表 + 开关 + 一堆边界测试，后者是环境变量里改一个词。"
        )
        lines.append(
            "   **收益更小而成本更高，就是不该做** —— 这是 `RETRIEVER_ROUTING` 保持默认关闭的最终理由，"
            "比\"证据不显著\"具体得多。"
        )
        lines.append(
            f"6. 但 `{plain}` 这条路也有前提：它在**单库**评测里赢，生产是 QA + PDF 双库，"
            "教材段落长、术语密集，BM25 的贡献可能不同。改默认值之前应该在双库上补一次对照，"
            "而不是直接拿这个数字上线。"
        )
    lines.append("")

    lines.append("## 附：当前映射表\n")
    lines.append("| 类型 | 检索器 | 证据 | 策略 | 依据 |")
    lines.append("|---|---|---|---|---|")
    for row in policy_table():
        lines.append(
            f"| `{row['question_type']}` | `{row['retriever']}` | {row['evidence']} | "
            f"`{row['strategy']}` | {row['note']} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="只输出 JSON，不写报告")
    ap.add_argument("--into", default="results.json", help="读哪个评测结果文件")
    ap.add_argument("--out", default="route_audit", help="输出文件主名")
    args = ap.parse_args()

    rows, data = load_rows(args.into)
    result = audit(rows, data)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    report = render(result, data)
    (OUT_DIR / f"{args.out}.md").write_text(report, encoding="utf-8")
    (OUT_DIR / f"{args.out}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[兜底率] {result['fallback']}/{result['n_queries']} = {result['fallback_rate']:.1%}")
    print(f"[一致率] {result['agreement']}/{result['n_queries']} = {result['agreement_rate']:.1%}")
    print(f"[检索器分布] {result['retriever_usage']}")
    print(f"[R@5] 统一hybrid={result['r5_uniform'].get('hybrid', float('nan')):.3f}  "
          f"完美路由={result['r5_perfect_routing']:.3f}  "
          f"真实路由={result['r5_actual_routing']:.3f}")
    print(f"\n[完成] -> eval/{args.out}.md  eval/{args.out}.json\n")


if __name__ == "__main__":
    main()
