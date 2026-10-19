"""第 1 步：最笨的检索 —— 数重叠的字符。

不引入模型、不引入向量库，就一个问题：**给 48 段医学资料，问一句话，
怎么找出最相关的那段？**

最直接的想法：两边有多少字是重合的。重得多就更相关。

这一步的目的是先有个能跑的东西，然后看清它到底哪里不行 ——
后面每一层都是为了修一个具体的毛病，而不是为了"架构好看"。

跑法：
    python -m rebuild.step1_literal
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuild.common import K, evaluate, header, load, print_detail, print_result  # noqa: E402

DOCS, QUERIES = load()


def grams(text: str, n: int = 2) -> set[str]:
    """把一段话切成连续的 n 个字，装进集合。

    为什么用 2-gram 而不是直接切词：中文没有空格，切词要依赖分词库，
    而这一步我想尽量不用任何依赖。2-gram 是零依赖的近似。
    """
    clean = "".join(ch for ch in text if not ch.isspace())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def search(query: str, k: int = K) -> list[str]:
    """按要求：query 的 2-gram 有多大比例出现在文档里，按这个比例排序。"""
    q_grams = grams(query)
    if not q_grams:
        return []
    scored = []
    for doc in DOCS:
        overlap = len(q_grams & grams(doc["text"])) / len(q_grams)
        scored.append((overlap, doc["key"]))
    scored.sort(reverse=True)
    return [key for _, key in scored[:k]]


def show_failures(n: int = 3) -> None:
    """把排名靠后的几条挑出来看 —— 这一步的价值主要在这里。"""
    print()
    print("  几个典型失败：")
    shown = 0
    for q in QUERIES:
        if q["style"] != "colloquial" or shown >= n:
            continue
        ranked = search(q["query"])
        if q["gold"] not in ranked:
            print()
            print(f"    问：{q['query']}")
            print(f"    该答：{q['gold']}")
            print(f"    实际给出：{ranked[:2]}")
            shown += 1


def main() -> None:
    header("第 1 步：数重叠的字符", "不依赖任何模型 —— 只看字面上有多少重合")

    r = evaluate(search, QUERIES, "1. 字面重叠（2-gram）")
    print()
    print_result(r)
    print_detail(r)

    show_failures()

    print()
    print("  看出来了：书面问法还行，口语问法几乎全错。")
    print("  原因很直接 —— 口语问法「血糖高的人跑步有用吗」和正文")
    print("  「运动治疗应循序渐进」几乎没有共同的字，数重叠数不出来。")
    print()
    print("  这不是实现写得不好，是**方法本身的天花板**：字面匹配只能匹配字面。")
    print("  → 下一步：换成向量，让「意思相近」也能匹配上。")


if __name__ == "__main__":
    main()
