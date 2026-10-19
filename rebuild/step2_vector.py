"""第 2 步：换成向量 —— 让「意思相近」也能匹配上。

第 1 步的问题不是写得糙，是方法的天花板：字面匹配只能匹配字面。
「血糖高的人跑步有用吗」和正文「运动治疗应循序渐进」几乎没有共同的字，
无论怎么数字数，都连不上。

向量做的事很简单：把每句话压成一串数字（比如 512 个），
让意思相近的两句话，这两串数字在空间里靠得近。

跑法：
    python -m rebuild.step2_vector
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedders import BgeEmbedder  # noqa: E402
from rebuild.common import K, evaluate, header, load, print_detail, print_result  # noqa: E402
from rebuild.step1_literal import search as literal_search  # noqa: E402

DOCS, QUERIES = load()


def main() -> None:
    header("第 2 步：换成向量", "不再数字，改成算「意思有多近」")

    print("  加载本地向量模型（第一次会稍慢，之后有缓存）...", flush=True)
    emb = BgeEmbedder()
    doc_mat = emb.encode([d["text"] for d in DOCS])
    print(f"  模型就绪：{emb.name}，每段文本变成 {emb.dim} 个数")

    # query 只有 96 条，一次编码完。真实服务里是在线编码，
    # 这里预编码只是为了快，不影响结论。
    qvecs = {q["query"]: v for q, v in zip(QUERIES, emb.encode([q["query"] for q in QUERIES]))}
    keys = [d["key"] for d in DOCS]

    def vector_search(query: str, k: int = K) -> list[str]:
        if query not in qvecs:
            qvecs[query] = emb.encode([query])[0]
        sims = doc_mat @ qvecs[query]  # 都归一化过，点积就是余弦相似度
        order = np.argsort(-sims)[:k]
        return [keys[i] for i in order]

    r1 = evaluate(literal_search, QUERIES, "1. 字面重叠（2-gram）")
    r2 = evaluate(vector_search, QUERIES, "2. 向量（bge-small-zh）")

    print()
    print_result(r1)
    print_detail(r1)
    print_result(r2, r1)
    print_detail(r2)

    print()
    print("  口语问法这一列的变化最明显 —— 这正是向量该解决的问题：")
    print(f"    书面 {r1['recall']['formal']:.3f} → {r2['recall']['formal']:.3f}")
    print(f"    口语 {r1['recall']['colloquial']:.3f} → {r2['recall']['colloquial']:.3f}")

    # 但向量也不是万能的，这里刻意挑一个它容易犯的错
    print()
    print("  但先别急着收工。向量有一个它自己的短板：**名字很像的东西分不开**。")
    print("  比如同一主题下「用药」和「饮食」两段，语义都在讲糖尿病，向量会觉得它们都挺像。")
    probe = "糖尿病患者的饮食要注意什么"
    top = vector_search(probe, 3)
    print()
    print(f"    问：{probe}")
    print(f"    向量给出：{top}")
    print("    注意看返回的东西是不是都在同一个主题里打转 —— 主题对了，但侧面不一定最准。")
    print()
    print("  还有一些问题上，纯字面反而更强：药名、剂量、指标缩写这类，")
    print("  「格列美脲」和「格列齐特」在向量空间里几乎是同一个点，但它们是两种药。")
    print("  → 下一步：把字面那一路也留着，两路一起用。")


if __name__ == "__main__":
    main()
