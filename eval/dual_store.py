"""双库模拟：把知识点用两种粒度组织，模拟生产的 QA 库 + PDF 库。

为什么要这个
------------
`core/routing.py` 的 `STRATEGIES` 表给每种问题类型分配了不同的 `qa_k / pdf_k`
（比如药物类 qa_k=15/pdf_k=1，定义类 qa_k=4/pdf_k=4）。但单库评测
（`run_eval.py`）根本测不了这两个数字——它只有一个库。

所以这两个数字一直是经验值。代码注释里还写着"数值依据见 docs/retrieval-tuning.md"，
而那个文档不存在。这是硬伤：等于承认了有个依据，但拿不出来。

这个模块把"双库"造出来，让那两个数字第一次可以被测量。

两个库的差别
------------
- **QA 库**：48 条，一条 = 一个知识点的**核心结论句**（约 100 字）。
  对应生产里人工整理的问答对：**精度高、切题**，但只回答最常问的那一层。
- **PDF 库**：8 条父文档，一条 = 一个主题下 6 个侧面完整正文（约 600 字），
  内部按 200 字切 chunk。
  对应生产里的教材段落：**内容全**（机制、分型、注意事项都有），
  但检索命中任意一个 chunk 就要把整段父文档返回，**必然带进其余 5 个侧面当噪声**。

所以这是一个真实的权衡：**QA 路精确但只给结论，PDF 路完整但一段塞六件事**。
`qa_k / pdf_k` 就是在调这个配比。

第一版做错了：两个库装的是完全相同的正文。那样细粒度必然赢
（同样内容，一条讲一件事当然比一条讲六件事好），整张网格全是 1.000，
一点区分度都没有。把 QA 库砍成结论句之后才测得出东西。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from eval.knowledge import POINTS, TOPICS

CHUNK_SIZE = 200


@dataclass(frozen=True)
class QaDoc:
    """QA 库的一条：一个知识点一条。"""

    doc_id: str
    text: str
    point_key: str


@dataclass(frozen=True)
class PdfParent:
    """PDF 库的一条父文档：一个主题下六个侧面合并。"""

    parent_id: str
    topic: str
    text: str
    point_keys: tuple[str, ...]


@dataclass(frozen=True)
class PdfChunk:
    """父文档切出来的子块，检索命中的是这个。"""

    chunk_id: str
    parent_id: str
    topic: str
    text: str


def build_qa_store() -> List[QaDoc]:
    """QA 库只放每个知识点的**第一句**，也就是核心结论。

    这一条是刻意这么设计的，第一版做错了：一开始两个库存的是完全相同的正文，
    结果细粒度必然赢——QA 库一条只讲一件事，PDF 库一条讲六件事，
    同样内容下 PDF 永远不划算。跑出来整张网格全是 1.000，毫无区分度。

    真实差别在于**覆盖度**：QA 库是人工整理过的问答对，答案短、切题，
    但只回答最常问的那一层；教材原文才展开机制、分型、注意事项。
    所以这里让 QA 库只有结论句，PDF 库保留全部句子 —— 这样"要不要
    多给 PDF 库一点位置"才成为一个真问题。
    """
    docs = []
    for p in POINTS:
        sents = sentence_split(p.content)
        docs.append(QaDoc(f"qa_{p.key}", sents[0] if sents else p.content, p.key))
    return docs


def build_pdf_store() -> List[PdfParent]:
    """按主题聚合六个侧面，拼成一段教材式正文。

    每段前面加一句主题导语，后面按侧面顺序拼接 —— 教材就是这么组织的，
    而且这会让"检索命中一个侧面"和"返回整段"之间的噪声更真实。
    """
    by_topic: Dict[str, List] = {}
    for p in POINTS:
        by_topic.setdefault(p.topic, []).append(p)

    parents: List[PdfParent] = []
    for topic, points in by_topic.items():
        points = sorted(points, key=lambda p: p.facet)
        head = f"【{TOPICS.get(topic, topic)}】本节讨论该病的六个方面：" \
               f"{'、'.join(p.facet for p in points)}。"
        body = "".join(p.content for p in points)
        parents.append(
            PdfParent(
                parent_id=f"pdf_{topic}",
                topic=topic,
                text=head + body,
                point_keys=tuple(p.key for p in points),
            )
        )
    return parents


def chunk_parents(parents: Sequence[PdfParent], size: int = CHUNK_SIZE) -> List[PdfChunk]:
    """按固定长度切块。刻意用定长切，不做句子对齐 ——
    生产中 PDF 抽取出来的就是这种质量，句子对齐反而是理想化。"""
    chunks: List[PdfChunk] = []
    for parent in parents:
        text = parent.text
        for i in range(0, len(text), size):
            chunks.append(
                PdfChunk(
                    chunk_id=f"{parent.parent_id}__c{i // size}",
                    parent_id=parent.parent_id,
                    topic=parent.topic,
                    text=text[i : i + size],
                )
            )
    return chunks


def rebuild_parents(hit_chunk_ids: Sequence[str], parents: Dict[str, PdfParent]) -> List[PdfParent]:
    """把命中的 chunk 还原成父文档，保持首次命中的顺序并去重。

    这是生产 `skills/parent_doc.py` 的行为：chunk 只是检索的抓手，
    真正进上下文的是整段父文档。
    """
    seen: List[str] = []
    for cid in hit_chunk_ids:
        pid = cid.split("__c")[0]
        if pid not in seen:
            seen.append(pid)
    return [parents[pid] for pid in seen if pid in parents]


def sentence_split(text: str) -> List[str]:
    """按中文句号切句，用来做要点级覆盖统计。"""
    parts = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？":
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return [s for s in parts if s]


def char_ngrams(text: str, n: int = 3) -> set:
    """字符 n-gram 集合，用来判断"这句话是否出现在上下文里"。

    为什么不用子串匹配：分块会从句子中间切断，子串匹配会因为边界不对
    而误判"没覆盖"。n-gram 覆盖率能容忍边界错位。
    """
    clean = "".join(ch for ch in text if not ch.isspace())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def coverage(needle: str, haystack: str, threshold: float = 0.8) -> bool:
    """needle 的 3-gram 有多少比例出现在 haystack 里。"""
    grams = char_ngrams(needle)
    if not grams:
        return False
    hit = sum(1 for g in grams if g in haystack)
    return hit / len(grams) >= threshold


def point_recall(context: str, point_key: str) -> float:
    """金标知识点的要点句有几成出现在上下文里。

    这是比"文档命中"更接近答案质量的指标：文档进了上下文，但被预算截断了一半，
    doc 级 recall 是 1.0，要点级 recall 可能只有 0.5 —— 后者才决定答案能不能说全。
    """
    point = next((p for p in POINTS if p.key == point_key), None)
    if point is None:
        return 0.0
    sents = sentence_split(point.content)
    if not sents:
        return 0.0
    return sum(1 for s in sents if coverage(s, context)) / len(sents)
