"""问题分类与路由策略。

改造要点：原来的 _select_strategy 六个分支全返回 "hybrid"，等于没有路由。
这里把"分类 → 策略 → 检索参数"拆成三张可配置表，
路由结果会真正改变 QA 路 / PDF 路的 k 值与上下文预算。

第二次改造（检索器选择）
------------------------
`eval/` 的离线评测显示：不存在通用最优的检索器。定义类问题上纯向量
R@5=0.941、混合检索只有 0.647；术语型上恰好相反（BM25 1.000 / 向量 0.900）。
既然差异存在，路由就该决定"用哪种检索器"，而不只是"取几条"。

但这个改造**默认关闭**，原因写在 RETRIEVERS 表里：n=17 时观察到 5:0 的
差距，McNemar p=0.0625，未达显著。在这种情况下把评测集上的 argmax 写死
进生产，就是拿用户的请求去验证我的过拟合。所以做成开关 + 证据分级：
开关打开才生效，`evidence` 标明每条映射的证据强度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QuestionType = Literal[
    "disease_definition", "clinical", "drug", "diet", "diagnosis", "prevention", "general", "emergency"
]

RetrieverMode = Literal["dense", "sparse", "hybrid"]

# 关键词规则：命中即归类。顺序敏感，先判急症。
_KEYWORDS: list[tuple[QuestionType, tuple[str, ...]]] = [
    (
        "emergency",
        (
            "胸痛", "胸口疼", "呼吸困难", "喘不上气", "意识不清", "昏迷", "抽搐",
            "大出血", "止不住血", "自杀", "服毒", "误服", "窒息", "晕倒", "休克",
        ),
    ),
    ("diet", ("饮食", "吃什么", "忌口", "禁忌", "营养", "食谱", "能不能吃")),
    ("diagnosis", ("区别", "鉴别", "还是", "怎么区分", "不同")),
    ("drug", ("药", "治疗", "用药", "剂量", "处方", "副作用", "抗生素")),
    ("prevention", ("预防", "感染", "传播", "传染", "疫苗", "筛查")),
    ("disease_definition", ("定义", "什么是", "是什么", "解释", "概念", "病因")),
    ("clinical", ("症状", "表现", "诊断", "检查", "指标", "正常值", "化验")),
]


@dataclass(frozen=True)
class RetrievalParams:
    """一次检索的具体参数。策略的价值就在于这些数字不同。"""

    qa_k: int
    pdf_k: int
    max_context_chars: int

    @property
    def use_qa(self) -> bool:
        return self.qa_k > 0

    @property
    def use_pdf(self) -> bool:
        return self.pdf_k > 0


# 策略 → 参数。数值依据见 docs/retrieval-tuning.md
STRATEGIES: dict[str, RetrievalParams] = {
    # 用药与饮食：QA 库里是结构化问答，精度更高，PDF 只取 1 段做机制补充
    "hybrid_qa_heavy": RetrievalParams(qa_k=15, pdf_k=1, max_context_chars=4000),
    # 定义与临床指标：教材段落更长更严谨，PDF 权重抬高，QA 少量兜底
    "hybrid_pdf_heavy": RetrievalParams(qa_k=4, pdf_k=4, max_context_chars=6000),
    # 鉴别诊断：需要多源对照，两路都放宽
    "hybrid_wide": RetrievalParams(qa_k=12, pdf_k=3, max_context_chars=6000),
    # 默认
    "hybrid": RetrievalParams(qa_k=10, pdf_k=3, max_context_chars=5000),
    # 单一来源
    "qa_only": RetrievalParams(qa_k=12, pdf_k=0, max_context_chars=3000),
    "pdf_only": RetrievalParams(qa_k=0, pdf_k=4, max_context_chars=4000),
}

# 问题类型 → 策略
ROUTES: dict[str, str] = {
    "drug": "hybrid_qa_heavy",
    "diet": "hybrid_qa_heavy",
    "disease_definition": "hybrid_pdf_heavy",
    "clinical": "hybrid_pdf_heavy",
    "diagnosis": "hybrid_wide",
    "prevention": "hybrid_wide",
    "general": "hybrid",
    "emergency": "qa_only",
}

# ────────────── 检索器选择（与上面的 k 值分配正交） ──────────────

DEFAULT_RETRIEVER: RetrieverMode = "hybrid"

# evidence 的含义：
#   measured    评测集上有直接对照，且差距方向与机制解释一致
#   hypothesis  有方向但样本不足以排除噪声，留在表里备查，不作为默认
#   default     没有证据支持偏离默认值，就是默认
Evidence = Literal["measured", "hypothesis", "default"]


@dataclass(frozen=True)
class RetrieverPolicy:
    mode: RetrieverMode
    evidence: Evidence
    note: str = ""


RETRIEVERS: dict[str, RetrieverPolicy] = {
    # ── 按 bge 评测结果（eval/report.md）重写 ──
    # 所有差异的 McNemar p 都 >= 0.0625，也就是**没有一条达到显著**。
    # 之所以还留下这张表，是因为它记录了"往哪个方向偏"，供后续扩样本验证；
    # 不是因为它是已验证的最优解。
    #
    # 药物类是唯一 hybrid 胜出的类型（1.000 / dense 0.920，2:0，p=0.5）。
    # 机制上说得通：药名彼此在向量空间里太近（都是"降糖药"），
    # BM25 靠字面才能把"格列美脲"和"格列齐特"分开。
    "drug": RetrieverPolicy(
        "hybrid", "hypothesis",
        "唯一 hybrid 胜出的类型（1.000 vs 0.920），2:0 p=0.5，样本不足",
    ),
    # 定义类：dense 1.000 / hybrid 0.706（5:0，p=0.0625）。
    # 全表最强的方向性证据，但 p 未达 0.05 —— 标 measured，不标 confirmed。
    "disease_definition": RetrieverPolicy(
        "dense", "measured",
        "dense 1.000 / hybrid 0.706，5:0，p=0.0625 未达显著",
    ),
    "clinical": RetrieverPolicy(
        "dense", "hypothesis", "dense 0.976 / hybrid 0.927，3:1 p=0.625",
    ),
    "diet": RetrieverPolicy(
        "dense", "hypothesis", "dense 1.000 / hybrid 0.882，2:0 p=0.5",
    ),
    "prevention": RetrieverPolicy(
        "dense", "hypothesis", "dense 1.000 / hybrid 0.938，1:0",
    ),
    # 评测集没有覆盖鉴别诊断问题，没有任何证据，不能凭直觉改
    "diagnosis": RetrieverPolicy("hybrid", "default", "评测集未覆盖该类型"),
    "general": RetrieverPolicy("hybrid", "default", ""),
    # 急症不进检索，占位只为让 resolve 不用判空
    "emergency": RetrieverPolicy("hybrid", "default", "急症走安全通道，不检索"),
}


@dataclass(frozen=True)
class RouteDecision:
    question_type: QuestionType
    strategy: str
    params: RetrievalParams
    retriever: RetrieverMode = DEFAULT_RETRIEVER
    emergency: bool = False
    matched: tuple[str, ...] = field(default_factory=tuple)

    def to_trace(self) -> dict[str, object]:
        return {
            "question_type": self.question_type,
            "strategy": self.strategy,
            "retriever": self.retriever,
            "qa_k": self.params.qa_k,
            "pdf_k": self.params.pdf_k,
            "emergency": self.emergency,
        }


def classify(query: str) -> tuple[QuestionType, tuple[str, ...]]:
    """按关键词规则分类，返回类型与命中的关键词（便于解释路由原因）。"""
    text = query.lower()
    for qtype, words in _KEYWORDS:
        hits = tuple(w for w in words if w in text)
        if hits:
            return qtype, hits
    return "general", ()


def resolve(
    query: str,
    *,
    retriever_routing: bool = False,
    default_retriever: RetrieverMode | None = None,
) -> RouteDecision:
    """分类 + 选策略，一步到位。

    `retriever_routing` 默认 False：检索器一律用 `default_retriever`（缺省 `hybrid`），
    与改造前完全一致。置 True 才按 RETRIEVERS 表选 —— 这个开关存在的理由是证据不够硬，
    见模块 docstring。

    `default_retriever` 可以在不改代码的前提下换全局默认（config.RETRIEVER_DEFAULT）。
    存在的理由：评测显示单库上纯向量 R@5 0.974、混合 0.905，如果要试纯向量，
    应该能改一行配置上线，而不是改代码重新发版。
    """
    qtype, matched = classify(query)
    strategy = ROUTES.get(qtype, "hybrid")
    policy = RETRIEVERS.get(qtype)
    fallback = default_retriever or DEFAULT_RETRIEVER
    retriever = policy.mode if (retriever_routing and policy) else fallback
    return RouteDecision(
        question_type=qtype,
        strategy=strategy,
        params=STRATEGIES[strategy],
        retriever=retriever,
        emergency=(qtype == "emergency"),
        matched=matched,
    )


def policy_table() -> list[dict[str, str]]:
    """把映射表摊平成可打印/可写文档的形式，给审计脚本与文档用。"""
    return [
        {
            "question_type": qtype,
            "retriever": p.mode,
            "evidence": p.evidence,
            "strategy": ROUTES.get(qtype, "hybrid"),
            "note": p.note,
        }
        for qtype, p in RETRIEVERS.items()
    ]

