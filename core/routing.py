"""问题分类与路由策略。

改造要点：原来的 _select_strategy 六个分支全返回 "hybrid"，等于没有路由。
这里把"分类 → 策略 → 检索参数"拆成三张可配置表，
路由结果会真正改变 QA 路 / PDF 路的 k 值与上下文预算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QuestionType = Literal[
    "disease_definition", "clinical", "drug", "diet", "diagnosis", "prevention", "general", "emergency"
]

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


@dataclass(frozen=True)
class RouteDecision:
    question_type: QuestionType
    strategy: str
    params: RetrievalParams
    emergency: bool = False
    matched: tuple[str, ...] = field(default_factory=tuple)

    def to_trace(self) -> dict[str, object]:
        return {
            "question_type": self.question_type,
            "strategy": self.strategy,
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


def resolve(query: str) -> RouteDecision:
    """分类 + 选策略，一步到位。"""
    qtype, matched = classify(query)
    strategy = ROUTES.get(qtype, "hybrid")
    return RouteDecision(
        question_type=qtype,
        strategy=strategy,
        params=STRATEGIES[strategy],
        emergency=(qtype == "emergency"),
        matched=matched,
    )
