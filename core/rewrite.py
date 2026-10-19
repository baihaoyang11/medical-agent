"""二次检索的 query 改写。

为什么不用 LLM 改写：
1. 二次检索本来就是低预算路径，再调一次模型成本翻倍；
2. 改写结果必须可复现，规则版能做单测；
3. 医学场景同义词很集中，小词典覆盖得住。

两档策略：
  attempt 1 → 去停用词，收敛到核心词
  attempt 2 → 去停用词 + 同义词扩展（提升召回）
"""

from __future__ import annotations

from typing import Iterable

# 口语化停用词：删掉是因为它们在 BM25 里只会稀释分数
STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "吗", "呢", "啊", "吧", "是", "有", "和", "与", "或",
        "请问", "我想", "想问", "知道", "了解", "一下", "怎么样", "怎么办",
        "怎么", "如何", "什么", "哪些", "为什么", "可以", "能不能", "需要",
        "应该", "这个", "那个", "有没有", "是不是",
    }
)

# 小规模医学同义词表。按"查询里出现 key → 追加扩展词"使用。
SYNONYMS: dict[str, tuple[str, ...]] = {
    "血糖": ("葡萄糖", "糖尿病"),
    "糖尿病": ("高血糖", "胰岛素"),
    "高血压": ("血压升高", "降压"),
    "血脂": ("胆固醇", "甘油三酯"),
    "感冒": ("上呼吸道感染",),
    "发烧": ("发热", "体温升高"),
    "拉肚子": ("腹泻",),
    "心梗": ("心肌梗死", "冠心病"),
    "中风": ("脑卒中", "脑梗死"),
    "肝功": ("肝功能", "转氨酶"),
    "肾功": ("肾功能", "肌酐"),
    "抗生素": ("抗菌药物", "消炎药"),
    "疫苗": ("接种", "免疫"),
}


def strip_stopwords(query: str, extra: Iterable[str] = ()) -> str:
    """去掉停用词，保留其余字符。按长度倒序替换，避免"是不是"被"是"先拆掉。"""
    text = query
    for word in sorted(set(STOPWORDS) | set(extra), key=len, reverse=True):
        text = text.replace(word, " ")
    return " ".join(text.split())


def expand_synonyms(query: str, *, max_terms: int = 3) -> str:
    """按同义词表追加扩展词，最多 max_terms 个，避免 query 被撑爆。"""
    extras: list[str] = []
    for key, syns in SYNONYMS.items():
        if key in query:
            for s in syns:
                if s not in query and s not in extras:
                    extras.append(s)
                if len(extras) >= max_terms:
                    return query + " " + " ".join(extras)
    return query + (" " + " ".join(extras) if extras else "")


def rewrite(query: str, attempt: int) -> str:
    """第 attempt 次检索使用的 query（attempt 从 1 开始）。"""
    base = strip_stopwords(query)
    if not base:
        # 全是停用词（比如"怎么了"）→ 退回原 query，删空反而更糟
        base = query.strip()
    if attempt <= 1:
        return base
    return expand_synonyms(base)
