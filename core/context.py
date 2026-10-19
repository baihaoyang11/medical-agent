"""上下文拼装与去重。

原来这两个函数是 `agents/medical_agent.py` 里的私有函数。挪出来的原因很实际：
`eval/dual_store.py` 要模拟"按预算拼上下文"这一步，而评测必须调**生产的**
那段逻辑 —— 自己复刻一份，测的就是脚本而不是系统，这是评测里最常见的自欺。

放在 `core/` 而不是 `agents/`，还有一个好处：这个模块不依赖 langgraph。
评测和单测只关心"怎么拼上下文"，没必要为了一个纯函数把整个图框架拉起来。
"""

from __future__ import annotations

from typing import Any, Dict, List


def dedupe(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 (doc_id, 内容前 60 字) 去重，保留首次出现。

    只用 doc_id 不够：同一父文档的不同段落 doc_id 相同但内容不同。
    """
    seen: set[tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for d in docs:
        content = str(d.get("content") or d.get("text") or "")
        key = (str(d.get("doc_id") or ""), content[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def select_within_budget(docs: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    """按预算挑出能进上下文的文档，返回挑中的那些。

    单独拆出来是因为评测需要知道"到底哪几段进了上下文"才能算噪声占比。
    如果让评测自己复刻一遍选取规则，两边一旦不一致，测出来的就是脚本的行为。
    """
    picked: List[Dict[str, Any]] = []
    used = 0
    for d in docs:
        content = str(d.get("content") or d.get("text") or "").strip()
        if not content:
            continue
        if used + len(content) > budget and picked:
            break
        picked.append(d)
        used += len(content)
    return picked


def compose_context(docs: List[Dict[str, Any]], budget: int) -> str:
    """按预算拼上下文。超预算时整段丢弃，而不是截断——半句话比没有更危险。

    规则细节：第一条即使超预算也会收进来，否则预算配小了会返回空上下文，
    那比"多花几个 token"糟糕得多。
    """
    parts = []
    for d in select_within_budget(docs, budget):
        content = str(d.get("content") or d.get("text") or "").strip()
        parts.append(content)
    return "\n\n---\n\n".join(parts)
