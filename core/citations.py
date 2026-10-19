"""引用构造。

医疗问答不给出来源是硬伤：用户没法核对，产品也没法追责。
这里把检索结果的 metadata 压成前端能直接渲染的 citations。
"""

from __future__ import annotations

from typing import Any, Iterable

DEFAULT_LIMIT = 5
DEFAULT_SNIPPET_CHARS = 120


def _snippet(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "…"


def build_citations(
    docs: Iterable[dict[str, Any]],
    *,
    limit: int = DEFAULT_LIMIT,
    snippet_chars: int = DEFAULT_SNIPPET_CHARS,
) -> list[dict[str, Any]]:
    """去重后按出现顺序生成引用列表。

    去重键用 (doc_id, page)：同一段父文档可能被多路召回多次。
    """
    seen: set[tuple[str, int]] = set()
    citations: list[dict[str, Any]] = []

    for doc in docs or []:
        doc_id = str(doc.get("doc_id") or doc.get("pk") or "")
        page = int(doc.get("page") or 0)
        key = (doc_id, page)
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            {
                "rank": len(citations) + 1,
                "doc_id": doc_id,
                "source": str(doc.get("source") or ""),
                "page": page,
                "snippet": _snippet(str(doc.get("content") or doc.get("text") or ""), snippet_chars),
            }
        )
        if len(citations) >= limit:
            break

    return citations


def render_for_prompt(citations: list[dict[str, Any]]) -> str:
    """把引用渲染成提示词里的编号清单，让模型回答时带 [1][2] 标记。"""
    if not citations:
        return ""
    lines = []
    for c in citations:
        where = c["source"] or "内部知识库"
        if c.get("page"):
            where = f"{where} 第 {c['page']} 页"
        lines.append(f"[{c['rank']}] {where}")
    return "\n".join(lines)
