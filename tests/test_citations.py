"""引用构造测试。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.citations import build_citations, render_for_prompt  # noqa: E402


def test_builds_ranked_citations():
    docs = [
        {"doc_id": "d1", "source": "生理学.pdf", "page": 12, "content": "血糖调节机制"},
        {"doc_id": "d2", "source": "生物化学.pdf", "page": 3, "content": "胰岛素的作用"},
    ]
    citations = build_citations(docs)
    assert [c["rank"] for c in citations] == [1, 2]
    assert citations[0]["source"] == "生理学.pdf"
    assert citations[0]["page"] == 12


def test_dedupes_same_doc_and_page():
    docs = [
        {"doc_id": "d1", "page": 1, "content": "甲"},
        {"doc_id": "d1", "page": 1, "content": "甲"},
    ]
    assert len(build_citations(docs)) == 1


def test_respects_limit():
    docs = [{"doc_id": f"d{i}", "page": i, "content": "内容"} for i in range(10)]
    assert len(build_citations(docs, limit=3)) == 3


def test_snippet_is_truncated():
    docs = [{"doc_id": "d1", "content": "很长的一段内容" * 50}]
    citation = build_citations(docs, snippet_chars=20)[0]
    assert len(citation["snippet"]) <= 21  # 20 + 省略号
    assert citation["snippet"].endswith("…")


def test_render_handles_empty():
    assert render_for_prompt([]) == ""
    text = render_for_prompt([{"rank": 1, "source": "生理学.pdf", "page": 5}])
    assert "生理学.pdf" in text and "第 5 页" in text
