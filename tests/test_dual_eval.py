"""双库评测的测试。

这套评测的价值全在"指标算得对"，所以测的是指标本身，不是跑一遍不报错：

1. `point_recall` 能不能区分"全进去了"和"只进去一半"
2. `rebuild_parents` 会不会重复返回同一篇父文档（重复会虚高上下文长度）
3. `chunk_parents` 切完之后内容有没有丢（定长切最容易丢尾巴）
4. `select_within_budget` 和 `compose_context` 必须用同一套选取规则
   —— 两者不一致的话，"噪声占比"算的就是另一件事
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.context import compose_context, select_within_budget  # noqa: E402
from eval import dual_store as DS  # noqa: E402
from eval.knowledge import POINTS  # noqa: E402


@pytest.fixture(scope="module")
def point():
    return next(p for p in POINTS if p.key == "diabetes_definition")


# ──────────────── 要点覆盖 ────────────────
def test_point_recall_full_context_is_one(point):
    assert DS.point_recall(point.content, point.key) == pytest.approx(1.0)


def test_point_recall_empty_context_is_zero(point):
    assert DS.point_recall("", point.key) == pytest.approx(0.0)


def test_point_recall_partial_context_is_between(point):
    """只给第一句 → 覆盖 1/N，既不应该是 0 也不应该是 1。

    这条是核心：如果实现退化成"子串包含就返回 1"，半截上下文会被判成满分，
    那整个评测就失去了意义。
    """
    first = DS.sentence_split(point.content)[0]
    val = DS.point_recall(first, point.key)
    assert 0.0 < val < 1.0


def test_point_recall_unknown_key_is_zero():
    assert DS.point_recall("随便什么内容", "not_a_real_key") == pytest.approx(0.0)


def test_coverage_tolerates_boundary_shift():
    """分块会从句中切断，子串匹配会误判"没覆盖"，3-gram 不该。

    但容忍是有限度的：截掉一两个字算覆盖，只给前半句不算。
    后者是刻意的——半句话的答案比没有更危险，这一点和 `core/context.py`
    "超预算整段丢弃、不做半句截断"是同一个判断。
    """
    sentence = "糖尿病是一组以慢性高血糖为特征的代谢性疾病。"
    assert DS.coverage(sentence, "前文前文" + sentence + "后文后文")
    assert DS.coverage(sentence, sentence[:-2])            # 尾部被切掉一点，仍算覆盖
    assert not DS.coverage(sentence, sentence[:15])        # 只有前半句，不算
    assert not DS.coverage(sentence, "完全是另一段和它无关的话")


# ──────────────── 语料构造 ────────────────
def test_qa_store_keeps_only_core_sentence(point):
    """QA 库只放第一句 —— 这是刻意的设定，不是 bug。

    第一版两个库存的是完全相同的正文，导致细粒度必然赢、
    整张网格全是 1.000、毫无区分度。所以这条测试守着这个设计。
    """
    docs = {d.point_key: d for d in DS.build_qa_store()}
    text = docs[point.key].text
    assert text == DS.sentence_split(point.content)[0]
    assert len(text) < len(point.content)


def test_pdf_parents_cover_all_points():
    parents = DS.build_pdf_store()
    keys = {k for p in parents for k in p.point_keys}
    assert keys == {p.key for p in POINTS}, "有知识点没被任何父文档覆盖"


def test_pdf_parent_keeps_full_text():
    """父文档必须是完整正文，否则"PDF 更完整"这个前提就不成立了。"""
    parents = {p.parent_id: p for p in DS.build_pdf_store()}
    for point in POINTS:
        parent = parents[f"pdf_{point.topic}"]
        assert point.content in parent.text


def test_chunking_does_not_drop_content():
    """定长切最容易丢尾巴。把所有 chunk 拼回去必须等于原文。"""
    parents = DS.build_pdf_store()
    chunks = DS.chunk_parents(parents, size=200)
    for parent in parents:
        joined = "".join(c.text for c in chunks if c.parent_id == parent.parent_id)
        assert joined == parent.text


def test_chunk_size_is_respected():
    chunks = DS.chunk_parents(DS.build_pdf_store(), size=200)
    assert all(len(c.text) <= 200 for c in chunks)


# ──────────────── 父文档还原 ────────────────
def test_rebuild_parents_dedupes_and_keeps_order():
    """同一父文档的多个 chunk 命中，只能返回一次 ——
    返回多次会让上下文长度虚高，"成本"那一列就失真了。
    """
    parents = {p.parent_id: p for p in DS.build_pdf_store()}
    ids = list(parents)[:2]
    chunk_ids = [f"{ids[0]}__c0", f"{ids[1]}__c0", f"{ids[0]}__c1"]
    rebuilt = DS.rebuild_parents(chunk_ids, parents)
    assert [p.parent_id for p in rebuilt] == ids


def test_rebuild_parents_ignores_unknown_id():
    parents = {p.parent_id: p for p in DS.build_pdf_store()}
    assert DS.rebuild_parents(["pdf_不存在__c0"], parents) == []


# ──────────────── 选取规则一致性 ────────────────
def test_select_and_compose_agree():
    """评测用 select_within_budget 算噪声、用 compose_context 取文本。
    两套规则一旦分叉，"噪声占比"就不是针对真正进上下文的那几段算的。
    """
    docs = [
        {"content": "a" * 100, "doc_id": "1"},
        {"content": "b" * 100, "doc_id": "2"},
        {"content": "c" * 100, "doc_id": "3"},
    ]
    picked = select_within_budget(docs, 250)
    assert [d["doc_id"] for d in picked] == ["1", "2"]
    assert compose_context(docs, 250) == "a" * 100 + "\n\n---\n\n" + "b" * 100


def test_first_doc_is_kept_even_if_over_budget():
    """预算配小了也不能返回空上下文——空上下文比多花几个 token 糟糕得多。"""
    docs = [{"content": "x" * 500, "doc_id": "1"}]
    assert len(select_within_budget(docs, 100)) == 1
