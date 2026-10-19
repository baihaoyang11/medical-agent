"""重排器测试。

重点不是"分数算得对"，而是三条工程契约：
    1. 排序稳定（同分不乱序 → 引用编号不抖）
    2. 主路挂了必须走备路，不能让重排放倒整条检索链路
    3. 空输入 / top_n 边界不炸
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rerank import (  # noqa: E402
    CrossEncoderReranker,
    FallbackReranker,
    KeywordReranker,
    NoOpReranker,
    build_reranker,
)

DOCS = [
    {"doc_id": "a", "content": "农业种植技术与土壤改良方法，深耕与轮作可以提高产量。"},
    {"doc_id": "b", "content": "高血压患者的日常监测与用药依从性很重要，需定期复查。"},
    {"doc_id": "c", "content": "糖尿病的饮食禁忌包括控制碳水化合物、避免高糖食物、规律进餐。"},
]


def test_noop_keeps_order():
    out = NoOpReranker().rerank("糖尿病的饮食禁忌", DOCS)
    assert [d["doc_id"] for d in out] == ["a", "b", "c"]


def test_noop_respects_top_n():
    out = NoOpReranker().rerank("任意", DOCS, top_n=2)
    assert len(out) == 2


def test_keyword_promotes_relevant_doc():
    """相关的那条在队尾也要提到第一。"""
    out = KeywordReranker(weight=0.8).rerank("糖尿病 饮食禁忌", DOCS)
    assert out[0]["doc_id"] == "c"
    assert "rerank_score" in out[0]


def test_keyword_does_not_mutate_input():
    before = [dict(d) for d in DOCS]
    KeywordReranker().rerank("糖尿病", DOCS)
    assert DOCS == before


def test_keyword_is_stable_on_ties():
    """完全无关时退化为原顺序，不能随机打乱。"""
    docs = [{"doc_id": str(i), "content": "无关内容"} for i in range(6)]
    out = KeywordReranker(weight=1.0).rerank("糖尿病", docs)
    assert [d["doc_id"] for d in out] == [str(i) for i in range(6)]


def test_keyword_empty_input():
    assert KeywordReranker().rerank("糖尿病", []) == []


def test_weight_is_validated():
    try:
        KeywordReranker(weight=1.5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("非法 weight 应该报错")


def test_fallback_switches_on_primary_failure():
    """cross-encoder 装不上时，结果集不能是空的，必须仍有排序。"""

    class Broken:
        name = "broken"
        degraded = False

        def rerank(self, query, docs, *, top_n=None):
            raise RuntimeError("模型加载失败")

    wrapper = FallbackReranker(Broken(), KeywordReranker(weight=0.8))
    assert wrapper.degraded is False

    out = wrapper.rerank("糖尿病 饮食禁忌", DOCS)
    assert wrapper.degraded is True
    assert len(out) == len(DOCS)
    assert out[0]["doc_id"] == "c"

    # 第二次直接走备路，不会重复抛异常
    out2 = wrapper.rerank("糖尿病 饮食禁忌", DOCS)
    assert out2[0]["doc_id"] == "c"


def test_cross_encoder_without_model_path_raises():
    reranker = CrossEncoderReranker(model_path=None)
    try:
        reranker.rerank("糖尿病", DOCS)
    except Exception as exc:
        assert "RERANK_MODEL_PATH" in str(exc) or "sentence" in str(exc)
    # 走 FallbackReranker 时则必须成功
    out = FallbackReranker(CrossEncoderReranker(model_path=None)).rerank("糖尿病 饮食", DOCS)
    assert out[0]["doc_id"] == "c"


def test_build_reranker_unknown_strategy_degrades():
    r = build_reranker("not-a-strategy")
    assert r.name == "keyword"


def test_build_reranker_noop():
    assert build_reranker("noop").name == "noop"


def test_build_reranker_cross_encoder_is_fallback_wrapped():
    r = build_reranker("cross_encoder", model_path="")
    assert r.name == "cross_encoder"
    out = r.rerank("糖尿病", DOCS)
    assert getattr(r, "degraded", False) is True
    assert len(out) == len(DOCS)
