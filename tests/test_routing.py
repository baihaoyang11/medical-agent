"""路由策略测试。

重点验证三件事：
1. 分类规则命中正确
2. 不同策略真的产出不同检索参数（旧版六个分支返回同一个值）
3. 急症能被识别并短路
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.routing import STRATEGIES, classify, resolve  # noqa: E402


def test_drug_question_routes_to_qa_heavy():
    decision = resolve("布洛芬的用药剂量和副作用是什么")
    assert decision.question_type == "drug"
    assert decision.strategy == "hybrid_qa_heavy"
    # QA 库是结构化问答，精度更高，所以 QA 权重必须高于 PDF
    assert decision.params.qa_k > decision.params.pdf_k


def test_definition_routes_to_pdf_heavy():
    decision = resolve("什么是糖尿病")
    assert decision.question_type == "disease_definition"
    assert decision.strategy == "hybrid_pdf_heavy"
    assert decision.params.pdf_k >= decision.params.qa_k


def test_diagnosis_routes_to_wide():
    decision = resolve("牛皮癣和湿疹怎么区分")
    assert decision.question_type == "diagnosis"
    assert decision.strategy == "hybrid_wide"


def test_emergency_short_circuits():
    decision = resolve("突然胸痛，呼吸困难，出冷汗")
    assert decision.question_type == "emergency"
    assert decision.emergency is True
    assert "胸痛" in decision.matched


def test_unknown_question_falls_back_to_default():
    decision = resolve("今天天气不错")
    assert decision.question_type == "general"
    assert decision.emergency is False


def test_strategies_are_actually_different():
    """这条是防回归的：旧实现六个分支返回同一个 "hybrid"。"""
    distinct = {(p.qa_k, p.pdf_k) for p in STRATEGIES.values()}
    assert len(distinct) >= 4, f"策略参数重复过多: {distinct}"


def test_every_route_points_to_a_defined_strategy():
    from core.routing import ROUTES

    for qtype, strategy in ROUTES.items():
        assert strategy in STRATEGIES, f"{qtype} 指向了未定义的策略 {strategy}"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("糖尿病患者的饮食禁忌", "diet"),
        ("心肌梗死的预防和筛查", "prevention"),
        ("血常规指标的正常值", "clinical"),
    ],
)
def test_classification_table(query: str, expected: str):
    qtype, _ = classify(query)
    assert qtype == expected
