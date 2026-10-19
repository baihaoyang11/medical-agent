"""质量判定与 query 改写测试。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.quality import assess, keyword_coverage  # noqa: E402
from core.rewrite import expand_synonyms, rewrite, strip_stopwords  # noqa: E402


# ───────────── 质量判定 ─────────────

def test_empty_result_fails():
    report = assess("糖尿病的饮食禁忌", [], "")
    assert report.passed is False
    assert report.doc_count == 0


def test_short_context_fails():
    report = assess("糖尿病", [{"content": "糖尿病"}], "糖尿病")
    assert report.passed is False
    assert any("过短" in r for r in report.reasons)


def test_long_but_irrelevant_context_fails():
    """这条对应旧实现的漏洞：只看字符数时，一段无关长文会被判为"质量很好"。"""
    irrelevant = "本文介绍了农业种植技术与土壤改良方法。" * 20
    report = assess("糖尿病的饮食禁忌", [{"content": irrelevant}], irrelevant)
    assert report.passed is False
    assert any("关键词覆盖" in r for r in report.reasons)


def test_relevant_context_passes():
    context = (
        "糖尿病的饮食禁忌包括：控制碳水化合物摄入，避免高糖食物与含糖饮料，"
        "主食可用杂粮替代精制米面，增加蔬菜和膳食纤维摄入，规律进餐并监测血糖。"
        "同时应限制油炸食品与高盐腌制食品，戒烟限酒，避免空腹饮酒引起低血糖。"
        "合并肾病的患者需在医生指导下限制蛋白质总量，合并高血压者每日食盐不超过五克。"
    )
    report = assess("糖尿病的饮食禁忌", [{"content": context}], context)
    assert report.passed is True, report.reasons
    assert report.coverage >= 0.3


def test_short_but_on_topic_context_fails_on_length():
    """短而准的片段依然要拦——一条 45 字的句子不足以支撑一段医学回答。"""
    context = "糖尿病的饮食禁忌包括：控制碳水化合物摄入，避免高糖食物，增加蔬菜摄入。"
    report = assess("糖尿病的饮食禁忌", [{"content": context}], context)
    assert report.passed is False
    assert report.coverage == 1.0
    assert any("过短" in r for r in report.reasons)


def test_coverage_is_bounded():
    assert 0.0 <= keyword_coverage("糖尿病", "") <= 1.0
    assert keyword_coverage("", "任意内容") == 0.0


# ───────────── query 改写 ─────────────

def test_strip_stopwords_removes_colloquialisms():
    result = strip_stopwords("请问糖尿病患者的饮食禁忌有哪些")
    assert "请问" not in result
    assert "有哪些" not in result
    assert "糖尿病" in result


def test_first_attempt_only_strips():
    assert rewrite("糖尿病的饮食禁忌有哪些", 1) == "糖尿病 饮食禁忌"


def test_second_attempt_adds_synonyms():
    expanded = rewrite("糖尿病的饮食禁忌", 2)
    assert "糖尿病" in expanded
    assert "高血糖" in expanded or "胰岛素" in expanded


def test_expand_respects_max_terms():
    out = expand_synonyms("糖尿病 高血压 血脂", max_terms=2)
    assert len(out.split()) <= 5


def test_all_stopwords_query_is_preserved():
    """全是停用词时不能返回空串，否则检索必然无结果。"""
    assert rewrite("怎么办", 1).strip() != ""
