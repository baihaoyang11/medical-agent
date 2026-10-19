"""Agent 编排集成测试。

用假 Skill 跑真实的 LangGraph 图，验证四件事：
1. 急症直接走安全通道，不检索不生成
2. 首次检索通过时 retry_count 为 0，且产出引用
3. 首次检索无结果时，真的会改写 query 再检索一次
4. 上下文预算按整段丢弃，不做半句截断

需要 langgraph；未安装时整个文件跳过，不影响其它单测。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("langgraph")

from agents.medical_agent import MedicalAgent  # noqa: E402
from core.context import compose_context, dedupe  # noqa: E402

LONG_DOC = (
    "糖尿病患者的饮食禁忌包括：控制碳水化合物总量，避免精制糖与含糖饮料；"
    "主食优先选择全谷物并控制份量；增加蔬菜与膳食纤维摄入，尤其是低升糖指数的绿叶菜；"
    "适量摄入优质蛋白，如鱼、蛋、豆制品；限制高脂高盐加工食品；"
    "规律三餐、避免暴饮暴食，并配合血糖监测与医生随访。"
)


class FakeSkill:
    """按调用次数依次返回预设批次，用来模拟"第一次召回为空、第二次命中"。"""

    def __init__(self, name: str, batches: list[list[dict]]) -> None:
        self.name = name
        self._batches = batches
        self.calls = 0

    def to_tool_definition(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    def execute(self, query: str = "", k: int = 0, **kwargs) -> dict:
        index = min(self.calls, len(self._batches) - 1)
        self.calls += 1
        return {
            "success": True,
            "data": self._batches[index],
            "error": None,
            "metadata": {"method": "fake", "count": len(self._batches[index])},
        }


class FakeAnswer:
    name = "answer_generation"

    def __init__(self) -> None:
        self.last_context = ""

    def to_tool_definition(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    def execute(self, query: str = "", context: str = "", **kwargs) -> dict:
        self.last_context = context
        return {"success": True, "data": "根据资料，应控制碳水并规律监测血糖 [1]。", "error": None, "metadata": {}}


def _agent(qa_batches, pdf_batches=None) -> tuple[MedicalAgent, FakeSkill, FakeAnswer]:
    qa = FakeSkill("hybrid_search", qa_batches)
    pdf = FakeSkill("parent_doc_search", pdf_batches or [[]])
    answer = FakeAnswer()
    return MedicalAgent(qa, pdf, answer), qa, answer


def test_emergency_skips_retrieval_and_generation():
    agent, qa, answer = _agent([[]])
    result = agent.run("突然胸痛，呼吸困难")

    assert result["emergency"] is True
    assert "120" in result["answer"]
    assert result["citations"] == []
    assert qa.calls == 0, "急症不应触发检索"
    assert answer.last_context == "", "急症不应触发生成"


def test_normal_question_passes_on_first_attempt():
    agent, qa, answer = _agent([[{"doc_id": "d1", "content": LONG_DOC, "source": "内科学.pdf", "page": 42}]])
    result = agent.run("糖尿病的饮食禁忌有哪些")

    assert result["retry_count"] == 0
    assert result["question_type"] == "diet"
    assert result["retrieval_strategy"] == "hybrid_qa_heavy"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["page"] == 42
    assert "[1]" in answer.last_context, "生成阶段应带上引用编号"


def test_empty_first_attempt_triggers_rewrite_and_retry():
    doc = {"doc_id": "d2", "content": LONG_DOC, "source": "内科学.pdf", "page": 7}
    agent, qa, _ = _agent([[], [doc]])
    result = agent.run("糖尿病的饮食禁忌")

    assert qa.calls >= 2, "首次无结果必须触发二次检索"
    assert result["retry_count"] >= 1
    assert len(result["citations"]) >= 1

    steps = "\n".join(result["thinking_steps"])
    assert "改写" in steps


def test_retry_stops_at_max():
    """始终召回为空时应在上限处停止，不能无限循环。"""
    agent, qa, _ = _agent([[]])
    result = agent.run("一个查不到答案的问题")

    assert result["retry_count"] <= 2
    assert qa.calls <= 3
    assert result["degraded"] is True


def test_dedupe_keeps_distinct_chunks_of_same_doc():
    docs = [
        {"doc_id": "d1", "content": "第一段内容" * 20},
        {"doc_id": "d1", "content": "第一段内容" * 20},
        {"doc_id": "d1", "content": "第二段完全不同的内容" * 10},
    ]
    assert len(dedupe(docs)) == 2


def test_compose_context_drops_whole_doc_over_budget():
    docs = [{"content": "甲" * 100}, {"content": "乙" * 100}, {"content": "丙" * 100}]
    context = compose_context(docs, budget=250)

    assert "甲" in context
    assert "丙" not in context, "超预算的文档应整段丢弃，而不是截断"
    assert len(context) <= 250
