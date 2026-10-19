"""检索器路由测试。

要验证的不是"配置表能读出来"，而是三件事：

1. **默认行为一点没变**。改造前所有问题走 hybrid，改造后开关关闭时也必须是 hybrid。
   这条是防漂移的，比任何新功能测试都重要 —— 新增的开关不能悄悄改变线上行为。
2. **开关打开后，mode 真的改变了检索路径**，而不是只改了日志里的一行字符串。
   dense 模式不该建 BM25 索引（几万文档要几秒），sparse 模式不该调 embedding
   （一次网络往返）。这两条断言才有实际价值。
3. **配置表自身是自洽的**：每个类型都有映射、mode 合法、证据等级标注齐全。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.routing import (  # noqa: E402
    DEFAULT_RETRIEVER,
    RETRIEVERS,
    ROUTES,
    policy_table,
    resolve,
)


# ────────────────────────────────────────────────────────────────
# 一、默认行为不变
# ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "query",
    [
        "什么是糖尿病",            # 定义类：表里被映射为 dense
        "糖尿病患者如何安排饮食",
        "二甲双胍有什么副作用",
        "突然胸痛，呼吸困难",
        "今天天气不错",
    ],
)
def test_default_mode_is_hybrid_for_every_type(query: str):
    """不开开关时，无论分类成什么，检索器都必须是 hybrid。

    改造后 disease_definition 在表里是 dense —— 如果 resolve 忘了判开关，
    这条会立刻红。这就是它存在的意义。
    """
    assert resolve(query).retriever == DEFAULT_RETRIEVER
    assert resolve(query, retriever_routing=False).retriever == DEFAULT_RETRIEVER


def test_switch_on_changes_only_mapped_types():
    """开开关后，只有表里标了非默认值的类型才会变。"""
    decision = resolve("什么是糖尿病", retriever_routing=True)
    assert decision.question_type == "disease_definition"
    assert decision.retriever == "dense"

    # 同一次调用里，k 值分配不受影响 —— 检索器与策略是两个正交维度
    assert decision.strategy == "hybrid_pdf_heavy"
    assert decision.params.qa_k == 4


def test_unmapped_type_stays_hybrid_when_switch_on():
    decision = resolve("今天天气不错", retriever_routing=True)
    assert decision.question_type == "general"
    assert decision.retriever == "hybrid"


def test_default_retriever_is_configurable():
    """全局默认要能改一行配置就换 —— 评测显示 dense 在单库上明显更强，
    如果无法低成本试，那条结论就只能躺在报告里。"""
    assert resolve("什么是糖尿病", default_retriever="dense").retriever == "dense"


def test_table_takes_precedence_over_default():
    """路由打开时，映射表优先于全局默认。

    药物类在表里是 hybrid，即便全局默认改成 dense 也不该被盖掉 ——
    否则"按类型选检索器"就没有意义了。
    """
    decision = resolve("二甲双胍有什么副作用", retriever_routing=True, default_retriever="dense")
    assert decision.question_type == "drug"
    assert decision.retriever == "hybrid"


def test_trace_includes_retriever():
    trace = resolve("什么是糖尿病", retriever_routing=True).to_trace()
    assert trace["retriever"] == "dense"
    # 线上排查要能回答"这条请求到底走了哪条路"，所以 trace 里必须有这三个键
    assert {"question_type", "strategy", "retriever"} <= set(trace)


# ────────────────────────────────────────────────────────────────
# 二、配置表自洽
# ────────────────────────────────────────────────────────────────
def test_every_route_has_a_retriever_policy():
    for qtype in ROUTES:
        assert qtype in RETRIEVERS, f"{qtype} 有策略但没有检索器映射"


def test_all_modes_are_valid():
    valid = {"dense", "sparse", "hybrid"}
    for qtype, policy in RETRIEVERS.items():
        assert policy.mode in valid, f"{qtype} 的 mode={policy.mode!r} 不合法"


def test_all_evidence_levels_are_valid():
    valid = {"measured", "hypothesis", "default"}
    for qtype, policy in RETRIEVERS.items():
        assert policy.evidence in valid, f"{qtype} 的 evidence={policy.evidence!r} 不合法"


def test_non_default_modes_must_carry_evidence():
    """偏离默认值的映射必须写明证据等级和依据。

    这条是给未来的自己看的：想改生产检索器，先回答"证据在哪"。
    """
    for qtype, policy in RETRIEVERS.items():
        if policy.mode != DEFAULT_RETRIEVER:
            assert policy.evidence in ("measured", "hypothesis"), f"{qtype} 改了默认值却没标证据"
            assert policy.note, f"{qtype} 改了默认值却没写依据"


def test_policy_table_is_serialisable():
    rows = policy_table()
    assert len(rows) == len(RETRIEVERS)
    for row in rows:
        assert set(row) == {"question_type", "retriever", "evidence", "strategy", "note"}


# ────────────────────────────────────────────────────────────────
# 三、mode 真的改变了检索路径（用替身，不碰 Milvus）
# ────────────────────────────────────────────────────────────────
class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, query: str):
        self.calls += 1
        return [0.1, 0.2, 0.3]


class _FakeMilvus:
    """只实现 _dense_search / _load_texts 用到的方法。"""

    def __init__(self, docs: list[dict] | None = None) -> None:
        self.search_calls = 0
        self.docs = docs or []

    def search(self, collection, data, anns_field="", limit=10, output_fields=None):
        self.search_calls += 1
        return [[{
            "id": "1",
            "distance": 0.87,
            "entity": {"doc_id": "a", "text": "糖尿病是一组以慢性高血糖为特征的代谢性疾病。"},
        }]]

    def query(self, collection, limit=1000, offset=0, output_fields=None):
        # 分页语义：第一次返回全部，第二次返回空让循环结束
        return self.docs if offset == 0 else []


def _skill(docs=None):
    pytest.importorskip("rank_bm25")
    from skills.hybrid_search import HybridSearchSkill

    emb = _FakeEmbeddings()
    client = _FakeMilvus(docs)
    return HybridSearchSkill(client, emb), emb, client


# 语料不能只有 1 篇：rank_bm25 在 N=1 时每个词的 IDF 都是负的，
# 特殊处理后又把负值留在原地，导致所有分数 <= 0 被 _bm25_search 过滤掉。
# 那是库在小语料上的边界行为，不是要找的 bug —— 用 4 篇让 IDF 回到正值。
_DOCS = [
    {"pk": 1, "doc_id": "a", "text": "糖尿病是一组以慢性高血糖为特征的代谢性疾病。"},
    {"pk": 2, "doc_id": "b", "text": "高血压是最常见的慢性病之一，需要长期管理。"},
    {"pk": 3, "doc_id": "c", "text": "哮喘的气道炎症具有可变气流受限的特征。"},
    {"pk": 4, "doc_id": "d", "text": "慢性肾脏病按肾小球滤过率分为五个阶段。"},
]


def test_dense_mode_skips_bm25_index():
    """dense 模式不该构建 BM25 索引 —— 那是全量拉取 + 分词，几万文档要几秒。"""
    skill, emb, client = _skill(_DOCS)
    result = skill.execute("什么是糖尿病", k=5, mode="dense")

    assert result["success"] is True
    assert result["metadata"]["requested"] == "dense"
    assert result["metadata"]["bm25_count"] == 0
    assert skill._bm25 is None, "dense 模式不应触发 BM25 索引构建"
    assert emb.calls == 1
    assert client.search_calls == 1


def test_sparse_mode_skips_embedding_call():
    """sparse 模式不该调 embedding —— 一次网络往返 + 计费，纯 BM25 用不上。"""
    skill, emb, client = _skill(_DOCS)
    result = skill.execute("糖尿病", k=5, mode="sparse")

    assert result["success"] is True
    assert result["metadata"]["requested"] == "sparse"
    assert emb.calls == 0, "sparse 模式不应调用 embedding"
    assert client.search_calls == 0, "sparse 模式不应走向量检索"
    assert result["metadata"]["bm25_count"] >= 1


def test_hybrid_mode_uses_both_paths():
    skill, emb, client = _skill(_DOCS)
    result = skill.execute("糖尿病", k=5, mode="hybrid")

    assert result["metadata"]["requested"] == "hybrid"
    assert emb.calls == 1
    assert client.search_calls == 1
    assert result["metadata"]["bm25_count"] >= 1


def test_invalid_mode_falls_back_to_hybrid():
    """配置写错不该让请求 500。降级到 hybrid 是安全的：它包含另外两路。"""
    skill, emb, client = _skill(_DOCS)
    result = skill.execute("糖尿病", k=5, mode="not-a-mode")

    assert result["success"] is True
    assert result["metadata"]["requested"] == "hybrid"


# ────────────────────────────────────────────────────────────────
# 四、Agent 层确实把 mode 传下去了
# ────────────────────────────────────────────────────────────────
class _RecordingSkill:
    name = "hybrid_search"

    def __init__(self) -> None:
        self.modes: list[str] = []

    def to_tool_definition(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    def execute(self, query: str = "", k: int = 0, **kwargs) -> dict:
        self.modes.append(kwargs.get("mode", "<missing>"))
        return {"success": True, "data": [], "error": None, "metadata": {"method": "fake"}}


class _FakeAnswer:
    name = "answer_generation"

    def to_tool_definition(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

    def execute(self, query: str = "", context: str = "", **kwargs) -> dict:
        return {"success": True, "data": "占位答案", "error": None, "metadata": {}}


def test_agent_passes_mode_to_skills():
    pytest.importorskip("langgraph")
    from agents.medical_agent import MedicalAgent

    qa = _RecordingSkill()
    pdf = _RecordingSkill()
    agent = MedicalAgent(qa, pdf, _FakeAnswer(), retriever_routing=True, max_retries=0)
    agent.run("什么是糖尿病")

    # 定义类在表里是 dense，两条检索路都该收到 dense
    assert qa.modes and all(m == "dense" for m in qa.modes)
    assert pdf.modes and all(m == "dense" for m in pdf.modes)


def test_agent_defaults_to_hybrid_without_switch():
    pytest.importorskip("langgraph")
    from agents.medical_agent import MedicalAgent

    qa = _RecordingSkill()
    pdf = _RecordingSkill()
    agent = MedicalAgent(qa, pdf, _FakeAnswer(), retriever_routing=False, max_retries=0)
    agent.run("什么是糖尿病")

    assert qa.modes and all(m == "hybrid" for m in qa.modes)
