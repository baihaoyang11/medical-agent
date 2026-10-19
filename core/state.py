"""Agent 状态定义 — LangGraph StateGraph 的核心数据结构。

相对旧版的改动：
- 增加 effective_query / used_queries / retry_count，支撑真正的二次检索
- 增加 merged_docs / quality / citations，让引用与质量判定有地方落
- emergency 单列，急症走安全通道，不进检索
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """Agent 全局状态。

    thinking_steps 用 Annotated[list, add] 声明 reducer：节点返回的列表会
    自动追加而不是覆盖，这样轨迹是完整累积的。
    """

    # 输入
    query: str
    effective_query: str

    # 路由
    question_type: str
    retrieval_strategy: str
    emergency: bool
    route_matched: List[str]

    # 二次检索控制
    retry_count: int
    used_queries: List[str]
    need_secondary_search: bool

    # 检索结果
    qa_docs: List[Dict[str, Any]]
    pdf_docs: List[Dict[str, Any]]
    merged_docs: List[Dict[str, Any]]
    final_context: str

    # 质量与引用
    quality: Dict[str, Any]
    citations: List[Dict[str, Any]]

    # 输出
    final_answer: str
    degraded: bool

    # 轨迹
    thinking_steps: Annotated[List[str], add]
