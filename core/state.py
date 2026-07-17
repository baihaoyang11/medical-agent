"""
Agent 状态定义 — LangGraph StateGraph 的核心数据结构

定义了 Agent 在各个节点之间传递的状态字段，
包括检索结果、思考步骤、最终答案等。
"""

from typing import TypedDict, List, Dict, Any, Annotated
from operator import add


class AgentState(TypedDict):
    """
    Agent 全局状态，在 LangGraph 各节点间流转

    使用 Annotated[list, add] 的字段会自动累加（而非覆盖），
    适合记录思考步骤这类需要追加的数据。
    """
    # 用户输入
    query: str

    # 检索结果
    qa_retrieved_docs: List[str]          # QA库检索到的文档内容
    pdf_retrieved_content: str            # PDF父文档召回的内容
    final_context: str                    # 拼接后的最终上下文

    # Agent 决策
    question_type: str                    # 问题类型: disease/drug/diagnosis/diet/prevention/general
    retrieval_strategy: str               # 检索策略: qa_only/pdf_only/hybrid
    need_secondary_search: bool           # 是否需要二次检索

    # 输出
    final_answer: str                     # 最终答案
    citations: List[Dict[str, str]]       # 引用来源列表

    # 思考链（面试展示用，使用 add reducer 自动累加）
    thinking_steps: Annotated[List[str], add]
